"""Dataset that reads ``precompute_features.py``'s cache, instead of
recomputing ``build_observation()``/``transform()`` from raw Parquet on every
``__getitem__`` call the way ``PolicyFeatureDataset`` does — see that script's
docstring for why this exists. Drop-in for ``PolicyFeatureDataset`` from
``bc_train.py``'s point of view: same ``__getitem__``/``episode_ids()``
shape, just backed by the cache file rather than a live Parquet reader.

Random access is the case that matters here, since training shuffles: each
``__getitem__`` is one ``pread`` of exactly that sample's bytes out of the
concatenated blob file, located via the manifest's offset index. See
``precompute_features.CACHE_FORMAT`` for what the earlier sharded layout cost
under shuffle."""

import io
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

import flat_codec
from precompute_features import BLOB_FILENAME, CACHE_FORMAT, LEGACY_PICKLE_FORMAT


class PrecomputedPolicyFeatureDataset(Dataset):
    def __init__(
        self,
        precomputed_dir: str | Path,
        cached_shards: int | None = None,
        warn_legacy: bool = True,
    ) -> None:
        """``cached_shards`` is obsolete and ignored — kept so existing callers
        keep working. There are no shards to cache now; the OS page cache
        handles reuse.

        ``warn_legacy=False`` suppresses the "this cache is in the slow format"
        notice. Only ``precompute_features.repack`` should pass it: repack opens
        a legacy cache *in order to* re-encode it, so telling it to go run a
        repack is advice to run the command already running."""
        self.dir = Path(precomputed_dir)
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"no manifest.json in {self.dir} — run precompute_features.py first"
            )
        self.manifest = json.loads(manifest_path.read_text())
        # A cache written before the feature schema changed would otherwise
        # surface as a confusing KeyError deep inside the model's forward pass.
        found = self.manifest.get("format")
        if found not in (CACHE_FORMAT, LEGACY_PICKLE_FORMAT):
            hint = (
                f"convert it in place (fast, no feature rebuild) with:\n"
                f"  python precompute_features.py --from-shards {self.dir}"
                if found == "blob-v1"
                else "re-run precompute_features.py to rebuild it"
            )
            raise ValueError(
                f"{self.dir} was written in format {found!r}, but this code "
                f"expects {CACHE_FORMAT!r} — {hint}"
            )
        # ``_templates is None`` is what selects the decode path in
        # __getitem__, so it is keyed off the *format* alone. warn_legacy only
        # silences the notice — conflating the two made a repack (which opens a
        # legacy cache deliberately, with the warning off) demand templates that
        # a legacy cache does not have.
        self._templates: list[flat_codec.Template] | None = None
        if found == LEGACY_PICKLE_FORMAT:
            # A v3 cache still trains, it is just ~12x more CPU per sample to
            # decode, which on this dataset is the difference between a
            # GPU-bound run and a 3-hour epoch. Loud, because the fix is cheap
            # and one-time.
            if warn_legacy:
                print(
                    f"WARNING: {self.dir} is in the legacy {LEGACY_PICKLE_FORMAT!r} "
                    f"format (one torch.save archive per sample, ~26ms of CPU each — "
                    f"training will be loader-bound). Re-encode it with:\n"
                    f"  python precompute_features.py --repack {self.dir} "
                    f"--out {self.dir}_v4 --num-workers 10",
                    flush=True,
                )
        else:
            templates = self.manifest.get("templates")
            if not templates:
                raise ValueError(
                    f"{self.dir}/manifest.json has format {found!r} but no "
                    f"feature-shape templates — the records cannot be decoded "
                    f"without them; re-run the precompute or repack"
                )
            self._templates = [flat_codec.Template(t) for t in templates]
        self._num_samples: int = self.manifest["num_samples"]
        self._episode_ids: list[Any] = self.manifest["episode_ids"]
        self._offsets: list[int] = self.manifest["offsets"]
        self._blob_path = self.dir / BLOB_FILENAME
        if not self._blob_path.is_file():
            raise FileNotFoundError(f"{self._blob_path} is missing from the cache")
        # Opened once and read with os.pread, which takes an explicit offset
        # and does not touch the shared file position — so the descriptor is
        # safe to inherit across DataLoader worker forks. A seek+read pair
        # would race between workers on the same fd.
        self._fd = os.open(self._blob_path, os.O_RDONLY)
        # No application-level cache on purpose: reads are exact-sized, and
        # the OS page cache already handles reuse better than the LRU that
        # used to live here (which had a ~0% hit rate under shuffle).

    def __len__(self) -> int:
        return self._num_samples

    def episode_ids(self) -> list[Any]:
        """Same contract as ``PolicyFeatureDataset.episode_ids()`` — the
        episode id of every sample, in sample-index order, recorded once at
        precompute time rather than re-read from Parquet."""
        return list(self._episode_ids)

    def __getitem__(self, idx: int) -> tuple[dict, torch.Tensor]:
        if not 0 <= idx < len(self):
            raise IndexError(f"Dataset index out of range: {idx}")
        start, end = self._offsets[idx], self._offsets[idx + 1]
        if self._templates is None:
            blob = os.pread(self._fd, end - start, start)
            # weights_only=False: the blob holds a plain tuple of dicts, which
            # torch's default weights_only=True path (aimed at untrusted model
            # checkpoints) rejects. This is a pickle read of already-built
            # tensors, not the build_observation()/transform() work the cache
            # exists to avoid.
            features, meta, target_action = torch.load(io.BytesIO(blob), weights_only=False)
            return {"features": features, "meta": meta}, target_action

        # unpack decompresses into a fresh (writable) bytearray, which is what
        # decode's tensors end up as views into.
        sig_index, body = flat_codec.unpack(os.pread(self._fd, end - start, start))
        features, meta, target_action = flat_codec.decode(body, self._templates[sig_index])
        return {"features": features, "meta": meta}, target_action

    def __del__(self):
        fd = getattr(self, "_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
