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

from precompute_features import BLOB_FILENAME, CACHE_FORMAT


class PrecomputedPolicyFeatureDataset(Dataset):
    def __init__(self, precomputed_dir: str | Path, cached_shards: int | None = None) -> None:
        """``cached_shards`` is obsolete and ignored — kept so existing callers
        keep working. There are no shards to cache now; the OS page cache
        handles reuse."""
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
        if found != CACHE_FORMAT:
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
        blob = os.pread(self._fd, end - start, start)
        # weights_only=False: the blob holds a plain tuple of dicts, which
        # torch's default weights_only=True path (aimed at untrusted model
        # checkpoints) rejects. This is a pickle read of already-built
        # tensors, not the build_observation()/transform() work the cache
        # exists to avoid.
        features, meta, target_action = torch.load(io.BytesIO(blob), weights_only=False)
        return {"features": features, "meta": meta}, target_action

    def __del__(self):
        fd = getattr(self, "_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
