"""Precompute ``PolicyFeatureDataset``'s per-sample feature tensors to disk.

Motivation: ``dataset.py.__getitem__`` does real work on *every* call — an
Arrow row -> Python dict conversion, then ``build_observation()``'s backward
scans through the episode for ``decision_chain``/``opponent_history``, then
``transform()``'s tensor construction. ``bc_train.py`` already measured this
as the actual training bottleneck (CPU-bound Python per sample, not Parquet
I/O — see its ``train()`` docstring comment), and every epoch redoes the
exact same work for the exact same rows, since ``player_name`` filtering
selects a fixed sample set. This script runs that work once and caches the
already-transformed tensors, so a later epoch's data loading becomes "load a
small ``torch.save`` shard from disk" instead — the per-batch padding in
``collate.py`` still happens at train time (it genuinely depends on what's in
the batch), only the expensive per-sample assembly is cached.

Run it once per ``(parquet, player_name, spec)`` combination you intend to
train with; ``bc_train.py --precomputed-dir`` then reads the cache instead of
the raw Parquet file. The manifest records that combination so a mismatched
precompute directory fails loudly rather than silently feeding the wrong
config to training.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.utils.data

import flat_codec
from dataset import PolicyFeatureDataset, transform
from observation import DEFAULT_SPEC

#: Marks the on-disk layout so a cache written by an older/newer version of
#: this script fails loudly in ``PrecomputedPolicyFeatureDataset`` instead of
#: being silently misread.
#:
#: ``bin-v2`` replaced a sharded layout (``blob-v1``: 500 samples per
#: ``.pt`` file, read through an LRU cache) because training reads
#: *randomly* — ``DataLoader(shuffle=True)`` — while that layout only made
#: *sequential* reads cheap. Pulling one 506KB sample meant loading its whole
#: 253MB shard, a ~500x read amplification that measured 2.4 samples/s
#: shuffled vs 26 sequential, i.e. ~9 hours per epoch with the GPU idle.
#: One concatenated file plus an offset index makes a random read exactly one
#: ``pread`` of that sample's own bytes, and leaves caching to the OS page
#: cache instead of a hand-rolled one that had a ~0% hit rate under shuffle.
#:
#: ``bin-v3`` is the same *layout* as v2 but different *values*: options now
#: carry the card id they refer to (``features.resolve_option_card``), where
#: before every option's ``card_id`` was the 0 sentinel. Keys and shapes are
#: unchanged, so nothing downstream would raise on a v2 cache — it would just
#: silently train on zeroed card identity. Hence the bump: this guard is the
#: only thing that catches a values-only change.
#:
#: ``bin-v4`` keeps the layout *and the values* of v3 and changes only how one
#: sample's tensors are packed into its bytes: ``flat_codec`` instead of
#: ``torch.save``. Measured on the pretraining cache, v3 stored a sample whose
#: tensors hold 21 KB of numbers in a 269 KB blob (12.8x, all zip-record
#: overhead for ~600 tiny tensors) and spent 26 ms of CPU per sample parsing it
#: back — against a 115 ms GPU step for a whole batch of 64, which is why
#: training stayed loader-bound *after* the precompute removed
#: ``transform()``. v4 is ~12x smaller on disk (which also lets the page cache
#: hold a meaningful fraction of it) and decodes to ``torch.frombuffer`` views.
#: Repack an existing v3 cache with ``--repack`` — pure I/O, no feature rebuild.
#: ``bin-v5`` adds per-record zlib compression to v4's flat encoding — another
#: 52x on real records, because the payload is dominated by structural zeros
#: (padded card slots, a repeating 60-deep decision chain, one all-zero float
#: field worth a sixth of the bytes). The full pretraining cache goes ~130GB ->
#: ~2.5GB, which is the difference between reading it off disk every epoch and
#: having the page cache hold all of it. Decompression is 0.25ms/sample against
#: ~2.3ms for the rest of the decode, so it is close to free.
CACHE_FORMAT = "bin-v5"

#: The format ``--repack`` reads. Still supported by
#: ``PrecomputedPolicyFeatureDataset`` so an interrupted repack leaves you with
#: a cache you can still train from.
LEGACY_PICKLE_FORMAT = "bin-v3"

#: All sample blobs concatenated, addressed by ``manifest["offsets"]``.
BLOB_FILENAME = "blobs.bin"


#: Skeleton keys this worker has already shipped to the parent. A skeleton is
#: ~10 KB of JSON and there are only ~36 distinct ones (one per
#: ``(own bench count, opponent bench count)`` pair), so sending one per *sample*
#: would put more bytes on the IPC queue than the samples themselves. Sending it
#: once per worker per distinct shape costs at most 36 x num_workers messages.
_SENT_SKELETONS: set[str] = set()


def _serialize_collate(batch):
    """Serialize one sample to a single bytes blob — **in the worker process**.

    ``collate_fn`` runs worker-side, which is the whole point of doing it
    here. A transformed sample is a deeply nested tree of *thousands* of tiny
    tensors (one per card slot per flag, times a 60-deep decision chain).
    Returning that tree directly makes ``torch.multiprocessing`` hand every
    single tensor across the queue as its own shared-memory object, and at
    ``num_workers>1`` that exhausts a per-process kernel limit long before
    the run finishes — an FD limit under the default ``file_descriptor``
    sharing strategy ("received 0 items of ancdata"), or ``vm.max_map_count``
    under ``file_system`` ("unable to mmap 72 bytes ... Cannot allocate
    memory"). Both are the same root cause: too many shared objects, not too
    little memory.

    Collapsing the tree into one ``bytes`` blob means exactly one plain
    (non-shared-memory) object crosses per sample, so neither limit is
    approached regardless of worker count. The blob is also what gets stored
    in the shard, so the main process never has to deserialize it either.

    ``episode_id`` is returned alongside, unserialized, so building the
    manifest doesn't require unpacking the blob.

    Returns ``(episode_id, skeleton_key, skeleton_or_None, compressed_body)``.
    The template index is not part of the body: only the parent sees every
    sample, so only the parent can assign indexes, and by this point the body is
    already compressed (see ``_stream_records``). The skeleton itself rides along
    the first time this worker encounters its shape.
    """
    observation, target_action = batch[0]
    skeleton, body = flat_codec.encode(
        observation["features"], observation["meta"], target_action
    )
    key = hashlib.blake2b(skeleton.encode(), digest_size=8).hexdigest()
    first_time = key not in _SENT_SKELETONS
    _SENT_SKELETONS.add(key)
    return (
        observation["meta"]["episode_id"],
        key,
        skeleton if first_time else None,
        body,
    )


def precompute(
    parquet_path: str,
    out_dir: str,
    player_name=None,
    shard_size: int = 2000,
    num_workers: int = 4,
    cached_row_groups: int = 16,
    opponent_history_size: int | None = None,
    decision_chain_size: int | None = None,
    limit: int | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = PolicyFeatureDataset(
        parquet_path, transform=transform, player_name=player_name,
        cached_row_groups=cached_row_groups,
        opponent_history_size=opponent_history_size,
        decision_chain_size=decision_chain_size,
    )
    if limit is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(limit, len(dataset))))
    print(f"{len(dataset)} samples to precompute", flush=True)

    # num_workers>0 parallelizes the CPU-bound build_observation()/transform()
    # work across subprocesses — the same fix bc_train.py's DataLoader uses,
    # just applied once here instead of on every epoch. shuffle=False keeps
    # results in dataset order (PyTorch's DataLoader preserves ordering across
    # workers when not shuffling), so shard files line up with the manifest's
    # episode_ids by plain sample index.
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=_serialize_collate,
    )

    episode_ids, offsets, templates = _stream_records(loader, out, len(dataset))
    _write_manifest(
        out, episode_ids, offsets, player_name, parquet_path,
        opponent_history_size, decision_chain_size, templates=templates,
    )


def _stream_records(loader, out: Path, total: int) -> tuple[list, list[int], list]:
    """Drain a ``_serialize_collate`` loader into ``out/blobs.bin``.

    Returns ``(episode_ids, offsets, templates)`` for the manifest. The template
    registry is built here rather than in the workers because index assignment
    has to be global; it is prepended to each compressed body on the way past,
    which is why the index sits outside the compressed region.
    """
    episode_ids = []
    # Cumulative byte offsets, so sample i occupies [offsets[i], offsets[i+1]).
    # One entry longer than the sample count; no separate length array needed.
    offsets = [0]
    template_index: dict[str, int] = {}
    templates: list = []
    t0 = time.time()

    with open(out / BLOB_FILENAME, "wb") as blobs:
        for i, (episode_id, key, skeleton, body) in enumerate(loader):
            index = template_index.get(key)
            if index is None:
                if skeleton is None:
                    # Only reachable if a worker's dedup memo and the parent's
                    # registry disagree, which would mean the record points at
                    # a template that was never stored.
                    raise RuntimeError(
                        f"sample {i} references unknown feature-shape {key} and "
                        f"carried no skeleton"
                    )
                index = template_index[key] = len(templates)
                templates.append(json.loads(skeleton))
            record = flat_codec.pack(index, body)

            episode_ids.append(episode_id)
            blobs.write(record)
            offsets.append(offsets[-1] + len(record))
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                done = i + 1
                rate = done / elapsed
                print(
                    f"  {done}/{total} ({elapsed:.0f}s, {rate:.1f} samples/s, "
                    f"{offsets[-1] / 1e9:.1f}GB written, "
                    f"{len(templates)} feature shapes, "
                    f"eta {(total - done) / max(rate, 1e-9) / 60:.0f}min, "
                    f"~{offsets[-1] / done * total / 1e9:.0f}GB total)",
                    flush=True,
                )

    print(
        f"wrote {len(episode_ids)} samples, {offsets[-1] / 1e9:.1f}GB to "
        f"{out / BLOB_FILENAME} in {time.time() - t0:.0f}s",
        flush=True,
    )
    return episode_ids, offsets, templates


def _write_manifest(
    out: Path, episode_ids: list, offsets: list[int], player_name,
    parquet_path, opponent_history_size, decision_chain_size,
    templates: list | None = None, cache_format: str = CACHE_FORMAT,
) -> None:
    manifest = {
        "format": cache_format,
        # The per-shape feature skeletons every record's ``sig`` field indexes
        # into (see flat_codec). Absent for the legacy pickled format.
        "templates": templates,
        "num_samples": len(episode_ids),
        "offsets": offsets,
        "episode_ids": episode_ids,
        "player_name": player_name,
        "parquet_path": str(parquet_path),
        "opponent_history_size": opponent_history_size or DEFAULT_SPEC.opponent_history_size,
        "decision_chain_size": decision_chain_size or DEFAULT_SPEC.decision_chain_size,
    }
    # Written last, so an interrupted run leaves no manifest and therefore no
    # cache that looks complete but isn't.
    (out / "manifest.json").write_text(json.dumps(manifest))


def convert_from_shards(cache_dir: str) -> None:
    """Rewrite an existing ``blob-v1`` sharded cache into the ``bin-v2``
    single-file layout, in place.

    Pure I/O — the expensive ``build_observation()``/``transform()`` work is
    already baked into the stored blobs, so this re-lays them out rather than
    recomputing them (minutes instead of re-running the whole precompute).
    Needs free space equal to the existing cache, since the shards are left
    untouched until you delete them.
    """
    directory = Path(cache_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != "blob-v1":
        raise SystemExit(
            f"{directory} has format {manifest.get('format')!r}; --from-shards "
            f"converts a 'blob-v1' sharded cache"
        )
    shards = manifest["shards"]
    print(f"converting {len(shards)} shards ({manifest['num_samples']} samples)", flush=True)

    offsets = [0]
    t0 = time.time()
    with open(directory / BLOB_FILENAME, "wb") as blobs:
        for shard_idx, name in enumerate(shards):
            for blob in torch.load(directory / name, weights_only=False):
                blobs.write(blob)
                offsets.append(offsets[-1] + len(blob))
            if (shard_idx + 1) % 20 == 0:
                print(
                    f"  {shard_idx + 1}/{len(shards)} shards, "
                    f"{offsets[-1] / 1e9:.1f}GB ({time.time() - t0:.0f}s)",
                    flush=True,
                )

    if len(offsets) - 1 != manifest["num_samples"]:
        raise SystemExit(
            f"recovered {len(offsets) - 1} samples but the manifest claims "
            f"{manifest['num_samples']} — refusing to overwrite it"
        )
    # Still the pickled per-sample format — this only re-lays-out the blobs it
    # was handed. Run --repack afterwards to get the current format.
    _write_manifest(
        directory, manifest["episode_ids"], offsets, manifest["player_name"],
        manifest["parquet_path"], manifest["opponent_history_size"],
        manifest["decision_chain_size"], cache_format=LEGACY_PICKLE_FORMAT,
    )
    print(
        f"wrote {directory / BLOB_FILENAME} ({offsets[-1] / 1e9:.1f}GB) in "
        f"{time.time() - t0:.0f}s\nthe old shards are now unused — reclaim them with:\n"
        f"  rm {directory}/shard_*.pt",
        flush=True,
    )


def repack(src_dir: str, out_dir: str, num_workers: int = 8) -> None:
    """Rewrite a ``bin-v3`` (pickled-per-sample) cache into ``bin-v4``.

    Like ``convert_from_shards``, this does **not** rebuild features: the
    expensive ``build_observation()``/``transform()`` work is already baked into
    the stored samples, so this just decodes each one and re-encodes it with
    ``flat_codec``. Values are bit-identical and sample order is preserved, so
    the episode-level train/val split for a given ``--seed`` is unchanged and an
    in-progress run can carry straight on against the new directory with
    ``bc_train.py --resume``.

    Writes to a *separate* ``out_dir`` and leaves the source untouched, so a
    failed or interrupted repack costs nothing but disk. Delete the source once
    you have trained a batch against the result.

    The wall-clock cost is dominated by unpickling the old blobs (~26 ms of CPU
    per sample), which is why this fans out over ``num_workers`` — reads are
    sequential, so the I/O side is not the constraint.
    """
    from precomputed_dataset import PrecomputedPolicyFeatureDataset

    source = Path(src_dir)
    out = Path(out_dir)
    if out.resolve() == source.resolve():
        raise SystemExit(
            "--repack writes a new cache; point --out at a different directory "
            "so an interrupted run cannot destroy the source"
        )
    out.mkdir(parents=True, exist_ok=True)

    dataset = PrecomputedPolicyFeatureDataset(source, warn_legacy=False)
    manifest = dataset.manifest
    print(
        f"repacking {len(dataset)} samples from {source} "
        f"(format {manifest['format']}) into {out}",
        flush=True,
    )
    # shuffle=False so records land in source order and the manifest's
    # episode_ids/sample indexes keep meaning the same thing.
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=_serialize_collate,
    )
    episode_ids, offsets, templates = _stream_records(loader, out, len(dataset))

    if episode_ids != list(manifest["episode_ids"]):
        raise SystemExit(
            "repacked episode_ids do not match the source manifest — refusing to "
            "write a manifest whose split would differ from the original's"
        )
    _write_manifest(
        out, episode_ids, offsets, manifest["player_name"], manifest["parquet_path"],
        manifest["opponent_history_size"], manifest["decision_chain_size"],
        templates=templates,
    )
    before = manifest["offsets"][-1]
    print(
        f"{before / 1e9:.1f}GB -> {offsets[-1] / 1e9:.1f}GB "
        f"({before / max(offsets[-1], 1):.1f}x smaller). Train against {out}, "
        f"then reclaim the old cache with:\n  rm -r {source}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="data/policy_decisions_pretraining.parquet")
    parser.add_argument(
        "--player-name", default="",
        help="restrict precomputed samples to this player; the default (empty) "
             "precomputes every player's decisions, which is what the "
             "pretraining stage trains on",
    )
    parser.add_argument("--out", default="data/precomputed_features_pretraining")
    parser.add_argument(
        "--from-shards", default=None, metavar="CACHE_DIR",
        help="skip feature building: convert an existing 'blob-v1' sharded "
             "cache in CACHE_DIR to the current single-file layout in place. "
             "Pure I/O, so this is much faster than rebuilding, and it needs "
             "free space equal to the cache until you delete the old shards.",
    )
    parser.add_argument(
        "--shard-size", type=int, default=500,
        help="unused; kept so existing invocations keep working. Samples now "
             "stream into one file addressed by an offset index, so there are "
             "no shards to size.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="parallel subprocesses building observations — this is a one-time "
             "cost, so it's worth spending more workers here than you would "
             "during training",
    )
    parser.add_argument("--cached-row-groups", type=int, default=16)
    parser.add_argument("--opponent-history-size", type=int, default=None)
    parser.add_argument("--decision-chain-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap dataset size (debug)")
    parser.add_argument(
        "--repack", default=None, metavar="CACHE_DIR",
        help="skip feature building: re-encode an existing 'bin-v3' cache in "
             "CACHE_DIR into the current format, writing to --out. ~12x smaller "
             "and ~20x cheaper to decode (see CACHE_FORMAT), values identical, "
             "so a run in progress can --resume straight onto the result.",
    )
    args = parser.parse_args()

    if args.repack:
        repack(args.repack, args.out, num_workers=args.num_workers)
        raise SystemExit(0)

    if args.from_shards:
        convert_from_shards(args.from_shards)
        raise SystemExit(0)

    precompute(
        args.parquet, args.out, player_name=(args.player_name or None),
        shard_size=args.shard_size, num_workers=args.num_workers,
        cached_row_groups=args.cached_row_groups,
        opponent_history_size=args.opponent_history_size,
        decision_chain_size=args.decision_chain_size,
        limit=args.limit,
    )
