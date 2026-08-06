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
import io
import json
import time
from pathlib import Path

import torch
import torch.utils.data

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
CACHE_FORMAT = "bin-v3"

#: All sample blobs concatenated, addressed by ``manifest["offsets"]``.
BLOB_FILENAME = "blobs.bin"


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
    """
    observation, target_action = batch[0]
    buffer = io.BytesIO()
    torch.save((observation["features"], observation["meta"], target_action), buffer)
    return observation["meta"]["episode_id"], buffer.getvalue()


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

    episode_ids = []
    # Cumulative byte offsets, so sample i occupies [offsets[i], offsets[i+1]).
    # One entry longer than the sample count; no separate length array needed.
    offsets = [0]
    t0 = time.time()

    with open(out / BLOB_FILENAME, "wb") as blobs:
        for i, (episode_id, blob) in enumerate(loader):
            episode_ids.append(episode_id)
            blobs.write(blob)
            offsets.append(offsets[-1] + len(blob))
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                done = i + 1
                rate = done / elapsed
                print(
                    f"  {done}/{len(dataset)} ({elapsed:.0f}s, {rate:.1f} samples/s, "
                    f"{offsets[-1] / 1e9:.1f}GB written, "
                    f"eta {(len(dataset) - done) / max(rate, 1e-9) / 60:.0f}min, "
                    f"~{offsets[-1] / done * len(dataset) / 1e9:.0f}GB total)",
                    flush=True,
                )

    _write_manifest(
        out, episode_ids, offsets, player_name, parquet_path,
        opponent_history_size, decision_chain_size,
    )
    print(
        f"wrote {len(episode_ids)} samples, {offsets[-1] / 1e9:.1f}GB to "
        f"{out / BLOB_FILENAME} in {time.time() - t0:.0f}s",
        flush=True,
    )


def _write_manifest(
    out: Path, episode_ids: list, offsets: list[int], player_name,
    parquet_path, opponent_history_size, decision_chain_size,
) -> None:
    manifest = {
        "format": CACHE_FORMAT,
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
    _write_manifest(
        directory, manifest["episode_ids"], offsets, manifest["player_name"],
        manifest["parquet_path"], manifest["opponent_history_size"],
        manifest["decision_chain_size"],
    )
    print(
        f"wrote {directory / BLOB_FILENAME} ({offsets[-1] / 1e9:.1f}GB) in "
        f"{time.time() - t0:.0f}s\nthe old shards are now unused — reclaim them with:\n"
        f"  rm {directory}/shard_*.pt",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="data/policy_decisions_lucario.parquet")
    parser.add_argument(
        "--player-name", default="Majkel1337",
        help="restrict precomputed samples to this player (pass an empty string "
             "to precompute every player's decisions)",
    )
    parser.add_argument("--out", default="data/precomputed_features_lucario")
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
    args = parser.parse_args()

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
