"""Concatenate policy-decision parquets, keeping one copy of each episode.

Why this exists: the replay tree and the parquet are not interchangeable
histories of the same games. ``data/episodes/`` (the daily Kaggle dumps) and
whatever ``replays/`` tree an older parquet was built from overlap only
partially — measured here, ``data/episodes`` held 26321 episodes of which just
656 were in ``policy_decisions_pretraining.parquet``, while that parquet held
3694 episodes present in no surviving replay directory. Re-running
``convert_replays.py`` over the new dump alone would therefore *lose* those
3694, and pointing it at both is impossible once the original JSON is gone. So
the merge happens at the parquet level, where every row still exists.

Dedup is by ``episode_id``, not by row, and **whole episodes** are taken from
exactly one input. That matters for two reasons:

- ``episode_split`` (bc_train.py) partitions train/val by ``episode_id``, so a
  duplicated episode lands wholly on one side and would silently get double
  weight in the loss — a sampling bias nobody chose.
- Row-level dedup would be wrong anyway: the same decision converted by two
  script versions can differ in a column while describing one real event, and
  keeping both halves of that pair invents a contradiction in the labels.

Earlier inputs win. Order the arguments accordingly: pass the parquet you
trust most first.

Streaming, one row group at a time, because a 4.4M-row parquet at 102 columns
does not want to be resident all at once — and the whole point of the merge is
that the result is bigger than either input.

Usage::

    python archetypes/pretraining/policy_network/merge_parquets.py \\
        data/policy_decisions_pretraining.parquet \\
        data/policy_decisions_episodes.parquet \\
        --out data/policy_decisions_pretraining_v2.parquet
"""

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def episode_ids(path: Path) -> set:
    """Every ``episode_id`` in ``path``, read as one column rather than one
    pass over the whole file — 102 columns of row data are irrelevant here."""
    table = pq.ParquetFile(path).read(columns=["episode_id"])
    return set(table.column("episode_id").to_pylist())


def merge(sources: list[Path], out: Path, compression: str = "zstd") -> None:
    if len(sources) < 2:
        raise SystemExit("pass at least two parquets to merge")

    readers = [pq.ParquetFile(path) for path in sources]
    schema = readers[0].schema_arrow
    for path, reader in zip(sources[1:], readers[1:]):
        if not reader.schema_arrow.equals(schema):
            raise SystemExit(
                f"{path} has a different schema from {sources[0]} — they were "
                f"written by different convert_replays.py versions, and "
                f"concatenating them would misalign columns. Re-convert one."
            )

    # Which episodes each source is responsible for: earlier sources win, so a
    # later one only contributes episodes nothing before it had.
    claimed: set = set()
    plans = []
    for path in sources:
        ids = episode_ids(path)
        plans.append(ids - claimed)
        claimed |= ids
    for path, plan in zip(sources, plans):
        print(f"{path}: {len(plan)} episode(s) kept", flush=True)

    t0 = time.time()
    written = 0
    writer = pq.ParquetWriter(out, schema, compression=compression)
    try:
        for path, reader, keep in zip(sources, readers, plans):
            if not keep:
                print(f"{path}: nothing to take, skipped", flush=True)
                continue
            for index in range(reader.num_row_groups):
                group = reader.read_row_group(index)
                # A row group spans episodes, so filter within it rather than
                # taking or dropping the group wholesale.
                mask = pc.is_in(group.column("episode_id"), value_set=pa.array(list(keep)))
                filtered = group.filter(mask)
                if filtered.num_rows:
                    writer.write_table(filtered)
                    written += filtered.num_rows
            print(
                f"{path}: done, {written} row(s) written so far "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    finally:
        writer.close()

    total_episodes = len(claimed)
    print(
        f"wrote {written} rows from {total_episodes} episode(s) to {out} "
        f"({out.stat().st_size / 1e9:.1f}GB, {time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="parquets to merge, best first")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--compression", default="zstd",
        choices=("snappy", "zstd", "gzip", "brotli", "lz4", "none"),
    )
    args = parser.parse_args()
    merge(args.sources, args.out, args.compression)
