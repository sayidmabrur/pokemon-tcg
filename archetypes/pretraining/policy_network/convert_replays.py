"""Convert Kaggle Pokémon TCG replay JSON files into policy-decision Parquet.

Usage (pretraining copy — the whole replay tree, every player kept):
    python archetypes/pretraining/policy_network/convert_replays.py

Both arguments default to the only pair this stage ever wants: ``replays/``
in, ``data/policy_decisions_pretraining.parquet`` out. Replays live one level
down, as ``replays/<submission id>/episode-<id>-replay.json``, and the source
is walked recursively — so the default sweeps every submission directory in
one pass and picks up new ones with no change to the command.

Converting a single ``replays/<submission id>/`` is possible but rarely what
you want: a submission directory contains only the games *that* submission
played, so it is a slice of the field rather than the field. The overlap
between those directories is also why ``replay_paths`` deduplicates — an
episode is listed under every submission that took part in it.

Nothing here filters by player or by decklist: the pretraining stage wants
both seats of every game (see bc_train.py's docstring). The per-archetype
fine-tune is what narrows to one expert, and it does so at training time via
``--player-name``, not here.

The replay stores an action beside the *resulting* observation.  This script
therefore pairs each active selection with the next action from that player.
Initial 60-card deck submissions and inactive/stale observations are excluded.
If any decision in an episode fails to pair with a valid action, the entire
episode is dropped rather than only the offending decision, since a broken
pairing indicates the episode's causal chain can't be trusted.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


import sys

sys.path.insert(0, str(Path(__file__).parent))

from features import OPTION_FIELDS, SELECTION_FIELDS  # noqa: E402


def _card_schema(pa: Any) -> Any:
    return pa.struct([
        pa.field("id", pa.int32()),
        pa.field("playerIndex", pa.int8()),
        pa.field("serial", pa.int32()),
    ])


def make_schema(pa: Any) -> Any:
    """Explicit schema: Arrow inference would lose option fields absent first."""
    card = _card_schema(pa)
    pokemon = pa.struct([
        pa.field("id", pa.int32()), pa.field("playerIndex", pa.int8()),
        pa.field("serial", pa.int32()), pa.field("hp", pa.int16()),
        pa.field("maxHp", pa.int16()), pa.field("appearThisTurn", pa.bool_()),
        pa.field("energies", pa.list_(pa.int8())),
        pa.field("energyCards", pa.list_(card)), pa.field("tools", pa.list_(card)),
        pa.field("preEvolution", pa.list_(card)),
    ])
    player = pa.struct([
        pa.field("active", pa.list_(pokemon)), pa.field("bench", pa.list_(pokemon)),
        pa.field("benchMax", pa.int8()), pa.field("deckCount", pa.int16()),
        pa.field("discard", pa.list_(card)), pa.field("prize", pa.list_(card)),
        pa.field("handCount", pa.int16()), pa.field("hand", pa.list_(card)),
        pa.field("poisoned", pa.bool_()), pa.field("burned", pa.bool_()),
        pa.field("asleep", pa.bool_()), pa.field("paralyzed", pa.bool_()),
        pa.field("confused", pa.bool_()),
    ])
    state = pa.struct([
        pa.field("energyAttached", pa.bool_()), pa.field("firstPlayer", pa.int8()),
        pa.field("looking", pa.list_(card)), pa.field("players", pa.list_(player)),
        pa.field("result", pa.int8()), pa.field("retreated", pa.bool_()),
        pa.field("stadium", pa.list_(card)), pa.field("stadiumPlayed", pa.bool_()),
        pa.field("supporterPlayed", pa.bool_()), pa.field("turn", pa.int16()),
        pa.field("turnActionCount", pa.int16()), pa.field("yourIndex", pa.int8()),
    ])
    option = pa.struct([
        pa.field("type", pa.int8()), pa.field("number", pa.int16()),
        pa.field("area", pa.int8()), pa.field("index", pa.int16()),
        pa.field("playerIndex", pa.int8()), pa.field("toolIndex", pa.int8()),
        pa.field("energyIndex", pa.int8()), pa.field("count", pa.int8()),
        pa.field("inPlayArea", pa.int8()), pa.field("inPlayIndex", pa.int8()),
        pa.field("attackId", pa.int32()), pa.field("cardId", pa.int32()),
        pa.field("serial", pa.int32()), pa.field("specialConditionType", pa.int8()),
    ])
    selection = pa.struct([
        pa.field("type", pa.int8()), pa.field("context", pa.int16()),
        pa.field("minCount", pa.int8()), pa.field("maxCount", pa.int8()),
        pa.field("remainDamageCounter", pa.int16()), pa.field("remainEnergyCost", pa.int16()),
        pa.field("deck", pa.list_(card)), pa.field("contextCard", card), pa.field("effect", card),
    ])
    return pa.schema([
        pa.field("episode_id", pa.int64()), pa.field("frame_index", pa.int32()),
        pa.field("player_index", pa.int8()), pa.field("player_name", pa.string()),
        pa.field("step", pa.int32()),
        pa.field("remaining_overage_time", pa.float32()), pa.field("state", state),
        pa.field("selection", selection), pa.field("options", pa.list_(option)),
        pa.field("target_action", pa.list_(pa.int16())), pa.field("reward", pa.float32()),
        pa.field("final_result", pa.int8()),
    ])


def normalise_option(option: dict[str, Any]) -> dict[str, Any]:
    return {field: option.get(field) for field in OPTION_FIELDS}


def is_decision(record: dict[str, Any], player_index: int) -> bool:
    observation = record.get("observation", {})
    current = observation.get("current")
    return (
        record.get("status") == "ACTIVE"
        and observation.get("select") is not None
        and current is not None
        and current.get("yourIndex") == player_index
    )


def is_valid_target(action: Any, selection: dict[str, Any], option_count: int) -> bool:
    if not isinstance(action, list):
        return False
    if not selection["minCount"] <= len(action) <= selection["maxCount"]:
        return False
    if len(set(action)) != len(action):
        return False
    return all(isinstance(index, int) and 0 <= index < option_count for index in action)


def episode_result(frames: list[list[dict[str, Any]]]) -> int:
    for frame in reversed(frames):
        for record in frame:
            current = record.get("observation", {}).get("current")
            if current is not None and current.get("result", -1) != -1:
                return current["result"]
    return -1


def decision_rows(replay_path: Path) -> tuple[list[dict[str, Any]], int]:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    frames = replay["steps"]
    pending: list[dict[str, Any] | None] = [None, None]
    rows: list[dict[str, Any]] = []
    rejected = 0
    result = episode_result(frames)
    episode_id = int(replay.get("info", {}).get("EpisodeId", -1))
    team_names = replay.get("info", {}).get("TeamNames", [])

    for frame_index, frame in enumerate(frames):
        for player_index, record in enumerate(frame):
            # The current action answers this player's previous active selection.
            previous = pending[player_index]
            if previous is not None:
                action = record.get("action")
                if is_valid_target(action, previous["selection"], len(previous["options"])):
                    rows.append({
                        **previous,
                        "target_action": action,
                        "reward": float(record.get("reward", 0.0)),
                        "final_result": result,
                    })
                else:
                    rejected += 1
                pending[player_index] = None

            if is_decision(record, player_index):
                observation = record["observation"]
                selection = observation["select"]
                pending[player_index] = {
                    "episode_id": episode_id,
                    "frame_index": frame_index,
                    "player_index": player_index,
                    "player_name": team_names[player_index] if player_index < len(team_names) else None,
                    "step": observation.get("step"),
                    "remaining_overage_time": observation.get("remainingOverageTime"),
                    "state": observation["current"],
                    "selection": {key: selection.get(key) for key in SELECTION_FIELDS},
                    "options": [normalise_option(option) for option in selection["option"]],
                }
    return rows, rejected


def replay_paths(source: Path) -> Iterable[Path]:
    """Every replay under ``source``, at most once per episode.

    ``rglob`` so a root like ``replays/`` sweeps all the per-submission
    subdirectories at once, which is the normal way to build the pretraining
    parquet.

    The dedup matters because those subdirectories overlap: an episode is
    listed under *every* submission that took part in it, so downloading two
    of your own submissions fetches the games between them twice. Measured on
    the current tree, 3715 files hold 3566 distinct episodes — 149 duplicates.
    Writing both copies would not corrupt the train/val split (``episode_split``
    keys on ``episode_id``, so the copies land on the same side) but it would
    silently give those episodes double weight in the loss, which is a
    sampling bias nobody asked for.

    Keyed on the ``episode-<id>-replay.json`` filename rather than on the
    parsed ``EpisodeId``, so duplicates cost nothing to detect — the point is
    to skip the parse, and a 15GB tree makes that worth doing. Files not
    matching that pattern are always kept: an unrecognised name is not
    evidence of a duplicate.
    """
    if source.is_file():
        yield source
        return

    seen: set[str] = set()
    for path in sorted(source.rglob("*.json")):
        # Two layouts in the wild: ``replays/<submission>/episode-<id>-replay.json``
        # from a per-submission download, and ``data/episodes/<date>/<id>.json``
        # from the daily Kaggle dumps. Both name the episode in the filename, so
        # both can be deduplicated without parsing. The daily dumps do not
        # currently repeat an episode across dates (checked: 26321 files, 26321
        # distinct ids), but they overlap by construction if you re-download a
        # date, and double-weighting an episode in the loss is silent.
        match = re.fullmatch(r"episode-(\d+)-replay\.json|(\d+)\.json", path.name)
        if match is not None:
            match = re.match(r"(\d+)", match.group(1) or match.group(2))
        if match is not None:
            if match.group(1) in seen:
                continue
            seen.add(match.group(1))
        yield path


def convert(
    source: Path,
    destination: Path,
    batch_size: int = 5_000,
    compression: str = "snappy",
) -> tuple[int, int, int]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Conversion requires pyarrow. Install it with `python -m pip install pyarrow`.") from exc

    paths = list(replay_paths(source))
    if not paths:
        raise FileNotFoundError(f"No replay JSON files found in {source}")

    schema = make_schema(pa)
    batch: list[dict[str, Any]] = []
    dropped_episodes = 0
    unreadable: list[Path] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(destination, schema, compression=compression)
    written = 0
    try:
        for path in paths:
            try:
                episode_rows, episode_rejected = decision_rows(path)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
                # A replay that will not parse is almost always a truncated
                # download, not a bug: the daily dumps are 20GB each and an
                # interrupted transfer leaves a file that looks complete until
                # the last few bytes. Measured on the first full tree, exactly
                # 1 of 26321 files was affected — so aborting the whole run for
                # it would throw away 6 hours of work over 0.004% of the data.
                # Skipped and *counted*, because a silently shrinking dataset is
                # the failure mode this would otherwise become: if a re-download
                # goes badly wrong, the tally at the end is what tells you.
                unreadable.append(path)
                print(f"  skipping unreadable {path}: {error}", flush=True)
                continue
            if episode_rejected:
                # A rejected pairing means this episode's action/observation
                # sequence is broken; keeping its other decisions would bias
                # the dataset with states whose causal chain can't be trusted.
                dropped_episodes += 1
                continue
            batch.extend(episode_rows)
            if len(batch) >= batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                written += len(batch)
                batch.clear()

        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            written += len(batch)
        elif written == 0:
            writer.write_table(pa.Table.from_pylist([], schema=schema))
    finally:
        writer.close()
    return len(paths), written, dropped_episodes, unreadable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Both positionals are optional here, and the defaults are the whole
    # pipeline: every replay on disk -> the parquet the pretraining stage
    # reads. Naming one ``replays/<submission id>/`` is the thing you almost
    # never want, because a submission directory holds only the games that
    # submission played — converting one in isolation silently trains on a
    # slice of the field rather than all of it.
    parser.add_argument(
        "source", type=Path, nargs="?", default=Path("replays"),
        help="replay JSON file, or a directory walked recursively "
             "(default: %(default)s — every submission directory at once)",
    )
    parser.add_argument(
        "destination", type=Path, nargs="?",
        default=Path("data/policy_decisions_pretraining.parquet"),
        help="output .parquet file (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=5_000, help="Rows buffered before writing")
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=("snappy", "zstd", "gzip", "brotli", "lz4", "none"),
        help="Parquet compression codec",
    )
    args = parser.parse_args()
    compression = None if args.compression == "none" else args.compression
    files, rows, dropped_episodes, unreadable = convert(
        args.source, args.destination, args.batch_size, compression
    )
    print(f"Wrote {rows} decision rows from {files} replay file(s) to {args.destination}.")
    if dropped_episodes:
        print(f"Dropped {dropped_episodes} episode(s) containing an unmatched/invalid pending decision.")
    if unreadable:
        print(
            f"Skipped {len(unreadable)} unreadable replay file(s) — almost "
            f"certainly truncated downloads. Re-download these dates and re-run "
            f"to recover them:"
        )
        for path in unreadable[:20]:
            print(f"  {path}")
        if len(unreadable) > 20:
            print(f"  ... and {len(unreadable) - 20} more")


if __name__ == "__main__":
    main()
