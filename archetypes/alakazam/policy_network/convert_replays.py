"""Convert Kaggle Pokémon TCG replay JSON files into policy-decision Parquet.

Usage:
    python archetypes/alakazam/policy_network/convert_replays.py \
        episode-87785476-replay.json data/policy_decisions.parquet

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
    if source.is_file():
        yield source
    else:
        yield from sorted(source.rglob("*.json"))


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(destination, schema, compression=compression)
    written = 0
    try:
        for path in paths:
            episode_rows, episode_rejected = decision_rows(path)
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
    return len(paths), written, dropped_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Replay JSON file or directory of replay JSON files")
    parser.add_argument("destination", type=Path, help="Output .parquet file")
    parser.add_argument("--batch-size", type=int, default=5_000, help="Rows buffered before writing")
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=("snappy", "zstd", "gzip", "brotli", "lz4", "none"),
        help="Parquet compression codec",
    )
    args = parser.parse_args()
    compression = None if args.compression == "none" else args.compression
    files, rows, dropped_episodes = convert(args.source, args.destination, args.batch_size, compression)
    print(f"Wrote {rows} decision rows from {files} replay file(s) to {args.destination}.")
    if dropped_episodes:
        print(f"Dropped {dropped_episodes} episode(s) containing an unmatched/invalid pending decision.")


if __name__ == "__main__":
    main()
