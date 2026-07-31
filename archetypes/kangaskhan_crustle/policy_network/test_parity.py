"""Prove the offline and live feature extractors produce identical output.

``dataset.py`` (imitation learning from replays) and ``live.py`` (self-play
against the ``cg`` engine) are deliberately separate — one seeks into Parquet,
the other accumulates a game as it happens.  A policy trained on one and run
on the other only works if the features are the same, so this test replays
recorded decisions through *both* paths and deep-compares every field.

Each Parquet row stores exactly what the engine handed the player at that
moment (``state`` / ``selection`` / ``options``), so it can be reassembled
into the live ``obs`` shape and pushed through ``LiveFeatureExtractor`` in
frame order.  Any drift between the two paths — a missing option key, a
differently-scoped history scan, an off-by-one in the backward walk — shows up
as a concrete field mismatch.

Usage:
    python archetypes/kangaskhan_crustle/policy_network/test_parity.py
    python archetypes/kangaskhan_crustle/policy_network/test_parity.py data/policy_decisions_kangaskhan_crustle.parquet 3
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from dataset import PolicyFeatureDataset
from live import LiveFeatureExtractor
from observation import OWN_FRAMES


def observation_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Reassemble the live ``obs`` dict this Parquet row was recorded from.

    ``convert_replays.py`` split one observation into three columns; this puts
    them back together.  ``option`` lives inside ``select`` in the raw engine
    shape, which is where ``split_observation`` expects to find it.
    """
    return {
        "current": row["state"],
        "select": {**row["selection"], "option": row["options"]},
    }


def differences(offline: Any, live: Any, path: str = "") -> list[str]:
    """Every leaf where the two observations disagree, as readable paths."""
    if isinstance(offline, dict) and isinstance(live, dict):
        found = []
        for key in sorted(set(offline) | set(live)):
            if key not in offline:
                found.append(f"{path}.{key}: missing offline (live={live[key]!r})")
            elif key not in live:
                found.append(f"{path}.{key}: missing live (offline={offline[key]!r})")
            else:
                found += differences(offline[key], live[key], f"{path}.{key}")
        return found
    if isinstance(offline, list) and isinstance(live, list):
        if len(offline) != len(live):
            return [f"{path}: length {len(offline)} offline vs {len(live)} live"]
        return [d for i, (a, b) in enumerate(zip(offline, live))
                for d in differences(a, b, f"{path}[{i}]")]
    return [] if offline == live else [f"{path}: {offline!r} offline vs {live!r} live"]


def episode_bounds(dataset: PolicyFeatureDataset, count: int) -> list[tuple[int, int]]:
    """``(start, stop)`` row ranges for the first ``count`` episodes.

    Episodes are written contiguously, so a forward walk finds the boundaries.
    """
    bounds, start = [], 0
    current = dataset._read_row(0)["episode_id"]
    for idx in range(1, len(dataset)):
        episode_id = dataset._read_row(idx)["episode_id"]
        if episode_id != current:
            bounds.append((start, idx))
            if len(bounds) == count:
                return bounds
            start, current = idx, episode_id
    bounds.append((start, len(dataset)))
    return bounds


def check_episode(dataset: PolicyFeatureDataset, start: int, stop: int) -> tuple[int, list[str]]:
    """Push one episode through the live extractor and compare every decision.

    The live extractor is fed the decisions in the same order the engine would
    have produced them, and told the action that was actually taken — the same
    information ``dataset.py`` reads out of the row it is about to featurise.
    """
    row = dataset._read_row(start)
    names = (row["player_name"], row["player_name"])
    extractor = LiveFeatureExtractor(player_names=names, spec=dataset.spec)
    extractor.reset(episode_id=row["episode_id"])

    problems = []
    for idx in range(start, stop):
        row = dataset._read_row(idx)
        offline, target = dataset[idx]
        live = extractor(observation_from_row(row))
        extractor.record_action(row["target_action"])

        problems += [f"row {idx} (frame {row['frame_index']}) features{d}"
                     for d in differences(offline["features"], live["features"])]
        # meta: frame_index is each source's own row counter (Parquet frame vs
        # live decision number), so only the POV pointer has to agree.
        if offline["meta"]["player_index"] != live["meta"]["player_index"]:
            problems.append(f"row {idx} meta.player_index disagrees")
        if target.tolist() != row["target_action"]:
            problems.append(f"row {idx} target_action disagrees with the row")
    return stop - start, problems


def check_episode_one_sided(
    dataset: PolicyFeatureDataset, start: int, stop: int, player_index: int
) -> tuple[int, list[str]]:
    """Feed the live extractor **only one player's** decisions and require the
    features to still match offline.

    This is the property the deployed agent depends on and the one the old
    single-stream test could not express.  In the competition harness
    ``agent()`` is called only for our own choices, so the opponent's frames
    never reach the extractor at all.  Under the default ``own_frames`` spec
    nothing in the observation is sourced from them, so dropping them must be
    a no-op — and if some future spec change reintroduces a dependency on the
    opponent's frames, this fails loudly instead of silently degrading the
    served policy relative to the trained one.
    """
    row = dataset._read_row(start)
    extractor = LiveFeatureExtractor(
        player_names=(row["player_name"], row["player_name"]), spec=dataset.spec
    )
    extractor.reset(episode_id=row["episode_id"])

    problems, checked = [], 0
    for idx in range(start, stop):
        row = dataset._read_row(idx)
        if row["player_index"] != player_index:
            continue  # the opponent's frame — invisible to a one-sided agent
        offline, _ = dataset[idx]
        live = extractor(observation_from_row(row))
        extractor.record_action(row["target_action"])
        checked += 1
        problems += [f"row {idx} (one-sided, player {player_index}) features{d}"
                     for d in differences(offline["features"], live["features"])]
    return checked, problems


def main() -> None:
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else "data/policy_decisions_kangaskhan_crustle.parquet"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    # Unfiltered on purpose: the live extractor has to be fed every decision in
    # the episode, both players', to reproduce the same history a full-file
    # backward scan sees.  Sample indexes therefore equal raw row indexes here.
    dataset = PolicyFeatureDataset(parquet_path)
    print(f"{parquet_path}: {len(dataset)} rows, spec={dataset.spec}")

    total, all_problems = 0, []
    one_sided_total = 0
    for start, stop in episode_bounds(dataset, episodes):
        episode_id = dataset._read_row(start)["episode_id"]
        checked, problems = check_episode(dataset, start, stop)
        total += checked
        all_problems += problems
        status = "OK" if not problems else f"{len(problems)} MISMATCHES"
        print(f"  episode {episode_id}: {checked} decisions -> {status}")

        # Same episode, but each seat replayed in isolation — the shape the
        # deployed agent actually runs in.
        if dataset.spec.opponent_history_source == OWN_FRAMES:
            for player_index in (0, 1):
                seat_checked, seat_problems = check_episode_one_sided(
                    dataset, start, stop, player_index
                )
                one_sided_total += seat_checked
                all_problems += seat_problems
                status = "OK" if not seat_problems else f"{len(seat_problems)} MISMATCHES"
                print(f"    seat {player_index} alone: {seat_checked} decisions -> {status}")

    print()
    if all_problems:
        print(f"FAIL: {len(all_problems)} mismatches across {total} decisions")
        for problem in all_problems[:20]:
            print(f"  {problem}")
        if len(all_problems) > 20:
            print(f"  ... and {len(all_problems) - 20} more")
        sys.exit(1)

    print(f"PASS: offline and live features are identical for all {total} decisions")
    if one_sided_total:
        print(f"  ...including {one_sided_total} replayed with only one seat's "
              f"frames visible (the deployed agent's view)")

    # Non-empty coverage: identical-but-always-empty history groups would pass
    # the comparison above while proving nothing.
    chains = [len(dataset[i][0]["features"]["decision_chain"]) for i in range(min(400, len(dataset)))]
    histories = [len(dataset[i][0]["features"]["opponent_history"]) for i in range(min(400, len(dataset)))]
    print(f"  decision_chain lengths over the first {len(chains)} rows: "
          f"min={min(chains)} max={max(chains)} nonempty={sum(1 for c in chains if c)}")
    print(f"  opponent_history lengths: "
          f"min={min(histories)} max={max(histories)} nonempty={sum(1 for h in histories if h)}")


if __name__ == "__main__":
    main()
