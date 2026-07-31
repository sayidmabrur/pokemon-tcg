"""Reconstruct a player's 60-card decklist straight from downloaded replays.

    python build_deck.py --replay-path replays/55063047
    python build_deck.py --replay-path replays/55063047 --out ragingbolt_deck.csv

This reads the replay JSONs directly, so it runs immediately after
``download_replay.sh`` — no Parquet conversion first. It replaces the
per-archetype ``archetypes/*/build_*_deck.py`` scripts, which each needed a
built parquet and a hardcoded player name.

Whose deck? By default the team appearing in the most episodes, which for a
single submission's replay directory is that submission's own agent (every one
of its episodes features it against a rotating cast of opponents). Pass
``--player-name`` to override, and see ``--list`` to just print the candidates.

How the counts are recovered: no frame ever states the decklist. A frame shows
only the player's *visible* zones — hand, discard, active, bench and everything
attached to them (energy, tools, the pre-evolution cards stacked underneath).
The library and the six prize cards stay hidden all game. So the most copies of
a card ever visible simultaneously is a lower bound on how many are in the
deck, and taking the maximum of that bound across many games converges upward
on the true count: a card has to be stuck in the prizes or undrawn in *every
single* game to stay undercounted.

That convergence is checkable rather than assumed — see ``validate()``.
"""

import argparse
import collections
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

#: Pokémon TCG rule: at most 4 copies of any card except basic energy, which
#: is unlimited. A reconstruction that exceeds this is wrong by construction.
_MAX_COPIES = 4
_DECK_SIZE = 60


def visible_cards(player: dict) -> collections.Counter:
    """Multiset of card ids visible in one player's own zones at one frame.

    ``prize`` is deliberately absent: every prize slot is facedown for the
    whole match, so it can contribute nothing. The deck library is likewise
    never enumerated.
    """
    counts = collections.Counter()
    for pokemon in (player.get("active") or []) + (player.get("bench") or []):
        if not pokemon:
            continue
        counts[pokemon["id"]] += 1
        # Attached cards are deck cards too — and ``preEvolution`` matters
        # especially: an Alakazam on the board means its Kadabra and Abra are
        # physically underneath it, and they'd be invisible otherwise.
        for key in ("energyCards", "tools", "preEvolution"):
            for card in pokemon.get(key) or []:
                if card:
                    counts[card["id"]] += 1
    for key in ("hand", "discard"):
        for card in player.get(key) or []:
            if card:
                counts[card["id"]] += 1
    return counts


def replay_files(replay_path: Path) -> list[Path]:
    if replay_path.is_file():
        return [replay_path]
    files = sorted(replay_path.glob("*.json"))
    if not files:
        raise SystemExit(f"no *.json replays found under {replay_path}")
    return files


def scan(files: list[Path]) -> tuple[dict, collections.Counter]:
    """Return ``{team: {episode: Counter}}`` and an episode count per team.

    One pass over every replay. Each frame is attributed to the team that was
    *acting* (``yourIndex``), since that is the only side whose hand is real —
    the engine redacts the opponent's.
    """
    per_team: dict = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    episodes_per_team: collections.Counter = collections.Counter()

    for path in files:
        try:
            replay = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {path.name}: {exc}")
            continue
        info = replay.get("info") or {}
        teams = info.get("TeamNames") or []
        episode_id = info.get("EpisodeId", path.stem)
        for team in teams:
            episodes_per_team[team] += 1

        for step in replay.get("steps") or []:
            for agent in step:
                current = (agent.get("observation") or {}).get("current")
                if not current or not current.get("players"):
                    continue
                index = current.get("yourIndex")
                if index is None or index >= len(teams):
                    continue
                frame = visible_cards(current["players"][index])
                episode = per_team[teams[index]][episode_id]
                for card_id, count in frame.items():
                    if count > episode[card_id]:
                        episode[card_id] = count
    return per_team, episodes_per_team


def reconstruct(per_episode: dict) -> tuple[collections.Counter, collections.Counter]:
    """Max each card's per-episode bound across every episode."""
    counts: collections.Counter = collections.Counter()
    support: collections.Counter = collections.Counter()
    for episode in per_episode.values():
        for card_id, count in episode.items():
            counts[card_id] = max(counts[card_id], count)
            support[card_id] += 1
    return counts, support


def load_card_data() -> dict:
    """``{card_id: (name, is_basic_energy)}`` from the engine's own database.

    Read from ``cg.api`` rather than ``EN_Card_Data.csv`` so the copy-limit
    check uses the authoritative card type instead of guessing from the name.
    Falls back to an empty map if the engine isn't importable, in which case
    the copy-limit check simply reports names as unknown.
    """
    try:
        import sys
        sys.path.insert(0, str(_REPO_ROOT))
        from cg.api import CardType, all_card_data
        return {
            c.cardId: (c.name, c.cardType == CardType.BASIC_ENERGY)
            for c in all_card_data()
        }
    except Exception as exc:  # pragma: no cover - engine optional
        print(f"  (cg.api unavailable: {exc}; skipping name/type lookup)")
        return {}


def validate(counts: collections.Counter, card_data: dict) -> list[str]:
    """Checks that would catch an undercount, not just a crash.

    The reconstruction is a lower bound, so the failure mode isn't a wrong
    card — it's a *missing* one, which is silent. The totals and copy limits
    are independent facts about legal decks that a bad reconstruction has no
    reason to satisfy.
    """
    problems = []
    total = sum(counts.values())
    if total != _DECK_SIZE:
        problems.append(
            f"total is {total}, not {_DECK_SIZE} — the reconstruction is a lower "
            f"bound, so {_DECK_SIZE - total} card(s) were never seen in any game "
            f"(stuck in prizes/library every time). Add more replays, or fill the "
            f"gap by hand."
        )
    for card_id, count in counts.items():
        name, is_basic_energy = card_data.get(card_id, ("?", False))
        if count > _MAX_COPIES and not is_basic_energy:
            problems.append(
                f"card {card_id} ({name}) reconstructed at {count} copies, above "
                f"the {_MAX_COPIES}-copy rule"
            )
    if not counts:
        problems.append("no cards found — check --player-name spelling")
    return problems


def slugify(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    return "_".join(part for part in slug.split("_") if part) or "deck"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-path", required=True, type=Path,
                        help="replay JSON file, or a directory of them (e.g. replays/55063047)")
    parser.add_argument("--player-name", default=None,
                        help="team whose deck to build; defaults to the team appearing "
                             "in the most episodes (the submission's own agent)")
    parser.add_argument("--out", default=None,
                        help="output deck CSV; defaults to <player-slug>_deck.csv at the repo root")
    parser.add_argument("--list", action="store_true",
                        help="just list the teams found and exit")
    args = parser.parse_args()

    files = replay_files(args.replay_path)
    print(f"scanning {len(files)} replay file(s) under {args.replay_path}...")
    per_team, episodes_per_team = scan(files)

    if not episodes_per_team:
        raise SystemExit("no team names found in these replays")

    if args.list:
        print("\nteams present:")
        for team, n in episodes_per_team.most_common():
            print(f"  {n:>5} episodes  {team}")
        return

    player = args.player_name or episodes_per_team.most_common(1)[0][0]
    if player not in per_team:
        raise SystemExit(
            f"no frames for {player!r}. Teams found: "
            f"{[t for t, _ in episodes_per_team.most_common(8)]}"
        )
    if args.player_name is None:
        runner_up = episodes_per_team.most_common(2)
        detail = f" (next: {runner_up[1][0]!r} in {runner_up[1][1]})" if len(runner_up) > 1 else ""
        print(f"auto-detected player: {player!r} — in {episodes_per_team[player]} "
              f"of {len(files)} episodes{detail}")

    card_data = load_card_data()
    counts, support = reconstruct(per_team[player])
    num_episodes = len(per_team[player])
    print(f"\n{player}: {num_episodes} episodes with visible frames\n")
    for card_id, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        name = card_data.get(card_id, ("???", False))[0]
        print(f"  {count}x  {card_id:>5}  {name:<34} seen in {support[card_id]}/{num_episodes} games")
    print(f"\ntotal {sum(counts.values())} cards, {len(counts)} distinct")

    for problem in validate(counts, card_data):
        print(f"CHECK FAILED: {problem}")

    out = Path(args.out) if args.out else _REPO_ROOT / f"{slugify(player)}_deck.csv"
    # Same format as the repo-root deck.csv that ``main.read_deck_csv`` parses:
    # one bare card id per line, no header.
    deck = [card_id for card_id, count in sorted(counts.items()) for _ in range(count)]
    out.write_text("\n".join(str(card_id) for card_id in deck) + "\n")
    print(f"wrote {out} ({len(deck)} lines)")


if __name__ == "__main__":
    main()
