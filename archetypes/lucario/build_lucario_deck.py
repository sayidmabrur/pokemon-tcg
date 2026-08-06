"""Reconstruct the decklist a replay player actually piloted, from the
replay frames themselves, and write it in ``deck.csv`` format.

Why this exists: the behavioral-cloning policy in ``policy_network/`` is
trained on one player's decisions, but the duel harness
(``duel_inference.py``) built both players' decks from the repo-root
``deck.csv`` — a *different* deck. Of that file's 25 distinct card ids only
5 ever appeared on the cloned player's own board, so the policy was being
asked to pilot 20 cards it had never seen anyone play, four of which had no
training exposure at all and whose embeddings were therefore still at their
random initialisation. Evaluating the clone on the deck it was cloned from
is the apples-to-apples comparison that isolates "did the model learn
anything" from "is the deck wrong".

How the counts are recovered: no frame ever states the decklist. A frame
shows only the player's *visible* zones — hand, discard, active, bench and
everything attached to them (energy, tools, the pre-evolution cards stacked
underneath). The library and the six prize cards stay hidden all game. So
the most copies of a card ever visible simultaneously is a lower bound on
how many are in the deck, and taking the maximum of that bound across many
games converges upward on the true count: a card has to be stuck in the
prizes or undrawn in *every single* game to stay undercounted.

That convergence is checkable rather than assumed — see ``validate()``.
"""

import argparse
import collections
import csv
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Pokémon TCG rule: at most 4 copies of any card except basic energy,
#: which is unlimited. A reconstruction that exceeds this is wrong by
#: construction, so it's worth asserting rather than trusting.
_MAX_COPIES = 4
_DECK_SIZE = 60


def visible_cards(player: dict) -> collections.Counter:
    """Multiset of card ids visible in one player's own zones at one frame.

    ``prize`` is deliberately absent: every prize slot is facedown for the
    whole match (confirmed in ``policy_network/dataset.py`` — 1,394,877
    slots checked, 0 ever revealed), so it can contribute nothing. The deck
    library is likewise never enumerated.
    """
    counts = collections.Counter()
    for pokemon in (player["active"] or []) + (player["bench"] or []):
        if pokemon is None:
            continue
        counts[pokemon["id"]] += 1
        # Attached cards are deck cards too — and ``preEvolution`` matters
        # especially: a Mega Lucario ex on the board means its Lucario and Riolu
        # are physically underneath it, and they'd be invisible otherwise.
        for key in ("energyCards", "tools", "preEvolution"):
            for card in pokemon[key] or []:
                if card:
                    counts[card["id"]] += 1
    for key in ("hand", "discard"):
        for card in player[key] or []:
            if card:
                counts[card["id"]] += 1
    return counts


def reconstruct(parquet_path: Path, player_name: str):
    """Return ``(counts, support, num_episodes)`` — copies per card id, how
    many episodes each id was seen in, and the episode total."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    per_episode: dict = collections.defaultdict(collections.Counter)

    for group in range(parquet.num_row_groups):
        for row in parquet.read_row_group(group).to_pylist():
            if row["player_name"] != player_name:
                continue
            state = row["state"]
            # Rows are stored already reoriented to the deciding player's
            # view, so ``yourIndex`` is this player — the same convention
            # dataset.py's feature builders rely on.
            frame = visible_cards(state["players"][state["yourIndex"]])
            episode = per_episode[row["episode_id"]]
            for card_id, count in frame.items():
                episode[card_id] = max(episode[card_id], count)

    counts = collections.Counter()
    support = collections.Counter()
    for episode in per_episode.values():
        for card_id, count in episode.items():
            counts[card_id] = max(counts[card_id], count)
            support[card_id] += 1
    return counts, support, len(per_episode)


def validate(counts: collections.Counter, names: dict) -> list[str]:
    """Checks that would catch an undercount, not just a crash.

    The reconstruction is a lower bound, so the failure mode isn't a wrong
    card — it's a *missing* one, which is silent. These three checks are
    what make the result trustworthy: the totals and copy limits are
    independent facts about legal decks that a bad reconstruction has no
    reason to satisfy.
    """
    problems = []
    total = sum(counts.values())
    if total != _DECK_SIZE:
        problems.append(
            f"total is {total}, not {_DECK_SIZE} — the reconstruction is a lower "
            f"bound, so {_DECK_SIZE - total} card(s) were never seen in any game "
            f"(stuck in prizes/library every time). Add more episodes, or fill "
            f"the gap by hand."
        )
    for card_id, count in counts.items():
        is_basic_energy = names.get(card_id, "").startswith("Basic ")
        if count > _MAX_COPIES and not is_basic_energy:
            problems.append(
                f"card {card_id} ({names.get(card_id, '?')}) reconstructed at "
                f"{count} copies, above the {_MAX_COPIES}-copy rule"
            )
    if not counts:
        problems.append("no cards found — check --player-name spelling")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=str(_REPO_ROOT / "data/policy_decisions_lucario.parquet"))
    parser.add_argument("--player-name", default="Majkel1337")
    parser.add_argument("--card-data", default=str(_REPO_ROOT / "EN_Card_Data.csv"))
    parser.add_argument("--out", default=str(_REPO_ROOT / "lucario_deck.csv"))
    args = parser.parse_args()

    names = {
        int(row["Card ID"]): row["Card Name"]
        for row in csv.DictReader(open(args.card_data, encoding="utf-8"))
    }

    counts, support, num_episodes = reconstruct(Path(args.parquet), args.player_name)
    print(f"{num_episodes} episodes for {args.player_name}")

    unknown = sorted(set(counts) - set(names))
    if unknown:
        print(f"WARNING: card ids absent from {args.card_data}: {unknown}")

    for card_id, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(
            f"  {count}x  {card_id:>5}  {names.get(card_id, '???'):<34} "
            f"seen in {support[card_id]}/{num_episodes} games"
        )
    print(f"total {sum(counts.values())} cards, {len(counts)} distinct")

    problems = validate(counts, names)
    for problem in problems:
        print(f"CHECK FAILED: {problem}")

    # Same format as the repo-root deck.csv that ``main.read_deck_csv``
    # parses: one bare card id per line, 60 lines, no header.
    deck = [card_id for card_id, count in sorted(counts.items()) for _ in range(count)]
    Path(args.out).write_text("\n".join(str(card_id) for card_id in deck) + "\n")
    print(f"wrote {args.out} ({len(deck)} lines)")


if __name__ == "__main__":
    main()
