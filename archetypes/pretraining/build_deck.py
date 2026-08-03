"""Reconstruct each replay player's decklist and cache it in ``decks/``.

The pretraining parquet holds every player across every game, and each of
them piloted their own 60 cards. Anything that has to *play* one of those
seats — a duel, an inference harness, an eval that pairs a cloned policy
against the deck it was cloned from — needs that specific list, so this
builds them on demand and caches one CSV per player under
``archetypes/pretraining/decks/<player name>.csv``.

Keyed by player name rather than agent id purely for readability: names are
what the parquet's ``player_name`` column already carries and what you type
on a command line. See ``deck_path`` for what happens to names that aren't
filesystem-safe.

How the counts are recovered (the method is ``build_crustle_deck.py``'s,
generalised from one player to all of them): no frame ever states a
decklist. A frame shows only that player's *visible* zones — hand, discard,
active, bench, and everything attached to them (energy, tools, the
pre-evolution cards stacked underneath). The library and the six prize cards
stay hidden all game. So the most copies of a card ever visible at once is a
lower bound on how many are in the deck, and taking the maximum of that
bound across many games converges upward on the truth: a card has to be
prized or undrawn in *every single* game to stay missing.

That "across many games" is the catch, and it is why this module talks about
completeness everywhere instead of just returning a list. A player with 900
episodes converges; a player with 16 does not, and their reconstruction will
be a partial deck of 40-odd cards. A short list is not a legal deck and the
engine will reject it, so ``ensure_deck`` refuses to hand one back unless you
pass ``allow_incomplete``. The failure is silent otherwise, which is exactly
the kind of thing that surfaces three hours into an eval as a confusing
engine error.

**One name is not one deck.** The bound above assumes every episode was
played with the same 60 cards, and for a lot of players that is false — a
competitor who submitted several agents over the competition appears under
one ``player_name`` with a different decklist behind each submission. The
parquet carries no agent id to separate them by, so this module recovers the
split from the cards themselves.

The detection is exact, not heuristic: under a single deck each card's
max-across-episodes cannot exceed its true count, so the reconstructed total
cannot exceed 60. A total above 60 is therefore *proof* of multiple decks
rather than evidence of one. Measured on the crustle parquet, Majkel1337
reconstructs to 177 cards over 93 episodes with 56 distinct ids (flg: 24, and
exactly 60) — three decks' worth under one name.

``cluster_episodes`` then partitions that player's games into deck groups,
greedily merging episodes while the merged multiset stays legal, and the
group with the most episodes becomes the deck written to ``decks/``. Keeping
one file per player is the readable layout asked for; the clustering is what
makes that layout mean something when a name spans several decks. Players
whose games *do* all agree — flg among them — collapse to a single cluster
and are unaffected.

Usage:
    # one player, building it if decks/ doesn't have it yet
    python archetypes/pretraining/build_deck.py --player-name flg

    # every player in the parquet, with a completeness summary
    python archetypes/pretraining/build_deck.py --all

From Python, the one function most callers want:
    from build_deck import ensure_deck
    deck = ensure_deck("flg")          # list[int], 60 ids, built + cached on miss
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import unicodedata
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default locations. Both are overridable on every entry point — the
#: defaults just spare callers from restating the layout.
DECKS_DIR = Path(__file__).resolve().parent / "decks"
DEFAULT_PARQUET = _REPO_ROOT / "data/policy_decisions_pretraining.parquet"
DEFAULT_CARD_DATA = _REPO_ROOT / "EN_Card_Data.csv"

#: Pokémon TCG rule: at most 4 copies of any card except basic energy, which
#: is unlimited. A reconstruction that exceeds this is wrong by construction,
#: so it is worth asserting rather than trusting.
_MAX_COPIES = 4
_DECK_SIZE = 60

#: Characters that cannot appear in a filename (or that make one a pain to
#: type). Player names in the data include spaces, ``&``, ``+``, ``@`` and
#: Japanese text; only the genuinely path-hostile ones are rewritten, since
#: the whole point of naming files by player is being able to read them.
_UNSAFE = re.compile(r"[/\\\x00-\x1f:*?\"<>|]+")


def deck_path(player_name: str, decks_dir: Path | None = None) -> Path:
    """Where ``player_name``'s deck is cached.

    Path separators and control characters are replaced with ``_`` and
    surrounding whitespace/dots are stripped (a leading dot would hide the
    file; a trailing one breaks on Windows). Everything else — spaces,
    Unicode, punctuation — is preserved, because a directory listing you can
    actually read is the reason this is keyed by name at all.

    Two different names *can* collapse onto one path (``a/b`` and ``a_b``).
    That is rare enough to not be worth an escaping scheme that would make
    every filename unreadable to avoid it, but ``build_all`` checks for it
    and refuses rather than letting one player silently overwrite another.
    """
    safe = _UNSAFE.sub("_", unicodedata.normalize("NFC", player_name)).strip(" .")
    if not safe:  # a name made entirely of stripped characters
        safe = "unnamed"
    return (decks_dir or DECKS_DIR) / f"{safe}.csv"


def visible_cards(player: dict) -> collections.Counter:
    """Multiset of card ids visible in one player's own zones at one frame.

    ``prize`` is deliberately absent: every prize slot is facedown for the
    whole match (confirmed in ``policy_network/dataset.py`` — 1,394,877 slots
    checked, 0 ever revealed), so it can contribute nothing. The deck library
    is likewise never enumerated.
    """
    counts = collections.Counter()
    for pokemon in (player["active"] or []) + (player["bench"] or []):
        if pokemon is None:
            continue
        counts[pokemon["id"]] += 1
        # Attached cards are deck cards too — and ``preEvolution`` matters
        # especially: an Alakazam on the board means its Kadabra and Abra are
        # physically underneath it, and they would be invisible otherwise.
        for key in ("energyCards", "tools", "preEvolution"):
            for card in pokemon[key] or []:
                if card:
                    counts[card["id"]] += 1
    for key in ("hand", "discard"):
        for card in player[key] or []:
            if card:
                counts[card["id"]] += 1
    return counts


def _legal(counts: collections.Counter, names: dict) -> bool:
    """Could ``counts`` be a subset of one legal 60-card deck?

    Two independent rules, both facts about decks rather than about this
    data: no more than 60 cards, and no more than 4 copies of anything that
    is not basic energy. Without ``names`` the copy rule is skipped (every id
    could be basic energy for all we know) and the size rule carries it.
    """
    if sum(counts.values()) > _DECK_SIZE:
        return False
    return all(
        n <= _MAX_COPIES or names.get(cid, "").startswith("Basic ")
        for cid, n in counts.items()
    )


def _merge(a: collections.Counter, b: collections.Counter) -> collections.Counter:
    """Per-card max, which is how two episodes' lower bounds combine."""
    out = a.copy()
    for card_id, count in b.items():
        out[card_id] = max(out[card_id], count)
    return out


def cluster_episodes(
    episodes: dict, names: dict | None = None
) -> list[tuple[collections.Counter, int]]:
    """Partition one player's per-episode bounds into decks.

    Greedy agglomeration: walk the episodes richest-first and drop each into
    the first existing cluster it can join without the merged multiset going
    illegal, otherwise start a new one. Richest-first matters — a game that
    revealed 30 cards pins down which deck it was, while one that revealed 3
    is compatible with almost anything, so seeding clusters with the
    informative games keeps the sparse ones from bridging two real decks into
    a chimera.

    Returns ``[(counts, num_episodes), ...]``, most episodes first. A player
    with one deck yields exactly one entry, which is the common case and the
    one that has to stay untouched.
    """
    names = names or {}
    clusters: list[list] = []  # [counts, episode_count]
    for _episode_id, counts in sorted(
        episodes.items(), key=lambda kv: -sum(kv[1].values())
    ):
        for cluster in clusters:
            merged = _merge(cluster[0], counts)
            if _legal(merged, names):
                cluster[0], cluster[1] = merged, cluster[1] + 1
                break
        else:
            clusters.append([counts.copy(), 1])
    clusters.sort(key=lambda c: (-c[1], -sum(c[0].values())))
    return [(counts, n) for counts, n in clusters]


class DeckReconstruction:
    """One player's reconstructed deck, plus what backs it up.

    ``counts`` is the copies-per-card-id multiset, ``support`` how many
    episodes each id turned up in, and ``num_episodes`` how many games fed
    the estimate. The last two are not decoration: a 60-card total built from
    12 games with most ids seen once is a very different claim from the same
    total built from 900, and ``complete`` alone cannot tell you which you
    have.
    """

    def __init__(self, player_name, counts, support, num_episodes,
                 num_decks=1, deck_episodes=None, total_episodes=None):
        self.player_name = player_name
        self.counts = counts
        self.support = support
        #: Episodes backing *this* deck (the dominant cluster), which is what
        #: the estimate actually rests on...
        self.num_episodes = num_episodes
        #: ...as opposed to every episode the player appears in. The two
        #: differ exactly when one name covers several decks.
        self.total_episodes = total_episodes if total_episodes is not None else num_episodes
        #: How many distinct decks this player's games split into. >1 means
        #: the other clusters were discarded, not merged.
        self.num_decks = num_decks
        #: Episode count per cluster, largest first — the shape of the split.
        self.deck_episodes = deck_episodes or [num_episodes]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def complete(self) -> bool:
        return self.total == _DECK_SIZE

    def as_deck(self) -> list[int]:
        """Flat card-id list, one entry per copy, sorted — the ``deck.csv``
        row order that ``main.read_deck_csv`` expects."""
        return [cid for cid, n in sorted(self.counts.items()) for _ in range(n)]

    def problems(self, names: dict | None = None) -> list[str]:
        """Checks that would catch an undercount, not just a crash.

        The reconstruction is a lower bound, so the failure mode is not a
        wrong card — it is a *missing* one, which is silent. The totals and
        copy limits are independent facts about legal decks that a bad
        reconstruction has no reason to satisfy.
        """
        names = names or {}
        found = []
        if not self.counts:
            found.append(
                f"no cards found for {self.player_name!r} — check the spelling "
                f"against the parquet's player_name column"
            )
        elif self.total > _DECK_SIZE:
            # Should be unreachable once clustering runs — a cluster is only
            # ever grown while it stays legal. Kept as an assertion in prose:
            # if it fires, the clustering is broken, not the data.
            found.append(
                f"total is {self.total}, ABOVE {_DECK_SIZE} — a single deck "
                f"cannot exceed 60 cards, so this player's episodes span "
                f"several decks and the clustering failed to separate them"
            )
        elif self.total < _DECK_SIZE:
            found.append(
                f"total is {self.total}, not {_DECK_SIZE} — the reconstruction "
                f"is a lower bound, so {_DECK_SIZE - self.total} card(s) were "
                f"never visible in any of the {self.num_episodes} episode(s) "
                f"backing this deck (prized or undrawn every time). More "
                f"episodes for this player is the only real fix"
            )
        if self.num_decks > 1:
            found.append(
                f"this player's {self.total_episodes} episodes split into "
                f"{self.num_decks} distinct decks ({self.deck_episodes} episodes "
                f"each) — one name, several agent submissions. Only the largest "
                f"is written; the rest are discarded"
            )
        for card_id, count in self.counts.items():
            if count > _MAX_COPIES and not names.get(card_id, "").startswith("Basic "):
                found.append(
                    f"card {card_id} ({names.get(card_id, '?')}) reconstructed at "
                    f"{count} copies, above the {_MAX_COPIES}-copy rule"
                )
        return found


def reconstruct(
    parquet_path: Path | str = DEFAULT_PARQUET,
    players: set[str] | None = None,
    names: dict | None = None,
) -> dict[str, DeckReconstruction]:
    """Reconstruct every player's deck in a *single* pass over the parquet.

    One pass for all players rather than one pass per player: the pretraining
    parquet holds a few hundred of them, and rescanning the whole file per
    name turns a minute into an afternoon. Pass ``players`` to restrict which
    names are accumulated — that trims memory, not I/O, since the scan has to
    visit every row either way.
    """
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    # (player, episode) -> per-episode max visible count, the inner bound.
    per_episode: dict = collections.defaultdict(collections.Counter)

    for group in range(parquet.num_row_groups):
        for row in parquet.read_row_group(group).to_pylist():
            name = row["player_name"]
            if players is not None and name not in players:
                continue
            state = row["state"]
            # Rows are stored already reoriented to the deciding player's
            # view, so ``yourIndex`` is this player — the same convention
            # dataset.py's feature builders rely on.
            frame = visible_cards(state["players"][state["yourIndex"]])
            episode = per_episode[(name, row["episode_id"])]
            for card_id, count in frame.items():
                episode[card_id] = max(episode[card_id], count)

    # Regroup by player, then split each player's episodes into decks. The
    # per-card max is only a valid bound *within* a deck, so the clustering
    # has to happen before the maxes are taken across episodes.
    by_player: dict = collections.defaultdict(dict)
    for (name, episode_id), seen in per_episode.items():
        by_player[name][episode_id] = seen

    out = {}
    for name, episodes in by_player.items():
        clusters = cluster_episodes(episodes, names)
        counts, num_episodes = clusters[0]
        # Support is counted over the dominant cluster's episodes only —
        # counting it across decks would advertise a card as well-attested
        # when the games attesting it were played with a different list.
        dominant = {
            ep: seen for ep, seen in episodes.items()
            if all(seen[c] <= counts[c] for c in seen)
        }
        support = collections.Counter()
        for seen in dominant.values():
            for card_id in seen:
                support[card_id] += 1
        out[name] = DeckReconstruction(
            name, counts, support, num_episodes,
            num_decks=len(clusters),
            deck_episodes=[n for _, n in clusters],
            total_episodes=len(episodes),
        )
    return out


def write_deck(deck: list[int], path: Path) -> None:
    """Write ``deck.csv`` format: one bare card id per line, no header —
    the same thing ``main.read_deck_csv`` parses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(card_id) for card_id in deck) + "\n")


def read_deck(path: Path) -> list[int]:
    """Inverse of ``write_deck``; blank lines tolerated."""
    return [int(line) for line in path.read_text().split() if line.strip()]


def ensure_deck(
    player_name: str,
    parquet_path: Path | str = DEFAULT_PARQUET,
    decks_dir: Path | None = None,
    allow_incomplete: bool = False,
    rebuild: bool = False,
) -> list[int]:
    """``player_name``'s decklist, reconstructing and caching it on a miss.

    This is the function the rest of the pretraining code should call: it
    reads ``decks/<player name>.csv`` when that exists and otherwise builds
    it from the parquet, so callers never have to care which happened.

    Raises ``ValueError`` when the reconstruction comes up short of 60 cards,
    rather than returning a deck the engine will reject. ``allow_incomplete``
    downgrades that to a caller's problem — reasonable when you only want the
    card *identities* (say, to check which archetype someone was piloting)
    and not a playable list.
    """
    path = deck_path(player_name, decks_dir)
    if path.exists() and not rebuild:
        deck = read_deck(path)
        if len(deck) != _DECK_SIZE and not allow_incomplete:
            raise ValueError(
                f"{path} holds {len(deck)} cards, not {_DECK_SIZE} — it was "
                f"cached from an incomplete reconstruction. Rebuild it with "
                f"more episodes, or pass allow_incomplete=True"
            )
        return deck

    found = reconstruct(parquet_path, players={player_name},
                        names=_load_card_names(DEFAULT_CARD_DATA))
    if player_name not in found:
        raise ValueError(
            f"no rows for player {player_name!r} in {parquet_path} — check the "
            f"spelling against the parquet's player_name column"
        )
    result = found[player_name]
    if not result.complete and not allow_incomplete:
        raise ValueError(
            f"reconstructed only {result.total}/{_DECK_SIZE} cards for "
            f"{player_name!r} from {result.num_episodes} episode(s); "
            f"nothing was cached. Pass allow_incomplete=True to take it anyway"
        )
    deck = result.as_deck()
    write_deck(deck, path)
    return deck


def build_all(
    parquet_path: Path | str = DEFAULT_PARQUET,
    decks_dir: Path | None = None,
    allow_incomplete: bool = False,
    rebuild: bool = False,
) -> dict[str, DeckReconstruction]:
    """Reconstruct and cache every player in the parquet, in one pass.

    Incomplete reconstructions are reported and skipped unless
    ``allow_incomplete`` — for most parquets that will be the majority of
    players, since most of them appear in only a handful of games.
    """
    results = reconstruct(parquet_path, names=_load_card_names(DEFAULT_CARD_DATA))

    # Two names can normalise onto one filename (see ``deck_path``). Refuse
    # the whole batch rather than let one player silently overwrite another.
    claimed: dict = {}
    for name in results:
        claimed.setdefault(deck_path(name, decks_dir), []).append(name)
    clashes = {p: names for p, names in claimed.items() if len(names) > 1}
    if clashes:
        detail = "; ".join(f"{p.name} <- {names}" for p, names in clashes.items())
        raise ValueError(f"player names collide on one filename: {detail}")

    for name, result in results.items():
        if not result.complete and not allow_incomplete:
            continue
        path = deck_path(name, decks_dir)
        if path.exists() and not rebuild:
            continue
        write_deck(result.as_deck(), path)
    return results


def _load_card_names(card_data: Path | str) -> dict:
    try:
        with open(card_data, encoding="utf-8") as handle:
            return {int(r["Card ID"]): r["Card Name"] for r in csv.DictReader(handle)}
    except FileNotFoundError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--player-name", default=None, help="one player to build")
    parser.add_argument("--all", action="store_true", help="build every player in the parquet")
    parser.add_argument("--decks-dir", default=str(DECKS_DIR))
    parser.add_argument("--card-data", default=str(DEFAULT_CARD_DATA),
                        help="only used to print card names and to exempt basic energy "
                             "from the 4-copy check")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="cache reconstructions that came up short of 60 cards; "
                             "they are NOT legal decks and the engine will reject them")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild even if decks/ already has the file")
    args = parser.parse_args()

    if bool(args.player_name) == bool(args.all):
        parser.error("pass exactly one of --player-name or --all")

    names = _load_card_names(args.card_data)
    decks_dir = Path(args.decks_dir)

    if args.player_name:
        results = reconstruct(args.parquet, players={args.player_name}, names=names)
        result = results.get(args.player_name)
        if result is None:
            parser.error(f"no rows for player {args.player_name!r} in {args.parquet}")
        print(f"{result.num_episodes} of {result.total_episodes} episodes back "
              f"{result.player_name}'s dominant deck "
              f"({result.num_decks} deck(s) detected)")
        for card_id, count in sorted(result.counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {count}x  {card_id:>5}  {names.get(card_id, '???'):<34} "
                  f"seen in {result.support[card_id]}/{result.num_episodes} games")
        print(f"total {result.total} cards, {len(result.counts)} distinct")
        for problem in result.problems(names):
            print(f"CHECK FAILED: {problem}")
        if result.complete or args.allow_incomplete:
            path = deck_path(result.player_name, decks_dir)
            write_deck(result.as_deck(), path)
            print(f"wrote {path} ({result.total} lines)")
        else:
            print("nothing written — pass --allow-incomplete to cache it anyway")
        return

    results = build_all(args.parquet, decks_dir, args.allow_incomplete, args.rebuild)
    complete = [r for r in results.values() if r.complete]
    print(f"{len(results)} players, {len(complete)} reconstructed to a full "
          f"{_DECK_SIZE} cards\n")
    for result in sorted(results.values(), key=lambda r: (-r.total, -r.num_episodes)):
        flag = "ok   " if result.complete else "SHORT"
        split = f"  [{result.num_decks} decks: {result.deck_episodes}]" if result.num_decks > 1 else ""
        print(f"  {flag} {result.total:>3}/{_DECK_SIZE} cards  "
              f"{result.num_episodes:>4}/{result.total_episodes:<4} eps  "
              f"{result.player_name}{split}")
    written = len(complete) if not args.allow_incomplete else len(results)
    print(f"\ncached {written} deck(s) under {decks_dir}")
    if len(complete) < len(results) and not args.allow_incomplete:
        print("players marked SHORT were skipped: a partial list is not a legal "
              "deck. Most of them simply do not have enough episodes to converge.")


if __name__ == "__main__":
    main()
