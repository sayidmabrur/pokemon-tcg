"""Measure the imitation-learned ``PolicyNetwork`` against ``PolicyRuleBased``.

**Pretraining variant.** The per-archetype copies of this file duel one deck,
because the policy they score was cloned from one expert piloting one list.
A pretrained policy is cloned from *every* player in the parquet, so scoring
it on any single decklist measures one arbitrary slice of what it learned. This
version therefore duels a **pool** of decks and defaults to all of them::

    # every deck in decks/, --episodes split across them and both seats
    python archetypes/pretraining/duel_inference.py --episodes 200

    # a named subset (name, unique substring, or path — see resolve_deck)
    python archetypes/pretraining/duel_inference.py --deck "Yushin Ito" --deck flg

    # a deterministic random sample, when all 44 is too slow
    python archetypes/pretraining/duel_inference.py --deck-sample 8 --seed 0

    python archetypes/pretraining/duel_inference.py --list-decks

Decks come from ``archetypes/pretraining/decks/`` via ``build_deck``, which
reconstructed one list per replay player and rebuilds any that is missing. Note
what a per-deck row does and does not tell you: the pool spreads a fixed episode
budget thin, so a single deck's rate over two or three games is noise. The
aggregate is the measurement; the per-deck breakdown is for spotting a deck the
policy cannot pilot *at all* (0% over every game, or a high declined-effect
rate), which is the failure the aggregate hides.

The BC policy is driven through the *same* observation path the submission
uses, deliberately: its ``LiveFeatureExtractor`` is fed only its own
decisions, never the opponent's, exactly as ``agent()`` is invoked in the
competition harness. Under the default ``own_frames`` spec that is provably
equivalent to feeding it both sides (``policy_network/test_parity.py``
checks it decision-by-decision), so nothing is lost — but wiring it this way
means the number printed here is a measurement of the thing that actually
gets submitted, not of a better-informed variant of it.

``PolicyRuleBased``'s deck stays fixed at ``crustle-agent-rule-based/deck.csv``
and is *not* drawn from the pool: its heuristics reference that list's card ids
directly, so handing it anything else makes it play badly for reasons that have
nothing to do with the policy under test, inflating the win rate.
"""

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import torch

# This file lives in archetypes/pretraining/ (not the repo root), so the repo
# root isn't on sys.path by default — needed for ``main``/``cg.game``. Its own
# directory is added explicitly rather than relied upon: that happens for free
# when this file is run as a script, but not when bc_train.py imports it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).parent / "policy_network"))

from cg.game import battle_finish, battle_select, battle_start

import build_deck
from collate import collate_features
from dataset import transform
from live import LiveFeatureExtractor
from policy_experimental import decode_action, load_policy, selection_counts


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# crustle-agent-rule-based/main.py shares the module name "main" with
# _REPO_ROOT/main.py, so it is loaded under a distinct module name rather
# than a plain ``import main`` to avoid clobbering sys.modules["main"].
_crustle_agent_dir = _REPO_ROOT / "crustle-agent-rule-based"
sys.path.insert(0, str(_crustle_agent_dir))
_crustle_agent_main = _load_module(
    "crustle_agent_rule_based_main", _crustle_agent_dir / "main.py"
)


class PolicyRuleBased:
    """Adapts crustle-agent-rule-based/main.py's plain rule_based_select/
    validate functions to the .act(obs) -> list[int] interface ``duel``
    expects (that module has no PolicyRuleBased class of its own)."""

    def act(self, obs: dict) -> list[int]:
        ctx_obs = _crustle_agent_main.to_observation_class(obs)
        choice = _crustle_agent_main.rule_based_select(ctx_obs)
        return _crustle_agent_main.validate(ctx_obs.select, choice)


def read_deck(path: Path) -> list[int]:
    lines = [line for line in path.read_text().split("\n") if line.strip()]
    if len(lines) != 60:
        raise SystemExit(f"{path} has {len(lines)} cards; the engine requires exactly 60")
    return [int(line) for line in lines]


def available_decks() -> list[str]:
    """Deck names cached in ``decks/``, which is what ``--deck`` matches
    against. Sorted so ``--deck-sample`` is reproducible from a seed: a
    directory listing's order is not."""
    return sorted(path.stem for path in build_deck.DECKS_DIR.glob("*.csv"))


def resolve_deck(spec: str) -> tuple[str, list[int]]:
    """Turn one ``--deck`` argument into ``(label, 60 card ids)``.

    Three spellings are accepted, in this order:

    1. **A path** to a deck csv, so any list on disk still works — the
       per-archetype decks, a hand-written file, whatever.
    2. **An exact player name.** Resolved through ``build_deck.ensure_deck``,
       which *builds and caches* the deck from the parquet if ``decks/`` does
       not have it yet, so a name from the parquet works even on a fresh
       checkout.
    3. **A unique case-insensitive substring** of a cached name. The names came
       from replay metadata and include spaces, CJK and parentheses
       ("李秉叡（ntumlnoob）", "e-toppo + kurupical"); requiring them typed
       exactly would make the flag unusable for half the pool. Ambiguity is an
       error listing the matches, never a silent pick of the first one.
    """
    path = Path(spec)
    if path.is_file():
        return path.stem, read_deck(path)

    names = available_decks()
    if spec in names:
        return spec, build_deck.ensure_deck(spec)

    matches = [name for name in names if spec.casefold() in name.casefold()]
    if len(matches) == 1:
        return matches[0], build_deck.ensure_deck(matches[0])
    if len(matches) > 1:
        raise SystemExit(
            f"--deck {spec!r} matches {len(matches)} decks: "
            f"{', '.join(matches)} — be more specific"
        )

    # Not a path, not a cached name, not a substring. It may still be a player
    # in the parquet whose deck has never been built, so try that before giving
    # up — but say so, since building reads the whole parquet and is slow.
    print(f"[decks] {spec!r} is not cached; reconstructing it from the parquet")
    try:
        return spec, build_deck.ensure_deck(spec)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(
            f"cannot resolve --deck {spec!r}: {error}\n"
            f"available ({len(names)}): {', '.join(names)}"
        ) from error


def resolve_deck_pool(
    specs: list[str] | None, sample: int | None, seed: int
) -> list[tuple[str, list[int]]]:
    """The decks to duel over: everything in ``decks/`` unless narrowed.

    Defaulting to the whole pool rather than one deck is the point of this
    variant — see the module docstring. ``sample`` takes a deterministic subset
    (its own ``Random(seed)``, so it does not consume the global stream that
    ``RandomPolicy`` draws from and thus cannot shift the baseline).
    """
    if specs:
        pool = [resolve_deck(spec) for spec in specs]
    else:
        names = available_decks()
        if not names:
            raise SystemExit(
                f"no decks in {build_deck.DECKS_DIR} — build them with:\n"
                f"  python archetypes/pretraining/build_deck.py --all"
            )
        pool = [(name, build_deck.ensure_deck(name)) for name in names]

    if sample is not None and sample < len(pool):
        pool = random.Random(seed).sample(pool, sample)
        pool.sort(key=lambda entry: entry[0])
    return pool


class BCPolicy:
    """The trained network behind a ``.act(obs) -> list[int]`` interface.

    Owns its own extractor and is fed only its own decisions — see the module
    docstring. That self-containment is the point: it makes this class a drop-in
    for ``PolicyRuleBased`` and keeps the duel loop from having to know
    anything about feature extraction.
    """

    def __init__(self, checkpoint: Path) -> None:
        if not checkpoint.is_file():
            raise SystemExit(
                f"no checkpoint at {checkpoint} — train one with "
                f"policy_network/bc_train.py, or pass --checkpoint"
            )
        self.network = load_policy(checkpoint)
        print(f"[BCPolicy] loaded {checkpoint}")
        self.extractor = LiveFeatureExtractor()
        self.decisions = 0
        self.empty_selections = 0

    def reset(self, episode_id: int) -> None:
        self.extractor.reset(episode_id=episode_id)

    def act(self, obs: dict) -> list[int]:
        observation = self.extractor(obs)
        features = collate_features([transform(observation)])
        with torch.no_grad():
            logits = self.network(features)
        min_count, max_count = selection_counts(features)
        options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)
        action = decode_action(logits, options_mask, min_count, max_count)[0]
        self.extractor.record_action(action)
        self.decisions += 1
        # An empty selection is legal only when minCount is 0, and it means
        # "decline this effect" — the failure mode a mis-set threshold causes,
        # so it is counted rather than left invisible.
        if not action:
            self.empty_selections += 1
        return action


class RandomPolicy:
    """Control condition. Without it a win rate has no floor to be read
    against: "the network beats the rule-based agent 12% of the time" only
    means something next to what picking legal moves at random scores."""

    def reset(self, episode_id: int) -> None:
        pass

    def act(self, obs: dict) -> list[int]:
        select = obs["select"]
        count = min(
            random.randint(select["minCount"], select["maxCount"]), len(select["option"])
        )
        return random.sample(range(len(select["option"])), count)


def wilson(wins: int, total: int) -> tuple[float, float]:
    """95% CI for a win rate. Wilson rather than the normal approximation,
    which collapses to zero width at 0 or 100% — precisely the results worth
    being careful about here."""
    if not total:
        return 0.0, 1.0
    rate = wins / total
    z = 1.96
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    spread = z / denominator * (rate * (1 - rate) / total + z**2 / (4 * total**2)) ** 0.5
    return max(center - spread, 0.0), min(center + spread, 1.0)


def duel(
    challenger, deck: list[int], opponent_deck: list[int],
    episodes: int, seat: int, quiet: bool,
) -> dict:
    """Play ``episodes`` matches of ``challenger`` vs ``PolicyRuleBased``.

    ``seat`` decides which side the challenger sits on. It matters: the
    engine gives the first player a real advantage, so a win rate measured
    from one seat only is partly a measurement of that seat.

    The two decks are separate because ``PolicyRuleBased`` is written for one
    specific decklist — its heuristics reference card ids directly. Handing it
    the Alakazam list makes it play badly for reasons that have nothing to do
    with how good the BC policy is, which would inflate the win rate.
    """
    opponent = PolicyRuleBased()
    policies = [None, None]
    policies[seat] = challenger
    policies[1 - seat] = opponent
    decks = [None, None]
    decks[seat] = deck
    decks[1 - seat] = opponent_deck

    wins = 0
    steps = []
    for episode in range(episodes):
        obs, _ = battle_start(decks[0], decks[1])
        challenger.reset(episode_id=episode)
        step = 0
        while obs["current"]["result"] == -1:
            player_index = obs["current"]["yourIndex"]
            action = policies[player_index].act(obs)
            obs = battle_select(action)
            step += 1
        result = obs["current"]["result"]
        battle_finish()

        if result == seat:
            wins += 1
        steps.append(step)
        if not quiet:
            won = "W" if result == seat else "L"
            print(f"  episode {episode + 1}/{episodes}: {won} ({step} steps)")

    low, high = wilson(wins, episodes)
    return {
        "wins": wins, "episodes": episodes, "rate": wins / max(episodes, 1),
        "ci": (low, high), "mean_steps": sum(steps) / max(len(steps), 1),
    }


def duel_pool(
    challenger, pool: list[tuple[str, list[int]]], opponent_deck: list[int],
    episodes: int, seats: list[int], quiet: bool,
) -> tuple[dict, list[dict]]:
    """Run ``challenger`` over every deck in ``pool``, splitting ``episodes``.

    Returns ``(totals, per_deck)``. ``episodes`` is the budget for the *whole*
    pool, not per deck, so raising ``--deck-sample`` narrows the pool instead of
    multiplying the runtime — with 44 decks and both seats, an episode count
    below 88 would round to zero games per combination, so it is clamped to one
    and the real total is reported rather than the requested one.
    """
    per_combo = max(1, episodes // max(len(pool) * len(seats), 1))
    totals = {"wins": 0, "episodes": 0}
    per_deck = []
    for name, deck in pool:
        deck_totals = {"wins": 0, "episodes": 0, "steps": 0.0}
        for seat in seats:
            outcome = duel(challenger, deck, opponent_deck, per_combo, seat, quiet)
            deck_totals["wins"] += outcome["wins"]
            deck_totals["episodes"] += outcome["episodes"]
            deck_totals["steps"] += outcome["mean_steps"] * outcome["episodes"]
        rate = deck_totals["wins"] / max(deck_totals["episodes"], 1)
        per_deck.append({
            "deck": name,
            "wins": deck_totals["wins"],
            "episodes": deck_totals["episodes"],
            "rate": rate,
            "mean_steps": deck_totals["steps"] / max(deck_totals["episodes"], 1),
        })
        totals["wins"] += deck_totals["wins"]
        totals["episodes"] += deck_totals["episodes"]
        # Printed even under --quiet, which suppresses *per-episode* lines. A
        # 44-deck pool is minutes of engine time; without one line per deck the
        # run looks hung, and the running total is the only way to tell a slow
        # duel from a stuck one.
        print(
            f"  [{len(per_deck)}/{len(pool)}] {name}: "
            f"{deck_totals['wins']}/{deck_totals['episodes']} = {rate:.1%} "
            f"(pool so far {totals['wins']}/{totals['episodes']})",
            flush=True,
        )
    return totals, per_deck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="total episodes across the whole deck pool and both seats, not per deck",
    )
    parser.add_argument(
        "--deck", action="append", default=None, metavar="NAME_OR_PATH",
        help="deck for the BC policy: a player name from decks/, a unique "
             "substring of one, or a path to a deck csv. Repeat for several. "
             "The default is every deck in decks/, since this checkpoint was "
             "pretrained across all of them",
    )
    parser.add_argument(
        "--deck-sample", type=int, default=None, metavar="N",
        help="duel a deterministic random N-deck subset of the pool (with "
             "--seed); useful when all of them is too slow",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="controls --deck-sample only",
    )
    parser.add_argument(
        "--list-decks", action="store_true",
        help="print the available deck names and exit",
    )
    parser.add_argument(
        "--opponent-deck", default=str(_REPO_ROOT / "crustle-agent-rule-based/deck.csv"),
        help="deck for PolicyRuleBased; leave it alone unless you know why — "
             "its heuristics reference this list's card ids directly",
    )
    parser.add_argument(
        "--checkpoint", default=str(_REPO_ROOT / "checkpoints/bc_pretrain.pt"),
        help="defaults to bc_train.py's own --out default for this stage",
    )
    parser.add_argument(
        "--seats", default="both", choices=("0", "1", "both"),
        help="which seat the policy plays; 'both' splits the episodes and "
             "reports each, since going first is an advantage",
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="skip the random-policy control (it is what makes the BC number readable)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress the per-episode W/L lines; the one-line-per-deck progress "
             "report is kept either way",
    )
    args = parser.parse_args()

    if args.list_decks:
        names = available_decks()
        print(f"{len(names)} decks in {build_deck.DECKS_DIR}:")
        for name in names:
            print(f"  {name}")
        return

    pool = resolve_deck_pool(args.deck, args.deck_sample, args.seed)
    opponent_deck = read_deck(Path(args.opponent_deck))
    seats = [0, 1] if args.seats == "both" else [int(args.seats)]
    per_combo = max(1, args.episodes // max(len(pool) * len(seats), 1))
    print(
        f"deck pool:     {len(pool)} deck(s), {per_combo} episode(s) each per seat "
        f"= {len(pool) * len(seats) * per_combo} episodes"
    )
    if len(pool) <= 6:
        for name, deck in pool:
            print(f"                 {name} ({len(set(deck))} distinct cards)")
    print(f"opponent deck: {args.opponent_deck} ({len(set(opponent_deck))} distinct cards)")

    contenders = [("BC policy", BCPolicy(Path(args.checkpoint)))]
    if not args.no_baseline:
        contenders.append(("random  ", RandomPolicy()))

    print()
    results = {}
    breakdowns = {}
    for name, policy in contenders:
        print(f"{name} over the pool, seats {seats} vs PolicyRuleBased")
        totals, per_deck = duel_pool(
            policy, pool, opponent_deck, args.episodes, seats, args.quiet
        )
        low, high = wilson(totals["wins"], totals["episodes"])
        print(
            f"  -> {totals['wins']}/{totals['episodes']} = "
            f"{totals['wins'] / max(totals['episodes'], 1):.1%} "
            f"(95% CI {low:.1%}-{high:.1%})"
        )
        results[name] = totals
        breakdowns[name] = per_deck

    print("\n" + "=" * 62)
    if len(pool) > 1:
        # Worst-first: a deck the policy cannot pilot at all is the thing this
        # breakdown exists to surface, and it is invisible in the aggregate.
        # Individual rates are noisy by construction (see the module docstring),
        # so the episode count is printed next to each one.
        print("per-deck (BC policy), worst first:")
        for row in sorted(breakdowns["BC policy"], key=lambda r: r["rate"]):
            print(
                f"  {row['rate']:6.1%}  {row['wins']}/{row['episodes']}  "
                f"{row['mean_steps']:3.0f} steps  {row['deck']}"
            )
        print()

    bc = contenders[0][1]
    if bc.decisions:
        rate = bc.empty_selections / bc.decisions
        print(
            f"declined effects: {bc.empty_selections}/{bc.decisions} decisions "
            f"({rate:.1%}) selected nothing — the expert declines ~2% of "
            f"optional selections, so a rate far above that means the policy "
            f"is skipping effects it should be using"
        )
    for name, totals in results.items():
        low, high = wilson(totals["wins"], totals["episodes"])
        print(
            f"{name}: {totals['wins']}/{totals['episodes']} = "
            f"{totals['wins'] / max(totals['episodes'], 1):.1%} "
            f"(95% CI {low:.1%}-{high:.1%})"
        )

    if "random  " in results and "BC policy" in results:
        bc, rnd = results["BC policy"], results["random  "]
        bc_low, _ = wilson(bc["wins"], bc["episodes"])
        _, rnd_high = wilson(rnd["wins"], rnd["episodes"])
        # Non-overlapping intervals is a deliberately conservative test: it
        # under-reports significance, so clearing it is meaningful while
        # failing it means "not shown", not "no difference".
        if bc_low > rnd_high:
            print("\nThe BC policy beats the random baseline (intervals do not overlap).")
        else:
            print(
                "\nThe BC policy is NOT distinguishable from random at this sample "
                "size — either it hasn't learned to play, or more episodes are needed."
            )


if __name__ == "__main__":
    main()
