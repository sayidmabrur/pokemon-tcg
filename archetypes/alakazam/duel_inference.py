"""Measure the imitation-learned ``PolicyNetwork`` against ``PolicyRuleBased``.

Run it::

    python archetypes/alakazam/duel_inference.py --episodes 100

The BC policy is driven through the *same* observation path the submission
uses, deliberately: its ``LiveFeatureExtractor`` is fed only its own
decisions, never the opponent's, exactly as ``agent()`` is invoked in the
competition harness. Under the default ``own_frames`` spec that is provably
equivalent to feeding it both sides (``policy_network/test_parity.py``
checks it decision-by-decision), so nothing is lost — but wiring it this way
means the number printed here is a measurement of the thing that actually
gets submitted, not of a better-informed variant of it.

The BC policy defaults to ``alakazam_deck.csv``, the list reconstructed from
the replays it was cloned from; playing anything else hands the network
cards it never saw its expert play, which measures the mismatch rather than
the policy. ``PolicyRuleBased`` defaults to
``crustle-agent-rule-based/deck.csv``, since its heuristics are written for
that specific decklist and play badly with any other.
"""

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import torch

# This file lives in archetypes/alakazam/ (not the repo root), so the repo
# root isn't on sys.path by default — needed for ``main``/``cg.game``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent / "policy_network"))

from cg.game import battle_finish, battle_select, battle_start

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
        # Weights and threshold load together — see load_policy.
        self.network, self.threshold = load_policy(checkpoint)
        print(f"[BCPolicy] loaded {checkpoint} (decode threshold {self.threshold:.2f})")
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
        action = decode_action(
            logits, options_mask, min_count, max_count, threshold=self.threshold
        )[0]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--deck", default=str(_REPO_ROOT / "alakazam_deck.csv"),
        help="deck for the BC policy",
    )
    parser.add_argument(
        "--opponent-deck", default=str(_REPO_ROOT / "crustle-agent-rule-based/deck.csv"),
        help="deck for PolicyRuleBased",
    )
    parser.add_argument("--checkpoint", default=str(_REPO_ROOT / "checkpoints/bc_policy.pt"))
    parser.add_argument(
        "--seats", default="both", choices=("0", "1", "both"),
        help="which seat the policy plays; 'both' splits the episodes and "
             "reports each, since going first is an advantage",
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="skip the random-policy control (it is what makes the BC number readable)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-episode lines")
    args = parser.parse_args()

    deck = read_deck(Path(args.deck))
    opponent_deck = read_deck(Path(args.opponent_deck))
    print(f"deck:          {args.deck} ({len(set(deck))} distinct cards)")
    print(f"opponent deck: {args.opponent_deck} ({len(set(opponent_deck))} distinct cards)")

    seats = [0, 1] if args.seats == "both" else [int(args.seats)]
    per_seat = max(1, args.episodes // len(seats))

    contenders = [("BC policy", BCPolicy(Path(args.checkpoint)))]
    if not args.no_baseline:
        contenders.append(("random  ", RandomPolicy()))

    print()
    results = {}
    for name, policy in contenders:
        totals = {"wins": 0, "episodes": 0}
        for seat in seats:
            print(f"{name} as player {seat}, {per_seat} episodes vs PolicyRuleBased")
            outcome = duel(policy, deck, opponent_deck, per_seat, seat, args.quiet)
            low, high = outcome["ci"]
            print(
                f"  -> {outcome['wins']}/{outcome['episodes']} = {outcome['rate']:.1%} "
                f"(95% CI {low:.1%}-{high:.1%}), {outcome['mean_steps']:.0f} steps/game"
            )
            totals["wins"] += outcome["wins"]
            totals["episodes"] += outcome["episodes"]
        results[name] = totals

    print("\n" + "=" * 62)
    bc = contenders[0][1]
    if bc.decisions:
        rate = bc.empty_selections / bc.decisions
        print(
            f"declined effects: {bc.empty_selections}/{bc.decisions} decisions "
            f"({rate:.1%}) selected nothing — the expert declines ~2% of "
            f"optional selections, so a rate far above that means the decode "
            f"threshold is too high (see bc_train.calibrate_threshold)"
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
