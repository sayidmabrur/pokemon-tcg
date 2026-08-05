"""Does inference-time search beat the raw BC policy? Measure, don't assume.

Same harness as ``duel_inference.py`` — same deck pool, same opponent, same
Wilson intervals, and it *imports* that module rather than copying it, so the
two differ only in the policy under test. Here the contender is ``PIMCPolicy``:
Perfect-Information Monte Carlo search on top of the same network.

**PIMC, not ISMCTS**, and the distinction is forced by the engine's API.
``cg.api.search_begin`` requires you to hand it *concrete* values for
everything you cannot see — the opponent's deck, prizes, hand, and a face-down
active — so a search state is a fully-observable world sampled from your
beliefs. That is determinization. Real information-set search would keep the
belief distribution inside the tree; this instead solves several perfect-
information games and votes, which imports two known pathologies:

- **Strategy fusion.** Within each sampled world the search assumes it will
  know the hidden cards later, so it happily plans lines that depend on
  information it will not actually have.
- **Non-locality.** The sampled worlds are drawn from a static prior, ignoring
  that a real opponent's play is itself evidence about which world you are in.

PIMC is nonetheless strong empirically in trick-taking card games, which is why
it is worth measuring here rather than dismissing. What this script exists to
answer is narrow and empirical: at what latency, and does the win rate move?

How the search is spent, and why each bound is here:

- Only decisions with ``maxCount == 1`` and more than one option are searched.
  Multi-select decisions have a combinatorial candidate space that a root
  enumeration cannot cover honestly, and forced decisions have nothing to
  choose. Everything else defers to the network, which is also what keeps the
  latency figure interpretable: ``--searched`` in the output reports how often
  search actually ran.
- Candidates are the network's **top-k options**, not all of them. A 45-option
  decision would otherwise cost 45 rollouts per determinization. The prior is
  doing the pruning, which is the point of having it.
- Rollouts are to a real terminal state, scored ±1 from the searching seat's
  point of view. No value network exists yet, so there is nothing to evaluate a
  cut-off leaf with — and truncating a rollout at a heuristic would measure the
  heuristic.

**Every default here is submittable.** That is the same rule
``duel_inference.py`` states for the plain policy — its extractor is fed only
its own decisions because the number printed has to describe the agent that
actually gets submitted, not a better-informed variant. A search harness breaks
that rule far more easily than a policy does, since the engine will happily
accept the opponent's real hidden cards if you hand them over. Two flags can
leave the submittable set, and both announce themselves loudly at startup:

``--belief oracle``   reads the opponent's true decklist. **Not submittable.**
                      It exists only to separate *does search help at all* from
                      *is my opponent model good enough*: if oracle search
                      cannot beat plain BC, no amount of opponent modelling
                      will. Any win rate it reports is unachievable.
``--rollout rulebased`` shippable in principle, but that agent hardcodes
                      crustle's card ids and deck composition, so on any other
                      deck it plays your list with heuristics for cards that
                      are not in it. Valid only when piloting crustle.

The defaults are therefore ``--belief pool`` (guess the opponent's deck from
the cards they have actually revealed, matched against ``decks/`` — possible
only because ``build_deck`` reconstructed the metagame's 44 lists) and
``--rollout random`` (weak, but at least deck-agnostic).

The rollout policy is the biggest lever on whether PIMC works at all: it
estimates the value of a position *under that policy*, so random play
systematically misjudges anything whose value depends on either side playing
well. Both available choices are bad, which is the honest state of affairs —
the real fix is a value head at the leaf instead of a rollout at all, and no
such head exists yet.

The per-episode time budget is also part of the submitted agent: the harness
gets 600s of overage per episode (``actTimeout`` is 0, so there is no per-action
cap), the observation carries ``remainingOverageTime``, and a search that spends
the pool forfeits. Search therefore stops once the remaining budget falls under
``reserve_seconds``, and ``skipped-out-of-budget`` reports how often that bit.

Usage::

    # the submittable comparison: BC vs BC+search, nothing the agent can't see
    python archetypes/pretraining/duel_inference_search.py \\
        --deck flg --episodes 300 --seats both --determinizations 4 --candidates 4

    # A/A control: run the search, discard its answer, play the network's move.
    # Must reproduce plain BC, or the search calls are perturbing the live game.
    python archetypes/pretraining/duel_inference_search.py \\
        --deck flg --episodes 20 --seats both --ignore-search

    # diagnostic upper bound only — announces that it is not submittable
    python archetypes/pretraining/duel_inference_search.py \\
        --deck flg --episodes 20 --belief oracle
"""

import argparse
import collections
import random
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "policy_network"))

import build_deck
import duel_inference as di
from cg import api
from cg.game import battle_finish, battle_select, battle_start
from collate import collate_features
from dataset import transform
from live import LiveFeatureExtractor
from policy_experimental import load_policy


#: Hard ceiling on rollout length. A rollout should terminate on its own in
#: ~200 selections, so this only catches a pathological loop — but without it
#: one such loop hangs the whole measurement with no output.
_MAX_ROLLOUT_STEPS = 600


def _random_selection(select, rng: random.Random) -> list[int]:
    """A uniformly random legal selection for one decision."""
    count = len(select.option)
    if count == 0:
        return []
    low = select.minCount
    high = min(select.maxCount, count)
    k = rng.randint(low, high) if high >= low else min(low, count)
    return rng.sample(range(count), min(k, count))


class _RuleBasedRollout:
    """``PolicyRuleBased``'s heuristics as a rollout policy.

    The rule-based agent consumes an ``Observation`` dataclass, and a
    ``SearchState`` already carries one — but it comes from *this* tree's
    ``cg.api`` while the agent vendored its own copy of the library. The
    classes are structurally identical and the agent only reads attributes, so
    passing it across works by duck typing. It is still wrapped: if any
    decision shape it was not written for makes it raise, that must degrade to
    a random choice rather than abort a 20-game measurement.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.failures = 0

    def __call__(self, observation) -> list[int]:
        try:
            choice = di._crustle_agent_main.rule_based_select(observation)
            return di._crustle_agent_main.validate(observation.select, choice)
        except Exception:
            self.failures += 1
            return _random_selection(observation.select, self.rng)


class PIMCPolicy:
    """BC network as prior + determinized rollout search, behind ``.act(obs)``.

    Drop-in for ``duel_inference.BCPolicy`` so ``duel``/``duel_pool`` need no
    changes — which is what makes the comparison in ``main`` apples-to-apples.
    """

    def __init__(
        self,
        checkpoint: Path,
        determinizations: int = 4,
        candidates: int = 4,
        rollout: str = "rulebased",
        belief: str = "oracle",
        skip_confident: float = 1.0,
        ignore_search: bool = False,
        deck_pool: list[tuple[str, list[int]]] | None = None,
        seed: int = 0,
    ) -> None:
        if not checkpoint.is_file():
            raise SystemExit(f"no checkpoint at {checkpoint}")
        self.network = load_policy(checkpoint)
        print(f"[PIMCPolicy] loaded {checkpoint}")
        self.extractor = LiveFeatureExtractor()
        self.determinizations = determinizations
        self.candidates = candidates
        self.belief = belief
        # Search only when the prior's top-1 probability is *below* this. Two
        # reasons, and the second matters more than the speed: most decisions in
        # this game are forced or obvious, so searching them burns 180ms to
        # re-derive what the network already knew; and every searched decision is
        # a chance for a noisy 6-rollout estimate to *overrule* a confident
        # prior, which is how search makes a policy worse rather than better.
        # 1.0 searches everything (the default, so behaviour is unchanged unless
        # asked for).
        self.skip_confident = skip_confident
        self.skipped_confident = 0
        # A/A control. Runs the search in full — same engine calls, same
        # search_begin/search_step/search_end traffic — then discards the answer
        # and plays the network's move. Any win-rate difference from plain BC
        # under this flag cannot be search *choosing* better, so it is evidence
        # that the search calls are perturbing the live battle rather than
        # merely reading it. That is the one failure mode a suspiciously good
        # result cannot be distinguished from any other way.
        self.ignore_search = ignore_search
        # Stop searching when this much of the episode's overage pool is left.
        # A submission that spends the pool loses the game on time, so the floor
        # is part of the agent, not a harness convenience.
        self.reserve_seconds = 60.0
        self.budget_skips = 0
        self.deck_pool = deck_pool or []
        self.rng = random.Random(seed)
        self.rollout_policy = (
            _RuleBasedRollout(self.rng) if rollout == "rulebased"
            else (lambda observation: _random_selection(observation.select, self.rng))
        )

        # Set by the harness before each episode: the searching seat's own 60
        # cards. Needed because determinizing *our* hidden cards (library and
        # prizes) requires knowing what is in them.
        self.own_deck: list[int] | None = None
        self.opponent_deck: list[int] | None = None  # --belief oracle only

        self.decisions = 0
        self.empty_selections = 0
        self.searched = 0
        self.fallbacks = 0
        self.rollouts = 0
        self.search_seconds = 0.0
        self.total_seconds = 0.0
        self.agreements = 0  # search picked what the network would have

    def reset(self, episode_id: int) -> None:
        self.extractor.reset(episode_id=episode_id)

    # ---------------------------------------------------------------- priors

    def _prior(self, obs: dict):
        """``(ranked option indexes, network's own selection)``.

        One forward pass serves both: the ranking prunes the candidate set and
        the full selection is the fallback whenever search declines to run, so
        a decision never costs two evaluations.
        """
        observation = self.extractor(obs)
        features = collate_features([transform(observation)])
        with torch.no_grad():
            logits = self.network(features)
        options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)
        from policy_experimental import decode_action, selection_counts

        min_count, max_count = selection_counts(features)
        action = decode_action(logits, options_mask, min_count, max_count)[0]
        # -inf at masked positions, so argsort never ranks a padded slot first.
        ranked = torch.argsort(logits[0], descending=True).tolist()
        valid = int(options_mask[0].sum().item())
        ranked = [i for i in ranked if i < valid]
        # Top-1 probability over the *valid* options only, used by the
        # confidence gate. Softmax over the raw row would include the -inf pads,
        # which is harmless, but restricting it keeps the number interpretable
        # as "how sure is the policy among the moves that exist".
        probabilities = torch.softmax(logits[0, :valid], dim=-1) if valid else None
        confidence = float(probabilities.max()) if valid else 1.0
        return ranked, action, confidence

    # -------------------------------------------------------- determinization

    def _hidden_split(self, own_cards: list[int], visible, deck_count: int, prize_count: int):
        """Deal ``own_cards`` minus ``visible`` into (deck, prize) guesses.

        The unseen part of a 60-card list is exactly library + prizes, so what
        is left after removing everything on the table is the pool both are
        drawn from. Shuffled, then split — which is precisely the uncertainty a
        determinization is supposed to resolve arbitrarily.
        """
        remaining = collections.Counter(own_cards) - collections.Counter(visible)
        pool = list(remaining.elements())
        self.rng.shuffle(pool)
        need = deck_count + prize_count
        if len(pool) < need:
            # The reconstruction disagrees with the engine's counts (a deck list
            # built from partial replays can be wrong). Pad with repeats of what
            # we do have rather than failing the decision outright.
            if not pool:
                return None, None
            pool = (pool * (need // len(pool) + 1))[:need]
        return pool[prize_count:need], pool[:prize_count]

    def _believed_opponent_deck(self, opponent_visible) -> list[int]:
        """The pool deck that best explains what the opponent has shown.

        Overlap of multisets, so a deck that plays two copies of a card the
        opponent has shown twice scores higher than one playing a single copy.
        Early in a game almost nothing is visible and this is nearly a uniform
        guess — which is a fair reflection of how much you actually know then.
        """
        if not self.deck_pool:
            raise SystemExit("--belief pool needs a non-empty deck pool")
        visible = collections.Counter(opponent_visible)
        best, best_score = None, -1
        for _, deck in self.deck_pool:
            score = sum((visible & collections.Counter(deck)).values())
            if score > best_score:
                best, best_score = deck, score
        return best

    def _determinize(self, observation):
        """One sampled world: the arguments ``search_begin`` demands."""
        state = observation.current
        me = state.yourIndex
        them = 1 - me
        mine, theirs = state.players[me], state.players[them]

        own_visible = build_deck.visible_cards(_as_dict(mine))
        own_deck_guess, own_prize_guess = self._hidden_split(
            self.own_deck, own_visible, mine.deckCount, len(mine.prize)
        )
        if own_deck_guess is None:
            return None

        opponent_list = (
            self.opponent_deck if self.belief == "oracle"
            else self._believed_opponent_deck(build_deck.visible_cards(_as_dict(theirs)))
        )
        opponent_visible = build_deck.visible_cards(_as_dict(theirs))
        # Their hand is hidden too, so it is dealt from the same unseen pool as
        # their library and prizes — hand first, then prizes, then library.
        remaining = collections.Counter(opponent_list) - collections.Counter(opponent_visible)
        pool = list(remaining.elements())
        self.rng.shuffle(pool)
        need = theirs.handCount + len(theirs.prize) + theirs.deckCount
        if len(pool) < need:
            if not pool:
                return None
            pool = (pool * (need // len(pool) + 1))[:need]
        hand = pool[: theirs.handCount]
        prize = pool[theirs.handCount : theirs.handCount + len(theirs.prize)]
        deck = pool[theirs.handCount + len(theirs.prize) : need]

        # A face-down active has to be named explicitly, and it must be a
        # Pokémon — hand the engine a basic from the believed list.
        active_guess = []
        if len(theirs.active) > 0 and theirs.active[0] is None:
            active_guess = [_first_basic(opponent_list)]

        return {
            # Ignored by the engine when it already knows our deck (the
            # deck-submission decision), which is why it is passed unconditionally.
            "your_deck": own_deck_guess,
            "your_prize": own_prize_guess,
            "opponent_deck": deck,
            "opponent_prize": prize,
            "opponent_hand": hand,
            "opponent_active": active_guess,
        }

    # -------------------------------------------------------------- rollouts

    def _rollout(self, state, seat: int) -> float:
        """Play ``state`` to the end under the rollout policy; ±1 for ``seat``."""
        observation = state.observation
        for _ in range(_MAX_ROLLOUT_STEPS):
            if observation.current.result != -1:
                break
            selection = self.rollout_policy(observation)
            state = api.search_step(state.searchId, selection)
            observation = state.observation
        self.rollouts += 1
        result = observation.current.result
        if result == -1:
            return 0.0  # hit the step ceiling: unresolved, score it a draw
        return 1.0 if result == seat else -1.0

    # ------------------------------------------------------------------- act

    def act(self, obs: dict) -> list[int]:
        started = time.time()
        ranked, network_action, confidence = self._prior(obs)
        select = obs["select"]
        num_options = len(select["option"])
        confident = confidence >= self.skip_confident
        if confident:
            self.skipped_confident += 1
        # The competition gives each agent a per-episode overage pool (600s,
        # with actTimeout 0 — no per-action cap), and the observation carries
        # what is left of it. A search that ignores that is not measuring a
        # submittable agent: it would run at full width until the pool ran dry
        # and then forfeit. Reserving a floor means the harness refuses to search
        # once the budget is nearly spent, exactly as the shipped agent must.
        remaining = obs.get("remainingOverageTime")
        out_of_budget = remaining is not None and remaining < self.reserve_seconds
        if out_of_budget:
            self.budget_skips += 1
        searchable = (
            select["maxCount"] == 1
            and num_options > 1
            and obs["current"]["result"] == -1
            and self.own_deck is not None
            and not confident
            and not out_of_budget
        )

        action = network_action
        if searchable:
            searched_action = self._search(obs, ranked)
            if searched_action is None:
                self.fallbacks += 1
            else:
                if not self.ignore_search:
                    action = searched_action
                self.searched += 1
                if searched_action == network_action:
                    self.agreements += 1

        self.extractor.record_action(action)
        self.decisions += 1
        if not action:
            self.empty_selections += 1
        self.total_seconds += time.time() - started
        return action

    def _search(self, obs: dict, ranked: list[int]) -> list[int] | None:
        """Root PIMC over the top-k candidates. ``None`` means "search could
        not run" (a determinization the engine rejected, say), which is
        reported separately from "search ran and agreed with the network" —
        conflating those two would make a broken search look like a harmless
        one."""
        observation = api.to_observation_class(obs)
        seat = obs["current"]["yourIndex"]
        candidates = [[i] for i in ranked[: self.candidates]]
        if len(candidates) < 2:
            return None

        started = time.time()
        totals = {tuple(c): 0.0 for c in candidates}
        counts = {tuple(c): 0 for c in candidates}
        worlds = 0
        for _ in range(self.determinizations):
            world = self._determinize(observation)
            if world is None:
                continue
            try:
                root = api.search_begin(observation, **world)
            except (ValueError, RuntimeError):
                # A rejected world is a bad guess, not a bug: an opponent deck
                # short of their real card count, a mis-reconstructed list.
                continue
            worlds += 1
            for candidate in candidates:
                try:
                    child = api.search_step(root.searchId, candidate)
                    totals[tuple(candidate)] += self._rollout(child, seat)
                    counts[tuple(candidate)] += 1
                except (ValueError, RuntimeError):
                    continue
            api.search_end()

        self.search_seconds += time.time() - started
        if not worlds:
            return None
        scored = [
            (totals[k] / counts[k], k) for k in totals if counts[k]
        ]
        if not scored:
            return None
        # max() on the value alone would break ties by dict order; ranked[0] is
        # the network's favourite, so ties resolve toward the prior.
        best_value = max(value for value, _ in scored)
        for candidate in candidates:  # in prior order
            key = tuple(candidate)
            if counts[key] and totals[key] / counts[key] == best_value:
                return list(key)
        return None

    def report(self) -> str:
        per_decision = self.total_seconds / max(self.decisions, 1)
        per_search = self.search_seconds / max(self.searched + self.fallbacks, 1)
        return (
            f"decisions={self.decisions} searched={self.searched} "
            f"({self.searched / max(self.decisions, 1):.0%}) "
            f"skipped-as-confident={self.skipped_confident} "
            f"skipped-out-of-budget={self.budget_skips} "
            f"fallbacks={self.fallbacks} rollouts={self.rollouts}\n"
            f"  latency: {per_decision * 1000:.0f}ms/decision overall, "
            f"{per_search * 1000:.0f}ms per searched decision\n"
            f"  search changed the network's pick on "
            f"{self.searched - self.agreements}/{max(self.searched, 1)} searched "
            f"decisions ({1 - self.agreements / max(self.searched, 1):.0%})"
        )


def two_proportion_test(wins_a: int, n_a: int, wins_b: int, n_b: int) -> tuple[float, float]:
    """``(difference, two-sided p)`` for arm A's win rate minus arm B's.

    A pooled two-proportion z-test. This is here because the non-overlapping-CI
    check inherited from ``duel_inference`` is *far* more conservative than it
    looks — two 95% intervals failing to overlap is roughly a 0.5% test, so it
    will call a genuine 10-point improvement "not shown" at any sample size you
    would actually run. Reporting the difference and its p-value separately lets
    you see the effect size and the evidence for it, instead of one boolean that
    conflates them.

    No pairing is available to reduce variance: the engine's ``battle_start``
    takes no seed, so the two arms cannot be made to play the *same* shuffles
    and coin flips. Every comparison here is therefore unpaired, which is
    precisely why the sample sizes below are as large as they are.
    """
    if not n_a or not n_b:
        return 0.0, 1.0
    rate_a, rate_b = wins_a / n_a, wins_b / n_b
    pooled = (wins_a + wins_b) / (n_a + n_b)
    se = (pooled * (1 - pooled) * (1 / n_a + 1 / n_b)) ** 0.5
    if se == 0:
        return rate_a - rate_b, 1.0
    z = (rate_a - rate_b) / se
    # Two-sided normal tail via erfc, so this needs no scipy.
    import math

    return rate_a - rate_b, math.erfc(abs(z) / math.sqrt(2))


def episodes_needed(effect: float, base_rate: float = 0.45) -> int:
    """Episodes *per arm* to detect ``effect`` at 80% power, alpha=0.05.

    Printed alongside a non-significant result so "not shown" comes with the
    cost of actually showing it, rather than leaving you to guess whether the
    next run should be 2x or 20x bigger.
    """
    if effect <= 0:
        return 0
    p1, p2 = base_rate, min(base_rate + effect, 0.999)
    variance = p1 * (1 - p1) + p2 * (1 - p2)
    return int(((1.96 + 0.84) ** 2 * variance) / effect**2) + 1


_BASIC_IDS: set[int] | None = None


def _first_basic(deck: list[int]) -> int:
    """A Basic Pokémon id from ``deck``, for naming a face-down active.

    ``search_begin`` rejects a non-Pokémon here, and a face-down active is by
    definition a Basic (it was played from hand face-down at setup). The card
    database is loaded once and cached: it is the engine's own table, so this
    agrees with whatever the engine considers basic rather than guessing from
    the reconstructed decklists.
    """
    global _BASIC_IDS
    if _BASIC_IDS is None:
        _BASIC_IDS = {card.cardId for card in api.all_card_data() if card.basic}
    for card_id in deck:
        if card_id in _BASIC_IDS:
            return card_id
    # No basic in the believed list means the belief is wrong, not that the
    # position is illegal — let the caller's search_begin reject this world.
    return deck[0] if deck else 0


def _as_dict(player) -> dict:
    """``build_deck.visible_cards`` reads a replay-shaped ``dict``; the search
    API hands out dataclasses. Convert shallowly rather than reimplementing the
    zone walk, so both paths stay in agreement about what "visible" means."""
    if isinstance(player, dict):
        return player
    return _dataclass_to_dict(player)


def _dataclass_to_dict(obj):
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {f: _dataclass_to_dict(getattr(obj, f)) for f in obj.__dataclass_fields__}
    return obj


def duel_vs(challenger, opponent, deck, opponent_deck, episodes, seat, quiet):
    """``duel_inference.duel``, but with the opponent injected.

    That module hardcodes ``PolicyRuleBased``, which makes it impossible to ask
    the only question that matters for the leaderboard: does search still help
    against an opponent it does not have a model of? Rolling out with
    ``rulebased`` while *playing* the rule-based agent gives the search a
    perfect policy model of its opponent for free — a result that cannot
    transfer. Swapping the opponent is how you find out what was real.
    """
    policies, decks = [None, None], [None, None]
    policies[seat], policies[1 - seat] = challenger, opponent
    decks[seat], decks[1 - seat] = deck, opponent_deck

    wins, steps = 0, []
    for episode in range(episodes):
        obs, _ = battle_start(decks[0], decks[1])
        for policy in policies:
            if hasattr(policy, "reset"):
                policy.reset(episode_id=episode)
        step = 0
        while obs["current"]["result"] == -1:
            action = policies[obs["current"]["yourIndex"]].act(obs)
            obs = battle_select(action)
            step += 1
        result = obs["current"]["result"]
        battle_finish()
        if result == seat:
            wins += 1
        steps.append(step)
        if not quiet:
            print(f"  episode {episode + 1}/{episodes}: "
                  f"{'W' if result == seat else 'L'} ({step} steps)")

    return {
        "wins": wins, "episodes": episodes, "rate": wins / max(episodes, 1),
        "mean_steps": sum(steps) / max(len(steps), 1),
    }


def make_opponent(kind: str, checkpoint: Path):
    """The opponent to measure against.

    ``bc`` is the important one: a competent, non-heuristic opponent that the
    search has no hardcoded model of, which is the closest available stand-in
    for the competition field (it was cloned from that field, after all).
    """
    if kind == "rulebased":
        return di.PolicyRuleBased()
    if kind == "random":
        return di.RandomPolicy()
    if kind == "bc":
        return di.BCPolicy(checkpoint)
    raise SystemExit(f"unknown opponent {kind!r}")


def duel_pool_with_decks(challenger, opponent, pool, opponent_deck, episodes, seats, quiet):
    """``duel_inference.duel_pool``, but telling the challenger which decks are
    in play before each matchup.

    Necessary because determinization needs the searcher's own list, and under
    ``--belief oracle`` the opponent's too — neither of which reaches a policy
    through the plain ``.act(obs)``/``.reset()`` interface.
    """
    per_combo = max(1, episodes // max(len(pool) * len(seats), 1))
    totals = {"wins": 0, "episodes": 0}
    per_deck = []
    for name, deck in pool:
        if hasattr(challenger, "own_deck"):
            challenger.own_deck = deck
            challenger.opponent_deck = opponent_deck
        deck_totals = {"wins": 0, "episodes": 0}
        for seat in seats:
            outcome = duel_vs(challenger, opponent, deck, opponent_deck, per_combo, seat, quiet)
            deck_totals["wins"] += outcome["wins"]
            deck_totals["episodes"] += outcome["episodes"]
        rate = deck_totals["wins"] / max(deck_totals["episodes"], 1)
        per_deck.append({"deck": name, **deck_totals, "rate": rate})
        totals["wins"] += deck_totals["wins"]
        totals["episodes"] += deck_totals["episodes"]
        print(
            f"  [{len(per_deck)}/{len(pool)}] {name}: "
            f"{deck_totals['wins']}/{deck_totals['episodes']} = {rate:.1%} "
            f"(pool so far {totals['wins']}/{totals['episodes']})",
            flush=True,
        )
    return totals, per_deck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--deck", action="append", default=None, metavar="NAME_OR_PATH")
    parser.add_argument("--deck-sample", type=int, default=None, metavar="N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--determinizations", type=int, default=4,
        help="sampled worlds per searched decision (D). Cost is D x candidates "
             "rollouts, so this and --candidates set the latency together",
    )
    parser.add_argument(
        "--candidates", type=int, default=4,
        help="top-k options from the network to actually search (k)",
    )
    parser.add_argument(
        "--rollout", default="random", choices=("random", "rulebased"),
        help="policy that plays out a determinized world. PIMC estimates the "
             "value of a position *under this policy*, so it is the dominant "
             "quality knob, not a detail. 'rulebased' is only valid when the "
             "searching seat pilots the crustle deck: that agent hardcodes card "
             "ids (CRUSTLE=345, DWEBBLE, LATIAS) and a fixed 60-card "
             "composition, so on any other deck it rolls out with heuristics for "
             "cards that are not in play. 'random' is the default because it is "
             "at least deck-agnostic — both are shippable, both are weak, and "
             "the real fix is a value head instead of a rollout",
    )
    parser.add_argument(
        "--ignore-search", action="store_true",
        help="A/A control: do all the search work, then play the network's move "
             "anyway. Should reproduce plain BC's win rate exactly; if it does "
             "not, the search calls are corrupting the live game",
    )
    parser.add_argument(
        "--skip-confident", type=float, default=1.0, metavar="P",
        help="skip search when the network's top-1 probability is >= P. Cuts "
             "latency on decisions that are already obvious, and stops a noisy "
             "few-rollout estimate from overruling a confident prior — the most "
             "likely way search makes things worse. 1.0 (default) searches "
             "everything; 0.9 is a reasonable first try",
    )
    parser.add_argument(
        "--belief", default="pool", choices=("pool", "oracle"),
        help="how the opponent's hidden cards are guessed. 'pool' (the default) "
             "uses only what a submission can see: the cards the opponent has "
             "actually revealed, matched against decks/. 'oracle' reads their "
             "true decklist and is NOT SUBMITTABLE — it is a diagnostic for "
             "separating 'does search help' from 'is my opponent model good', "
             "and any win rate it reports is unachievable in the competition",
    )
    parser.add_argument(
        "--opponent", default="rulebased", choices=("rulebased", "bc", "random"),
        help="who to measure against. 'rulebased' is the historical yardstick but "
             "is also what --rollout rulebased models, so that pairing hands the "
             "search a perfect model of its opponent and cannot show whether a "
             "gain transfers. 'bc' plays your own network — a competent opponent "
             "the search has no model of, and the closest stand-in for the "
             "competition field",
    )
    parser.add_argument(
        "--opponent-deck", default=str(di._REPO_ROOT / "crustle-agent-rule-based/deck.csv"),
    )
    parser.add_argument("--checkpoint", default=str(di._REPO_ROOT / "checkpoints/bc_pretrain.pt"))
    parser.add_argument("--seats", default="both", choices=("0", "1", "both"))
    parser.add_argument(
        "--no-baseline-bc", action="store_true",
        help="skip the plain-BC arm. Only useful for latency accounting: without "
             "it there is nothing to compare the win rate against",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    pool = di.resolve_deck_pool(args.deck, args.deck_sample, args.seed)
    opponent_deck = di.read_deck(Path(args.opponent_deck))
    seats = [0, 1] if args.seats == "both" else [int(args.seats)]
    per_combo = max(1, args.episodes // max(len(pool) * len(seats), 1))
    print(
        f"deck pool:     {len(pool)} deck(s) x {len(seats)} seat(s) x {per_combo} "
        f"= {len(pool) * len(seats) * per_combo} episodes per arm"
    )
    print(
        f"search:        D={args.determinizations} candidates={args.candidates} "
        f"rollout={args.rollout} belief={args.belief}"
    )
    print(f"opponent:      {args.opponent}")
    if args.rollout == "rulebased" and args.opponent == "rulebased":
        print(
            "  *** --rollout rulebased vs --opponent rulebased: the search's "
            "model of its opponent IS the opponent. Any gain measured here is "
            "specific to that agent and will not transfer. ***"
        )
    if args.belief == "oracle":
        print(
            "  *** --belief oracle reads the opponent's true decklist. This is "
            "NOT implementable in a submission; the win rate below is a "
            "diagnostic upper bound, not a result. ***"
        )
    if args.rollout == "rulebased" and any(
        "crustle" not in name.casefold() for name, _ in pool
    ):
        print(
            "  *** --rollout rulebased on a non-crustle deck: that agent "
            "hardcodes crustle's card ids, so it rolls out your deck with "
            "heuristics for cards that are not in it. ***"
        )

    arms = []
    searcher = PIMCPolicy(
        Path(args.checkpoint), args.determinizations, args.candidates,
        args.rollout, args.belief, args.skip_confident, args.ignore_search,
        pool, args.seed,
    )
    arms.append(("BC+PIMC ", searcher))
    if not args.no_baseline_bc:
        arms.append(("BC only ", di.BCPolicy(Path(args.checkpoint))))

    print()
    results = {}
    for name, policy in arms:
        print(f"{name} over the pool, seats {seats} vs PolicyRuleBased")
        started = time.time()
        totals, _ = duel_pool_with_decks(
            policy, make_opponent(args.opponent, Path(args.checkpoint)),
            pool, opponent_deck, args.episodes, seats, args.quiet
        )
        low, high = di.wilson(totals["wins"], totals["episodes"])
        print(
            f"  -> {totals['wins']}/{totals['episodes']} = "
            f"{totals['wins'] / max(totals['episodes'], 1):.1%} "
            f"(95% CI {low:.1%}-{high:.1%}) in {time.time() - started:.0f}s"
        )
        results[name] = totals

    print("\n" + "=" * 66)
    print(searcher.report())
    if isinstance(searcher.rollout_policy, _RuleBasedRollout) and searcher.rollout_policy.failures:
        print(
            f"  NOTE: the rule-based rollout policy failed and fell back to "
            f"random on {searcher.rollout_policy.failures} selection(s)"
        )
    for name, totals in results.items():
        low, high = di.wilson(totals["wins"], totals["episodes"])
        print(
            f"{name}: {totals['wins']}/{totals['episodes']} = "
            f"{totals['wins'] / max(totals['episodes'], 1):.1%} "
            f"(95% CI {low:.1%}-{high:.1%})"
        )

    if len(results) == 2:
        search, plain = results["BC+PIMC "], results["BC only "]
        difference, p_value = two_proportion_test(
            search["wins"], search["episodes"], plain["wins"], plain["episodes"]
        )
        print(
            f"\nsearch - BC = {difference:+.1%}  (two-proportion z-test p={p_value:.3f}, "
            f"n={search['episodes']} vs {plain['episodes']})"
        )
        if p_value < 0.05 and difference > 0:
            print("Search beats plain BC (p < 0.05).")
        elif p_value < 0.05:
            print("Search is significantly WORSE than plain BC (p < 0.05).")
        else:
            base = plain["wins"] / max(plain["episodes"], 1)
            print("Not significant. To detect an effect of this size you would need")
            for effect in (0.15, 0.10, 0.05):
                print(
                    f"  {effect:+.0%}: ~{episodes_needed(effect, base)} episodes per arm"
                )
            print(
                "The observed difference is not evidence of an improvement — at "
                "these sample sizes it is within what luck produces."
            )


if __name__ == "__main__":
    main()
