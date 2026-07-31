"""``agent()`` entry point for the packaged submission — the behavioral-
cloning ``PolicyNetwork`` playing live.

This file is the submission's ``main.py``; ``make_submission.py`` copies it
under that name alongside everything it imports. It is kept separate from
the repo-root ``main.py`` (the random-baseline template) so the template
stays untouched and this stays diffable.

Two things about the competition harness shape the design:

*Only our own decisions are visible.* Offline, ``LiveFeatureExtractor`` is
driven with both players' frames — the duel loop feeds it every decision
from both sides. Here ``agent()`` is invoked only when it is our turn to
choose, so the opponent's individual decisions are never seen. The
extractor still works (it derives everything from the board state carried in
each ``obs``), but ``opponent_history`` becomes a diff between *our*
successive views of the opponent's board rather than one entry per opponent
decision. That is a genuine train/serve difference, not a bug that can be
fixed here — the information simply is not in the input. See the note in
``make_submission.py`` about validating against it.

*A crash forfeits the match.* Every failure path therefore falls back to a
legal random selection rather than propagating: a weak move scores far
better than no move.
"""

import os
import random
import sys
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    # ``kaggle_environments`` does not import this file — it reads the source
    # and ``exec``s it against a bare globals dict, which has no ``__file__``.
    # Touching it unguarded raises NameError at module level, before any
    # fallback inside agent() could ever run, so the whole submission fails
    # to load rather than merely playing badly.
    _ROOT = Path("/kaggle_simulations/agent")
    if not _ROOT.is_dir():
        _ROOT = Path.cwd()
# The competition unpacks the bundle at a fixed path; when the harness runs
# from some other working directory the plain relative import still has to
# resolve, so the bundle root goes on sys.path explicitly.
# Each directory is tested independently: the bundle root is often already
# on sys.path (Python puts the running script's directory there), and
# short-circuiting on that would skip the policy directory too — which is
# not implied by it, since the policy modules import each other by bare name
# rather than as a package.
for _candidate in (_ROOT, Path("/kaggle_simulations/agent")):
    if not _candidate.is_dir():
        continue
    for _entry in (_candidate, _candidate / "archetypes/alakazam/policy_network"):
        if _entry.is_dir() and str(_entry) not in sys.path:
            sys.path.insert(0, str(_entry))

from cg.api import Observation, to_observation_class


def _resolve(filename: str) -> str:
    """Locate a bundled data file, mirroring the template's lookup: relative
    to the working directory first, then the harness's unpack path."""
    if os.path.exists(filename):
        return filename
    for base in (_ROOT, Path("/kaggle_simulations/agent")):
        candidate = base / filename
        if candidate.exists():
            return str(candidate)
    return filename


def read_deck_csv() -> list[int]:
    """Read the bundled decklist — 60 bare card ids, one per line.

    This is the deck reconstructed from the replays of the player the policy
    was cloned from (see ``archetypes/alakazam/build_alakazam_deck.py``).
    Piloting any other deck would hand the network cards it never saw that
    player play, which is exactly the mismatch this bundle exists to avoid.
    """
    with open(_resolve("deck.csv"), "r") as file:
        lines = file.read().split("\n")
    return [int(lines[i]) for i in range(60)]


class _Policy:
    """Lazily-loaded network plus the per-episode feature memory.

    Loading is deferred to the first real decision because the harness's
    first call only asks for the decklist, and paying torch import + weight
    load there risks a startup timeout for no benefit.
    """

    def __init__(self) -> None:
        self.network = None
        self.threshold = None
        self.extractor = None
        self.episode = 0
        self.failed = False

    def reset(self) -> None:
        """Start a new match. Called when the harness asks for a decklist,
        which is the one unambiguous episode-start signal available."""
        self.episode += 1
        if self.extractor is not None:
            self.extractor.reset(episode_id=self.episode)

    def _load(self) -> None:
        from live import LiveFeatureExtractor
        from policy_experimental import load_policy

        # load_policy also reads the decode threshold calibrated for these
        # weights. It is not a cosmetic setting: left at the naive 0.5 the
        # policy declines ~44% of optional effects outright (it plays the
        # card and takes nothing), because with one target option among N the
        # per-option probabilities are all well under 0.5 by construction.
        self.network, self.threshold = load_policy(_resolve("bc_policy.pt"))
        if self.extractor is None:
            self.extractor = LiveFeatureExtractor()
            self.extractor.reset(episode_id=self.episode)

    def act(self, obs_dict: dict) -> list[int]:
        import torch

        from collate import collate_features
        from dataset import transform
        from policy_experimental import decode_action, selection_counts

        if self.network is None:
            self._load()

        observation = self.extractor(obs_dict)
        features = collate_features([transform(observation)])
        with torch.no_grad():
            logits = self.network(features)

        min_count, max_count = selection_counts(features)
        options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)
        action = decode_action(
            logits, options_mask, min_count, max_count, threshold=self.threshold
        )[0]
        # The chain the *next* decision reads back only lines up if the move
        # actually submitted is the one recorded.
        self.extractor.record_action(action)
        return action


_policy = _Policy()


def _fallback(obs: "Observation") -> list[int]:
    """A legal-by-construction selection, used whenever the network path
    fails. Deliberately allocated no smarts: its only job is to keep the
    match going so one bad decision doesn't forfeit the rest of it."""
    count = random.randint(obs.select.minCount, obs.select.maxCount)
    count = min(count, len(obs.select.option))
    return random.sample(range(len(obs.select.option)), count)


def agent(obs_dict: dict) -> list[int]:
    """Return the option indexes to select for this decision.

    Each element is >= 0 and < ``len(obs.select.option)``; the length is
    within ``[minCount, maxCount]`` with no duplicates.
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        # The harness opens a match by asking for the decklist rather than a
        # move — the signal that any carried-over episode memory is stale.
        _policy.reset()
        return read_deck_csv()

    if _policy.failed:
        return _fallback(obs)

    try:
        action = _policy.act(obs_dict)
    except Exception as exc:  # noqa: BLE001 — a live match must not crash
        # Latch the failure: if the model or its feature pipeline is broken,
        # it will be broken every decision, and retrying it hundreds of
        # times per match risks a timeout on top of the errors. One warning,
        # then the fallback for the rest of the run.
        print(f"[agent] policy failed ({exc!r}) — falling back", file=sys.stderr)
        _policy.failed = True
        return _fallback(obs)

    # The engine rejects an out-of-bracket or duplicated selection outright,
    # so a decode that somehow violated it would forfeit. decode_action
    # enforces this already; this re-check costs nothing and covers the case
    # where the bracket recovered from the features disagrees with the raw
    # obs (they are the same numbers by construction — this catches the
    # "by construction" turning out to be false).
    if not (obs.select.minCount <= len(action) <= obs.select.maxCount) or len(set(action)) != len(action):
        print(
            f"[agent] decoded {len(action)} options outside "
            f"[{obs.select.minCount}, {obs.select.maxCount}] — falling back",
            file=sys.stderr,
        )
        return _fallback(obs)
    return action
