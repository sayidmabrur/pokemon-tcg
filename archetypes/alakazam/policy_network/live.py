"""Live counterpart of ``dataset.py``: turn the ``cg`` engine's raw ``obs``
into exactly the observation dict ``PolicyFeatureDataset`` yields.

Usage in a self-play/eval loop::

    feature_extractor = LiveFeatureExtractor()   # full-match defaults (60/60)
    obs, _ = battle_start(p0, p1)
    feature_extractor.reset()
    while obs["current"]["result"] == -1:
        observation = feature_extractor(obs)      # same shape as dataset[i][0]
        action = policy.act(obs)
        feature_extractor.record_action(action)   # so it lands in decision_chain
        obs = battle_select(action)

Both ``opponent_history`` and ``decision_chain`` span the whole match, so
they need memory of earlier decisions, which a single ``obs`` doesn't carry.
This class keeps that memory as a list of rows in the *same* shape
``convert_replays.py`` writes to Parquet, then hands them to
``observation.build_observation`` — the identical call ``dataset.py`` makes.
Assembling features is deliberately not this module's job: sharing one
implementation is what makes offline and live features the same by
construction rather than by two implementations happening to agree.

Feeding it both players' decisions is optional, and which to do is a
property of the ``ObservationSpec``, not of this class.  Under the default
``own_frames`` spec a policy's ``opponent_history`` is built only from its
own frames, so an extractor that sees just one player's decisions produces
the same features as one that sees both — which is what makes this usable in
the competition harness, where ``agent()`` is never called for the
opponent's choices.  Under an ``opponent_frames`` spec the opponent's frames
must be fed in, or their history comes back empty.  ``__call__`` derives the
deciding player from ``obs["current"]["yourIndex"]``, so a shared instance is
correct for both seats either way.
"""

from typing import Any

from features import split_observation
from observation import DEFAULT_SPEC, ObservationSpec, build_observation


class LiveFeatureExtractor:
    """Per-episode decision memory over live ``obs`` dicts.

    This class owns exactly one thing the offline path does not: turning a
    stream of engine observations into the row sequence the backward history
    scans need, since a single ``obs`` carries no memory of earlier
    decisions.  Assembling features from those rows is ``observation.py``'s
    job, shared with ``dataset.py`` — so "what the policy sees" is defined
    once rather than reimplemented per call site.

    ``record_action`` is optional: skip it and ``decision_chain`` entries
    carry ``target_action: None`` (the selection/options are still there).
    Call it to match the offline dataset exactly.
    """

    def __init__(
        self,
        opponent_history_size: int | None = None,
        decision_chain_size: int | None = None,
        player_names: tuple[str | None, str | None] = (None, None),
        spec: ObservationSpec = DEFAULT_SPEC,
    ) -> None:
        overrides = {}
        if opponent_history_size is not None:
            overrides["opponent_history_size"] = opponent_history_size
        if decision_chain_size is not None:
            overrides["decision_chain_size"] = decision_chain_size
        # Pass the *same* spec used to build the training set. Defaults
        # matching is not enough of a guarantee once specs start varying.
        self.spec = spec.variant(**overrides) if overrides else spec
        self.player_names = player_names
        self._episode_id = -1
        self._rows: list[dict[str, Any]] = []

    def reset(self, episode_id: int | None = None) -> None:
        """Start a new episode.  Rows from the previous one are dropped, so a
        scan can never walk across an episode boundary."""
        self._episode_id = self._episode_id + 1 if episode_id is None else episode_id
        self._rows = []

    @property
    def rows(self) -> list[dict[str, Any]]:
        """This episode's decision rows so far, in parquet row shape (the last
        one's ``target_action`` is ``None`` until ``record_action``)."""
        return self._rows

    def _read_row(self, index: int) -> dict[str, Any]:
        return self._rows[index]

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Append this decision to the episode memory and return its
        ``{"features": ..., "meta": ...}`` observation."""
        if obs.get("select") is None:
            raise ValueError(
                "obs has no 'select' — there is no decision to featurise "
                "(the episode is over or this observation is inactive)."
            )
        state, selection, options = split_observation(obs)
        player_index = state["yourIndex"]
        row = {
            "episode_id": self._episode_id,
            "frame_index": len(self._rows),
            "player_index": player_index,
            "player_name": self.player_names[player_index],
            "state": state,
            "selection": selection,
            "options": options,
            "target_action": None,
        }
        self._rows.append(row)
        return build_observation(self._read_row, len(self._rows) - 1, row, self.spec)

    def record_action(self, action: list[int]) -> None:
        """Attach the action chosen for the most recent observation, so it
        shows up as ``target_action`` in later ``decision_chain`` entries."""
        if not self._rows:
            raise RuntimeError("record_action called before any observation was extracted.")
        self._rows[-1]["target_action"] = list(action)
