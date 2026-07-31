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

Both ``opponent_history`` and ``decision_chain`` span the whole match: the
history is one board diff per opponent turn, the chain is this player's own
last ``decision_chain_size`` decisions from their own perspective.

They need memory of earlier decisions
in the episode, which a single ``obs`` doesn't carry.  This class keeps that
memory as a list of rows in the *same* shape ``convert_replays.py`` writes to
Parquet, and runs the identical ``features.opponent_history`` /
``features.decision_chain`` scans over it — so a policy sees bit-for-bit the
same transform offline and live.

Both players' decisions go through one extractor: it's a single observation
stream, and building either player's opponent history requires having seen
the other player's frames.  ``__call__`` derives the deciding player from
``obs["current"]["yourIndex"]``, so the shared instance is correct for both.
"""

from typing import Any

from features import (
    decision_chain,
    extract_features,
    opponent_history,
    split_observation,
)


class LiveFeatureExtractor:
    """Callable feature extractor with per-episode decision memory.

    ``record_action`` is optional: skip it and ``decision_chain`` entries
    carry ``target_action: None`` (the selection/options are still there).
    Call it to match the offline dataset exactly.
    """

    def __init__(
        self,
        opponent_history_size: int = 60,
        decision_chain_size: int = 60,
        player_names: tuple[str | None, str | None] = (None, None),
    ) -> None:
        self.opponent_history_size = opponent_history_size
        self.decision_chain_size = decision_chain_size
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
        idx = len(self._rows) - 1

        features = extract_features(state, selection, options, player_index)
        if self.opponent_history_size > 0:
            features["opponent_history"] = opponent_history(
                self._read_row, idx, row, self.opponent_history_size
            )
        if self.decision_chain_size > 0:
            features["decision_chain"] = decision_chain(
                self._read_row, idx, row, self.decision_chain_size
            )
        meta = {
            "episode_id": row["episode_id"],
            "frame_index": row["frame_index"],
            "player_index": player_index,
            "player_name": row["player_name"],
        }
        return {"features": features, "meta": meta}

    def record_action(self, action: list[int]) -> None:
        """Attach the action chosen for the most recent observation, so it
        shows up as ``target_action`` in later ``decision_chain`` entries."""
        if not self._rows:
            raise RuntimeError("record_action called before any observation was extracted.")
        self._rows[-1]["target_action"] = list(action)
