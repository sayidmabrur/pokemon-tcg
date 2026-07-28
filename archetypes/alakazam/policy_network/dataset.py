"""PyTorch dataset for the decision-level Pokémon TCG Parquet dataset."""

from pathlib import Path
from typing import Any, Callable
from bisect import bisect_right

import torch
from torch.utils.data import Dataset

from features import EMPTY_BOARD_STATE, board_state, diff_board_state, extract_features


class PolicyFeatureDataset(Dataset):
    """Load decision samples written by ``convert_replays.py`` and reshape them
    into the feature groups a policy network consumes.

    Each sample is ``(features, meta)`` for the observation, plus the target:
      - ``features["state"]``: the deciding player's own board (hand included).
      - ``features["opponent_state"]``: the opponent's board, same shape, hand
        redacted to ``None`` by the game engine already (only ``handCount``
        known).
      - ``features["global_state"]``: match-level context not scoped to
        either player (turn, stadium, energy/supporter-played flags, etc).
      - ``features["decision_context"]``: the selection being made and the
        options offered — only ever the deciding player's, never the
        opponent's.
      - ``features["opponent_history"]``: a list of per-turn diffs of the
        opponent's board over their last ``opponent_history_size`` turns
        (oldest first), built from the resulting board state after each of
        their turns — never their raw ``target_action``, since that's an
        internal option index the opponent never actually gets to observe
        about themselves; diffed against ``EMPTY_BOARD_STATE`` for their
        first captured turn so every entry has the same shape. Empty unless
        ``opponent_history_size > 0``.
      - ``meta``: bookkeeping only (``episode_id``, ``frame_index``,
        ``player_index``, ``player_name``) — everything is already reoriented
        to the deciding player's POV, so ``player_index`` is not a feature
        the model should condition on, only a raw-slot pointer for tracing
        rows back through an episode (e.g. building opponent history).

    The target is the list of option *indexes* selected by that player.  The
    index is local to ``features["decision_context"]["options"]``; it is not
    a card ID.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        transform: Callable[[dict[str, Any]], Any] | None = None,
        opponent_history_size: int = 0,
    ) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # Keep importing the module possible without pyarrow.
            raise ImportError(
                "PolicyFeatureDataset requires pyarrow. Install it with "
                "`python -m pip install pyarrow`."
            ) from exc

        parquet_path = Path(parquet_path)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Parquet dataset not found: {parquet_path}")

        self._parquet = pq.ParquetFile(parquet_path)
        self._row_group_offsets = [0]
        for row_group in range(self._parquet.num_row_groups):
            self._row_group_offsets.append(
                self._row_group_offsets[-1] + self._parquet.metadata.row_group(row_group).num_rows
            )
        self._cached_row_group: int | None = None
        self._cached_rows: list[dict[str, Any]] = []
        self.transform = transform
        self.opponent_history_size = opponent_history_size

    def __len__(self) -> int:
        return self._row_group_offsets[-1]

    def _read_row(self, idx: int) -> dict[str, Any]:
        row_group = bisect_right(self._row_group_offsets, idx) - 1
        if row_group != self._cached_row_group:
            self._cached_rows = self._parquet.read_row_group(row_group).to_pylist()
            self._cached_row_group = row_group
        return self._cached_rows[idx - self._row_group_offsets[row_group]]

    def _opponent_history(self, idx: int, row: dict[str, Any]) -> list[dict[str, Any]]:
        """Scan backward through this episode collecting the opponent's board
        state after each of their most recent turns, then diff consecutively.

        Rows for one episode are written contiguously in ascending
        frame_index order by ``convert_replays.py``, so this is a plain
        backward walk, not a search — scanning stays within the same or an
        adjacent row group in practice, since episodes (~150-250 rows) are
        far smaller than a row group.
        """
        episode_id, opponent_index = row["episode_id"], 1 - row["player_index"]
        snapshots: list[tuple[int, dict[str, Any]]] = []
        seen_turns: set[int] = set()
        cursor = idx - 1
        while cursor >= 0 and len(snapshots) < self.opponent_history_size:
            prior = self._read_row(cursor)
            if prior["episode_id"] != episode_id:
                break
            turn = prior["state"]["turn"]
            # Scanning backward, the first frame seen for a given turn is
            # that turn's *last* frame chronologically — exactly the final
            # board state for that turn.
            if prior["player_index"] == opponent_index and turn not in seen_turns:
                seen_turns.add(turn)
                snapshots.append((turn, board_state(prior["state"], opponent_index)))
            cursor -= 1
        snapshots.reverse()

        history = []
        previous_board = EMPTY_BOARD_STATE
        for turn, board in snapshots:
            history.append({"turn": turn, **diff_board_state(previous_board, board)})
            previous_board = board
        return history

    def __getitem__(self, idx: int) -> tuple[Any, torch.Tensor]:
        if not 0 <= idx < len(self):
            raise IndexError(f"Dataset index out of range: {idx}")

        row = self._read_row(idx)
        player_index = row["player_index"]
        features = extract_features(row["state"], row["selection"], row["options"], player_index)
        if self.opponent_history_size > 0:
            features["opponent_history"] = self._opponent_history(idx, row)
        meta = {
            "episode_id": row["episode_id"],
            "frame_index": row["frame_index"],
            "player_index": player_index,
            "player_name": row["player_name"],
        }
        observation = {"features": features, "meta": meta}
        if self.transform is not None:
            observation = self.transform(observation)
        return observation, torch.tensor(row["target_action"], dtype=torch.long)


def decision_collate(
    batch: list[tuple[Any, torch.Tensor]],
) -> tuple[list[Any], list[torch.Tensor]]:
    """Collate variable-size boards/options without incorrectly padding them.

    A future state/option encoder can replace this with its own padded-batch
    collation.  Keeping each sample separate here preserves the option-index
    targets exactly.
    """
    observations, actions = zip(*batch)
    return list(observations), list(actions)
