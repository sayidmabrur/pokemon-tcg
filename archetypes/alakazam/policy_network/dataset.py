"""PyTorch dataset for the decision-level Pokémon TCG Parquet dataset."""

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable
from bisect import bisect_right

import torch
from torch.utils.data import Dataset

from features import decision_chain, extract_features, opponent_history


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
        first captured turn so every entry has the same shape. The default
        ``opponent_history_size`` of 60 is above any realistic turn count, so
        it covers the whole match; pass 0 to switch the group off.
      - ``features["decision_chain"]``: this same actor's own last
        ``decision_chain_size`` decisions across the whole match (oldest
        first) — turn/selection/options/chosen target, since this is the
        actor's own past choices, not something being inferred about the
        opponent. The opponent's interleaved decisions are filtered out, not
        treated as a boundary, so this is purely the deciding player's
        history from their own perspective; each entry's ``turn`` says which
        turn it came from. Pass 0 to switch the group off.
      - ``meta``: bookkeeping only (``episode_id``, ``frame_index``,
        ``player_index``, ``player_name``) — everything is already reoriented
        to the deciding player's POV, so ``player_index`` is not a feature
        the model should condition on, only a raw-slot pointer for tracing
        rows back through an episode (e.g. building opponent history).

    The target is the list of option *indexes* selected by that player.  The
    index is local to ``features["decision_context"]["options"]``; it is not
    a card ID.

    ``player_name`` restricts the *samples* to one agent (or several) for
    imitation learning, so every target is a decision that agent actually
    made.  It does not restrict what the feature builders may read: the
    opponent's rows stay visible to the backward scans, since dropping them
    would erase the opponent's turns from ``opponent_history``.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        transform: Callable[[dict[str, Any]], Any] | None = None,
        opponent_history_size: int = 60,
        decision_chain_size: int = 60,
        player_name: str | Iterable[str] | None = None,
        cached_row_groups: int = 4,
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
        # Two caches, because the expensive step is Arrow -> Python, not I/O.
        # Row groups are held as Arrow tables (columnar, cheap to keep around)
        # and converted a single row at a time; converting a whole 1000-row
        # group of nested structs to read a few rows costs ~250ms.
        # Row groups also need slack: a backward history scan walks off the
        # front of the group its sample sits in, so a single-group cache is
        # evicted and refilled on every boundary crossing — fatal under a
        # shuffling DataLoader.
        self._table_cache: OrderedDict[int, Any] = OrderedDict()
        self._table_cache_size = max(1, cached_row_groups)
        self._row_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._row_cache_size = 4096
        self.transform = transform
        self.opponent_history_size = opponent_history_size
        self.decision_chain_size = decision_chain_size
        self.player_name = player_name

        self._row_indexes: list[int] | None = None
        if player_name is not None:
            wanted = [player_name] if isinstance(player_name, str) else list(player_name)
            names = self._parquet.read(columns=["player_name"]).to_pandas()["player_name"]
            self._row_indexes = names.index[names.isin(wanted)].tolist()

    def __len__(self) -> int:
        if self._row_indexes is None:
            return self._row_group_offsets[-1]
        return len(self._row_indexes)

    def raw_index(self, idx: int) -> int:
        """Map a sample index to its index in the underlying Parquet file.

        The two differ once ``player_name`` filtering is on.  Feature builders
        walk the *raw* space so the opponent's rows remain visible.
        """
        return idx if self._row_indexes is None else self._row_indexes[idx]

    @staticmethod
    def _touch(cache: OrderedDict, key: int, value: Any, limit: int) -> Any:
        """Insert ``key`` as the most recently used entry, evicting the oldest."""
        cache[key] = value
        if len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _read_row(self, idx: int) -> dict[str, Any]:
        """Read a *raw* Parquet row.  Not sample-indexed — see ``raw_index``."""
        row = self._row_cache.get(idx)
        if row is not None:
            self._row_cache.move_to_end(idx)
            return row

        row_group = bisect_right(self._row_group_offsets, idx) - 1
        table = self._table_cache.get(row_group)
        if table is None:
            table = self._touch(
                self._table_cache, row_group,
                self._parquet.read_row_group(row_group), self._table_cache_size,
            )
        else:
            self._table_cache.move_to_end(row_group)

        offset = idx - self._row_group_offsets[row_group]
        row = table.slice(offset, 1).to_pylist()[0]
        return self._touch(self._row_cache, idx, row, self._row_cache_size)

    def __getitem__(self, sample_idx: int) -> tuple[Any, torch.Tensor]:
        if not 0 <= sample_idx < len(self):
            raise IndexError(f"Dataset index out of range: {sample_idx}")

        idx = self.raw_index(sample_idx)
        row = self._read_row(idx)
        player_index = row["player_index"]
        features = extract_features(row["state"], row["selection"], row["options"], player_index)
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
