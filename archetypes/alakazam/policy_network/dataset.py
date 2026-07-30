"""PyTorch dataset for the decision-level Pokémon TCG Parquet dataset."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from bisect import bisect_right

import torch
from torch.utils.data import Dataset

from features import decision_chain, extract_features, opponent_history
from vocab import (
    EnergyType,
    OptionsVocab,
    SelectionVocab,
    _card_id,
    card_type_flags,
    pad_options,
    stack_selections,
)


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
        except (
            ImportError
        ) as exc:  # Keep importing the module possible without pyarrow.
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
                self._row_group_offsets[-1]
                + self._parquet.metadata.row_group(row_group).num_rows
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
            wanted = (
                [player_name] if isinstance(player_name, str) else list(player_name)
            )
            names = self._parquet.read(columns=["player_name"]).to_pandas()[
                "player_name"
            ]
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
                self._table_cache,
                row_group,
                self._parquet.read_row_group(row_group),
                self._table_cache_size,
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
        features = extract_features(
            row["state"], row["selection"], row["options"], player_index
        )
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



def pad_target_actions(decision_chain):
    """``target_action`` is one index per single-select decision but several
    for a multi-select one (e.g. discarding/ordering multiple cards), so it's
    ragged the same way ``options`` is — pad to ``(chain_len, max_targets)``
    with a validity mask rather than assuming a fixed width of 1."""
    per_decision = [
        torch.tensor(decision["target_action"], dtype=torch.long) for decision in decision_chain
    ]
    chain_len = len(per_decision)
    max_targets = max((t.numel() for t in per_decision), default=0)
    target_action = torch.full((chain_len, max_targets), -1, dtype=torch.long)
    mask = torch.zeros((chain_len, max_targets), dtype=torch.bool)
    for i, t in enumerate(per_decision):
        target_action[i, : t.numel()] = t
        mask[i, : t.numel()] = True
    return target_action, mask


def _with_card_flags(fields: dict, id_key: str, prefix: str, card_flag_fields=None) -> dict:
    """Join a ``card_id`` tensor already present in ``fields`` against the
    static ``CardData`` lookup (stage/type/energy_type/ex/mega_ex/tera/
    ace_spec), adding each as ``f"{prefix}_{flag}"`` so the model sees e.g. a
    Supporter vs. a Mega ex Pokémon vs. a Darkness-type card, not just an
    opaque id. ``card_flag_fields`` narrows which flags get looked up (see
    ``vocab.card_type_flags``) for slots where some are structurally dead."""
    fields.update(
        {
            f"{prefix}_{flag}": value
            for flag, value in card_type_flags(fields[id_key], card_flag_fields).items()
        }
    )
    return fields


def _card_id_list_with_flags(cards, prefix, card_flag_fields=None):
    """A ragged list of ``Card`` (entries may be ``None`` for a facedown/
    hidden card, e.g. ``prize``) -> a single ``f"{prefix}_card_id"`` id
    tensor plus its joined ``CardData`` flags."""
    ids = torch.tensor(
        [_card_id(card["id"] if card is not None else None) for card in cards],
        dtype=torch.long,
    )
    return _with_card_flags(
        {f"{prefix}_card_id": ids}, f"{prefix}_card_id", f"{prefix}_card", card_flag_fields
    )


def transform_pokemon(pokemon):
    """One board Pokémon (active or bench slot).

    ``serial`` is kept (unlike ``hand``'s dropped serial) since board
    Pokémon need cross-turn identity tracking (matching this same Pokémon
    to itself next turn), which is exactly what ``opponent_history``'s
    ``diff_board_state`` already relies on ``serial`` for. ``energies`` is
    the attached ``EnergyType`` list (real enum values 0-11, no "unset"
    sentinel needed — absence is an empty list, not a ``None`` entry).
    ``energyCards``/``tools``/``preEvolution`` are ragged ``Card`` lists,
    encoded the same way as ``discard``/``prize``/``hand``."""
    fields = {
        "card_id": torch.tensor(_card_id(pokemon["id"]), dtype=torch.long),
        "serial": torch.tensor(pokemon["serial"], dtype=torch.long),
        "hp": _normalize(torch.tensor(pokemon["hp"], dtype=torch.long), _NORMALIZATION_CAPS["hp"]),
        "max_hp": _normalize(
            torch.tensor(pokemon["maxHp"], dtype=torch.long), _NORMALIZATION_CAPS["max_hp"]
        ),
        "appear_this_turn": torch.tensor(int(pokemon["appearThisTurn"]), dtype=torch.long),
        "energies": torch.tensor(pokemon["energies"], dtype=torch.long),
    }
    _with_card_flags(fields, "card_id", "card")
    # Confirmed against all_card_data(): every one of the 20 energy cards is
    # always BASIC_ENERGY/SPECIAL_ENERGY (never a Pokémon), so stage/ex/
    # mega_ex/tera are always NOT_APPLICABLE/False here — only type/
    # energy_type/ace_spec can actually vary (3 real ACE SPEC energy cards).
    fields.update(
        _card_id_list_with_flags(
            pokemon["energyCards"], "energy", card_flag_fields=("type", "energy_type", "ace_spec")
        )
    )
    fields.update(_card_id_list_with_flags(pokemon["tools"], "tool"))
    fields.update(_card_id_list_with_flags(pokemon["preEvolution"], "pre_evolution"))
    return fields


_EMPTY_POKEMON = {
    "id": None, "serial": 0, "hp": 0, "maxHp": 0, "appearThisTurn": False,
    "energies": [], "energyCards": [], "tools": [], "preEvolution": [],
}


def transform_active_pokemon(active):
    """``active`` is 0-or-1 entries (``Pokemon | None``) — confirmed against
    the dataset it is genuinely empty ~5.8% of the time (18,039 of 308,884
    player-states: momentarily no active Pokémon right after a KO, before a
    new one is set), so it can't just be assumed present. But it's never
    *more* than one either, unlike ``bench`` — so this returns a single
    ``transform_pokemon``-shaped dict (sentinel-filled when absent) plus a
    ``present`` flag, not a list, since there's nothing ragged to represent
    here."""
    present = bool(active) and active[0] is not None
    fields = transform_pokemon(active[0] if present else _EMPTY_POKEMON)
    fields["present"] = torch.tensor(int(present), dtype=torch.long)
    return fields


def transform_player_state(player_state):
    """A player's own public+private board (``features.board_state``) —
    used for both ``state`` (deciding player, hand included) and
    ``opponent_state`` (hand redacted to ``None`` by the engine already;
    only ``hand_count`` is known there, so ``hand_card_id`` comes back an
    empty list even when ``hand_count`` > 0 — that distinction lives in
    ``hand_count``, not in the ragged id list, exactly like ``looking``
    vs. ``looking_card_ids`` in ``transform_global_state``).
    """
    active_pokemon = transform_active_pokemon(player_state["active"])
    bench_pokemon = [transform_pokemon(p) for p in player_state["bench"]]

    fields = {
        "active_pokemon": active_pokemon,
        "bench_pokemon": bench_pokemon,
        "bench_max": _normalize(
            torch.tensor(player_state["benchMax"], dtype=torch.long), _NORMALIZATION_CAPS["bench_max"]
        ),
        "deck_count": _normalize(
            torch.tensor(player_state["deckCount"], dtype=torch.long), _NORMALIZATION_CAPS["deck_count"]
        ),
        "hand_count": _normalize(
            torch.tensor(player_state["handCount"], dtype=torch.long), _NORMALIZATION_CAPS["hand_count"]
        ),
        # Every prize slot is confirmed always facedown (``None``) across the
        # whole dataset (1,394,877 slots checked, 0 revealed) — the game
        # never reveals prize identity here, so a card_id/CardData-flags join
        # would be pure sentinel noise. The only real signal is how many
        # remain.
        "prize_count": _normalize(
            torch.tensor(len(player_state["prize"]), dtype=torch.long), _NORMALIZATION_CAPS["prize_count"]
        ),
        "poisoned": torch.tensor(int(player_state["poisoned"]), dtype=torch.long),
        "burned": torch.tensor(int(player_state["burned"]), dtype=torch.long),
        "asleep": torch.tensor(int(player_state["asleep"]), dtype=torch.long),
        "paralyzed": torch.tensor(int(player_state["paralyzed"]), dtype=torch.long),
        "confused": torch.tensor(int(player_state["confused"]), dtype=torch.long),
    }
    fields.update(_card_id_list_with_flags(player_state["discard"], "discard"))
    fields.update(_card_id_list_with_flags(player_state["hand"] or [], "hand"))
    return fields


_STATUS_CONDITIONS = ("poisoned", "burned", "asleep", "paralyzed", "confused")


def _pad_card_list_chain(card_lists, prefix, card_flag_fields=None):
    """A chain (one entry per opponent turn) of ragged ``Card`` lists (e.g.
    ``discarded_cards``) -> ``(chain_len, max_width)`` id tensor + mask +
    joined card flags, the same padding shape as ``pad_options`` uses for a
    decision chain's options."""
    chain_len = len(card_lists)
    max_width = max((len(cards) for cards in card_lists), default=0)
    ids = torch.zeros((chain_len, max_width), dtype=torch.long)
    mask = torch.zeros((chain_len, max_width), dtype=torch.bool)
    for i, cards in enumerate(card_lists):
        for j, card in enumerate(cards):
            ids[i, j] = _card_id(card["id"])
            mask[i, j] = True
    return _with_card_flags(
        {f"{prefix}_card_id": ids, f"{prefix}_mask": mask}, f"{prefix}_card_id", f"{prefix}_card",
        card_flag_fields,
    )


def _pad_pokemon_list_chain(pokemon_lists, prefix):
    """A chain of ragged ``Pokemon`` lists (``new_pokemon``/``removed_pokemon``
    — a KO'd/played-this-turn event, not full state) -> padded identity/hp
    fields only. Each Pokémon's own attachments (energies/tools/etc.) are
    dropped here since this is a what-changed-since-last-turn summary, not
    full state — the *current* board already carries that via
    ``transform_pokemon`` in ``transform_player_state``."""
    chain_len = len(pokemon_lists)
    max_width = max((len(pokemon) for pokemon in pokemon_lists), default=0)
    card_ids = torch.zeros((chain_len, max_width), dtype=torch.long)
    serials = torch.zeros((chain_len, max_width), dtype=torch.long)
    hps = torch.zeros((chain_len, max_width), dtype=torch.long)
    max_hps = torch.zeros((chain_len, max_width), dtype=torch.long)
    mask = torch.zeros((chain_len, max_width), dtype=torch.bool)
    for i, pokemon_list in enumerate(pokemon_lists):
        for j, pokemon in enumerate(pokemon_list):
            card_ids[i, j] = _card_id(pokemon["id"])
            serials[i, j] = pokemon["serial"]
            hps[i, j] = pokemon["hp"]
            max_hps[i, j] = pokemon["maxHp"]
            mask[i, j] = True
    fields = {
        f"{prefix}_card_id": card_ids,
        f"{prefix}_serial": serials,
        f"{prefix}_hp": _normalize(hps, _NORMALIZATION_CAPS["hp"]),
        f"{prefix}_max_hp": _normalize(max_hps, _NORMALIZATION_CAPS["max_hp"]),
        f"{prefix}_mask": mask,
    }
    return _with_card_flags(fields, f"{prefix}_card_id", f"{prefix}_card")


def _pad_energy_attached_chain(energy_attached_lists):
    """A chain of ragged ``energy_attached`` events
    (``{serial, id, new_energy_types}``) -> padded identity fields plus a
    per-``(turn, slot)`` count vector over ``EnergyType`` (12 members) —
    a count, not a multi-hot flag, since two energies of the same type can
    land on one Pokémon in a single turn (e.g. Energy Search + a manual
    attach) and a bool would silently collapse that to one."""
    chain_len = len(energy_attached_lists)
    max_width = max((len(events) for events in energy_attached_lists), default=0)
    card_ids = torch.zeros((chain_len, max_width), dtype=torch.long)
    serials = torch.zeros((chain_len, max_width), dtype=torch.long)
    mask = torch.zeros((chain_len, max_width), dtype=torch.bool)
    new_energy_type_counts = torch.zeros((chain_len, max_width, len(EnergyType)), dtype=torch.long)
    for i, events in enumerate(energy_attached_lists):
        for j, event in enumerate(events):
            card_ids[i, j] = _card_id(event["id"])
            serials[i, j] = event["serial"]
            mask[i, j] = True
            for energy_type in event["new_energy_types"]:
                new_energy_type_counts[i, j, energy_type] += 1
    fields = {
        "energy_attached_card_id": card_ids,
        "energy_attached_serial": serials,
        "energy_attached_mask": mask,
        "energy_attached_new_energy_type_counts": _normalize(
            new_energy_type_counts, _NORMALIZATION_CAPS["energy_attach_count"]
        ),
    }
    return _with_card_flags(fields, "energy_attached_card_id", "energy_attached_card")


def _pad_status_applied_chain(status_lists):
    chain_len = len(status_lists)
    out = torch.zeros((chain_len, len(_STATUS_CONDITIONS)), dtype=torch.bool)
    for i, statuses in enumerate(status_lists):
        for status in statuses:
            out[i, _STATUS_CONDITIONS.index(status)] = True
    return out


def transform_opponent_history(history):
    """``features.opponent_history`` — a diff per opponent turn (oldest
    first), not a full board snapshot (see ``diff_board_state``): only
    what's knowable (discards, Pokémon that appeared/left, energy gained,
    hp lost, new status conditions) plus counts for what's genuinely hidden
    (``hand_count``/``deck_count``/``prize_count``)."""
    turns = torch.tensor([entry["turn"] for entry in history], dtype=torch.long)
    hand_counts = torch.tensor([entry["hand_count"] for entry in history], dtype=torch.long)
    deck_counts = torch.tensor([entry["deck_count"] for entry in history], dtype=torch.long)
    prize_counts = torch.tensor([entry["prize_count"] for entry in history], dtype=torch.long)
    hp_lost = torch.tensor([entry["hp_lost"] for entry in history], dtype=torch.long)

    fields = {
        "turn": _normalize(turns, _NORMALIZATION_CAPS["turn"]),
        "hand_count": _normalize(hand_counts, _NORMALIZATION_CAPS["hand_count"]),
        "deck_count": _normalize(deck_counts, _NORMALIZATION_CAPS["deck_count"]),
        "prize_count": _normalize(prize_counts, _NORMALIZATION_CAPS["prize_count"]),
        "hp_lost": _normalize(hp_lost, _NORMALIZATION_CAPS["hp_lost"]),
        "status_applied": _pad_status_applied_chain([entry["status_applied"] for entry in history]),
    }
    fields.update(_pad_card_list_chain([entry["discarded_cards"] for entry in history], "discarded"))
    fields.update(_pad_pokemon_list_chain([entry["new_pokemon"] for entry in history], "new_pokemon"))
    fields.update(
        _pad_pokemon_list_chain([entry["removed_pokemon"] for entry in history], "removed_pokemon")
    )
    fields.update(_pad_energy_attached_chain([entry["energy_attached"] for entry in history]))
    return fields


def transform_global_state(global_state):
    """Match-level context, not scoped to either player.

    ``stadium`` is a 0-or-1-length list (confirmed against the parquet: 0 in
    ~44% of rows, 1 in the rest — never assume it's always populated), so
    it's encoded as a single ``stadium_card_id`` (0 = no stadium in play).
    ``looking`` is similarly ragged (0-7 cards observed, ``None`` when not
    looking, and individual entries can themselves be ``None`` for a
    facedown card) — encoded as a plain list of card ids with facedown/no
    card both mapping to the ``_card_id`` sentinel 0, since there's no chain
    to align it against and thus no fixed width to pad to.
    """
    stadium = global_state["stadium"]
    stadium_card_id = _card_id(stadium[0]["id"]) if stadium else 0
    looking = global_state["looking"] or []
    looking_card_ids = torch.tensor(
        [_card_id(card["id"] if card is not None else None) for card in looking],
        dtype=torch.long,
    )
    turn = torch.tensor(global_state["turn"], dtype=torch.long)
    turn_action_count = torch.tensor(global_state["turnActionCount"], dtype=torch.long)
    fields = {
        "turn": _normalize(turn, _NORMALIZATION_CAPS["turn"]),
        "turn_action_count": _normalize(turn_action_count, _NORMALIZATION_CAPS["turn_action_count"]),
        "first_player": torch.tensor(global_state["firstPlayer"] + 1, dtype=torch.long),
        "stadium_card_id": torch.tensor(stadium_card_id, dtype=torch.long),
        "stadium_played": torch.tensor(int(global_state["stadiumPlayed"]), dtype=torch.long),
        "supporter_played": torch.tensor(int(global_state["supporterPlayed"]), dtype=torch.long),
        "energy_attached": torch.tensor(int(global_state["energyAttached"]), dtype=torch.long),
        "retreated": torch.tensor(int(global_state["retreated"]), dtype=torch.long),
        "looking_card_ids": looking_card_ids,
        "result": torch.tensor(global_state["result"] + 1, dtype=torch.long),
    }
    _with_card_flags(fields, "stadium_card_id", "stadium_card")
    _with_card_flags(fields, "looking_card_ids", "looking_card")
    return fields


#: Fixed, domain-informed caps for min-max scaling to [0, 1] — NOT computed
#: from this batch/dataset's observed min/max, since that would shift
#: depending on what happened to be sampled and wouldn't generalize. Each
#: cap is chosen from a real game-rule bound (e.g. a 60-card deck) or, where
#: no hard rule exists, comfortable headroom above the max actually observed
#: in ``data/policy_decisions.parquet`` (noted per field) so a rare
#: unseen-but-larger value clamps to 1.0 instead of over/undershooting.
_NORMALIZATION_CAPS = {
    "number": 10.0,  # observed max 6
    "count": 3.0,  # observed max 2 (single/double energy)
    "min_count": 60.0,  # bounded by deck size (60-card deck, a fixed TCG rule)
    "max_count": 60.0,
    "remain_damage_counter": 10.0,  # observed max 6
    "remain_energy_cost": 6.0,  # observed max 3
    "deck_size": 60.0,  # exact deck size cap (fixed TCG rule)
    # ``turn`` increments once per *player*-turn (alternating), and each
    # player can take at most ~60 of their own turns before decking out
    # (drawing with an empty 60-card deck) — so the shared counter tops out
    # around 2x that. Observed max in the data is 90, consistent with it.
    "turn": 120.0,
    # No clean rule bounds actions-per-turn the way deck size bounds ``turn``
    # — observed max is 51, so this reuses the same 60 scale for consistency
    # rather than inventing an unrelated constant.
    "turn_action_count": 60.0,
    "bench_max": 10.0,  # observed max 8 (standard rule is 5; some effects raise it)
    "deck_count": 60.0,  # exact deck size cap (fixed TCG rule)
    "hand_count": 60.0,  # observed max 31; deck-size scale as the bound
    "prize_count": 6.0,  # fixed rule: matches start with 6 prize cards
    "hp": 400.0,  # CardData.hp observed max 380 across all_card_data()
    "max_hp": 400.0,
    # hp_lost sums drops across every Pokémon that took damage in one
    # opponent turn (spread attacks can hit several), so it isn't bounded by
    # a single card's max HP the way "hp"/"max_hp" are — generous headroom
    # for a multi-target turn rather than a single-hit bound.
    "hp_lost": 1000.0,
    # Per-(turn, slot, energy type) attach count — normally 1, occasionally
    # 2 for a double-attach turn; empirically sampled max is 4 (3000-row
    # scan of data/policy_decisions.parquet), so headroom above that.
    "energy_attach_count": 8.0,
}


def _normalize(value: torch.Tensor, cap: float) -> torch.Tensor:
    """Min-max scale to [0, 1] via a fixed cap. Sentinel-filled (``-1``,
    padding) entries divide out negative and then clamp to 0 — same "rely on
    the mask, not the value, to know what's real" convention already used
    for the ``-1``/``0`` sentinels elsewhere in this module."""
    return (value.float() / cap).clamp(0.0, 1.0)


def _with_normalized_options(fields: dict) -> dict:
    """``number``/``count`` are the only true magnitudes among options'
    fields — ``index``/``energy_index``/``in_play_index``/``attack_id``/
    ``serial`` are position/id pointers meant for an embedding table, not
    continuous quantities, so they're left as raw ints. The raw int is
    replaced in place (not kept alongside), since the model only ever
    trains on the [0, 1] value, never the unscaled magnitude."""
    fields["number"] = _normalize(fields["number"], _NORMALIZATION_CAPS["number"])
    fields["count"] = _normalize(fields["count"], _NORMALIZATION_CAPS["count"])
    return fields


def _with_normalized_selection(fields: dict) -> dict:
    for key in ("min_count", "max_count", "remain_damage_counter", "remain_energy_cost", "deck_size"):
        fields[key] = _normalize(fields[key], _NORMALIZATION_CAPS[key])
    return fields


def _options_with_card_flags(options):
    return _with_normalized_options(_with_card_flags(pad_options(options), "card_id", "card"))


def _selections_with_card_flags(selections):
    fields = stack_selections(selections)
    _with_card_flags(fields, "context_card_id", "context_card")
    _with_card_flags(fields, "effect_card_id", "effect_card")
    return _with_normalized_selection(fields)


def transform_decision_context(decision_context):
    """Same encoding as one ``decision_chain`` entry, but for the single
    current-timestep decision — no ``turn``/``target_action`` here since
    that's the sample's target, not part of the context."""
    options = OptionsVocab.from_options(decision_context["options"])
    selection = SelectionVocab.from_selection(decision_context["selection"])
    return {
        "options": _options_with_card_flags([options]),
        "selection": _selections_with_card_flags([selection]),
    }


def transform_decision_chain(decision_chain):
    turns = torch.tensor([decision["turn"] for decision in decision_chain], dtype=torch.long)
    turn_action_counts = torch.tensor(
        [decision["turn_action_count"] for decision in decision_chain], dtype=torch.long
    )
    target_action, target_action_mask = pad_target_actions(decision_chain)
    options = [OptionsVocab.from_options(decision["options"]) for decision in decision_chain]
    selections = [SelectionVocab.from_selection(decision["selection"]) for decision in decision_chain]
    return {
        "turn": _normalize(turns, _NORMALIZATION_CAPS["turn"]),
        "turn_action_count": _normalize(turn_action_counts, _NORMALIZATION_CAPS["turn_action_count"]),
        "target_action": target_action,
        "target_action_mask": target_action_mask,
        "options": _options_with_card_flags(options),
        "selection": _selections_with_card_flags(selections),
    }

def transform(row):
    """Assemble the six trainable feature groups from one raw observation.

    Returns only what the model actually consumes — no raw ``row``/``meta``
    passthrough — each group already normalized/flag-joined by its own
    ``transform_*`` function."""
    return {
        "decision_chain": transform_decision_chain(row["features"]["decision_chain"]),
        "decision_context": transform_decision_context(row["features"]["decision_context"]),
        "global_state": transform_global_state(row["features"]["global_state"]),
        "opponent_history": transform_opponent_history(row["features"]["opponent_history"]),
        "opponent_state": transform_player_state(row["features"]["opponent_state"]),
        "state": transform_player_state(row["features"]["state"]),
    }

from pprint import pprint

dataset = PolicyFeatureDataset(
    "data/policy_decisions.parquet", player_name="Yushin Ito", transform=transform)
pprint(dataset[79])