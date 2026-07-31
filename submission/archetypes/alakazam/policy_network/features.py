"""POV-normalised feature extraction, shared by the offline parquet dataset
(``dataset.py``) and the live ``cg`` game engine (``cg/game.py``).

Both sources hand this module the same shape: a ``state`` dict matching
``observation["current"]`` from the engine (or ``row["state"]`` from the
parquet, which is exactly that field renamed), a ``selection`` dict, an
``options`` list, and the deciding player's index.  Sharing this logic means
the transform a policy trained offline sees is bit-for-bit what it gets fed
during live play.
"""

from typing import Any, Callable


#: Every option field the engine can emit.  The live engine omits fields that
#: don't apply to a given option, while ``convert_replays.py`` writes them as
#: explicit ``None``; normalising to this dense shape in ``_remap_options``
#: keeps offline and live option dicts key-for-key identical.
OPTION_FIELDS = (
    "type", "number", "area", "index", "playerIndex", "toolIndex",
    "energyIndex", "count", "inPlayArea", "inPlayIndex", "attackId",
    "cardId", "serial", "specialConditionType",
)

SELECTION_FIELDS = (
    "type", "context", "minCount", "maxCount", "remainDamageCounter",
    "remainEnergyCost", "deck", "contextCard", "effect",
)

GLOBAL_STATE_FIELDS = (
    "turn", "turnActionCount", "firstPlayer", "stadium", "stadiumPlayed",
    "supporterPlayed", "energyAttached", "retreated", "looking", "result",
)


def _strip_player_index(node: Any) -> Any:
    """Drop the redundant absolute ``playerIndex`` from card/Pokémon structs.

    Every card and Pokémon struct carries a ``playerIndex`` (0/1) that always
    matches whichever board branch it's nested under (``state`` vs
    ``opponent_state``) — it adds no information once the tree is already
    split by POV, and being a raw absolute index it risks reintroducing the
    same slot-labeling bias ``player_index`` metadata was pulled out to avoid.
    Detected by shape (``id`` + ``serial`` present) so option structs, which
    carry a differently-shaped ``playerIndex`` with real relative meaning,
    are left untouched — see ``_remap_options``.
    """
    if isinstance(node, dict):
        if "id" in node and "serial" in node:
            node = {key: value for key, value in node.items() if key != "playerIndex"}
        return {key: _strip_player_index(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_strip_player_index(item) for item in node]
    return node


def _remap_options(options: list[dict[str, Any]], player_index: int) -> list[dict[str, Any]]:
    """Replace each option's absolute ``playerIndex`` with a POV-relative flag.

    An option can target either player's Pokémon (e.g. an attack target), so
    unlike card/Pokémon structs this field carries real signal — it just
    needs to be relative to the deciding player, not an absolute slot index.

    Fields are also filled out to the full ``OPTION_FIELDS`` set so a live
    option (which omits inapplicable keys) and a replay-converted option
    (which stores them as ``None``) come out identical.
    """
    remapped = []
    for option in options:
        option = {field: option.get(field) for field in OPTION_FIELDS}
        raw = option.pop("playerIndex", None)
        option["targets_opponent"] = None if raw is None else raw != player_index
        remapped.append(option)
    return remapped


def board_state(state: dict[str, Any], player_index: int) -> dict[str, Any]:
    """A player's own public+private board — same shape for self and opponent.

    Only ``hand`` differs by perspective: the engine already returns ``None``
    for the opponent's hand (see cg/api.py), so no redaction happens here.
    """
    return _strip_player_index(state["players"][player_index])


def _global_state(state: dict[str, Any]) -> dict[str, Any]:
    return _strip_player_index({field: state[field] for field in GLOBAL_STATE_FIELDS})


STATUS_CONDITIONS = ("poisoned", "burned", "asleep", "paralyzed", "confused")

EMPTY_BOARD_STATE: dict[str, Any] = {
    "active": [], "bench": [], "benchMax": 0, "deckCount": 0,
    "discard": [], "prize": [], "handCount": 0, "hand": None,
    "poisoned": False, "burned": False, "asleep": False,
    "paralyzed": False, "confused": False,
}


def _pokemon_by_serial(board: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pokemon = [p for p in board["active"] if p is not None] + board["bench"]
    return {p["serial"]: p for p in pokemon}


def diff_board_state(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Summarise what changed on a board between two snapshots (e.g. two of
    the opponent's turns), matching Pokémon by ``serial`` since bench order
    isn't stable across turns.

    ``hand_count``/``deck_count``/``prize_count`` stay counts — their
    contents are genuinely hidden, so there's nothing to list. Everything
    with a knowable identity (discarded cards, Pokémon that appeared or left
    the board, which Pokémon gained which energy type) is returned as the
    actual cards, not a count: e.g. an attacker like Mega Abomasnow ex reads
    off *which* energy type was discarded, so "3 cards discarded" is useless
    but "3 Water Energy discarded" tells you they're closing in on the
    attack cost. A Pokémon that evolved rather than being freshly played
    shows up in ``new_pokemon`` with its ``preEvolution`` field populated —
    evolution assigns a new serial, so it's indistinguishable from a fresh
    play at the membership-diff level, but the struct itself still carries
    that link.
    """
    previous_pokemon = _pokemon_by_serial(previous)
    current_pokemon = _pokemon_by_serial(current)
    previous_serials, current_serials = previous_pokemon.keys(), current_pokemon.keys()
    kept_serials = previous_serials & current_serials

    previous_discard_serials = {card["serial"] for card in previous["discard"]}
    energy_attached = [
        {
            "serial": serial,
            "id": current_pokemon[serial]["id"],
            "new_energy_types": current_pokemon[serial]["energies"][len(previous_pokemon[serial]["energies"]):],
        }
        for serial in kept_serials
        if len(current_pokemon[serial]["energies"]) > len(previous_pokemon[serial]["energies"])
    ]

    return {
        "hand_count": current["handCount"],
        "deck_count": current["deckCount"],
        "prize_count": len(current["prize"]),
        "discarded_cards": [
            card for card in current["discard"] if card["serial"] not in previous_discard_serials
        ],
        "new_pokemon": [current_pokemon[serial] for serial in current_serials - previous_serials],
        "removed_pokemon": [previous_pokemon[serial] for serial in previous_serials - current_serials],
        "energy_attached": energy_attached,
        "hp_lost": sum(
            max(0, previous_pokemon[s]["hp"] - current_pokemon[s]["hp"]) for s in kept_serials
        ),
        "status_applied": [
            condition for condition in STATUS_CONDITIONS
            if current[condition] and not previous[condition]
        ],
    }


def decision_context(
    selection: dict[str, Any], options: list[dict[str, Any]], player_index: int
) -> dict[str, Any]:
    """POV-normalised selection/options for whoever is making *this* decision.

    Reused both for the current row's own decision and for building a
    same-actor decision chain within a turn (``dataset.py``) — legitimate to
    look at raw past decisions of the actor *itself*, unlike doing the same
    for the opponent (see ``diff_board_state`` for why opponent history is a
    board diff, not their raw selection/options).
    """
    return {
        "selection": _strip_player_index(selection),
        "options": _remap_options(options, player_index),
    }


def extract_features(
    state: dict[str, Any],
    selection: dict[str, Any],
    options: list[dict[str, Any]],
    player_index: int,
) -> dict[str, Any]:
    """Build the POV-normalised feature groups a policy network consumes.

      - ``state``: the deciding player's own board (hand included).
      - ``opponent_state``: the opponent's board, same shape, hand already
        redacted to ``None`` by the game engine (only ``handCount`` known).
      - ``global_state``: match-level context not scoped to either player.
      - ``decision_context``: the selection being made and the options
        offered — only ever the deciding player's, never the opponent's.
    """
    return {
        "state": board_state(state, player_index),
        "opponent_state": board_state(state, 1 - player_index),
        "global_state": _global_state(state),
        "decision_context": decision_context(selection, options, player_index),
    }


def extract_features_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper for the live engine's raw ``obs`` dict.

    ``obs`` (as returned by ``cg.game.battle_start``/``battle_select``) has
    ``current`` and ``select`` at the top level — the same shape as
    ``record["observation"]`` in a replay JSON.
    """
    state, selection, options = split_observation(observation)
    return extract_features(state, selection, options, state["yourIndex"])


def split_observation(
    observation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Split a live ``obs`` into the ``(state, selection, options)`` triple the
    parquet stores as three separate columns — same field selection as
    ``convert_replays.decision_rows`` so a live row and a converted row are
    interchangeable."""
    state = observation["current"]
    select = observation.get("select") or {}
    selection = {field: select.get(field) for field in SELECTION_FIELDS}
    return state, selection, list(select.get("option") or [])


# --------------------------------------------------------------------------
# History over a sequence of decision rows
#
# A "row" here is the parquet row shape: ``episode_id``, ``player_index``,
# ``state``, ``selection``, ``options``, ``target_action``.  Both the offline
# dataset (rows read lazily out of Parquet) and live play (rows appended as
# the episode unfolds) drive these through a ``read_row(index)`` callable, so
# the two paths cannot drift.
# --------------------------------------------------------------------------


def opponent_history(
    read_row: Callable[[int], dict[str, Any]],
    idx: int,
    row: dict[str, Any],
    size: int,
    source: str = "own_frames",
) -> list[dict[str, Any]]:
    """Scan backward from ``idx`` collecting the opponent's board state once
    per turn for the most recent ``size`` turns, then diff consecutively
    (oldest first).

    Rows for one episode are contiguous in ascending frame order, so this is a
    plain backward walk, not a search.  The first captured turn is diffed
    against ``EMPTY_BOARD_STATE`` so every entry has the same shape.

    ``source`` chooses *which frames* the snapshots are taken at — see
    ``observation.py``, which owns that choice and passes it down:

    ``"opponent_frames"`` snapshots at the opponent's own decision frames,
    giving roughly one entry per opponent turn.  Requires the opponent's
    frames to be present in the row sequence at all, which holds for recorded
    replays but not for a policy invoked only for its own decisions.

    ``"own_frames"`` snapshots at *this* player's frames instead, so each
    entry becomes "how their board changed since I last acted".  The
    opponent's board is public, so this is always observable — a player can
    read the other side of the table without being told what was chosen to
    get there.

    Either way the snapshot is ``board_state(..., opponent_index)``: the
    subject is always the opponent's board.  Only the sampling instants
    differ.

    Note the engine's ``turn`` counter increments once per *player* turn, so
    grouping by it usually yields one snapshot per turn of the chosen actor.
    The exception is a forced reaction interjected during the other player's
    turn: that produces a snapshot labelled with the other player's turn
    number.  It stays chronologically ordered and shape-consistent, so the
    diffs remain valid — they just aren't strictly end-of-turn boundaries.
    """
    if size <= 0:
        return []
    episode_id, player_index = row["episode_id"], row["player_index"]
    opponent_index = 1 - player_index
    # The board captured is the opponent's in both modes; this is whose
    # *frames* we sample it at.
    frame_owner = opponent_index if source == "opponent_frames" else player_index
    snapshots: list[tuple[int, dict[str, Any]]] = []
    seen_turns: set[int] = set()
    cursor = idx - 1
    while cursor >= 0 and len(snapshots) < size:
        prior = read_row(cursor)
        if prior["episode_id"] != episode_id:
            break
        turn = prior["state"]["turn"]
        # Scanning backward, the first frame seen for a given turn is that
        # turn's *last* frame chronologically — the final board for that turn.
        if prior["player_index"] == frame_owner and turn not in seen_turns:
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


def decision_chain(
    read_row: Callable[[int], dict[str, Any]],
    idx: int,
    row: dict[str, Any],
    size: int,
) -> list[dict[str, Any]]:
    """Scan backward for this same actor's ``size`` most recent earlier
    decisions in the episode (oldest first) — the actor's own decision memory
    over the whole match, not just the current turn.

    Legitimate to include the raw selection/options and chosen
    ``target_action`` here, since it's the actor's own memory of what it
    already chose, not something inferred about somebody else (contrast
    ``opponent_history``, which is a board diff for that reason).

    The scan stops only at the episode boundary.  Decisions by the *other*
    player are skipped rather than ending the scan, so the result is a clean
    sequence of this actor's choices with the opponent's interleaved frames
    filtered out.  Each entry carries its own ``turn``, so the model can still
    tell which decisions belong to the turn in progress and which are older —
    the turn boundary is recoverable, it just isn't a cutoff.
    """
    if size <= 0:
        return []
    episode_id, player_index = row["episode_id"], row["player_index"]
    chain = []
    cursor = idx - 1
    while cursor >= 0 and len(chain) < size:
        prior = read_row(cursor)
        if prior["episode_id"] != episode_id:
            break
        if prior["player_index"] == player_index:
            chain.append({
                "turn": prior["state"]["turn"],
                "turn_action_count": prior["state"]["turnActionCount"],
                **decision_context(prior["selection"], prior["options"], player_index),
                "target_action": prior["target_action"],
            })
        cursor -= 1
    chain.reverse()
    return chain
