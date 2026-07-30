"""Enum mirrors and tensor encodings for ``Option``/``SelectData`` fields.

The parquet dataset stores these fields as the plain ints the game engine
emits over the wire (see ``cg/api.py``'s ``to_dataclass``/``json_to_dataclass``
round trip) rather than as strings, so encoding them is a matter of picking a
padding/"unknown" sentinel per field, not building a string vocabulary.

The IntEnums below (``OptionType``, ``AreaType``, etc.) are a hand-copied
mirror of ``cg/api.py`` — trivial to keep in sync by eyeballing that file, and
this way a plain dataclass/enum reader doesn't need the rest of this module's
engine dependency. The *sizes* that can't safely be eyeballed and copied
(``CARD_ID_VOCAB_SIZE``, ``ATTACK_ID_VOCAB_SIZE``) are instead read live from
``cg.api.all_card_data()``/``all_attack()`` — the native ``cg.dll``/
``libcg.so`` simulator's own database — since those counts can grow as the
competition's card pool grows, and guessing them from replay data or
``EN_Card_Data.csv`` risks silently undersizing the embedding table the day a
new card/attack id outside the sampled range shows up.
"""

import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import torch

# ``cg`` lives at the repo root, three levels up from this file
# (archetypes/alakazam/policy_network/vocab.py) — not on sys.path when this
# module is imported from within policy_network (e.g. ``python dataset.py``
# run from the repo root only adds policy_network/ itself, per Python's
# script-directory rule).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cg.api import all_attack, all_card_data  # noqa: E402


class EnergyType(IntEnum):
    COLORLESS = 0
    GRASS = 1
    FIRE = 2
    WATER = 3
    LIGHTNING = 4
    PSYCHIC = 5
    FIGHTING = 6
    DARKNESS = 7
    METAL = 8
    DRAGON = 9
    RAINBOW = 10  # Every type.
    TEAM_ROCKET = 11  # Psychic and Darkness.


class CardType(IntEnum):
    POKEMON = 0
    ITEM = 1
    TOOL = 2  # Pokémon Tool.
    SUPPORTER = 3
    STADIUM = 4
    BASIC_ENERGY = 5
    SPECIAL_ENERGY = 6


class AreaType(IntEnum):
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    ENERGY = 8
    TOOL = 9
    PRE_EVOLUTION = 10
    PLAYER = 11
    LOOKING = 12


class OptionType(IntEnum):
    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    SKILL = 15
    SPECIAL_CONDITION = 16


class SelectType(IntEnum):
    MAIN = 0
    CARD = 1
    ATTACHED_CARD = 2
    CARD_OR_ATTACHED_CARD = 3
    ENERGY = 4
    SKILL = 5
    ATTACK = 6
    EVOLVE = 7
    COUNT = 8
    YES_NO = 9
    SPECIAL_CONDITION = 10


class SelectContext(IntEnum):
    MAIN = 0
    SETUP_ACTIVE_POKEMON = 1
    SETUP_BENCH_POKEMON = 2
    SWITCH = 3
    TO_ACTIVE = 4
    TO_BENCH = 5
    TO_FIELD = 6
    TO_HAND = 7
    DISCARD = 8
    TO_DECK = 9
    TO_DECK_BOTTOM = 10
    TO_PRIZE = 11
    NOT_MOVE = 12
    DAMAGE_COUNTER = 13
    DAMAGE_COUNTER_ANY = 14
    DAMAGE = 15
    REMOVE_DAMAGE_COUNTER = 16
    HEAL = 17
    EVOLVES_FROM = 18
    EVOLVES_TO = 19
    DEVOLVE = 20
    ATTACH_FROM = 21
    ATTACH_TO = 22
    DETACH_FROM = 23
    LOOK = 24
    EFFECT_TARGET = 25
    DISCARD_ENERGY_CARD = 26
    DISCARD_TOOL_CARD = 27
    SWITCH_ENERGY_CARD = 28
    DISCARD_CARD_OR_ATTACHED_CARD = 29
    DISCARD_ENERGY = 30
    TO_HAND_ENERGY = 31
    TO_DECK_ENERGY = 32
    SWITCH_ENERGY = 33
    SKILL_ORDER = 34
    ATTACK = 35
    DISABLE_ATTACK = 36
    EVOLVE = 37
    DRAW_COUNT = 38
    DAMAGE_COUNTER_COUNT = 39
    REMOVE_DAMAGE_COUNTER_COUNT = 40
    IS_FIRST = 41
    MULLIGAN = 42
    ACTIVATE = 43
    FIRST_EFFECT = 44
    MORE_DEVOLVE = 45
    COIN_HEAD = 46
    AFFECT_SPECIAL_CONDITION = 47
    RECOVER_SPECIAL_CONDITION = 48
    # cg/api.py notes new members may be appended during the competition.


#: Card/attack id ranges come from the game engine itself — the authoritative
#: source, not the replay data or EN_Card_Data.csv — via
#: ``cg.api.all_card_data()`` / ``cg.api.all_attack()``, which call straight
#: into the native ``cg.dll``/``libcg.so`` simulator and return its full
#: card/attack database. Both id spaces are contiguous 1..N (verified by
#: sorting the ids returned and checking against ``range(1, max+1)``), so a
#: dense embedding table indexed by id works directly; 0 is reserved as the
#: "no card"/"no attack" sentinel for optional fields.
MAX_CARD_ID = len(all_card_data())
MAX_ATTACK_ID = len(all_attack())
CARD_ID_VOCAB_SIZE = MAX_CARD_ID + 1
ATTACK_ID_VOCAB_SIZE = MAX_ATTACK_ID + 1


class CardStage(IntEnum):
    """Pokémon evolution stage. Trainer/Energy cards (``basic``/``stage1``/
    ``stage2`` all ``False`` in ``CardData`` — confirmed against
    ``all_card_data()``: 206 of 1267 cards) get ``NOT_APPLICABLE``, same as
    the ``card_id`` 0 padding/no-card sentinel."""

    NOT_APPLICABLE = 0
    BASIC = 1
    STAGE1 = 2
    STAGE2 = 3


CARD_STAGE_VOCAB_SIZE = len(CardStage)

#: ``CardType``/``EnergyType`` are always populated in ``CardData`` (never
#: ``None`` — confirmed against all 1267 cards), but their own members start
#: at 0, so a card_id-0 "no card" sentinel would collide with a real
#: ``POKEMON``/``COLORLESS`` value if looked up unshifted. +1 here, same
#: convention as every other Optional-enum vocab size in this module.
CARD_TYPE_VOCAB_SIZE = len(CardType) + 1
CARD_ENERGY_TYPE_VOCAB_SIZE = len(EnergyType) + 1

#: Per-``card_id`` static lookup tables (``cg.api.CardData``'s ``cardType``/
#: ``energyType``/``ex``/``megaEx``/``tera``/``aceSpec``/stage flags), built
#: once from ``all_card_data()`` so a board Pokémon's ``id`` can be joined
#: against its card-type flags without shipping the whole card database
#: through the dataset. Index 0 is the ``card_id`` "no card" sentinel, so
#: every table is sized ``CARD_ID_VOCAB_SIZE`` with index 0 left at its
#: default (0/False) — already correct for the shifted ``card_type``/
#: ``energy_type`` tables too, since 0 there means "no card" not a real enum
#: member.
_CARD_STAGE = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_TYPE = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_ENERGY_TYPE = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
#: ``torch.long`` 0/1, not ``torch.bool`` — these are model-input features
#: (unlike a padding/validity mask), so they follow the same ``int(bool)``
#: convention used for every other boolean feature in this codebase
#: (``poisoned``, ``stadium_played``, etc.) rather than the mask dtype.
_CARD_EX = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_MEGA_EX = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_TERA = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_ACE_SPEC = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
for _card in all_card_data():
    if _card.basic:
        _stage = CardStage.BASIC
    elif _card.stage1:
        _stage = CardStage.STAGE1
    elif _card.stage2:
        _stage = CardStage.STAGE2
    else:
        _stage = CardStage.NOT_APPLICABLE
    _CARD_STAGE[_card.cardId] = _stage
    _CARD_TYPE[_card.cardId] = _card.cardType + 1
    _CARD_ENERGY_TYPE[_card.cardId] = _card.energyType + 1
    _CARD_EX[_card.cardId] = int(_card.ex)
    _CARD_MEGA_EX[_card.cardId] = int(_card.megaEx)
    _CARD_TERA[_card.cardId] = int(_card.tera)
    _CARD_ACE_SPEC[_card.cardId] = int(_card.aceSpec)
del _card, _stage


def card_type_flags(
    card_id: torch.Tensor, fields: tuple[str, ...] | None = None
) -> dict[str, torch.Tensor]:
    """Look up static ``CardData`` type flags for a tensor of ``card_id``s
    (any shape) — ``card_id`` 0 (the "no card" sentinel) resolves to
    ``CardStage.NOT_APPLICABLE``/0 (no ``CardType``/``EnergyType``)/``False``
    for every flag.

    ``fields`` restricts which flags get looked up, for card slots where
    some are confirmed structurally dead: e.g. an attached Energy card is
    always ``BASIC_ENERGY``/``SPECIAL_ENERGY`` (never a Pokémon), so
    ``stage``/``ex``/``mega_ex``/``tera`` are ``False``/``NOT_APPLICABLE``
    for all 20 energy cards in ``all_card_data()`` — but ``ace_spec`` isn't
    (3 real ACE SPEC energy cards exist), so it stays available to request.
    """
    all_flags = {
        "stage": _CARD_STAGE[card_id],
        "type": _CARD_TYPE[card_id],
        "energy_type": _CARD_ENERGY_TYPE[card_id],
        "ex": _CARD_EX[card_id],
        "mega_ex": _CARD_MEGA_EX[card_id],
        "tera": _CARD_TERA[card_id],
        "ace_spec": _CARD_ACE_SPEC[card_id],
    }
    if fields is None:
        return all_flags
    return {flag: all_flags[flag] for flag in fields}

#: +1 on every enum used as an Optional field, to reserve 0 as "field not set
#: for this option/selection" — every one of these IntEnums' own members
#: already start at 0, so 0 can't double as both a real value and "unset".
OPTION_TYPE_VOCAB_SIZE = len(OptionType)
AREA_VOCAB_SIZE = len(AreaType) + 1
SELECT_TYPE_VOCAB_SIZE = len(SelectType)
SELECT_CONTEXT_VOCAB_SIZE = len(SelectContext) + 1  # +1 slack: enum may grow

#: 0/1 real values, 2 = unknown (option field was None).
TARGETS_OPPONENT_VOCAB_SIZE = 3

#: Sentinel for optional plain-magnitude int fields (index/count/id fields
#: with no fixed vocab) that carry no enum of their own.
NO_VALUE = -1


def _area(value: int | None) -> int:
    return 0 if value is None else int(value)


def _card_id(value: int | None) -> int:
    return 0 if value is None else int(value)


def _magnitude(value: int | None) -> int:
    return NO_VALUE if value is None else int(value)


def _targets_opponent(value: bool | None) -> int:
    return 2 if value is None else int(value)


@dataclass
class OptionsVocab:
    """One decision's option list, as parallel ``(num_options,)`` tensors.

    Fields mirror ``cg.api.Option`` (see ``features.OPTION_FIELDS``), except
    ``playerIndex`` which ``features._remap_options`` already turns into the
    POV-relative ``targets_opponent``.
    """

    type: torch.Tensor
    number: torch.Tensor
    area: torch.Tensor
    index: torch.Tensor
    targets_opponent: torch.Tensor
    energy_index: torch.Tensor
    count: torch.Tensor
    in_play_area: torch.Tensor
    in_play_index: torch.Tensor
    attack_id: torch.Tensor
    card_id: torch.Tensor
    serial: torch.Tensor

    @classmethod
    def from_options(cls, options: list[dict[str, Any]]) -> "OptionsVocab":
        return cls(
            type=torch.tensor([int(o["type"]) for o in options], dtype=torch.long),
            number=torch.tensor([_magnitude(o["number"]) for o in options], dtype=torch.long),
            area=torch.tensor([_area(o["area"]) for o in options], dtype=torch.long),
            index=torch.tensor([_magnitude(o["index"]) for o in options], dtype=torch.long),
            targets_opponent=torch.tensor(
                [_targets_opponent(o["targets_opponent"]) for o in options], dtype=torch.long
            ),
            energy_index=torch.tensor([_magnitude(o["energyIndex"]) for o in options], dtype=torch.long),
            count=torch.tensor([_magnitude(o["count"]) for o in options], dtype=torch.long),
            in_play_area=torch.tensor([_area(o["inPlayArea"]) for o in options], dtype=torch.long),
            in_play_index=torch.tensor([_magnitude(o["inPlayIndex"]) for o in options], dtype=torch.long),
            attack_id=torch.tensor([_magnitude(o["attackId"]) for o in options], dtype=torch.long),
            card_id=torch.tensor([_card_id(o["cardId"]) for o in options], dtype=torch.long),
            serial=torch.tensor([_magnitude(o["serial"]) for o in options], dtype=torch.long),
        )


@dataclass
class SelectionVocab:
    """One decision's selection, as scalar tensors.

    ``deck`` is encoded as its length (0 unless selecting from the deck);
    ``contextCard``/``effect`` as the referenced card's id (0 if unset) — the
    game state elsewhere already carries each card's full struct, so nothing
    beyond identity is needed to place these mid-decision.
    """

    type: torch.Tensor
    context: torch.Tensor
    min_count: torch.Tensor
    max_count: torch.Tensor
    remain_damage_counter: torch.Tensor
    remain_energy_cost: torch.Tensor
    deck_size: torch.Tensor
    context_card_id: torch.Tensor
    effect_card_id: torch.Tensor

    @classmethod
    def from_selection(cls, selection: dict[str, Any]) -> "SelectionVocab":
        deck = selection["deck"]
        context_card = selection["contextCard"]
        effect = selection["effect"]
        return cls(
            type=torch.tensor(int(selection["type"]), dtype=torch.long),
            context=torch.tensor(int(selection["context"]), dtype=torch.long),
            min_count=torch.tensor(int(selection["minCount"]), dtype=torch.long),
            max_count=torch.tensor(int(selection["maxCount"]), dtype=torch.long),
            remain_damage_counter=torch.tensor(int(selection["remainDamageCounter"]), dtype=torch.long),
            remain_energy_cost=torch.tensor(int(selection["remainEnergyCost"]), dtype=torch.long),
            deck_size=torch.tensor(0 if deck is None else len(deck), dtype=torch.long),
            context_card_id=torch.tensor(_card_id(context_card and context_card["id"]), dtype=torch.long),
            effect_card_id=torch.tensor(_card_id(effect and effect["id"]), dtype=torch.long),
        )


def pad_options(per_decision_options: list[OptionsVocab]) -> dict[str, torch.Tensor]:
    """Stack a chain of ``OptionsVocab`` (one per decision, ragged option
    counts) into ``(chain_len, max_options)`` tensors plus a validity mask.

    Padded slots get each field's own "unset" sentinel (0 for enum-ish
    fields, ``NO_VALUE`` for plain magnitudes) — the mask is what a model
    should actually gate on, since 0 doubles as a real value for some fields.
    """
    chain_len = len(per_decision_options)
    max_options = max((o.type.numel() for o in per_decision_options), default=0)

    def stack(field: str, fill: int) -> torch.Tensor:
        out = torch.full((chain_len, max_options), fill, dtype=torch.long)
        for i, options in enumerate(per_decision_options):
            values = getattr(options, field)
            out[i, : values.numel()] = values
        return out

    mask = torch.zeros((chain_len, max_options), dtype=torch.bool)
    for i, options in enumerate(per_decision_options):
        mask[i, : options.type.numel()] = True

    magnitude_fields = (
        "number", "index", "energy_index", "count",
        "in_play_index", "attack_id", "serial",
    )
    zero_filled_fields = ("area", "targets_opponent", "card_id")

    result = {"type": stack("type", 0), "options_mask": mask}
    for field in magnitude_fields:
        result[field] = stack(field, NO_VALUE)
    for field in zero_filled_fields:
        result[field] = stack(field, 0)
    return result


def stack_selections(selections: list[SelectionVocab]) -> dict[str, torch.Tensor]:
    """Stack a chain of per-decision ``SelectionVocab`` into ``(chain_len,)`` tensors.

    A decision chain is empty for the first decision of an episode (no prior
    history yet) — ``torch.stack`` rejects an empty list, so that case needs
    its own empty ``(0,)`` tensor per field rather than stacking nothing.
    """
    fields = (
        "type", "context", "min_count", "max_count", "remain_damage_counter",
        "remain_energy_cost", "deck_size", "context_card_id", "effect_card_id",
    )
    if not selections:
        return {field: torch.empty(0, dtype=torch.long) for field in fields}
    return {field: torch.stack([getattr(s, field) for s in selections]) for field in fields}
