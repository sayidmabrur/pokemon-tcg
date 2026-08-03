"""Policy encoder — one architecture family per feature group:

  decision_chain     -> per-step set-pooled features -> TransformerEncoder (sequence)
  decision_context   -> per-option MLP (scored against the pooled state -> action logits)
  global_state       -> flat MLP (fixed-size scalars/categoricals)
  opponent_history   -> per-turn set-pooled diffs -> TransformerEncoder (sequence)
  state/opponent_state -> per-Pokémon MLP + set pooling (permutation-invariant board)

Deep-but-narrow: hidden width ``D`` stays modest while the two sequence
encoders (decision_chain, opponent_history) go 8 layers deep.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import PolicyFeatureDataset, transform
from vocab import (
    AREA_VOCAB_SIZE,
    ATTACK_ID_VOCAB_SIZE,
    CARD_ENERGY_TYPE_VOCAB_SIZE,
    CARD_ID_VOCAB_SIZE,
    CARD_RESISTANCE_VOCAB_SIZE,
    CARD_STAGE_VOCAB_SIZE,
    CARD_TYPE_VOCAB_SIZE,
    CARD_WEAKNESS_VOCAB_SIZE,
    OPTION_TYPE_VOCAB_SIZE,
    SELECT_CONTEXT_VOCAB_SIZE,
    SELECT_TYPE_VOCAB_SIZE,
    TARGETS_OPPONENT_VOCAB_SIZE,
    EnergyType,
)

D = 64  # shared embedding/hidden width

#: Dropout probability, applied to every ReLU-terminated MLP block *and*
#: passed explicitly to the two ``TransformerEncoderLayer`` stacks.
#:
#: The transformers always had this — ``nn.TransformerEncoderLayer`` defaults
#: to ``dropout=0.1`` — but the MLP path (``CardEmbed``, the per-group encoder
#: heads, ``fuse``) had none, so most of the parameter count was unregularized.
#: That showed up as a widening train/val gap: on the crustle run, train loss
#: fell monotonically 0.622 -> 0.564 over epochs 18-22 while val loss bottomed
#: at 0.828 (epoch 19) and rose every epoch after, with val exact flat at
#: ~0.71. Threading one value through makes the amount explicit and tunable
#: from ``bc_train.py`` instead of being an inherited library default on some
#: layers and absent on the rest.
#:
#: Every ``nn.Dropout`` below is *appended* after a block's trailing ReLU
#: rather than inserted mid-``Sequential``, which keeps each ``Linear`` at the
#: index it already had — so existing checkpoints still load. ``score`` is left
#: alone deliberately: it emits the logits, and dropout on an output head just
#: adds noise to the thing being ranked.
DEFAULT_DROPOUT = 0.1


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``x`` (..., L, D) over L, respecting a bool ``mask`` (..., L).
    An all-False row (nothing valid) falls back to zeros rather than NaN."""
    mask = mask.float().unsqueeze(-1)
    denom = mask.sum(dim=-2).clamp(min=1.0)
    return (x * mask).sum(dim=-2) / denom


#: Divisor for the sum half of ``_masked_mean_sum``. Card lists here are hands,
#: discards and benches — a few to a few dozen entries — so this keeps the
#: summed vector in roughly the same range as the mean half instead of letting
#: it dominate the shared Linear that consumes both.
_SUM_SCALE = 10.0


def _masked_mean_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Concat a masked mean with a scaled masked sum, returning ``2 * D``.

    Mean alone is *multiplicity-blind*: a hand holding two Boss's Orders pools
    to the same vector as a hand holding one, because the duplicate entries
    average back to the same point. Counting resources is central to TCG
    decisions ("do I have a second gust?", "how many bodies on the bench?"),
    so the sum is carried alongside — it grows with copies where the mean does
    not, and keeping both means identity and quantity are separable.
    """
    mask_f = mask.float().unsqueeze(-1)
    summed = (x * mask_f).sum(dim=-2)
    denom = mask_f.sum(dim=-2).clamp(min=1.0)
    return torch.cat([summed / denom, summed / _SUM_SCALE], dim=-1)


#: Upper bound for the learned position tables. The observation spec caps both
#: sequences at 60 (``ObservationSpec.decision_chain_size`` /
#: ``opponent_history_size``); 64 leaves slack so a longer spec doesn't index
#: out of range.
MAX_SEQ_LEN = 64


class ReversePositionalEmbedding(nn.Module):
    """Learned position embedding indexed by *recency*, not absolute slot.

    ``nn.TransformerEncoder`` is permutation-invariant on its own — with no
    position signal, a 60-step decision chain is processed as an unordered
    bag, which is not a sequence model at all. The chains here are stored
    oldest-first and right-padded, so absolute slot 0 means "oldest" and the
    slot holding the *most recent* decision moves depending on how long the
    chain happens to be. Indexing from the end instead makes position 0 always
    "what I just did", which is the stable, meaningful frame — and it is
    recency that decisions actually condition on.
    """

    def __init__(self, dim: int, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.embed = nn.Embedding(max_len, dim)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``x`` is (B, L, D); ``mask`` (B, L) marks real steps."""
        lengths = mask.sum(dim=-1, keepdim=True)  # (B, 1)
        slots = torch.arange(x.shape[1], device=x.device).unsqueeze(0)  # (1, L)
        # Most recent valid step -> 0, the one before it -> 1, ... Padding
        # lands on arbitrary indices but is masked out downstream anyway.
        positions = (lengths - 1 - slots).clamp(0, self.max_len - 1)
        return x + self.embed(positions)


def _safe_key_padding_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    """``nn.TransformerEncoder``'s ``src_key_padding_mask`` (True = ignore)
    from a validity mask (True = real) — but a batch row that's entirely
    padding (e.g. one sample's chain is shorter than the batch's max, padded
    to empty) would softmax over an all -inf row internally and produce
    NaN. Force-unmask position 0 for those rows; the *caller*'s pooling
    still uses the true ``valid_mask`` (all-False there), so it correctly
    contributes zero — this only prevents NaN inside the transformer."""
    key_padding_mask = ~valid_mask
    fully_padded = ~valid_mask.any(dim=-1)
    if fully_padded.any():
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[fully_padded, 0] = False
    return key_padding_mask


class CardEmbed(nn.Module):
    """Shared card representation: id + CardData-derived categorical flags,
    used for every ``card_id`` slot (Pokémon, options, hand, stadium, ...).
    Missing narrowed flags (e.g. an energy card has no ``stage``/``ex``) fall
    back to zeros so the same module works for full and narrowed joins."""

    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.id = nn.Embedding(CARD_ID_VOCAB_SIZE, dim)
        self.stage = nn.Embedding(CARD_STAGE_VOCAB_SIZE, dim // 4)
        self.type_embed = nn.Embedding(CARD_TYPE_VOCAB_SIZE, dim // 4)
        self.energy_type = nn.Embedding(CARD_ENERGY_TYPE_VOCAB_SIZE, dim // 4)
        # Weakness/resistance are the game's damage multipliers (x2 / -30);
        # without them the model cannot evaluate a matchup at all.
        self.weakness = nn.Embedding(CARD_WEAKNESS_VOCAB_SIZE, dim // 4)
        self.resistance = nn.Embedding(CARD_RESISTANCE_VOCAB_SIZE, dim // 4)
        self.proj = nn.Linear(dim + 5 * (dim // 4) + 7, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, fields: dict) -> torch.Tensor:
        """``fields`` is already sliced to plain keys (``id``/``stage``/...)
        by ``_card_fields`` — missing narrowed flags fall back to zeros."""
        card_id = fields["id"]
        zeros = torch.zeros_like(card_id)
        float_zeros = torch.zeros_like(card_id, dtype=torch.float)
        stage = fields.get("stage", zeros)
        stage_norm = fields.get("stage_norm", float_zeros)
        ctype = fields.get("type", zeros)
        energy_type = fields.get("energy_type", zeros)
        weakness = fields.get("weakness", zeros)
        resistance = fields.get("resistance", zeros)
        flags = torch.stack(
            [
                fields.get("ex", zeros).float(),
                fields.get("mega_ex", zeros).float(),
                fields.get("tera", zeros).float(),
                fields.get("ace_spec", zeros).float(),
                stage_norm,
                # Printed HP on the shared HP/damage scale, and retreat cost —
                # "how hard is this to kill" and "how hard is it to escape".
                fields.get("hp_norm", float_zeros),
                fields.get("retreat_cost_norm", float_zeros),
            ],
            dim=-1,
        )
        out = torch.cat(
            [
                self.id(card_id), self.stage(stage), self.type_embed(ctype),
                self.energy_type(energy_type), self.weakness(weakness),
                self.resistance(resistance), flags,
            ],
            dim=-1,
        )
        return self.drop(F.relu(self.proj(out)))


def _card_fields(fields: dict, prefix: str) -> dict:
    """Slice out ``{prefix}_*`` keys into the flat ``{"id": ..., "stage": ...}``
    dict ``CardEmbed`` expects, from a flat dict keyed like ``f"{prefix}_id"``."""
    return {k[len(prefix) + 1:]: v for k, v in fields.items() if k.startswith(prefix + "_")}


class OptionEncoder(nn.Module):
    """One option -> vector. Options are a *set* within a decision (order is
    meaningless — pooled with a mask), not a sequence."""

    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.card = CardEmbed(dim, dropout)
        self.type_embed = nn.Embedding(OPTION_TYPE_VOCAB_SIZE, dim // 4)
        self.area = nn.Embedding(AREA_VOCAB_SIZE, dim // 4)
        self.targets_opponent = nn.Embedding(TARGETS_OPPONENT_VOCAB_SIZE, dim // 4)
        # ``attack_id`` is an id, so it gets an embedding table like every
        # other id — it used to be fed as the scalar ``attack_id / 10.0``.
        self.attack = nn.Embedding(ATTACK_ID_VOCAB_SIZE, dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim + 4 * (dim // 4) + 8 + len(EnergyType), dim), nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, options: dict) -> torch.Tensor:
        card = self.card(_card_fields(options, "card"))
        # ``index``/``energy_index``/``in_play_index``/``serial`` are
        # position/id pointers (NO_VALUE=-1 sentinel), not a fixed vocab —
        # crude /10 scaling here (not a proper embedding) is the "minimalist"
        # part, but they matter a lot: these are usually what actually
        # distinguishes one option from another (e.g. "which hand card by
        # index"), unlike type/area/card_id which are often shared across
        # every option in a decision.
        #
        # ``attack_damage_norm``/``attack_energy_cost_norm`` are the static
        # ``Attack`` properties joined in ``dataset._with_attack_flags``.
        # Damage is on the same scale as every HP field (see ``vocab.HP_CAP``),
        # so "damage >= defender HP" is a comparison the model can make
        # directly instead of having to memorize it per attack id.
        scalars = torch.stack(
            [
                options["number"], options["count"],
                options["index"].float() / 10.0, options["energy_index"].float() / 10.0,
                options["in_play_index"].float() / 10.0,
                options["serial"].float() / 10.0,
                options["attack_damage_norm"], options["attack_energy_cost_norm"],
            ],
            dim=-1,
        )
        out = torch.cat(
            [
                card,
                self.type_embed(options["type"]),
                self.area(options["area"]),
                self.targets_opponent(options["targets_opponent"]),
                self.attack(options["attack_id_safe"]),
                scalars,
                # Per-energy-type cost vector: paying {G}{C}{C} is a different
                # problem from paying {F}{C}{C} given what's on the board.
                options["attack_energy_type_counts"],
            ],
            dim=-1,
        )
        return self.mlp(out)


class SelectionEncoder(nn.Module):
    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.type_embed = nn.Embedding(SELECT_TYPE_VOCAB_SIZE, dim // 4)
        self.context = nn.Embedding(SELECT_CONTEXT_VOCAB_SIZE, dim // 4)
        self.context_card = CardEmbed(dim, dropout)
        self.effect_card = CardEmbed(dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim + 2 * (dim // 4) + 5, dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, selection: dict) -> torch.Tensor:
        scalars = torch.stack(
            [
                selection["min_count"], selection["max_count"], selection["remain_damage_counter"],
                selection["remain_energy_cost"], selection["deck_size"],
            ],
            dim=-1,
        )
        out = torch.cat(
            [
                self.type_embed(selection["type"]),
                self.context(selection["context"]),
                self.context_card(_card_fields(selection, "context_card")),
                self.effect_card(_card_fields(selection, "effect_card")),
                scalars,
            ],
            dim=-1,
        )
        return self.mlp(out)


class DecisionChainEncoder(nn.Module):
    """Actor's own last N decisions — a genuine temporal sequence, so this is
    the one group that gets a ``TransformerEncoder``."""

    def __init__(self, dim=D, nhead=2, layers=8, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.option = OptionEncoder(dim, dropout)
        self.selection = SelectionEncoder(dim, dropout)
        # 3 * dim: the menu summary, the selection, and *which option was
        # actually chosen* (see forward) — plus turn/turn_action_count.
        self.in_proj = nn.Linear(3 * dim + 2, dim)
        self.position = ReversePositionalEmbedding(dim)
        # norm_first=True (pre-LN). torch defaults to post-LN, which at this
        # depth needs LR warmup to train stably and otherwise varies wildly
        # run to run; pre-LN is stable at depth on its own. bc_train.py adds
        # warmup and grad clipping on top.
        encoder_layer = nn.TransformerEncoderLayer(
            dim, nhead, dim_feedforward=2 * dim, batch_first=True, norm_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)

    def forward(self, decision_chain: dict) -> torch.Tensor:
        chain_mask = decision_chain["chain_mask"]  # (B, chain_len)
        if chain_mask.shape[1] == 0:  # every sample in the batch has an empty chain
            return torch.zeros(chain_mask.shape[0], D, device=chain_mask.device)
        option_vecs = self.option(decision_chain["options"])  # (B, chain_len, max_options, D)
        option_summary = _masked_mean(option_vecs, decision_chain["options"]["options_mask"])
        selection_summary = self.selection(decision_chain["selection"])  # (B, chain_len, D)
        # The chain is the actor's memory of its own past *choices*, but the
        # two summaries above only describe the menu it was offered — every
        # option pooled together, with no indication of which one it took.
        # ``target_action`` holds those choices as option indexes, so gather
        # the chosen options' own encodings and pool them: that is the signal
        # that makes this a decision history rather than a menu history.
        chosen_summary = self._chosen(option_vecs, decision_chain)
        step = torch.cat(
            [option_summary, chosen_summary, selection_summary,
             decision_chain["turn"].unsqueeze(-1),
             decision_chain["turn_action_count"].unsqueeze(-1)],
            dim=-1,
        )
        step = F.relu(self.in_proj(step))  # (B, chain_len, D)
        step = self.position(step, chain_mask)
        encoded = self.transformer(step, src_key_padding_mask=_safe_key_padding_mask(chain_mask))
        return _masked_mean(encoded, chain_mask)

    @staticmethod
    def _chosen(option_vecs: torch.Tensor, decision_chain: dict) -> torch.Tensor:
        """Pool the encodings of the options the actor actually selected.

        ``target_action`` is (B, L, T) option indexes padded with -1 (a
        decision can select several options), with ``target_action_mask``
        marking the real ones. Indexes are clamped into range before the
        gather so the -1 padding cannot index backwards; those slots are then
        excluded by the mask.
        """
        targets = decision_chain["target_action"]
        target_mask = decision_chain["target_action_mask"]
        num_options = option_vecs.shape[-2]
        if targets.shape[-1] == 0 or num_options == 0:
            return option_vecs.new_zeros((*option_vecs.shape[:2], option_vecs.shape[-1]))
        index = targets.clamp(0, num_options - 1)
        index = index.unsqueeze(-1).expand(*index.shape, option_vecs.shape[-1])
        chosen = option_vecs.gather(dim=2, index=index)  # (B, L, T, D)
        return _masked_mean(chosen, target_mask)


class DecisionContextEncoder(nn.Module):
    """The current decision — same option/selection encoders as the chain,
    but this also exposes per-option vectors (for scoring against the
    pooled state to produce action logits), not just a pooled summary."""

    def __init__(self, dim=D, nhead=2, layers=2, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.option = OptionEncoder(dim, dropout)
        self.selection = SelectionEncoder(dim, dropout)
        # Self-attention *across the options of this decision*. Without it each
        # option is encoded in isolation and scored as ``score(option_i,
        # state)``, so option i never sees option j except through a mean-pooled
        # summary — yet "is this the best play" is inherently comparative:
        # whether Ultra Ball is right depends on what else is on the menu.
        # Deliberately NO positional encoding here, unlike the two sequence
        # encoders: options are a *set*, so permutation-equivariance is correct
        # (the ordering of the list carries no meaning beyond the index
        # pointer, which is already a feature).
        encoder_layer = nn.TransformerEncoderLayer(
            dim, nhead, dim_feedforward=2 * dim, batch_first=True, norm_first=True,
            dropout=dropout,
        )
        self.option_attention = nn.TransformerEncoder(encoder_layer, layers)

    def forward(self, decision_context: dict):
        # ``options``/``selection`` carry a leading size-1 "chain position"
        # dim (this is always a single decision, not a real chain) — squeeze
        # dim 1 specifically, not dim 0 (that's the batch dim).
        option_vecs = self.option(decision_context["options"]).squeeze(1)  # (B, max_options, D)
        options_mask = decision_context["options"]["options_mask"].squeeze(1)  # (B, max_options)
        option_vecs = self.option_attention(
            option_vecs, src_key_padding_mask=_safe_key_padding_mask(options_mask)
        )
        option_summary = _masked_mean(option_vecs, options_mask)
        selection_summary = self.selection(decision_context["selection"]).squeeze(1)  # (B, D)
        pooled = torch.cat([option_summary, selection_summary], dim=-1)
        return pooled, option_vecs, options_mask


class GlobalStateEncoder(nn.Module):
    """Fixed-size scalars/categoricals — a flat MLP, no sequence/set structure."""

    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.first_player = nn.Embedding(3, dim // 4)
        self.result = nn.Embedding(4, dim // 4)
        self.stadium_card = CardEmbed(dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim + 2 * (dim // 4) + 6, dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, global_state: dict) -> torch.Tensor:
        # ``looking_card``'s id field is ``looking_card_ids`` (plural, from
        # ``transform_global_state``), unlike every other card slot's
        # ``..._id`` — so ``_card_fields`` slices it out as ``ids``, not the
        # ``id`` key ``CardEmbed`` expects. Patch it back before embedding.
        looking_fields = _card_fields(global_state, "looking_card")
        looking_fields["id"] = looking_fields.pop("ids")
        looking_vecs = self.stadium_card(looking_fields)  # (B, max_looking, D)
        looking_card = _masked_mean(looking_vecs, global_state["looking_card_mask"])
        bits = torch.stack(
            [
                global_state["turn"], global_state["turn_action_count"],
                global_state["stadium_played"].float(), global_state["supporter_played"].float(),
                global_state["energy_attached"].float(), global_state["retreated"].float(),
            ],
            dim=-1,
        )
        out = torch.cat(
            [
                # Concatenated, not summed: the stadium in play and the cards
                # being looked at are unrelated facts, and adding two CardEmbed
                # outputs makes them indistinguishable from each other.
                self.stadium_card(_card_fields(global_state, "stadium_card")),
                looking_card,
                self.first_player(global_state["first_player"]),
                self.result(global_state["result"]),
                bits,
            ],
            dim=-1,
        )
        return self.mlp(out)


class PokemonEncoder(nn.Module):
    """One board Pokémon (active or bench slot) -> vector."""

    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.card = CardEmbed(dim, dropout)
        self.energy = nn.Embedding(len(EnergyType), dim // 4)
        self.energy_card = CardEmbed(dim, dropout)
        self.tool_card = CardEmbed(dim, dropout)
        self.pre_evolution_card = CardEmbed(dim, dropout)
        # Every attached list is pooled mean+sum: how *many* energy are on a
        # Pokémon decides whether its attack is payable at all, and a mean
        # over the attached energy is identical for one Fire and three Fire.
        self.mlp = nn.Sequential(
            nn.Linear(dim + 2 * (dim // 4) + 6 * dim + 2, dim), nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _pool_list(self, module, pokemon: dict, prefix: str) -> torch.Tensor:
        vecs = module(_card_fields(pokemon, prefix))  # (..., max_width, D)
        return _masked_mean_sum(vecs, pokemon[f"{prefix}_mask"])

    def forward(self, pokemon: dict) -> torch.Tensor:
        energy_vecs = self.energy(pokemon["energies"])  # (..., max_energies, dim // 4)
        energy_vec = _masked_mean_sum(energy_vecs, pokemon["energies_mask"])
        out = torch.cat(
            [
                self.card(_card_fields(pokemon, "card")),
                energy_vec,
                self._pool_list(self.energy_card, pokemon, "energy_card"),
                self._pool_list(self.tool_card, pokemon, "tool_card"),
                self._pool_list(self.pre_evolution_card, pokemon, "pre_evolution_card"),
                pokemon["hp"].unsqueeze(-1), pokemon["max_hp"].unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.mlp(out)


class PlayerStateEncoder(nn.Module):
    """A player's board (own or opponent's, same shape) — active/bench
    Pokémon are a permutation-invariant *set*, so mean-pooled, not a
    sequence model."""

    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.pokemon = PokemonEncoder(dim, dropout)
        self.hand_card = CardEmbed(dim, dropout)
        self.discard_card = CardEmbed(dim, dropout)
        # active (dim) + bench/hand/discard pooled mean+sum (2 * dim each) + 9
        # status scalars. Hand and bench counts are decision-critical: "a
        # second Ultra Ball" and "a fourth body on the bench" are exactly the
        # facts a mean pool erases.
        self.mlp = nn.Sequential(
            nn.Linear(7 * dim + 9, dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def _pool_cards(self, module, fields: dict, prefix: str) -> torch.Tensor:
        vecs = module(_card_fields(fields, prefix))  # (B, max_width, D)
        return _masked_mean_sum(vecs, fields[f"{prefix}_mask"])

    def forward(self, player_state: dict) -> torch.Tensor:
        active = (
            self.pokemon(player_state["active_pokemon"])
            * player_state["active_pokemon"]["present"].unsqueeze(-1)
        )
        # ``bench_pokemon`` is already a batched dict of (B, max_bench, ...)
        # tensors (collate.py's collate_bench_pokemon) — one PokemonEncoder
        # call over the whole grid, not a Python loop per slot, since every
        # op inside it (CardEmbed, _masked_mean) is already leading-dim
        # agnostic.
        bench_vecs = self.pokemon(player_state["bench_pokemon"])  # (B, max_bench, D)
        bench_vec = _masked_mean_sum(
            bench_vecs, player_state["bench_pokemon"]["bench_pokemon_mask"]
        )
        hand_vec = self._pool_cards(self.hand_card, player_state, "hand_card")
        discard_vec = self._pool_cards(self.discard_card, player_state, "discard_card")
        status = torch.stack(
            [
                player_state["bench_max"], player_state["deck_count"], player_state["hand_count"],
                player_state["prize_count"], player_state["poisoned"].float(),
                player_state["burned"].float(), player_state["asleep"].float(),
                player_state["paralyzed"].float(), player_state["confused"].float(),
            ],
            dim=-1,
        )
        out = torch.cat([active, bench_vec, hand_vec, discard_vec, status], dim=-1)
        return self.mlp(out)


class OpponentHistoryEncoder(nn.Module):
    """Per-opponent-turn diffs — a temporal sequence, so ``TransformerEncoder``
    again, with each turn's ragged card/Pokémon lists mean-pooled first."""

    def __init__(self, dim=D, nhead=2, layers=8, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.discarded_card = CardEmbed(dim, dropout)
        self.new_pokemon_card = CardEmbed(dim, dropout)
        self.removed_pokemon_card = CardEmbed(dim, dropout)
        self.energy_attached_card = CardEmbed(dim, dropout)
        # 8 * dim: four card groups, each pooled as mean+sum (2 * dim apiece) —
        # "they discarded three Water Energy" is a different fact from
        # "they discarded a Water Energy", and a mean cannot tell them apart.
        self.in_proj = nn.Linear(8 * dim + 5 + 5, dim)
        self.position = ReversePositionalEmbedding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            dim, nhead, dim_feedforward=2 * dim, batch_first=True, norm_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)

    def forward(self, history: dict) -> torch.Tensor:
        history_mask = history["history_mask"]  # (B, chain_len)
        if history_mask.shape[1] == 0:  # every sample in the batch has empty history
            return torch.zeros(history_mask.shape[0], D, device=history_mask.device)
        discarded = _masked_mean_sum(
            self.discarded_card(_card_fields(history, "discarded_card")), history["discarded_mask"]
        )
        new_pokemon = _masked_mean_sum(
            self.new_pokemon_card(_card_fields(history, "new_pokemon_card")),
            history["new_pokemon_mask"],
        )
        removed_pokemon = _masked_mean_sum(
            self.removed_pokemon_card(_card_fields(history, "removed_pokemon_card")),
            history["removed_pokemon_mask"],
        )
        energy_attached = _masked_mean_sum(
            self.energy_attached_card(_card_fields(history, "energy_attached_card")),
            history["energy_attached_mask"],
        )
        step = torch.cat(
            [
                discarded, new_pokemon, removed_pokemon, energy_attached,
                history["status_applied"].float(),
                torch.stack(
                    [history["turn"], history["hand_count"], history["deck_count"],
                     history["prize_count"], history["hp_lost"]],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        step = F.relu(self.in_proj(step))  # (B, chain_len, D)
        step = self.position(step, history_mask)
        encoded = self.transformer(step, src_key_padding_mask=_safe_key_padding_mask(history_mask))
        return _masked_mean(encoded, history_mask)


#: Finite stand-in for -inf on masked-out options. ``log_softmax`` over a row
#: containing -inf is fine mathematically but produces NaN gradients through
#: the masked entries, so the competition is restricted with a large negative
#: number instead (softmax weight ~1e-40, i.e. numerically excluded).
_MASK_LOGIT = -1e9


#: Option fields that decide whether two options are the *same play*.
#: Deliberately excludes ``index``/``serial``/``energy_index``: those are
#: pointers into a pile of cards, so two options differing only there
#: (hand slot 3 vs slot 5, both holding an Ultra Ball) do exactly the same
#: thing. ``in_play_index`` is kept — it names *which* Pokémon is targeted,
#: and two board Pokémon differ in HP and attached energy even when they
#: share a card id. The ``card_*``/``attack_*`` flags are omitted as
#: redundant: they are pure functions of ``card_id``/``attack_id_safe``.
_EQUIVALENCE_FIELDS = (
    "type", "area", "targets_opponent", "card_id",
    "number", "count", "attack_id_safe", "in_play_index",
)


def equivalence_mask(
    options: dict, targets: torch.Tensor, options_mask: torch.Tensor
) -> torch.Tensor:
    """``(B, N)`` bool marking every option that is the same play as the
    expert's chosen option.

    Measured on ``policy_decisions.parquet``, **50.5%** of single-select
    decisions offer at least one option behaviourally identical to the one
    the expert took, in tie groups 2-8 wide — and the expert resolves those
    ties essentially arbitrarily (lowest index only 44.3% of the time). So
    scoring a prediction against the one *index* the expert happened to click
    charges the model for plays that are literally the same move, and caps
    exact-index accuracy at ~63.7% no matter how good the policy is.

    Only the first target is expanded, so this is meaningful for
    single-select decisions; callers restrict its use accordingly.
    """
    feats = torch.stack(
        [options[field].squeeze(1).float() for field in _EQUIVALENCE_FIELDS], dim=-1
    )  # (B, N, F)
    target_index = targets.argmax(dim=-1)
    chosen = feats.gather(
        1, target_index.view(-1, 1, 1).expand(-1, 1, feats.shape[-1])
    )  # (B, 1, F)
    return (feats == chosen).all(dim=-1) & options_mask


def masked_selection_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    equivalent: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log-likelihood of the expert's selection, per decision.

    The right likelihood depends on what kind of choice the decision is, and
    these two cases are genuinely different distributions:

    *Single-select* (93.6% of decisions in ``policy_decisions.parquet``: the
    expert picks exactly one of a mean 8.2 options) is **categorical**. The
    options compete for one slot, so softmax cross-entropy is its likelihood
    — and softmax is what encodes that competition. Scoring it as N
    independent Bernoullis instead, as ``masked_bce_loss`` did, is both the
    wrong model and badly conditioned: with ~1 positive among 8 options, ~86%
    of the gradient comes from easy negatives, which compresses every
    probability toward zero. That is why a decode threshold had to be swept
    and calibrated per checkpoint, landing near 0.02-0.12 with a flat curve —
    the ranking was barely separated. Under cross-entropy the probabilities
    are a real distribution over options and ``argmax`` is simply correct.

    *Multi-select* (a discard-two, an ordering) really is a set choice, so it
    keeps a Bernoulli likelihood — summed over valid options rather than
    averaged, which makes it a joint log-likelihood on the same scale as the
    cross-entropy term so the two can be averaged together per decision
    without one silently dominating. Decisions selecting *nothing* (0.3%,
    declining an optional effect) fall here too, where an all-zero target is
    exactly right.

    ``equivalent`` (from ``equivalence_mask``) makes the single-select term
    indifferent between options that are the *same play*: instead of
    maximising the probability of one arbitrary index, it maximises the total
    probability of the whole tie group, ``-log sum_{i in group} p_i``. Since
    the expert resolves those ties arbitrarily, the index-specific objective
    was asking the model to fit coin flips — capacity spent on noise, and a
    hard ceiling on the metric. This keeps the likelihood proper (the group
    probabilities are a partition of the same softmax) while dropping the part
    that was never learnable.
    """
    safe = logits.masked_fill(~mask, _MASK_LOGIT)
    num_positive = targets.sum(dim=-1)
    single = num_positive == 1

    per_decision = logits.new_zeros(logits.shape[0])
    if single.any():
        if equivalent is None:
            per_decision[single] = F.cross_entropy(
                safe[single], targets[single].argmax(dim=-1), reduction="none"
            )
        else:
            log_probs = F.log_softmax(safe[single], dim=-1)
            group = equivalent[single]
            per_decision[single] = -torch.logsumexp(
                log_probs.masked_fill(~group, _MASK_LOGIT), dim=-1
            )
    multi = ~single
    if multi.any():
        per_option = F.binary_cross_entropy_with_logits(
            logits[multi].masked_fill(~mask[multi], 0.0), targets[multi], reduction="none"
        )
        per_decision[multi] = (per_option * mask[multi]).sum(dim=-1)
    return per_decision.mean()


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits over valid (masked-in) option positions only.

    ``logits`` comes back from ``PolicyNetwork`` with masked-out positions
    set to ``-inf`` (correct for inference — argmax/topk naturally ignore
    them). But ``F.binary_cross_entropy_with_logits`` internally computes
    ``-x * z``, and ``-inf * 0`` (a masked logit times its zero target) is
    ``NaN``, not the ``0`` it conceptually should contribute — so those
    ``-inf``s have to be swapped for a finite placeholder (any value works,
    since the masked term is then explicitly excluded from the mean, not
    just hoped-to-cancel) before computing the loss."""
    safe_logits = logits.masked_fill(~mask, 0.0)
    per_option = F.binary_cross_entropy_with_logits(safe_logits, targets, reduction="none")
    return (per_option * mask).sum() / mask.sum().clamp(min=1)


#: Probability above which an option is selected, for the decisions where the
#: count is genuinely free — optional (``minCount == 0``) and multi-select
#: (``maxCount > 1``), together ~17% of decisions. The other ~83% are
#: single-select, where the bracket forces one pick and this is ignored
#: entirely: that case is a plain argmax over the categorical distribution
#: ``masked_selection_loss`` fits with softmax cross-entropy.
#:
#: A fixed default is enough now. It was not under the old ``masked_bce_loss``,
#: which scored a 1-of-N choice as N independent Bernoullis and so compressed
#: every probability toward zero — that left the useful cut on a knife edge
#: (0.5 declined 44% of optional effects outright) and needed a per-checkpoint
#: sweep to place. Cross-entropy removed the compression, and the model's
#: probabilities are now well enough separated that this value does not matter:
#: measured over 210 live decisions, every threshold from 0.02 to 0.15 decodes
#: *identically*, and only at 0.30+ does anything change at all. Calibrating a
#: parameter that provably changes nothing is worse than a constant — it
#: implies a precision that isn't there, and a stale calibration file then
#: reads as a live misconfiguration.
DEFAULT_THRESHOLD = 0.15


def load_policy(checkpoint, map_location="cpu") -> "PolicyNetwork":
    """Load a checkpoint's weights, ready for inference."""
    network = PolicyNetwork()
    network.load_state_dict(torch.load(checkpoint, map_location=map_location))
    network.eval()
    return network


def decode_action(
    logits: torch.Tensor,
    options_mask: torch.Tensor,
    min_count: torch.Tensor,
    max_count: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[list[int]]:
    """Turn a batch of option logits into the actual selections to submit.

    The engine imposes a hard ``[minCount, maxCount]`` bracket per decision,
    and that bracket is what decides how this behaves:

    When ``maxCount == minCount == 1`` — the 93.6% single-select case that
    ``masked_selection_loss`` trains with softmax cross-entropy — the clamps
    below force ``count == 1`` and the top-scoring option is taken. That is
    plain ``argmax`` over the categorical distribution the loss actually fit,
    and ``threshold`` cannot affect it at all.

    ``threshold`` only comes into play where the count is genuinely free:
    multi-select decisions (``maxCount > 1``), which the loss still models as
    independent Bernoullis, so a per-option probability cut is the matching
    decode there.

    It is deliberately NOT allowed to empty a selection. ``minCount == 0``
    decisions (an optional effect, e.g. a Pokédex or a Hilda) are the ones
    where a low-confidence row could decode to nothing at all, and on those
    the threshold is reading a quantity training never constrained: an
    optional decision the expert *accepted* carries a single positive target,
    so ``masked_selection_loss`` scores it through the softmax cross-entropy
    branch — and softmax CE is invariant to adding a constant to every logit
    in a row. Only the differences between logits are fit; the absolute level
    is free, and the only thing pulling on it is the Bernoulli branch, whose
    mostly-zero targets drag it down. ``sigmoid(logit) > threshold`` then asks
    that unpinned level a question it cannot answer.

    Measured on ``policy_decisions_crustle.parquet`` the damage is large: the
    expert declines an optional selection 4.0% of the time, while the
    unfloored decode returned nothing on 29.0% of them, and flooring the count
    at one lifted exact-match on that subset from 61.0% to 82.0%.

    The floor concedes the genuine declines (4.0%, hence the 96% ceiling it
    implies) in exchange for the 25 points of spurious ones. Recovering them
    properly means making "decline" an explicit null option so it competes
    inside the same softmax the loss actually fits, rather than being inferred
    from an uncalibrated absolute probability — that is a retrain, not a
    decode change.

    Within the clamp options are taken in descending confidence, so when the
    threshold under-selects the next-most-confident options fill the gap, and
    when it over-selects the least confident ones are dropped first.

    Returns one list of option indexes per batch element (unlike the rest of
    this module it is not a tensor, because the counts are ragged).
    """
    probs = torch.sigmoid(logits).masked_fill(~options_mask, -1.0)
    num_valid = options_mask.sum(-1)

    count = (probs > threshold).sum(-1)
    # Floored at one, not at ``min_count`` — see above. ``num_valid`` still
    # clamps it back to zero below for a decision offering no valid option.
    count = torch.maximum(count, min_count.clamp(min=1))
    count = torch.minimum(count, max_count)
    count = torch.minimum(count, num_valid).clamp(min=0)

    order = probs.argsort(dim=-1, descending=True)
    return [order[i, : int(count[i])].tolist() for i in range(order.shape[0])]


def selection_counts(features: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the raw ``(minCount, maxCount)`` bracket of the *current*
    decision from a collated feature batch, as ``(B,)`` long tensors.

    ``dataset.transform`` stores them ``_normalize``d by the fixed
    ``min_count``/``max_count`` cap of 60 (deck size), and carries a length-1
    "chain" dim for the single current decision — this undoes both, so
    ``decode_action`` can be fed straight from a batch without the caller
    having to know that encoding."""
    selection = features["decision_context"]["selection"]
    min_count = (selection["min_count"] * 60.0).round().long().reshape(-1)
    max_count = (selection["max_count"] * 60.0).round().long().reshape(-1)
    return min_count, max_count


class PolicyNetwork(nn.Module):
    """Fuses every feature group into one state vector, then scores the
    current decision's options against it (a pointer-style classifier over
    a variable-size option set, not a fixed action space)."""

    def __init__(self, dim=D, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.decision_chain = DecisionChainEncoder(dim, dropout=dropout)
        self.decision_context = DecisionContextEncoder(dim, dropout=dropout)
        self.global_state = GlobalStateEncoder(dim, dropout)
        self.opponent_history = OpponentHistoryEncoder(dim, dropout=dropout)
        self.player_state = PlayerStateEncoder(dim, dropout)  # shared weights: self & opponent boards
        # 7*dim: decision_chain + ctx_pooled (2*dim) + global_state +
        # opponent_history + own board + opponent board.
        self.fuse = nn.Sequential(nn.Linear(7 * dim, dim), nn.ReLU(), nn.Dropout(dropout))
        self.score = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, features: dict):
        ctx_pooled, option_vecs, options_mask = self.decision_context(features["decision_context"])
        state = self.fuse(
            torch.cat(
                [
                    self.decision_chain(features["decision_chain"]),
                    ctx_pooled,
                    self.global_state(features["global_state"]),
                    self.opponent_history(features["opponent_history"]),
                    # Concatenated, NOT averaged. Sharing the encoder is right
                    # (a board is a board), but averaging its two outputs is
                    # symmetric in (self, opponent) — it made the logits
                    # provably identical when the two boards were swapped, i.e.
                    # the model could not tell whose 300 HP attacker it was
                    # looking at. Concatenation keeps the encoder shared and
                    # the two roles distinct.
                    self.player_state(features["state"]),
                    self.player_state(features["opponent_state"]),
                ],
                dim=-1,
            )
        )
        query = state.unsqueeze(1).expand(-1, option_vecs.size(1), -1)  # (B, max_options, D)
        logits = self.score(torch.cat([option_vecs, query], dim=-1)).squeeze(-1)  # (B, max_options)
        logits = logits.masked_fill(~options_mask, float("-inf"))
        return logits


if __name__ == "__main__":
    # Smoke test with a real (small) batch — forward() is batch-only now;
    # see bc_train.py for the actual training loop with collate.py.
    from collate import collate_features, pad_stack

    dataset = PolicyFeatureDataset(
        "data/policy_decisions_crustle.parquet", player_name="flg", transform=transform)
    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    samples = [dataset[i] for i in (77, 78, 79)]
    features = collate_features([obs["features"] for obs, _ in samples])
    num_options = features["decision_context"]["options"]["options_mask"].shape[-1]
    targets = pad_stack(
        [torch.zeros(num_options).index_fill_(0, action, 1.0) for _, action in samples], 0.0
    )
    print("targets:", targets)

    options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)
    for step in range(100):
        logits = policy(features)
        loss = masked_selection_loss(logits, targets, options_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == 99:
            print(f"step {step}: loss={loss.item():.4f}")
