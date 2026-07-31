"""Minimalist policy encoder — proves the feature pipeline is trainable
end-to-end, one architecture family per feature group:

  decision_chain     -> per-step set-pooled features -> TransformerEncoder (sequence)
  decision_context   -> per-option MLP (scored against the pooled state -> action logits)
  global_state       -> flat MLP (fixed-size scalars/categoricals)
  opponent_history   -> per-turn set-pooled diffs -> TransformerEncoder (sequence)
  state/opponent_state -> per-Pokémon MLP + set pooling (permutation-invariant board)

Kept intentionally small (dims, layers) — this is a trainability check, not a
tuned architecture.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import PolicyFeatureDataset, transform
from vocab import (
    AREA_VOCAB_SIZE,
    CARD_ENERGY_TYPE_VOCAB_SIZE,
    CARD_ID_VOCAB_SIZE,
    CARD_STAGE_VOCAB_SIZE,
    CARD_TYPE_VOCAB_SIZE,
    OPTION_TYPE_VOCAB_SIZE,
    SELECT_CONTEXT_VOCAB_SIZE,
    SELECT_TYPE_VOCAB_SIZE,
    TARGETS_OPPONENT_VOCAB_SIZE,
    EnergyType,
)

D = 32  # shared embedding/hidden width — small on purpose


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``x`` (..., L, D) over L, respecting a bool ``mask`` (..., L).
    An all-False row (nothing valid) falls back to zeros rather than NaN."""
    mask = mask.float().unsqueeze(-1)
    denom = mask.sum(dim=-2).clamp(min=1.0)
    return (x * mask).sum(dim=-2) / denom


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

    def __init__(self, dim=D):
        super().__init__()
        self.id = nn.Embedding(CARD_ID_VOCAB_SIZE, dim)
        self.stage = nn.Embedding(CARD_STAGE_VOCAB_SIZE, dim // 4)
        self.type_embed = nn.Embedding(CARD_TYPE_VOCAB_SIZE, dim // 4)
        self.energy_type = nn.Embedding(CARD_ENERGY_TYPE_VOCAB_SIZE, dim // 4)
        self.proj = nn.Linear(dim + 3 * (dim // 4) + 5, dim)

    def forward(self, fields: dict) -> torch.Tensor:
        """``fields`` is already sliced to plain keys (``id``/``stage``/...)
        by ``_card_fields`` — missing narrowed flags fall back to zeros."""
        card_id = fields["id"]
        zeros = torch.zeros_like(card_id)
        stage = fields.get("stage", zeros)
        stage_norm = fields.get("stage_norm", torch.zeros_like(card_id, dtype=torch.float))
        ctype = fields.get("type", zeros)
        energy_type = fields.get("energy_type", zeros)
        flags = torch.stack(
            [
                fields.get("ex", zeros).float(),
                fields.get("mega_ex", zeros).float(),
                fields.get("tera", zeros).float(),
                fields.get("ace_spec", zeros).float(),
                stage_norm,
            ],
            dim=-1,
        )
        out = torch.cat(
            [self.id(card_id), self.stage(stage), self.type_embed(ctype), self.energy_type(energy_type), flags],
            dim=-1,
        )
        return F.relu(self.proj(out))


def _card_fields(fields: dict, prefix: str) -> dict:
    """Slice out ``{prefix}_*`` keys into the flat ``{"id": ..., "stage": ...}``
    dict ``CardEmbed`` expects, from a flat dict keyed like ``f"{prefix}_id"``."""
    return {k[len(prefix) + 1:]: v for k, v in fields.items() if k.startswith(prefix + "_")}


class OptionEncoder(nn.Module):
    """One option -> vector. Options are a *set* within a decision (order is
    meaningless — pooled with a mask), not a sequence."""

    def __init__(self, dim=D):
        super().__init__()
        self.card = CardEmbed(dim)
        self.type_embed = nn.Embedding(OPTION_TYPE_VOCAB_SIZE, dim // 4)
        self.area = nn.Embedding(AREA_VOCAB_SIZE, dim // 4)
        self.targets_opponent = nn.Embedding(TARGETS_OPPONENT_VOCAB_SIZE, dim // 4)
        self.mlp = nn.Sequential(nn.Linear(dim + 3 * (dim // 4) + 7, dim), nn.ReLU())

    def forward(self, options: dict) -> torch.Tensor:
        card = self.card(_card_fields(options, "card"))
        # ``index``/``energy_index``/``in_play_index``/``attack_id``/``serial``
        # are position/id pointers (NO_VALUE=-1 sentinel), not a fixed vocab —
        # crude /60 scaling here (not a proper embedding) is the "minimalist"
        # part, but they matter a lot: these are usually what actually
        # distinguishes one option from another (e.g. "which hand card by
        # index"), unlike type/area/card_id which are often shared across
        # every option in a decision.
        scalars = torch.stack(
            [
                options["number"], options["count"],
                options["index"].float() / 10.0, options["energy_index"].float() / 10.0,
                options["in_play_index"].float() / 10.0, options["attack_id"].float() / 10.0,
                options["serial"].float() / 10.0,
            ],
            dim=-1,
        )
        out = torch.cat(
            [
                card,
                self.type_embed(options["type"]),
                self.area(options["area"]),
                self.targets_opponent(options["targets_opponent"]),
                scalars,
            ],
            dim=-1,
        )
        return self.mlp(out)


class SelectionEncoder(nn.Module):
    def __init__(self, dim=D):
        super().__init__()
        self.type_embed = nn.Embedding(SELECT_TYPE_VOCAB_SIZE, dim // 4)
        self.context = nn.Embedding(SELECT_CONTEXT_VOCAB_SIZE, dim // 4)
        self.context_card = CardEmbed(dim)
        self.effect_card = CardEmbed(dim)
        self.mlp = nn.Sequential(nn.Linear(2 * dim + 2 * (dim // 4) + 5, dim), nn.ReLU())

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

    def __init__(self, dim=D, nhead=2, layers=1):
        super().__init__()
        self.option = OptionEncoder(dim)
        self.selection = SelectionEncoder(dim)
        self.in_proj = nn.Linear(2 * dim + 2, dim)
        encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_feedforward=2 * dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)

    def forward(self, decision_chain: dict) -> torch.Tensor:
        chain_mask = decision_chain["chain_mask"]  # (B, chain_len)
        if chain_mask.shape[1] == 0:  # every sample in the batch has an empty chain
            return torch.zeros(chain_mask.shape[0], D, device=chain_mask.device)
        option_vecs = self.option(decision_chain["options"])  # (B, chain_len, max_options, D)
        option_summary = _masked_mean(option_vecs, decision_chain["options"]["options_mask"])
        selection_summary = self.selection(decision_chain["selection"])  # (B, chain_len, D)
        step = torch.cat(
            [option_summary, selection_summary, decision_chain["turn"].unsqueeze(-1),
             decision_chain["turn_action_count"].unsqueeze(-1)],
            dim=-1,
        )
        step = F.relu(self.in_proj(step))  # (B, chain_len, D)
        encoded = self.transformer(step, src_key_padding_mask=_safe_key_padding_mask(chain_mask))
        return _masked_mean(encoded, chain_mask)


class DecisionContextEncoder(nn.Module):
    """The current decision — same option/selection encoders as the chain,
    but this also exposes per-option vectors (for scoring against the
    pooled state to produce action logits), not just a pooled summary."""

    def __init__(self, dim=D):
        super().__init__()
        self.option = OptionEncoder(dim)
        self.selection = SelectionEncoder(dim)

    def forward(self, decision_context: dict):
        # ``options``/``selection`` carry a leading size-1 "chain position"
        # dim (this is always a single decision, not a real chain) — squeeze
        # dim 1 specifically, not dim 0 (that's the batch dim).
        option_vecs = self.option(decision_context["options"]).squeeze(1)  # (B, max_options, D)
        options_mask = decision_context["options"]["options_mask"].squeeze(1)  # (B, max_options)
        option_summary = _masked_mean(option_vecs, options_mask)
        selection_summary = self.selection(decision_context["selection"]).squeeze(1)  # (B, D)
        pooled = torch.cat([option_summary, selection_summary], dim=-1)
        return pooled, option_vecs, options_mask


class GlobalStateEncoder(nn.Module):
    """Fixed-size scalars/categoricals — a flat MLP, no sequence/set structure."""

    def __init__(self, dim=D):
        super().__init__()
        self.first_player = nn.Embedding(3, dim // 4)
        self.result = nn.Embedding(4, dim // 4)
        self.stadium_card = CardEmbed(dim)
        self.mlp = nn.Sequential(nn.Linear(dim + 2 * (dim // 4) + 6, dim), nn.ReLU())

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
                self.stadium_card(_card_fields(global_state, "stadium_card")) + looking_card,
                self.first_player(global_state["first_player"]),
                self.result(global_state["result"]),
                bits,
            ],
            dim=-1,
        )
        return self.mlp(out)


class PokemonEncoder(nn.Module):
    """One board Pokémon (active or bench slot) -> vector."""

    def __init__(self, dim=D):
        super().__init__()
        self.card = CardEmbed(dim)
        self.energy = nn.Embedding(len(EnergyType), dim // 4)
        self.energy_card = CardEmbed(dim)
        self.tool_card = CardEmbed(dim)
        self.pre_evolution_card = CardEmbed(dim)
        self.mlp = nn.Sequential(nn.Linear(dim + dim // 4 + 3 * dim + 2, dim), nn.ReLU())

    def _pool_list(self, module, pokemon: dict, prefix: str) -> torch.Tensor:
        vecs = module(_card_fields(pokemon, prefix))  # (..., max_width, D)
        return _masked_mean(vecs, pokemon[f"{prefix}_mask"])

    def forward(self, pokemon: dict) -> torch.Tensor:
        energy_vecs = self.energy(pokemon["energies"])  # (..., max_energies, dim // 4)
        energy_vec = _masked_mean(energy_vecs, pokemon["energies_mask"])
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

    def __init__(self, dim=D):
        super().__init__()
        self.pokemon = PokemonEncoder(dim)
        self.hand_card = CardEmbed(dim)
        self.discard_card = CardEmbed(dim)
        self.mlp = nn.Sequential(nn.Linear(4 * dim + 9, dim), nn.ReLU())

    def _pool_cards(self, module, fields: dict, prefix: str) -> torch.Tensor:
        vecs = module(_card_fields(fields, prefix))  # (B, max_width, D)
        return _masked_mean(vecs, fields[f"{prefix}_mask"])

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
        bench_vec = _masked_mean(bench_vecs, player_state["bench_pokemon"]["bench_pokemon_mask"])
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

    def __init__(self, dim=D, nhead=2, layers=1):
        super().__init__()
        self.discarded_card = CardEmbed(dim)
        self.new_pokemon_card = CardEmbed(dim)
        self.removed_pokemon_card = CardEmbed(dim)
        self.energy_attached_card = CardEmbed(dim)
        self.in_proj = nn.Linear(4 * dim + 5 + 5, dim)
        encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_feedforward=2 * dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)

    def forward(self, history: dict) -> torch.Tensor:
        history_mask = history["history_mask"]  # (B, chain_len)
        if history_mask.shape[1] == 0:  # every sample in the batch has empty history
            return torch.zeros(history_mask.shape[0], D, device=history_mask.device)
        discarded = _masked_mean(
            self.discarded_card(_card_fields(history, "discarded_card")), history["discarded_mask"]
        )
        new_pokemon = _masked_mean(
            self.new_pokemon_card(_card_fields(history, "new_pokemon_card")),
            history["new_pokemon_mask"],
        )
        removed_pokemon = _masked_mean(
            self.removed_pokemon_card(_card_fields(history, "removed_pokemon_card")),
            history["removed_pokemon_mask"],
        )
        energy_attached = _masked_mean(
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
        encoded = self.transformer(step, src_key_padding_mask=_safe_key_padding_mask(history_mask))
        return _masked_mean(encoded, history_mask)


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


def decode_action(
    logits: torch.Tensor,
    options_mask: torch.Tensor,
    min_count: torch.Tensor,
    max_count: torch.Tensor,
    threshold: float = 0.5,
) -> list[list[int]]:
    """Turn a batch of option logits into the actual selections to submit.

    The training objective (``masked_bce_loss``) is a *per-option* sigmoid:
    it teaches "is this option part of the selection?", one option at a
    time. The matching decode is therefore a threshold — take every option
    the model is more than ``threshold`` confident about — and NOT a
    top-``k`` for some externally chosen ``k``. Picking ``k`` some other way
    (a fixed number, or worse a random one) throws away the only thing the
    count-relevant part of the model learned, and makes the policy's
    behaviour inconsistent with what the loss was ever scored on.

    The engine still imposes a hard ``[minCount, maxCount]`` bracket per
    decision, so the thresholded count is clamped into it, and further
    clamped to the number of genuinely valid options. Within the clamp,
    options are taken in descending confidence — so when the threshold
    under-selects, the next-most-confident options fill the gap, and when it
    over-selects, the least confident ones are dropped first.

    Returns one list of option indexes per batch element (unlike the rest of
    this module it is not a tensor, because the counts are ragged).
    """
    probs = torch.sigmoid(logits).masked_fill(~options_mask, -1.0)
    num_valid = options_mask.sum(-1)

    count = (probs > threshold).sum(-1)
    count = torch.maximum(count, min_count)
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

    def __init__(self, dim=D):
        super().__init__()
        self.decision_chain = DecisionChainEncoder(dim)
        self.decision_context = DecisionContextEncoder(dim)
        self.global_state = GlobalStateEncoder(dim)
        self.opponent_history = OpponentHistoryEncoder(dim)
        self.player_state = PlayerStateEncoder(dim)  # shared weights: self & opponent boards
        self.fuse = nn.Sequential(nn.Linear(6 * dim, dim), nn.ReLU())  # ctx_pooled is 2*dim
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
                    (self.player_state(features["state"]) + self.player_state(features["opponent_state"])) / 2,
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
        "data/policy_decisions.parquet", player_name="Yushin Ito", transform=transform)
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
        loss = masked_bce_loss(logits, targets, options_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == 99:
            print(f"step {step}: loss={loss.item():.4f}")
