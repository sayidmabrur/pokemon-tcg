"""Behavioral-cloning training loop over the offline replay dataset.

Uses ``PolicyFeatureDataset``/``transform`` (dataset.py) for features,
``collate.py`` to batch several samples together (padding every ragged
dimension to the batch's max, with a validity mask), and ``PolicyNetwork``
(policy_experimental.py) as the model. Real batching is what actually lets
CUDA help: a single-sample loop is all tiny ops (small embeddings, short
sequences) dominated by kernel-launch/H2D-transfer overhead rather than
compute, so the GPU barely engages — see collate.py's docstring for how the
padding/masking is built.
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing
import torch.utils.data

from collate import collate_features, pad_stack
from dataset import PolicyFeatureDataset, transform
from precomputed_dataset import PrecomputedPolicyFeatureDataset
from policy_experimental import (
    DEFAULT_DROPOUT,
    PolicyNetwork,
    decode_action,
    equivalence_mask,
    masked_selection_loss,
    selection_counts,
)

# duel_inference.py lives in archetypes/lucario/ (one level up from this
# file's policy_network/ directory) — needed for the end-of-epoch match
# against PolicyRuleBased, which is a much better read on playing strength
# than the loss/exact-match numbers above (see run_epoch_duel).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duel_inference

# Batches reach the main process as ~520 separate tensors, and torch's default
# "file_descriptor" sharing strategy passes each one as an SCM_RIGHTS file
# descriptor over a socket. That handshake fails outright — "RuntimeError:
# received 0 items of ancdata" from ``recvfds`` — when a worker dies mid-send,
# which on a memory-tight box happens at worker-fork time, before the first
# batch is ever produced. It is intermittent for exactly that reason: it
# depends on how much RAM is free at the instant the workers fork.
#
# "file_system" passes tensors through named shared-memory files instead, so
# there is no fd handshake left to fail. That strategy is safe *here* because
# the number of simultaneously-live shared objects is bounded — ~520 tensors
# per batch times the handful of batches in flight, each freed as its batch is
# consumed. It was NOT safe in precompute_features.py, where the main process
# accumulated thousands of samples' worth of worker-backed tensors at once and
# blew past vm.max_map_count (see that module's CACHE_FORMAT note). Bounded
# versus unbounded is the whole difference between the two cases.
torch.multiprocessing.set_sharing_strategy("file_system")


def collate_batch(batch: list[tuple[dict, torch.Tensor]]):
    samples, target_actions = zip(*batch)
    features = collate_features([s["features"] for s in samples])
    num_options = features["decision_context"]["options"]["options_mask"].shape[-1]
    targets = pad_stack(
        [torch.zeros(num_options).index_fill_(0, action, 1.0) for action in target_actions], 0.0
    )
    return features, targets


def _to_device(obj, device):
    """Move a ``collate_features()``-shaped batch to ``device``. Plain
    ``.to(device)`` only works on a single tensor — this recurses through
    the nested dicts the real structure is built from."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


def episode_split(
    dataset: PolicyFeatureDataset, val_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split sample indexes into (train, val) by *episode*, never by row.

    Every row of a given match lands on the same side of the split — see
    ``PolicyFeatureDataset.episode_ids`` for why a row-wise split silently
    leaks and reports an inflated validation score."""
    episode_ids = dataset.episode_ids()
    unique = sorted(set(episode_ids))
    random.Random(seed).shuffle(unique)
    num_val = max(1, round(len(unique) * val_frac)) if val_frac > 0 else 0
    val_episodes = set(unique[:num_val])

    train_idx, val_idx = [], []
    for sample_idx, episode_id in enumerate(episode_ids):
        (val_idx if episode_id in val_episodes else train_idx).append(sample_idx)
    return train_idx, val_idx


@torch.no_grad()
def evaluate(policy: PolicyNetwork, loader, device: str) -> dict[str, float]:
    """Policy-quality metrics on a held-out split.

    The loss the training loop prints is *not* a measure of how well the
    policy plays: it is an average negative log-likelihood, and it keeps
    dropping long after the argmax has stopped improving. These are the
    numbers that actually track playing strength:

    ``top1``       the single highest-scoring option is one the expert
                   really did pick — the closest thing to "would it make
                   this move".
    ``exact``      the full decoded selection (via ``decode_action``, the
                   same decode used at inference time) matches the
                   expert's selection exactly, as a set.
    ``chance``     top1 for a policy picking uniformly at random among the
                   valid options. ``top1`` is only meaningful next to this:
                   most decisions offer few options, so a large top1 can
                   still be no better than guessing.
    ``count_mae``  mean absolute error in *how many* options get selected.
                   Separated out because a selection can be perfect on
                   which options matter and still be submitted with the
                   wrong count.
    """
    policy.eval()
    totals = {"loss": 0.0, "top1": 0.0, "top1eq": 0.0, "exact": 0.0,
              "chance": 0.0, "count_mae": 0.0}
    num_samples = 0
    num_batches = 0

    for features, targets in loader:
        features = _to_device(features, device)
        targets = targets.to(device)
        options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)

        logits = policy(features)
        equivalent = equivalence_mask(
            features["decision_context"]["options"], targets, options_mask
        )
        totals["loss"] += masked_selection_loss(logits, targets, options_mask).item()
        num_batches += 1

        # A masked-out position is -inf, so argmax can never land on one.
        prediction = logits.argmax(dim=-1, keepdim=True)
        top1 = targets.gather(1, prediction).squeeze(1)
        totals["top1"] += top1.sum().item()
        # Credit a prediction that is the *same play* as the expert's, not
        # merely the same index — see equivalence_mask for why the index-exact
        # number is capped around 63.7% regardless of policy quality.
        totals["top1eq"] += equivalent.gather(1, prediction).squeeze(1).sum().item()

        num_valid = options_mask.sum(-1).clamp(min=1)
        num_target = targets.sum(-1)
        totals["chance"] += (num_target / num_valid).sum().item()

        min_count, max_count = selection_counts(features)
        predictions = decode_action(logits, options_mask, min_count, max_count)
        for i, prediction in enumerate(predictions):
            expert = set(targets[i].nonzero().flatten().tolist())
            totals["exact"] += float(set(prediction) == expert)
            totals["count_mae"] += abs(len(prediction) - len(expert))

        num_samples += targets.shape[0]

    policy.train()
    num_samples = max(num_samples, 1)
    return {
        "loss": totals["loss"] / max(num_batches, 1),
        "top1": totals["top1"] / num_samples,
        "top1eq": totals["top1eq"] / num_samples,
        "exact": totals["exact"] / num_samples,
        "chance": totals["chance"] / num_samples,
        "count_mae": totals["count_mae"] / num_samples,
    }


class _InProcessBCPolicy:
    """Same ``.act(obs)``/``.reset(episode_id)`` interface as
    ``duel_inference.BCPolicy``, but wraps the network still being trained
    directly rather than round-tripping it through a checkpoint file — so
    the end-of-epoch duel measures the exact in-memory weights just
    trained, not a stale save from a previous epoch."""

    def __init__(self, network: PolicyNetwork, device: str):
        self.network = network
        self.device = device
        self.extractor = duel_inference.LiveFeatureExtractor()
        self.decisions = 0
        self.empty_selections = 0

    def reset(self, episode_id: int) -> None:
        self.extractor.reset(episode_id=episode_id)

    def act(self, obs: dict) -> list[int]:
        observation = self.extractor(obs)
        features = _to_device(collate_features([transform(observation)]), self.device)
        with torch.no_grad():
            logits = self.network(features)
        min_count, max_count = selection_counts(features)
        options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)
        action = decode_action(logits, options_mask, min_count, max_count)[0]
        self.extractor.record_action(action)
        self.decisions += 1
        if not action:
            self.empty_selections += 1
        return action


def _deck_label(path: Path) -> str:
    """Log-friendly deck path: enough of it to identify which deck, since
    several archetypes' decklists are all named deck.csv."""
    try:
        return str(path.resolve().relative_to(duel_inference._REPO_ROOT))
    except ValueError:
        return str(path)


def run_epoch_duel(
    policy: PolicyNetwork,
    device: str,
    deck: list[int],
    opponent_deck: list[int],
    episodes: int,
    seats: str,
) -> dict[str, float]:
    """Play ``episodes`` games of the current weights against
    ``PolicyRuleBased`` — the number that actually matters (win rate),
    versus the proxy metrics ``evaluate()`` reports against the offline
    expert labels. Slower than ``evaluate()`` since it runs the real game
    engine turn-by-turn, so keep ``episodes`` modest for a per-epoch check."""
    policy.eval()
    challenger = _InProcessBCPolicy(policy, device)
    seat_list = [0, 1] if seats == "both" else [int(seats)]
    per_seat = max(1, episodes // len(seat_list))
    wins, total = 0, 0
    for seat in seat_list:
        outcome = duel_inference.duel(challenger, deck, opponent_deck, per_seat, seat, quiet=True)
        wins += outcome["wins"]
        total += outcome["episodes"]
    policy.train()
    rate = wins / max(total, 1)
    empty_rate = challenger.empty_selections / max(challenger.decisions, 1)
    return {"wins": wins, "episodes": total, "rate": rate, "empty_rate": empty_rate}


def train(
    parquet_path: str = "data/policy_decisions_lucario.parquet",
    player_name: str = "Majkel1337",
    epochs: int = 1,
    lr: float = 1e-3,
    dropout: float = DEFAULT_DROPOUT,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    batch_size: int = 64,
    log_every: int = 0,
    limit: int | None = None,
    device: str | None = None,
    num_workers: int = 4,
    shuffle: bool = True,
    cached_row_groups: int = 16,
    val_frac: float = 0.1,
    seed: int = 0,
    out: str | None = None,
    duel_episodes: int = 20,
    duel_seats: str = "both",
    duel_deck: str | None = None,
    duel_opponent_deck: str | None = None,
    precomputed_dir: str | None = None,
    cached_shards: int = 4,
) -> PolicyNetwork:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}, batch_size={batch_size}", flush=True)

    duel_deck_ids, duel_opponent_deck_ids = None, None
    if duel_episodes > 0:
        duel_deck_path = Path(duel_deck or duel_inference._REPO_ROOT / "lucario_deck.csv")
        duel_opponent_deck_path = Path(
            duel_opponent_deck or duel_inference._REPO_ROOT / "crustle-agent-rule-based/deck.csv"
        )
        duel_deck_ids = duel_inference.read_deck(duel_deck_path)
        duel_opponent_deck_ids = duel_inference.read_deck(duel_opponent_deck_path)
        print(
            f"end-of-epoch duel: {duel_episodes} episodes/epoch vs PolicyRuleBased "
            # Parent-qualified, not just .name: the rule-based opponent's deck
            # lives at crustle-agent-rule-based/deck.csv, whose basename is the
            # bare "deck.csv" — indistinguishable in a log from the repo-root
            # deck.csv, which is a different archetype entirely. Printing only
            # the basename made a correct config read as a mirror match.
            f"({_deck_label(duel_deck_path)} vs {_deck_label(duel_opponent_deck_path)})",
            flush=True,
        )

    t0 = time.time()
    if precomputed_dir is not None:
        # Skips build_observation()/transform() entirely — see
        # precompute_features.py's docstring. Every epoch after the first
        # is where this actually pays off, since PolicyFeatureDataset would
        # otherwise redo that per-sample work from scratch each time.
        dataset = PrecomputedPolicyFeatureDataset(precomputed_dir, cached_shards=cached_shards)
        print(
            f"loaded precomputed dataset from {precomputed_dir} in "
            f"{time.time() - t0:.1f}s — {len(dataset)} samples "
            f"(player_name={dataset.manifest['player_name']!r})",
            flush=True,
        )
    else:
        dataset = PolicyFeatureDataset(
            parquet_path, player_name=player_name, transform=transform,
            cached_row_groups=cached_row_groups,
        )
        print(f"dataset ready in {time.time() - t0:.1f}s — {len(dataset)} samples", flush=True)

    train_idx, val_idx = episode_split(dataset, val_frac, seed)
    if limit is not None:
        train_idx = train_idx[:limit]
        val_idx = val_idx[: max(1, limit // 10)] if val_idx else []
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx) if val_idx else None
    print(
        f"split by episode (seed={seed}): {len(train_idx)} train / {len(val_idx)} val samples",
        flush=True,
    )

    # The bottleneck here is CPU-bound Python work per sample — building
    # ``transform()``'s nested dict and then ``collate_features()``'s
    # padding — not disk I/O (measured: a stable ~1.2s/batch with zero
    # degradation over time rules out row-group-cache thrashing). That work
    # runs single-threaded on the main process with ``num_workers=0``,
    # leaving the GPU idle between batches; ``num_workers>0`` is meant to run
    # it in parallel subprocesses instead. In practice the measured benefit
    # was inconsistent (a big win at one worker count, none at another in a
    # back-to-back test) — plausibly IPC/pickling cost for the deeply nested
    # per-batch dict eating into the parallelism gain, or an OS disk-cache
    # confound between runs, not confirmed either way. Benchmark this
    # yourself with --log-every and compare a few --num-workers values
    # before trusting any specific setting.
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch,
        num_workers=num_workers,
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_batch,
            num_workers=num_workers,
        )
        if val_set is not None
        else None
    )
    print(f"{len(loader)} batches/epoch, shuffle={shuffle}, num_workers={num_workers}", flush=True)

    policy = PolicyNetwork(dropout=dropout).to(device)
    # AdamW, not Adam+weight_decay: under Adam the L2 term is folded into the
    # adaptive denominator, so parameters with large gradient history get
    # decayed less — the opposite of the intent. AdamW decouples it.
    #
    # Biases, LayerNorm and embedding tables are excluded. Decaying a bias or
    # a LayerNorm gain just fights the layer's own calibration, and pulling
    # embedding rows toward zero penalizes *rare* cards hardest (they get few
    # gradient updates to counteract the decay) — exactly the ids where the
    # model has least to spare.
    decayed, not_decayed = [], []
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        skip = param.ndim <= 1 or ".norm" in name or "embed" in name.lower()
        (not_decayed if skip else decayed).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": weight_decay},
            {"params": not_decayed, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    print(
        f"dropout={dropout} weight_decay={weight_decay} "
        f"({len(decayed)} decayed / {len(not_decayed)} exempt tensors)",
        flush=True,
    )
    # Linear warmup then cosine decay. The two sequence encoders are 8-layer
    # transformers; even pre-LN, a deep stack starting at full LR takes large
    # early steps on attention weights that are still random, which is what
    # makes results vary run to run rather than converge. Warmup is the
    # standard fix and costs a few hundred batches.
    total_steps = max(1, epochs * len(loader))
    warmup_steps = min(max(1, int(0.03 * total_steps)), 500)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    print(
        f"schedule: {total_steps} steps, {warmup_steps} warmup, cosine decay, "
        f"grad clip {grad_clip}",
        flush=True,
    )
    best_exact = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        t_epoch = time.time()
        for i, (features, targets) in enumerate(loader):
            features = _to_device(features, device)
            targets = targets.to(device)
            options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)

            logits = policy(features)
            loss = masked_selection_loss(logits, targets, options_mask)

            optimizer.zero_grad()
            loss.backward()
            # Clip before stepping: a single outlier batch (a 44-option
            # decision, an unusually long chain) can otherwise land one huge
            # update that the rest of the epoch spends recovering from.
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_grad_norm += float(grad_norm)
            num_batches += 1
            # log_every=0 (the default) reports once per epoch instead of
            # mid-epoch. The per-epoch line below carries the same numbers
            # aggregated, so nothing is lost by leaving this off.
            if log_every and (i + 1) % log_every == 0:
                elapsed = time.time() - t_epoch
                print(
                    f"epoch {epoch} batch {i + 1}/{len(loader)}: "
                    f"avg loss={total_loss / num_batches:.4f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"|g|={float(grad_norm):.2f}, "
                    f"{elapsed / num_batches:.2f}s/batch",
                    flush=True,
                )

        train_loss = total_loss / max(num_batches, 1)
        epoch_seconds = time.time() - t_epoch
        print(
            f"epoch {epoch} done: train loss={train_loss:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"|g|avg={total_grad_norm / max(num_batches, 1):.2f} "
            f"({num_batches} batches, {epoch_seconds:.0f}s, "
            f"{epoch_seconds / max(num_batches, 1):.2f}s/batch)",
            flush=True,
        )

        if duel_episodes > 0:
            duel_result = run_epoch_duel(
                policy, device, duel_deck_ids, duel_opponent_deck_ids, duel_episodes, duel_seats,
            )
            print(
                f"epoch {epoch}   duel vs PolicyRuleBased: "
                f"{duel_result['wins']}/{duel_result['episodes']} = {duel_result['rate']:.1%} "
                f"(declined {duel_result['empty_rate']:.1%})",
                flush=True,
            )

        if val_loader is None:
            continue

        metrics = evaluate(policy, val_loader, device)
        # Read top1 against chance, not against 0 — and watch val loss vs
        # train loss for the overfitting gap that a train-loss-only curve
        # cannot show you.
        print(
            f"epoch {epoch}   val: loss={metrics['loss']:.4f} "
            f"top1={metrics['top1']:.3f} (chance {metrics['chance']:.3f}) "
            f"top1eq={metrics['top1eq']:.3f} "
            f"exact={metrics['exact']:.3f} count_mae={metrics['count_mae']:.2f}",
            flush=True,
        )
        # Select on exact-match, the metric closest to "submits the move the
        # expert submitted" — val loss keeps improving on the easy negatives
        # after the actual decisions have stopped getting better.
        if out is not None and metrics["exact"] > best_exact:
            best_exact = metrics["exact"]
            best_epoch = epoch
            torch.save(policy.state_dict(), out)
            print(f"  saved {out} (best exact={best_exact:.3f})", flush=True)

    if out is not None:
        if val_loader is None:
            torch.save(policy.state_dict(), out)
            print(f"saved {out} (final epoch — no validation split)", flush=True)
        else:
            final_out = f"{out}.final"
            torch.save(policy.state_dict(), final_out)
            print(
                f"saved {final_out} (final epoch); {out} holds epoch {best_epoch} "
                f"(best val exact={best_exact:.3f})",
                flush=True,
            )

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="data/policy_decisions_lucario.parquet")
    parser.add_argument("--player-name", default="Majkel1337")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--dropout", type=float, default=DEFAULT_DROPOUT,
        help="dropout for every MLP block and both transformer stacks "
             "(0 disables it entirely)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-2,
        help="AdamW decoupled weight decay; biases, norms and embeddings are "
             "exempt (0 disables it)",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="max global grad norm; the 8-layer sequence encoders are the "
             "reason this matters (see the schedule set up in train())",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="cap dataset size (debug)")
    parser.add_argument("--out", default="bc_policy.pt")
    parser.add_argument("--device", default=None, help="defaults to cuda if available, else cpu")
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="parallel data-loading processes — the real fix for slow training "
             "(measured ~4x wall-clock speedup at 8 workers on a 12-core machine); "
             "0 runs preprocessing single-threaded on the main process, leaving "
             "the GPU idle between batches",
    )
    parser.add_argument(
        "--log-every", type=int, default=0,
        help="print a progress line every N batches; 0 (the default) prints "
             "only the per-epoch summary, which already reports the same "
             "loss/lr/grad-norm/timing aggregated over the epoch",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.1,
        help="fraction of *episodes* (not rows) held out for validation; 0 disables it",
    )
    parser.add_argument("--seed", type=int, default=42, help="controls the episode split")
    parser.add_argument("--no-shuffle", action="store_true", help="disable shuffling (debug)")
    parser.add_argument("--cached-row-groups", type=int, default=16)
    parser.add_argument(
        "--duel-episodes", type=int, default=20,
        help="games vs PolicyRuleBased to play at the end of every epoch (0 disables); "
             "this is the actual win-rate signal, slower than --val-frac's offline metrics "
             "since it runs the real engine turn-by-turn",
    )
    parser.add_argument(
        "--duel-seats", default="both", choices=("0", "1", "both"),
        help="which seat the policy plays in the end-of-epoch duel; 'both' splits "
             "--duel-episodes across both",
    )
    parser.add_argument(
        "--duel-deck", default=None,
        help="deck for the policy in the end-of-epoch duel; defaults to lucario_deck.csv",
    )
    parser.add_argument(
        "--duel-opponent-deck", default=None,
        help="deck for PolicyRuleBased in the end-of-epoch duel; defaults to "
             "crustle-agent-rule-based/deck.csv",
    )
    parser.add_argument(
        "--precomputed-dir", default=None,
        help="load samples from a precompute_features.py cache instead of "
             "reading --parquet live; skips build_observation()/transform() on "
             "every epoch, which is the actual training bottleneck (see "
             "precompute_features.py). --player-name/--parquet are ignored "
             "when this is set, since the cache already fixes both.",
    )
    parser.add_argument(
        "--cached-shards", type=int, default=4,
        help="LRU shard cache size for --precomputed-dir (ignored otherwise)",
    )
    args = parser.parse_args()

    # Saving lives inside train() now: with a validation split, the
    # checkpoint worth keeping is the best epoch, which only the loop knows.
    train(
        args.parquet, args.player_name, args.epochs, args.lr,
        dropout=args.dropout, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size, log_every=args.log_every, limit=args.limit,
        device=args.device, num_workers=args.num_workers, shuffle=not args.no_shuffle,
        cached_row_groups=args.cached_row_groups,
        val_frac=args.val_frac, seed=args.seed, out=args.out,
        duel_episodes=args.duel_episodes, duel_seats=args.duel_seats,
        duel_deck=args.duel_deck, duel_opponent_deck=args.duel_opponent_deck,
        precomputed_dir=args.precomputed_dir, cached_shards=args.cached_shards,
    )
