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
import random
import time

import torch
import torch.utils.data

from collate import collate_features, pad_stack
from dataset import PolicyFeatureDataset, transform
from policy_experimental import (
    PolicyNetwork,
    decode_action,
    masked_bce_loss,
    selection_counts,
)


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

    The BCE number the training loop prints is *not* a measure of how well
    the policy plays. It is a per-option average over a target that is
    ~1 selected option among N, so it is dominated by the easy negatives:
    a model that answers "no" to every option already scores well on it,
    and the loss keeps dropping long after the argmax has stopped
    improving. These are the numbers that actually track playing strength:

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
    totals = {"loss": 0.0, "top1": 0.0, "exact": 0.0, "chance": 0.0, "count_mae": 0.0}
    num_samples = 0
    num_batches = 0

    for features, targets in loader:
        features = _to_device(features, device)
        targets = targets.to(device)
        options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)

        logits = policy(features)
        totals["loss"] += masked_bce_loss(logits, targets, options_mask).item()
        num_batches += 1

        # A masked-out position is -inf, so argmax can never land on one.
        top1 = targets.gather(1, logits.argmax(dim=-1, keepdim=True)).squeeze(1)
        totals["top1"] += top1.sum().item()

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
        "exact": totals["exact"] / num_samples,
        "chance": totals["chance"] / num_samples,
        "count_mae": totals["count_mae"] / num_samples,
    }


def train(
    parquet_path: str = "data/policy_decisions.parquet",
    player_name: str = "Yushin Ito",
    epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 64,
    log_every: int = 10,
    limit: int | None = None,
    device: str | None = None,
    num_workers: int = 4,
    shuffle: bool = True,
    cached_row_groups: int = 16,
    val_frac: float = 0.1,
    seed: int = 0,
    out: str | None = None,
) -> PolicyNetwork:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}, batch_size={batch_size}", flush=True)

    t0 = time.time()
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

    policy = PolicyNetwork().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    best_exact = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        t_epoch = time.time()
        for i, (features, targets) in enumerate(loader):
            features = _to_device(features, device)
            targets = targets.to(device)
            options_mask = features["decision_context"]["options"]["options_mask"].squeeze(1)

            logits = policy(features)
            loss = masked_bce_loss(logits, targets, options_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            if (i + 1) % log_every == 0:
                elapsed = time.time() - t_epoch
                print(
                    f"epoch {epoch} batch {i + 1}/{len(loader)}: "
                    f"avg loss={total_loss / num_batches:.4f}, "
                    f"{elapsed / num_batches:.2f}s/batch",
                    flush=True,
                )

        train_loss = total_loss / max(num_batches, 1)
        print(f"epoch {epoch} done: train loss={train_loss:.4f}", flush=True)

        if val_loader is None:
            continue

        metrics = evaluate(policy, val_loader, device)
        # Read top1 against chance, not against 0 — and watch val loss vs
        # train loss for the overfitting gap that a train-loss-only curve
        # cannot show you.
        print(
            f"epoch {epoch}   val: loss={metrics['loss']:.4f} "
            f"top1={metrics['top1']:.3f} (chance {metrics['chance']:.3f}) "
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
    parser.add_argument("--parquet", default="data/policy_decisions.parquet")
    parser.add_argument("--player-name", default="Yushin Ito")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--val-frac", type=float, default=0.1,
        help="fraction of *episodes* (not rows) held out for validation; 0 disables it",
    )
    parser.add_argument("--seed", type=int, default=0, help="controls the episode split")
    parser.add_argument("--no-shuffle", action="store_true", help="disable shuffling (debug)")
    parser.add_argument("--cached-row-groups", type=int, default=16)
    args = parser.parse_args()

    # Saving lives inside train() now: with a validation split, the
    # checkpoint worth keeping is the best epoch, which only the loop knows.
    train(
        args.parquet, args.player_name, args.epochs, args.lr,
        batch_size=args.batch_size, log_every=args.log_every, limit=args.limit,
        device=args.device, num_workers=args.num_workers, shuffle=not args.no_shuffle,
        cached_row_groups=args.cached_row_groups,
        val_frac=args.val_frac, seed=args.seed, out=args.out,
    )
