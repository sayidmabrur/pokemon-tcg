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
import time

import torch
import torch.utils.data

from collate import collate_features, pad_stack
from dataset import PolicyFeatureDataset, transform
from policy_experimental import PolicyNetwork, masked_bce_loss


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
) -> PolicyNetwork:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}, batch_size={batch_size}", flush=True)

    t0 = time.time()
    dataset = PolicyFeatureDataset(
        parquet_path, player_name=player_name, transform=transform,
        cached_row_groups=cached_row_groups,
    )
    if limit is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(limit, len(dataset))))
    print(f"dataset ready in {time.time() - t0:.1f}s — {len(dataset)} samples", flush=True)

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
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch,
        num_workers=num_workers,
    )
    print(f"{len(loader)} batches/epoch, shuffle={shuffle}, num_workers={num_workers}", flush=True)

    policy = PolicyNetwork().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

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

        print(f"epoch {epoch} done: avg loss={total_loss / max(num_batches, 1):.4f}", flush=True)

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
    parser.add_argument("--no-shuffle", action="store_true", help="disable shuffling (debug)")
    parser.add_argument("--cached-row-groups", type=int, default=16)
    args = parser.parse_args()

    policy = train(
        args.parquet, args.player_name, args.epochs, args.lr,
        batch_size=args.batch_size, log_every=args.log_every, limit=args.limit,
        device=args.device, num_workers=args.num_workers, shuffle=not args.no_shuffle,
        cached_row_groups=args.cached_row_groups,
    )
    torch.save(policy.state_dict(), args.out)
    print(f"saved {args.out}", flush=True)
