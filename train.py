"""Train a world model on frame prediction with L1 loss and Adam.

    python train.py                        # CNN baseline -> checkpoints/cnn_best.pt
    python train.py --model convgru        # ConvGRU      -> checkpoints/convgru_best.pt
    python train.py --epochs 100 --device cuda
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import WindowDataset, load_episodes, to_float
from model import MODELS, build_model, count_parameters, rollout


def pick_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def prediction_loss(model, batch, device):
    """L1 between the predicted and the actual frames. The same loss for every model."""
    context, actions, targets = (x.to(device, non_blocking=True) for x in batch)
    return F.l1_loss(rollout(model, to_float(context), actions), to_float(targets))


def run_epoch(model, loader, device, optimizer=None):
    """One pass over loader. Trains when an optimizer is given, otherwise only measures loss."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            loss = prediction_loss(model, batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * len(batch[0])
            count += len(batch[0])
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="cnn", choices=sorted(MODELS))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1,
        help="frames predicted per training example; 1 is next-frame prediction, more than 1 "
        "also trains on the model's own predictions",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    window = dict(context_frames=model.context_frames, rollout_steps=args.rollout_steps)
    train_set = WindowDataset(load_episodes(args.data_dir, "train"), **window)
    val_set = WindowDataset(load_episodes(args.data_dir, "val"), **window)
    if len(train_set) == 0 or len(val_set) == 0:
        raise SystemExit("Episodes are too short to cut any training or validation examples.")
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, pin_memory=pin)
    print(
        f"model {args.model}: {count_parameters(model):,} trainable parameters, device {device}, "
        f"{len(train_set)} train and {len(val_set)} val examples"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{args.model}_best.pt"
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        print(
            f"epoch {epoch:3d}  train L1 {train_loss:.5f}  val L1 {val_loss:.5f}  "
            f"{time.time() - start:5.1f}s{'  saved' if improved else ''}",
            flush=True,
        )
    print(f"best val L1 {best_val:.5f}, checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
