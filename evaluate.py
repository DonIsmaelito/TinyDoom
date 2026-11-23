"""Evaluate autoregressive rollouts: MAE and PSNR at 1, 5, and 10 steps.

    python evaluate.py                                          # checkpoints/cnn_best.pt on test
    python evaluate.py --checkpoint checkpoints/convgru_best.pt
    python evaluate.py --checkpoint checkpoints/cnn_best.pt checkpoints/convgru_best.pt

A rollout starts at a frame that has 3 real frames before it and 10 recorded actions after it.
The model then predicts 10 frames from the actions alone, feeding its own predictions back in.
Every model is scored on the same rollouts: the ConvGRU observes all 4 real frames first, the
CNN starts from the last of them. Every possible start frame in the split is used.

The last table is a reference that needs no model: it repeats the last real frame for all 10
steps. A model is only useful where it beats this.
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from data import SPLITS, WindowDataset, load_episodes, to_float
from model import MODELS, build_model, rollout
from train import pick_device

STEPS = (1, 5, 10)
CONTEXT_FRAMES = max(m.context_frames for m in MODELS.values())


def frame_metrics(prediction, target):
    """Per-frame MAE and PSNR for float frames in [0, 1], shape (..., 3, H, W)."""
    error = prediction - target
    mae = error.abs().mean(dim=(-3, -2, -1))
    mse = error.pow(2).mean(dim=(-3, -2, -1))
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
    return mae, psnr


def repeat_last_frame(context, actions):
    """The reference that needs no model: every predicted frame is the last real frame."""
    steps = actions.shape[1] - context.shape[1] + 1
    return context[:, -1:].expand(-1, steps, -1, -1, -1)


@torch.no_grad()
def score_rollouts(predict, episodes, device="cpu", steps=STEPS, batch_size=64, stride=1):
    """Mean MAE and PSNR over all rollouts at each step in steps.

    predict(context, actions) returns the predicted frames, like `rollout` does for a model.
    With stride S, only every S-th start frame of each episode is used.
    """
    windows = WindowDataset(
        episodes, context_frames=CONTEXT_FRAMES, rollout_steps=max(steps), stride=stride
    )
    if len(windows) == 0:
        raise ValueError(
            f"No episode has {CONTEXT_FRAMES} frames followed by {max(steps)} actions."
        )
    index = [s - 1 for s in steps]
    maes, psnrs = [], []
    for context, actions, targets in DataLoader(windows, batch_size=batch_size):
        prediction = predict(to_float(context.to(device)), actions.to(device))
        mae, psnr = frame_metrics(
            prediction[:, index].clamp(0.0, 1.0), to_float(targets[:, index].to(device))
        )
        maes.append(mae.cpu())
        psnrs.append(psnr.cpu())
    mae = torch.cat(maes).mean(dim=0)
    psnr = torch.cat(psnrs).mean(dim=0)
    return {
        "rollouts": len(windows),
        "steps": {
            str(s): {"mae": mae[j].item(), "psnr": psnr[j].item()} for j, s in enumerate(steps)
        },
    }


def evaluate(model, episodes, device="cpu", steps=STEPS, batch_size=64, stride=1):
    """Mean MAE and PSNR of the model's rollouts at each step in steps."""
    model.eval()
    return score_rollouts(
        lambda context, actions: rollout(model, context, actions),
        episodes,
        device,
        steps,
        batch_size,
        stride,
    )


def evaluate_repeat_last_frame(episodes, device="cpu", steps=STEPS, batch_size=64, stride=1):
    """The same metrics on the same rollouts for repeating the last real frame."""
    return score_rollouts(repeat_last_frame, episodes, device, steps, batch_size, stride)


def print_table(result):
    print(f"{'step':>4}  {'MAE':>8}  {'PSNR':>6}")
    for step, m in result["steps"].items():
        print(f"{step:>4}  {m['mae']:8.5f}  {m['psnr']:6.2f}")


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(checkpoint["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", nargs="+", default=["checkpoints/cnn_best.pt"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--out", help="optional path for a JSON copy of the metrics")
    args = parser.parse_args()

    device = pick_device(args.device)
    episodes = load_episodes(args.data_dir, args.split)
    print(f"split {args.split}: {len(episodes)} episodes")
    print("MAE is on pixel values in [0, 1]. PSNR is in dB, averaged over frames.")

    reference = evaluate_repeat_last_frame(episodes, device)
    results = []
    for path in args.checkpoint:
        model, checkpoint = load_model(path, device)
        result = evaluate(model, episodes, device)
        print(
            f"\n{path}: model {checkpoint['model']}, epoch {checkpoint['epoch']}, "
            f"val L1 {checkpoint['val_loss']:.5f}, {result['rollouts']} rollouts"
        )
        print_table(result)
        results.append(
            dict(
                checkpoint=path,
                model=checkpoint["model"],
                split=args.split,
                **result,
                repeat_last_frame=reference["steps"],
            )
        )

    print(f"\nrepeat the last real frame (reference, no model), {reference['rollouts']} rollouts")
    print_table(reference)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
