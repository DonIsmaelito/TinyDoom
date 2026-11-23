"""Export the two trained models and a few test episodes for the browser demo in demo/.

    python export_demo.py

Reads checkpoints/cnn_best.pt, checkpoints/convgru_best.pt, and data/test, and writes into demo/:

    models/cnn.onnx, models/convgru.onnx   one prediction step each, for ONNX Runtime Web
    episodes/episode_<id>.png              every frame of an episode in one lossless image
    manifest.json                          what the page loads: models, episodes, actions,
                                           good start frames, and the numbers to check against
    clip.webp                              real | CNN | ConvGRU side by side, shown by the page
                                           while the models load and if they fail to load

The export fails if a 10-step rollout through an ONNX file differs from `model.rollout()` by
more than 1e-5 on any pixel value.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from data import ACTION_NAMES, FRAME_SIZE, WindowDataset, load_episodes, to_float
from evaluate import CONTEXT_FRAMES, STEPS, evaluate, evaluate_repeat_last_frame, load_model
from model import count_parameters, rollout

MODEL_LABELS = {"cnn": "CNN", "convgru": "ConvGRU"}
PARITY_TOLERANCE = 1e-5
SHEET_COLUMNS = 16
# In the fixture the page gets a real frame every FIXTURE_INTERVAL steps, so one stretch of
# predictions is one evaluate.py rollout.
FIXTURE_INTERVAL = max(STEPS)
# A start frame must leave this many actions after it, so the page has something to play.
MIN_ACTIONS_AFTER_START = 30
STARTS_PER_EPISODE = 4
START_SPACING = 10


class CnnStep(nn.Module):
    """One prediction step of a model without memory: (frame, action) -> next_frame."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frame, action):
        return self.model(frame, action)[0].clamp(0.0, 1.0)


class RecurrentStep(nn.Module):
    """One prediction step of a model with memory: (frame, action, hidden) -> next_frame, hidden.

    An all-zero hidden is the same as no hidden yet. To observe a real frame, run this step
    and ignore next_frame.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frame, action, hidden):
        next_frame, hidden = self.model(frame, action, hidden)
        return next_frame.clamp(0.0, 1.0), hidden


def hidden_shape(model):
    """Shape of the model's memory for one 64 x 64 frame, or None if it has no memory."""
    frame = torch.zeros(1, 3, FRAME_SIZE, FRAME_SIZE)
    with torch.no_grad():
        hidden = model(frame, torch.zeros(1, dtype=torch.long))[1]
    return None if hidden is None else list(hidden.shape)


def export_onnx(model, path):
    """Write one prediction step of the model to path. Returns the hidden shape or None."""
    model.eval()
    frame = torch.zeros(1, 3, FRAME_SIZE, FRAME_SIZE)
    action = torch.zeros(1, dtype=torch.long)
    shape = hidden_shape(model)
    if shape is None:
        module, args = CnnStep(model), (frame, action)
        input_names, output_names = ["frame", "action"], ["next_frame"]
    else:
        module, args = RecurrentStep(model), (frame, action, torch.zeros(shape))
        input_names, output_names = ["frame", "action", "hidden"], ["next_frame", "hidden_out"]
    torch.onnx.export(
        module,
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamo=False,
    )
    return shape


def open_session(path):
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def onnx_rollout(session, context, actions, context_frames):
    """`model.rollout()` done the way the page does it, one ONNX step at a time.

    context: float32 (C, 3, H, W) in [0, 1], C >= context_frames real frames.
    actions: int64 (C - 1 + K,). Returns float32 (K, 3, H, W).
    """
    recurrent = len(session.get_inputs()) == 3
    hidden = None
    if recurrent:
        hidden = np.zeros(session.get_inputs()[2].shape, dtype=np.float32)

    def step(frame, action):
        nonlocal hidden
        feed = {"frame": frame[None], "action": np.array([action], dtype=np.int64)}
        if recurrent:
            feed["hidden"] = hidden
            next_frame, hidden = session.run(None, feed)
        else:
            (next_frame,) = session.run(None, feed)
        return next_frame[0]

    last = len(context) - 1
    for i in range(len(context) - context_frames, last):
        step(context[i], actions[i])
    frame = context[last]
    predictions = []
    for action in actions[last:]:
        frame = step(frame, action)
        predictions.append(frame)
    return np.stack(predictions)


def check_parity(model, session, episodes, max_rollouts=8):
    """Largest difference on any pixel value between ONNX and PyTorch 10-step rollouts."""
    windows = WindowDataset(
        episodes, context_frames=CONTEXT_FRAMES, rollout_steps=FIXTURE_INTERVAL
    )
    if len(windows) == 0:
        raise ValueError("The exported episodes are too short for one 10-step rollout.")
    picks = np.linspace(0, len(windows) - 1, min(max_rollouts, len(windows))).astype(int)
    worst = 0.0
    for i in picks:
        context, actions, _ = windows[int(i)]
        context = to_float(context)
        with torch.no_grad():
            expected = rollout(model, context[None], actions[None])[0].clamp(0.0, 1.0).numpy()
        got = onnx_rollout(session, context.numpy(), actions.numpy(), model.context_frames)
        worst = max(worst, float(np.abs(got - expected).max()))
    return worst


def episode_sheet(frames, columns=SHEET_COLUMNS):
    """All frames of an episode in one image: frame i is at row i // columns, column i % columns."""
    count, height, width, _ = frames.shape
    rows = -(-count // columns)
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        y, x = (i // columns) * height, (i % columns) * width
        sheet[y : y + height, x : x + width] = frame
    return Image.fromarray(sheet)


def read_sheet(path, count, columns=SHEET_COLUMNS, size=FRAME_SIZE):
    """The inverse of episode_sheet: uint8 (count, size, size, 3)."""
    sheet = np.asarray(Image.open(path).convert("RGB"))
    return np.stack(
        [
            sheet[(i // columns) * size :, (i % columns) * size :][:size, :size]
            for i in range(count)
        ]
    )


def start_scores(frames, horizon=10):
    """How good each frame is as a place to start watching, higher is better.

    Many frames are one flat wall texture, where a prediction and the real frame look alike
    whatever the model does. A good start has large areas of different color for the next
    `horizon` frames, and the view changes during them.
    """
    f = frames.astype(np.float32) / 255.0
    count, size = len(f), frames.shape[1]
    cell = size // 8
    cells = f.reshape(count, 8, cell, 8, cell, 3).mean(axis=(2, 4)).reshape(count, 64, 3)
    layout = cells.std(axis=1).mean(axis=1)
    change = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2, 3))
    scores = np.zeros(count)
    for t in range(count - horizon):
        scores[t] = layout[t : t + horizon + 1].min() * change[t : t + horizon].mean()
    return scores


def pick_starts(frames, count=STARTS_PER_EPISODE, spacing=START_SPACING):
    """Up to `count` good start frames, best first, at least `spacing` frames apart."""
    scores = start_scores(frames)
    last = len(frames) - 1 - MIN_ACTIONS_AFTER_START
    candidates = sorted(range(CONTEXT_FRAMES - 1, last + 1), key=lambda t: -scores[t])
    picks = []
    for t in candidates:
        if picks and scores[t] < 0.25 * scores[picks[0]]:
            break
        if all(abs(t - p) >= spacing for p in picks):
            picks.append(t)
        if len(picks) == count:
            break
    return picks, [float(scores[t]) for t in picks]


def choose_episodes(episodes, count):
    """The `count` episodes with the best start frames, in episode order."""
    long_enough = [
        ep for ep in episodes if len(ep) >= CONTEXT_FRAMES - 1 + MIN_ACTIONS_AFTER_START
    ]
    if not long_enough:
        raise ValueError(f"No episode has {CONTEXT_FRAMES - 1 + MIN_ACTIONS_AFTER_START} actions.")
    ranked = sorted(long_enough, key=lambda ep: -sum(pick_starts(ep.frames)[1]))
    return sorted(ranked[:count], key=lambda ep: ep.episode_id)


def demo_fixture(models, episodes):
    """The evaluate.py numbers for exactly the rollouts the page plays.

    The page starts each exported episode at its first frame that has 3 real frames before
    it, shows the models a real frame every 10 steps, and stops an episode when fewer than 10
    actions are left. Each stretch of 10 predictions is then one evaluate.py rollout, so this
    is evaluate.py on the exported episodes using every 10th start frame.
    """
    reference = evaluate_repeat_last_frame(episodes, stride=FIXTURE_INTERVAL)
    fixture = {
        "interval": FIXTURE_INTERVAL,
        "first_frame": CONTEXT_FRAMES - 1,
        "rollouts": reference["rollouts"],
        "repeat_last_frame": reference["steps"],
    }
    for name, model in models.items():
        fixture[name] = evaluate(model, episodes, stride=FIXTURE_INTERVAL)["steps"]
    return fixture


def split_metrics(models, episodes, split):
    """The evaluate.py numbers on the whole split, for the page to print."""
    reference = evaluate_repeat_last_frame(episodes)
    metrics = {
        "split": split,
        "episodes": len(episodes),
        "rollouts": reference["rollouts"],
        "repeat_last_frame": reference["steps"],
    }
    for name, model in models.items():
        metrics[name] = evaluate(model, episodes)["steps"]
    return metrics


def to_uint8(frames):
    """float (N, 3, H, W) in [0, 1] -> uint8 (N, H, W, 3)."""
    return (frames.clamp(0.0, 1.0) * 255.0).round().byte().permute(0, 2, 3, 1).numpy()


def render_clip(models, episodes, starts, path, steps_per_episode=40, scale=3, fps=5):
    """A looping side-by-side clip: real | each model, with a real frame every 10 steps."""
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=12)
    size = FRAME_SIZE * scale
    gap, top, bottom = 8, 24, 24
    names = list(models)
    labels = ["Real game"] + [MODEL_LABELS.get(name, name) for name in names]
    width = len(labels) * size + (len(labels) + 1) * gap
    height = top + size + bottom

    def picture(panels, caption, highlight):
        image = Image.new("RGB", (width, height), (16, 16, 20))
        draw = ImageDraw.Draw(image)
        for i, (label, panel) in enumerate(zip(labels, panels)):
            x = gap + i * (size + gap)
            draw.text((x, 4), label, fill=(235, 235, 235), font=font)
            tile = Image.fromarray(panel).resize((size, size), Image.NEAREST)
            image.paste(tile, (x, top))
        color = (255, 196, 64) if highlight else (190, 190, 190)
        draw.text((gap, top + size + 5), caption, fill=color, font=small)
        return image

    pictures = []
    for episode, start in zip(episodes, starts):
        frames = to_float(torch.from_numpy(episode.frames).permute(0, 3, 1, 2))
        actions = torch.from_numpy(episode.actions)
        stop = min(len(episode), start + steps_per_episode)
        t = start
        while t + FIXTURE_INTERVAL <= stop:
            first = t - (CONTEXT_FRAMES - 1)
            context = frames[first : t + 1][None]
            window = actions[first : t + FIXTURE_INTERVAL][None]
            with torch.no_grad():
                predicted = [to_uint8(rollout(m, context, window)[0]) for m in models.values()]
            real = episode.frames[t]
            pictures.append(
                picture([real] * len(labels), "The models get a real frame", True)
            )
            for k in range(1, FIXTURE_INTERVAL + 1):
                caption = f"The models are {k} step{'s' if k > 1 else ''} past a real frame"
                panels = [episode.frames[t + k]] + [p[k - 1] for p in predicted]
                pictures.append(picture(panels, caption, False))
            t += FIXTURE_INTERVAL
    if not pictures:
        raise ValueError("No episode is long enough for the clip.")
    pictures[0].save(
        path,
        save_all=True,
        append_images=pictures[1:],
        duration=int(1000 / fps),
        loop=0,
        quality=80,
    )
    return len(pictures)


def manifest_text(manifest):
    """Indented JSON, with each list of numbers on one line."""
    text = json.dumps(manifest, indent=1)
    return re.sub(r"\[[\d\s,.eE+-]+\]", lambda m: re.sub(r"\s+", " ", m[0]), text) + "\n"


def export_demo(
    checkpoints, data_dir, out_dir, split="test", num_episodes=4, episode_ids=None
):
    """Write everything the page needs into out_dir. Returns the manifest."""
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)

    all_episodes = load_episodes(data_dir, split)
    if episode_ids:
        by_id = {ep.episode_id: ep for ep in all_episodes}
        missing = [i for i in episode_ids if i not in by_id]
        if missing:
            raise ValueError(f"No episode {missing} in {data_dir}/{split}.")
        episodes = [by_id[i] for i in episode_ids]
    else:
        episodes = choose_episodes(all_episodes, num_episodes)

    models, model_entries = {}, []
    for path in checkpoints:
        model, checkpoint = load_model(path, "cpu")
        name = checkpoint["model"]
        if name in models:
            raise ValueError(f"Two checkpoints hold a {name} model.")
        models[name] = model
        file = f"models/{name}.onnx"
        shape = export_onnx(model, out_dir / file)
        difference = check_parity(model, open_session(out_dir / file), episodes)
        print(f"{file}: largest difference from model.rollout() {difference:.2e}")
        if not difference <= PARITY_TOLERANCE:
            raise RuntimeError(
                f"{file} differs from model.rollout() by {difference:.2e} on a pixel value, "
                f"more than {PARITY_TOLERANCE:.0e}. Nothing in {out_dir} should be used."
            )
        model_entries.append(
            {
                "name": name,
                "label": MODEL_LABELS.get(name, name),
                "file": file,
                "parameters": count_parameters(model),
                "context_frames": model.context_frames,
                "hidden_shape": shape,
                "epoch": checkpoint["epoch"],
                "val_loss": checkpoint["val_loss"],
                "bytes": (out_dir / file).stat().st_size,
                "parity": difference,
            }
        )

    episode_entries = []
    for ep in episodes:
        file = f"episodes/episode_{ep.episode_id:04d}.png"
        episode_sheet(ep.frames).save(out_dir / file, optimize=True)
        if not np.array_equal(read_sheet(out_dir / file, len(ep.frames)), ep.frames):
            raise RuntimeError(f"{file} does not decode back to the episode's frames.")
        starts, _ = pick_starts(ep.frames)
        episode_entries.append(
            {
                "id": ep.episode_id,
                "sheet": file,
                "frames": len(ep.frames),
                "actions": ep.actions.tolist(),
                "starts": starts,
            }
        )
        print(f"{file}: {len(ep.frames)} frames, start frames {starts}")

    manifest = {
        "frame_size": FRAME_SIZE,
        "sheet_columns": SHEET_COLUMNS,
        "action_names": list(ACTION_NAMES),
        "context_frames": CONTEXT_FRAMES,
        "models": model_entries,
        "episodes": episode_entries,
        "fixture": demo_fixture(models, episodes),
        "metrics": split_metrics(models, all_episodes, split),
        "clip": "clip.webp",
    }
    starts = [entry["starts"][0] for entry in episode_entries]
    count = render_clip(models, episodes, starts, out_dir / manifest["clip"])
    print(f"{manifest['clip']}: {count} pictures")
    (out_dir / "manifest.json").write_text(manifest_text(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        default=["checkpoints/cnn_best.pt", "checkpoints/convgru_best.pt"],
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="demo")
    parser.add_argument("--episodes", type=int, default=4, help="how many episodes to export")
    parser.add_argument(
        "--episode-ids", type=int, nargs="+", help="export these episodes instead of choosing"
    )
    args = parser.parse_args()

    manifest = export_demo(
        args.checkpoint, args.data_dir, args.out_dir, args.split, args.episodes, args.episode_ids
    )
    fixture = manifest["fixture"]
    print(
        f"\n{fixture['rollouts']} rollouts on the exported episodes, "
        f"a real frame every {fixture['interval']} steps. MAE:"
    )
    names = [entry["name"] for entry in manifest["models"]] + ["repeat_last_frame"]
    print(f"{'step':>4}  " + "  ".join(f"{name:>17}" for name in names))
    for step in fixture["repeat_last_frame"]:
        print(f"{step:>4}  " + "  ".join(f"{fixture[n][step]['mae']:17.5f}" for n in names))
    print(f"\nWrote {args.out_dir}/manifest.json. Serve the page with:")
    print(f"    python -m http.server --directory {args.out_dir} 8000")


if __name__ == "__main__":
    main()
