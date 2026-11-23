"""Build a failure gallery of the worst (and a few best) 10-step rollouts.

    python gallery.py --checkpoint checkpoints/cnn_best.pt --out-dir gallery/

Uses the same rollouts as evaluate.py, ranks them by step-10 MAE, and writes per-rollout
strips, a contact sheet, and a JSON summary.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data import ACTION_NAMES, FRAME_SIZE, SPLITS, load_episodes
from evaluate import STEPS, collect_rollouts, evaluate, load_model
from train import pick_device

ROLLOUT_STEPS = max(STEPS)
SCALE = 4
LABEL_HEIGHT = 36
ROW_GAP = 2
STRIP_PAD = 8
CONTACT_GAP = 12


def chw_to_hwc_uint8(tensor):
    """CHW float [0,1] or uint8 -> H W C uint8."""
    if tensor.is_floating_point():
        array = (tensor.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    else:
        array = tensor.numpy().astype(np.uint8)
    return np.transpose(array, (1, 2, 0))


def upscale_nearest(frame_hwc, scale):
    image = Image.fromarray(frame_hwc, mode="RGB")
    return image.resize((frame_hwc.shape[1] * scale, frame_hwc.shape[0] * scale), Image.NEAREST)


def abs_diff_row(real_hwc, pred_hwc):
    diff = np.abs(real_hwc.astype(np.int16) - pred_hwc.astype(np.int16)).astype(np.uint8)
    scale = diff.max()
    if scale == 0:
        return diff
    return (diff.astype(np.float32) / scale * 255).astype(np.uint8)


def render_strip(record, model_name, scale=SCALE):
    """One rollout: real, predicted, abs-diff rows; labels under each step column."""
    steps = len(record["per_step_mae"])
    cell_w = cell_h = FRAME_SIZE
    up_w, up_h = cell_w * scale, cell_h * scale

    rows = []
    for row_name, source in (
        ("real", "targets"),
        ("pred", "predictions"),
    ):
        tiles = []
        for t in range(steps):
            if source == "targets":
                frame = chw_to_hwc_uint8(record["targets"][t])
            else:
                frame = chw_to_hwc_uint8(record["predictions"][t])
            tiles.append(upscale_nearest(frame, scale))
        rows.append(tiles)

    diff_tiles = []
    for t in range(steps):
        real = chw_to_hwc_uint8(record["targets"][t])
        pred = chw_to_hwc_uint8(record["predictions"][t])
        diff_tiles.append(upscale_nearest(abs_diff_row(real, pred), scale))
    rows.append(diff_tiles)

    width = steps * up_w + (steps - 1) * ROW_GAP + 2 * STRIP_PAD
    height = 3 * up_h + 2 * ROW_GAP + LABEL_HEIGHT + 2 * STRIP_PAD
    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title = (
        f"{model_name}  ep {record['episode_id']}  start {record['start_index']}  "
        f"step-10 MAE {record['per_step_mae'][-1]:.5f}"
    )
    draw.text((STRIP_PAD, 4), title, fill=(220, 220, 220), font=font)

    y0 = STRIP_PAD + 16
    row_labels = ("real", "pred", "diff")
    for row_idx, tiles in enumerate(rows):
        y = y0 + row_idx * (up_h + ROW_GAP)
        for col, tile in enumerate(tiles):
            x = STRIP_PAD + col * (up_w + ROW_GAP)
            canvas.paste(tile, (x, y))
        draw.text((2, y + up_h // 2), row_labels[row_idx], fill=(180, 180, 180), font=font)

    label_y = y0 + 3 * up_h + 2 * ROW_GAP
    for col in range(steps):
        x = STRIP_PAD + col * (up_w + ROW_GAP)
        action = ACTION_NAMES[record["actions"][col]]
        mae = record["per_step_mae"][col]
        text = f"{action}\n{mae:.4f}"
        draw.multiline_text((x, label_y), text, fill=(200, 200, 200), font=font, spacing=2)

    return canvas


def build_contact_sheet(images, columns=2):
    if not images:
        raise ValueError("No images for contact sheet.")
    columns = min(columns, len(images))
    col_widths = [0] * columns
    for idx, im in enumerate(images):
        col_widths[idx % columns] = max(col_widths[idx % columns], im.width)
    row_heights = []
    for row_start in range(0, len(images), columns):
        row_heights.append(
            max(images[j].height for j in range(row_start, min(row_start + columns, len(images))))
        )
    sheet_w = sum(col_widths) + (columns - 1) * CONTACT_GAP + 2 * STRIP_PAD
    sheet_h = sum(row_heights) + (len(row_heights) - 1) * CONTACT_GAP + 2 * STRIP_PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 16, 16))
    y = STRIP_PAD
    for row_start, row_h in zip(range(0, len(images), columns), row_heights):
        x = STRIP_PAD
        for col, idx in enumerate(range(row_start, min(row_start + columns, len(images)))):
            sheet.paste(images[idx], (x, y))
            x += col_widths[col] + CONTACT_GAP
        y += row_h + CONTACT_GAP
    return sheet


def select_rollouts(merged, worst_k=8, best_k=2):
    ranked = sorted(merged, key=lambda r: r["rank_mae"], reverse=True)
    worst = ranked[:worst_k]
    best = ranked[-best_k:] if best_k else []
    chosen, seen = [], set()
    for label, group in (("worst", worst), ("best", best)):
        for item in group:
            key = (item["episode_id"], item["start_index"])
            if key in seen:
                continue
            seen.add(key)
            chosen.append({**item, "selection": label})
    return chosen


def run_gallery(
    checkpoint_paths,
    episodes,
    device,
    out_dir,
    worst_k=8,
    best_k=2,
    batch_size=64,
    rank_checkpoint_index=0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for path in checkpoint_paths:
        model, checkpoint = load_model(path, device)
        records = collect_rollouts(model, episodes, device, ROLLOUT_STEPS, batch_size)
        loaded.append((path, checkpoint["model"], records))

    rank_model = loaded[rank_checkpoint_index][1]
    keyed = {name: recs for _, name, recs in loaded}
    primary_records = loaded[rank_checkpoint_index][2]
    merged = []
    for rec in primary_records:
        entry = {
            "episode_id": rec["episode_id"],
            "start_index": rec["start_index"],
            "actions": rec["actions"],
            "targets": rec["targets"],
            "models": {},
        }
        for _, name, records in loaded:
            match = next(
                r
                for r in records
                if r["episode_id"] == rec["episode_id"] and r["start_index"] == rec["start_index"]
            )
            entry["models"][name] = {
                "per_step_mae": match["per_step_mae"],
                "predictions": match["predictions"],
            }
        entry["rank_mae"] = entry["models"][rank_model]["per_step_mae"][-1]
        merged.append(entry)

    selected = select_rollouts(merged, worst_k=worst_k, best_k=best_k)
    summary = {
        "rollout_steps": ROLLOUT_STEPS,
        "rank_model": rank_model,
        "mean_step10_mae_by_model": {
            name: float(np.mean([r["per_step_mae"][-1] for r in records]))
            for _, name, records in loaded
        },
        "rollouts": [
        {
            "selection": pick["selection"],
            "episode_id": pick["episode_id"],
            "start_index": pick["start_index"],
            "actions": pick["actions"],
            "rank_step10_mae": pick["rank_mae"],
            "per_step_mae": {
                name: pick["models"][name]["per_step_mae"] for _, name, _ in loaded
            },
        }
        for pick in selected
        ],
    }

    strip_paths = []
    contact_images = []
    for _, model_name, _ in loaded:
        model_strips = []
        for pick in selected:
            render_record = {
                **pick,
                "per_step_mae": pick["models"][model_name]["per_step_mae"],
                "predictions": pick["models"][model_name]["predictions"],
            }
            strip = render_strip(render_record, model_name)
            filename = (
                f"{model_name}-ep{pick['episode_id']:04d}-t{pick['start_index']:03d}-"
                f"{pick['selection']}.png"
            )
            path = out_dir / filename
            strip.save(path)
            strip_paths.append(path)
            model_strips.append(strip)

        contact = build_contact_sheet(model_strips, columns=2)
        contact_path = out_dir / f"{model_name}-contact-sheet.png"
        contact.save(contact_path)
        contact_images.append(contact_path)

    json_path = out_dir / "gallery.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    return summary, strip_paths, contact_images, json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="gallery")
    parser.add_argument("--worst-k", type=int, default=8)
    parser.add_argument("--best-k", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = pick_device(args.device)
    episodes = load_episodes(args.data_dir, args.split)

    # Verify means match evaluate.py before writing images.
    for path in args.checkpoint:
        model, checkpoint = load_model(path, device)
        agg = evaluate(model, episodes, device, batch_size=args.batch_size)
        print(
            f"{path} ({checkpoint['model']}): {agg['rollouts']} rollouts, "
            f"step-10 MAE {agg['steps']['10']['mae']:.5f}"
        )

    summary, strips, contacts, json_path = run_gallery(
        args.checkpoint,
        episodes,
        device,
        args.out_dir,
        worst_k=args.worst_k,
        best_k=args.best_k,
        batch_size=args.batch_size,
    )
    print(f"\nWrote {len(strips)} strips, {len(contacts)} contact sheet(s), {json_path}")
    for name, mae in summary["mean_step10_mae_by_model"].items():
        print(f"  {name} mean step-10 MAE (from collect_rollouts): {mae:.5f}")


if __name__ == "__main__":
    main()