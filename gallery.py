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
LABEL_HEIGHT = 72
ROW_LABEL_WIDTH = 112
ROW_GAP = 4
STRIP_PAD = 12
CONTACT_GAP = 12
DIFF_MAX = 255  # fixed uint8 scale for |real - pred| per RGB channel


def load_fonts():
    """Readable poster labels; falls back to default if DejaVu is missing."""
    try:
        title = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        row = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        column = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        title = row = column = ImageFont.load_default()
    return title, row, column


def rollout_interval(start_index, rollout_steps=ROLLOUT_STEPS):
    """Inclusive frame indices of the 10 predicted targets (t+1 .. t+rollout_steps)."""
    return start_index + 1, start_index + rollout_steps


def windows_overlap(start_a, start_b, rollout_steps=ROLLOUT_STEPS):
    a0, a1 = rollout_interval(start_a, rollout_steps)
    b0, b1 = rollout_interval(start_b, rollout_steps)
    return not (a1 < b0 or b1 < a0)


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
    """Per-channel |real - pred| on uint8 RGB; fixed 0..255 scale (no per-frame normalization)."""
    return np.abs(real_hwc.astype(np.int16) - pred_hwc.astype(np.int16)).astype(np.uint8)


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

    grid_x0 = STRIP_PAD + ROW_LABEL_WIDTH
    width = grid_x0 + steps * up_w + (steps - 1) * ROW_GAP + STRIP_PAD
    height = 3 * up_h + 2 * ROW_GAP + LABEL_HEIGHT + STRIP_PAD + 28
    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    title_font, row_font, column_font = load_fonts()

    title = (
        f"{model_name}  episode {record['episode_id']}  start frame {record['start_index']}  "
        f"step-10 MAE {record['per_step_mae'][-1]:.5f}"
    )
    draw.text((STRIP_PAD, 6), title, fill=(235, 235, 235), font=title_font)

    y0 = 28
    row_labels = ("Real", "Predicted", "Difference")
    for row_idx, tiles in enumerate(rows):
        y = y0 + row_idx * (up_h + ROW_GAP)
        for col, tile in enumerate(tiles):
            x = grid_x0 + col * (up_w + ROW_GAP)
            canvas.paste(tile, (x, y))
        draw.text(
            (STRIP_PAD, y + up_h // 2 - 10),
            row_labels[row_idx],
            fill=(230, 230, 230),
            font=row_font,
        )

    label_y = y0 + 3 * up_h + 2 * ROW_GAP + 4
    for col in range(steps):
        x = grid_x0 + col * (up_w + ROW_GAP)
        action = ACTION_NAMES[record["actions"][col]]
        mae = record["per_step_mae"][col]
        text = f"{action}\nMAE {mae:.4f}"
        draw.multiline_text(
            (x, label_y),
            text,
            fill=(235, 235, 235),
            font=column_font,
            spacing=4,
        )

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


def raw_ranking(merged):
    ranked = sorted(merged, key=lambda r: r["rank_mae"], reverse=True)
    return [
        {
            "rank": index + 1,
            "episode_id": item["episode_id"],
            "start_index": item["start_index"],
            "step10_mae": float(item["rank_mae"]),
        }
        for index, item in enumerate(ranked)
    ]


def pick_non_overlapping(candidates, k, picked_by_episode, rollout_steps, label):
    chosen = []
    for item in candidates:
        if len(chosen) >= k:
            break
        episode_id = item["episode_id"]
        start_index = item["start_index"]
        starts = picked_by_episode.get(episode_id, [])
        if any(windows_overlap(start_index, other, rollout_steps) for other in starts):
            continue
        chosen.append({**item, "selection": label})
        picked_by_episode.setdefault(episode_id, []).append(start_index)
    return chosen


def select_rollouts(
    merged, worst_k=8, best_k=2, dedupe_overlap=True, rollout_steps=ROLLOUT_STEPS
):
    ranked_desc = sorted(merged, key=lambda r: r["rank_mae"], reverse=True)
    ranked_asc = list(reversed(ranked_desc))
    ranking = raw_ranking(merged)

    if not dedupe_overlap:
        worst = [{**item, "selection": "worst"} for item in ranked_desc[:worst_k]]
        best = [{**item, "selection": "best"} for item in ranked_asc[:best_k]]
        return worst + best, ranking

    worst = pick_non_overlapping(
        ranked_desc, worst_k, {}, rollout_steps, "worst"
    )
    best = pick_non_overlapping(
        ranked_asc, best_k, {}, rollout_steps, "best"
    )
    return worst + best, ranking


def run_gallery(
    checkpoint_paths,
    episodes,
    device,
    out_dir,
    worst_k=8,
    best_k=2,
    batch_size=64,
    rank_checkpoint_index=0,
    dedupe_overlap=True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    json_old = out_dir / "gallery.json"
    if json_old.exists():
        json_old.unlink()

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

    selected, ranking = select_rollouts(
        merged, worst_k=worst_k, best_k=best_k, dedupe_overlap=dedupe_overlap
    )
    summary = {
        "rollout_steps": ROLLOUT_STEPS,
        "dedupe_overlap": dedupe_overlap,
        "rank_model": rank_model,
        "raw_ranking": ranking,
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
    parser.add_argument(
        "--no-dedupe-overlap",
        action="store_false",
        dest="dedupe_overlap",
        help="allow overlapping windows from the same episode in the gallery",
    )
    parser.set_defaults(dedupe_overlap=True)
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
        dedupe_overlap=args.dedupe_overlap,
    )
    print(f"\nWrote {len(strips)} strips, {len(contacts)} contact sheet(s), {json_path}")
    for name, mae in summary["mean_step10_mae_by_model"].items():
        print(f"  {name} mean step-10 MAE (from collect_rollouts): {mae:.5f}")


if __name__ == "__main__":
    main()