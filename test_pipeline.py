"""Fast checks of the pipeline on small synthetic episodes.

    pytest

Only test_collect_real_episodes needs ViZDoom. It is skipped when ViZDoom is not installed.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from data import (
    FRAME_SIZE,
    NUM_ACTIONS,
    SPLITS,
    Episode,
    WindowDataset,
    assign_splits,
    load_episodes,
    save_episode,
    split_counts,
)
from evaluate import collect_rollouts, evaluate, frame_metrics
from gallery import run_gallery, select_rollouts, windows_overlap
from model import build_model, count_parameters, rollout
from train import prediction_loss

PARAMETERS = {"cnn": 371_235, "convgru": 786_147}


def make_episode(episode_id, num_actions, seed=0):
    """Synthetic episode: a random pattern that scrolls sideways on turns and down on forward."""
    rng = np.random.default_rng(seed + episode_id)
    coarse = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    frame = np.kron(coarse, np.ones((8, 8, 1), dtype=np.uint8))
    actions = rng.integers(NUM_ACTIONS, size=num_actions)
    shifts = {0: (4, 0), 1: (4, 1), 2: (-4, 1)}
    frames = [frame]
    for action in actions:
        shift, axis = shifts[int(action)]
        frames.append(np.roll(frames[-1], shift, axis=axis))
    return Episode(
        episode_id=episode_id,
        frames=np.stack(frames),
        actions=actions.astype(np.int64),
        positions=np.zeros((num_actions + 1, 3), dtype=np.float32),
        angles=np.zeros(num_actions + 1, dtype=np.float32),
    )


def make_indexed_episode(episode_id, num_actions):
    """Episode whose frames spell out where they came from: red = episode id, green = step."""
    frames = np.zeros((num_actions + 1, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    frames[..., 0] = episode_id
    frames[..., 1] = np.arange(num_actions + 1)[:, None, None]
    episode = make_episode(episode_id, num_actions)
    episode.frames = frames
    episode.actions = (np.arange(num_actions) + episode_id) % NUM_ACTIONS
    return episode


@pytest.mark.parametrize("name", sorted(PARAMETERS))
def test_model_output_shapes_and_parameter_count(name):
    model = build_model(name)
    assert count_parameters(model) == PARAMETERS[name]

    frame = torch.rand(2, 3, FRAME_SIZE, FRAME_SIZE)
    action = torch.randint(NUM_ACTIONS, (2,))
    next_frame, hidden = model(frame, action)
    assert next_frame.shape == frame.shape
    if name == "cnn":
        assert hidden is None
    else:
        assert hidden.shape == (2, 96, FRAME_SIZE // 4, FRAME_SIZE // 4)

    context = torch.rand(2, model.context_frames, 3, FRAME_SIZE, FRAME_SIZE)
    actions = torch.randint(NUM_ACTIONS, (2, model.context_frames - 1 + 5))
    assert rollout(model, context, actions).shape == (2, 5, 3, FRAME_SIZE, FRAME_SIZE)


def test_convgru_remembers_earlier_frames_and_cnn_does_not():
    torch.manual_seed(0)
    context = torch.rand(1, 4, 3, FRAME_SIZE, FRAME_SIZE)
    changed = context.clone()
    changed[:, 0] = torch.rand(3, FRAME_SIZE, FRAME_SIZE)
    actions = torch.randint(NUM_ACTIONS, (1, 6))
    with torch.no_grad():
        cnn, convgru = build_model("cnn"), build_model("convgru")
        assert torch.equal(rollout(cnn, context, actions), rollout(cnn, changed, actions))
        assert not torch.equal(
            rollout(convgru, context, actions), rollout(convgru, changed, actions)
        )


@pytest.mark.parametrize("name", sorted(PARAMETERS))
def test_training_steps_reduce_loss(name):
    torch.manual_seed(0)
    model = build_model(name)
    dataset = WindowDataset([make_episode(0, 11)], context_frames=model.context_frames)
    batch = torch.utils.data.default_collate([dataset[i] for i in range(8)])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(10):
        loss = prediction_loss(model, batch, "cpu")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert losses[-1] < losses[0]


def test_split_counts():
    assert split_counts(60) == [42, 8, 10]
    assert split_counts(120) == [84, 16, 20]
    for n in (1, 6, 7, 59, 61, 100):
        assert sum(split_counts(n)) == n
    assert assign_splits(60) == ["train"] * 42 + ["val"] * 8 + ["test"] * 10


def test_dataset_loading_and_whole_episode_split(tmp_path):
    episodes = [make_indexed_episode(i, num_actions=5 + i % 3) for i in range(60)]
    for episode, split in zip(episodes, assign_splits(len(episodes))):
        save_episode(tmp_path, split, episode)

    loaded = {split: load_episodes(tmp_path, split) for split in SPLITS}
    assert [len(loaded[split]) for split in SPLITS] == [42, 8, 10]
    ids = [{ep.episode_id for ep in loaded[split]} for split in SPLITS]
    assert set.union(*ids) == set(range(60))
    assert sum(len(s) for s in ids) == 60, "an episode appears in more than one split"

    for ep in loaded["val"]:
        original = episodes[ep.episode_id]
        assert np.array_equal(ep.frames, original.frames)
        assert np.array_equal(ep.actions, original.actions)
        assert ep.frames.dtype == np.uint8 and ep.frames.shape[1:] == (FRAME_SIZE, FRAME_SIZE, 3)

    # One example per transition, and no example mixes frames from two episodes.
    transitions = WindowDataset(loaded["train"])
    assert len(transitions) == sum(len(ep) for ep in loaded["train"])
    windows = WindowDataset(loaded["train"], context_frames=4, rollout_steps=2)
    assert len(windows) == sum(len(ep) - 4 for ep in loaded["train"])
    for context, actions, targets in windows:
        assert context.shape == (4, 3, FRAME_SIZE, FRAME_SIZE) and targets.shape[0] == 2
        frames = torch.cat([context, targets])
        episode_id = int(frames[0, 0, 0, 0])
        first_step = int(frames[0, 1, 0, 0])
        assert (frames[:, 0] == episode_id).all()
        assert frames[:, 1, 0, 0].tolist() == list(range(first_step, first_step + 6))
        expected = episodes[episode_id].actions[first_step : first_step + 5]
        assert actions.tolist() == expected.tolist()


def test_missing_split_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_episodes(tmp_path, "train")


def test_frame_metrics():
    target = torch.full((2, 3, 8, 8), 0.5)
    mae, psnr = frame_metrics(target + 0.1, target)
    assert torch.allclose(mae, torch.tensor(0.1))
    assert torch.allclose(psnr, torch.tensor(20.0), atol=1e-4)
    mae, psnr = frame_metrics(target, target)
    assert torch.all(mae == 0) and torch.all(psnr == 100.0)


class Brighten(nn.Module):
    """Stand-in model that adds a fixed amount to the frame at every step."""

    context_frames = 1

    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(self, frame, action, hidden=None):
        return frame + self.amount, None


def test_rollout_evaluation_reports_mae_and_psnr_at_1_5_10():
    # Every frame is one grey level brighter than the one before it.
    episodes = [make_indexed_episode(i, num_actions=20) for i in range(2)]
    for ep in episodes:
        ep.frames[:] = np.arange(21, dtype=np.uint8)[:, None, None, None]

    perfect = evaluate(Brighten(1 / 255), episodes)
    assert perfect["rollouts"] == 2 * (20 - 3 - 10 + 1)
    assert set(perfect["steps"]) == {"1", "5", "10"}
    for step in ("1", "5", "10"):
        assert perfect["steps"][step]["mae"] < 1e-6
        assert perfect["steps"][step]["psnr"] > 90

    # A model that never changes the frame falls behind by one grey level per step.
    frozen = evaluate(Brighten(0.0), episodes)
    for step in (1, 5, 10):
        assert frozen["steps"][str(step)]["mae"] == pytest.approx(step / 255, rel=1e-4)
    assert frozen["steps"]["1"]["psnr"] > frozen["steps"]["5"]["psnr"] > frozen["steps"]["10"]["psnr"]


@pytest.mark.parametrize("name", sorted(PARAMETERS))
def test_rollout_evaluation_runs_for_each_model(name):
    torch.manual_seed(0)
    result = evaluate(build_model(name), [make_episode(0, 14)])
    assert result["rollouts"] == 2
    for step in ("1", "5", "10"):
        assert np.isfinite(result["steps"][step]["mae"])
        assert np.isfinite(result["steps"][step]["psnr"])


def test_collect_rollouts_step10_mae_matches_evaluate():
    episodes = [make_episode(0, 20), make_episode(1, 18)]
    torch.manual_seed(0)
    model = build_model("cnn")
    agg = evaluate(model, episodes, device="cpu")
    records = collect_rollouts(model, episodes, device="cpu")
    mean10 = sum(r["per_step_mae"][-1] for r in records) / len(records)
    assert len(records) == agg["rollouts"]
    assert mean10 == pytest.approx(agg["steps"]["10"]["mae"], rel=0, abs=1e-6)


def test_select_rollouts_dedupes_overlapping_windows():
    merged = []
    for start in range(8, 18):
        merged.append(
            {
                "episode_id": 57,
                "start_index": start,
                "rank_mae": 1.0 - start * 0.01,
                "actions": [0] * 10,
                "targets": None,
                "models": {},
            }
        )
    for start in range(30, 40):
        merged.append(
            {
                "episode_id": 53,
                "start_index": start,
                "rank_mae": 0.5 - start * 0.001,
                "actions": [0] * 10,
                "targets": None,
                "models": {},
            }
        )
    selected, ranking = select_rollouts(merged, worst_k=8, best_k=2, dedupe_overlap=True)
    worst = [r for r in selected if r["selection"] == "worst"]
    assert len(worst) == 2
    assert worst[0]["episode_id"] == 57 and worst[0]["start_index"] == 8
    assert worst[1]["episode_id"] == 53 and worst[1]["start_index"] == 30
    by_ep = {}
    for item in worst:
        by_ep.setdefault(item["episode_id"], []).append(item["start_index"])
    for starts in by_ep.values():
        for i, s0 in enumerate(starts):
            for s1 in starts[i + 1 :]:
                assert not windows_overlap(s0, s1)
    assert len(ranking) == len(merged)
    assert ranking[0]["start_index"] == 8

    dense_worst, _ = select_rollouts(
        merged, worst_k=8, best_k=0, dedupe_overlap=False
    )
    assert len(dense_worst) == 8
    assert dense_worst[0]["start_index"] == 8
    assert dense_worst[1]["start_index"] == 9


def test_gallery_writes_strips_and_json(tmp_path):
    episodes = [make_episode(0, 20), make_episode(1, 20)]
    for ep in episodes:
        save_episode(tmp_path, "test", ep)
    loaded = load_episodes(tmp_path, "test")
    torch.manual_seed(0)
    model = build_model("cnn")
    checkpoint = tmp_path / "cnn_untrained.pt"
    torch.save(
        {
            "model": "cnn",
            "state_dict": model.state_dict(),
            "epoch": 0,
            "val_loss": 1.0,
        },
        checkpoint,
    )
    out_dir = tmp_path / "gallery"
    summary, strips, contacts, json_path = run_gallery(
        [str(checkpoint)], loaded, "cpu", out_dir, worst_k=2, best_k=1, batch_size=8
    )
    agg = evaluate(model, loaded, device="cpu")
    assert summary["mean_step10_mae_by_model"]["cnn"] == pytest.approx(
        agg["steps"]["10"]["mae"], rel=0, abs=1e-6
    )
    assert len(strips) == 3
    assert all(p.exists() for p in strips)
    assert len(contacts) == 1 and contacts[0].exists()
    assert json_path.exists()
    assert len(summary["rollouts"]) == 3
    for record in summary["rollouts"]:
        assert len(record["actions"]) == 10
        assert len(record["per_step_mae"]["cnn"]) == 10


def test_collect_real_episodes(tmp_path):
    pytest.importorskip("vizdoom")
    from collect import collect

    metadata = collect(tmp_path, num_episodes=6, max_actions=12)
    assert metadata["episodes"] == {"train": 4, "val": 1, "test": 1}
    for split in SPLITS:
        for ep in load_episodes(tmp_path, split):
            assert 1 <= len(ep) <= 12
            assert ep.frames.shape == (len(ep) + 1, FRAME_SIZE, FRAME_SIZE, 3)
            assert ep.frames.dtype == np.uint8
            assert set(ep.actions.tolist()) <= set(range(NUM_ACTIONS))
            assert ep.positions.shape == (len(ep) + 1, 3)
            assert ep.angles.shape == (len(ep) + 1,)
            assert not np.array_equal(ep.frames[0], ep.frames[-1])
