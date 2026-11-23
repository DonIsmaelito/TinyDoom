"""Checks of the demo export on random-weight models and small synthetic episodes.

    pytest test_demo.py

Skipped when onnxruntime is not installed.
"""

import json

import numpy as np
import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")

import export_demo  # noqa: E402
from data import FRAME_SIZE, NUM_ACTIONS, save_episode  # noqa: E402
from evaluate import evaluate, evaluate_repeat_last_frame, load_model  # noqa: E402
from export_demo import (  # noqa: E402
    export_onnx,
    onnx_rollout,
    open_session,
    pick_starts,
    read_sheet,
)
from model import build_model, rollout  # noqa: E402
from test_pipeline import make_episode  # noqa: E402

MODELS = ("cnn", "convgru")
EPISODE_ACTIONS = (45, 52, 38, 60)


def write_random_checkpoints(directory, seed=0):
    """Untrained models of the real size, saved the way train.py saves its best checkpoint."""
    torch.manual_seed(seed)
    paths = []
    for name in MODELS:
        path = directory / f"{name}_best.pt"
        torch.save(
            {
                "model": name,
                "state_dict": build_model(name).state_dict(),
                "epoch": 1,
                "val_loss": 0.5,
                "args": {},
            },
            path,
        )
        paths.append(str(path))
    return paths


def export_random_demo(work_dir, out_dir, num_episodes=3):
    """Export random-weight models and synthetic test episodes into out_dir."""
    episodes = [make_episode(i, n) for i, n in enumerate(EPISODE_ACTIONS)]
    for episode in episodes:
        save_episode(work_dir / "data", "test", episode)
    checkpoints = write_random_checkpoints(work_dir)
    manifest = export_demo.export_demo(
        checkpoints, work_dir / "data", out_dir, num_episodes=num_episodes
    )
    return manifest, checkpoints, episodes


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("demo")
    out_dir = work_dir / "out"
    manifest, checkpoints, episodes = export_random_demo(work_dir, out_dir)
    return {
        "out": out_dir,
        "manifest": manifest,
        "models": {name: load_model(path, "cpu")[0] for name, path in zip(MODELS, checkpoints)},
        "episodes": {ep.episode_id: ep for ep in episodes},
    }


def exported_episodes(exported):
    return [exported["episodes"][entry["id"]] for entry in exported["manifest"]["episodes"]]


def test_export_writes_every_file_the_manifest_names(exported):
    out, manifest = exported["out"], exported["manifest"]
    assert json.loads((out / "manifest.json").read_text()) == manifest
    for entry in manifest["models"] + manifest["episodes"]:
        assert (out / entry.get("file", entry.get("sheet"))).stat().st_size > 0
    assert (out / manifest["clip"]).stat().st_size > 0
    assert [m["name"] for m in manifest["models"]] == list(MODELS)
    assert len(manifest["episodes"]) == 3


@pytest.mark.parametrize("name", MODELS)
def test_onnx_inputs_and_outputs(exported, name):
    session = open_session(exported["out"] / "models" / f"{name}.onnx")
    inputs = [(i.name, i.type, i.shape) for i in session.get_inputs()]
    outputs = [(o.name, o.type, o.shape) for o in session.get_outputs()]
    frame = [1, 3, FRAME_SIZE, FRAME_SIZE]
    hidden = [1, 96, 16, 16]
    if name == "cnn":
        assert inputs == [("frame", "tensor(float)", frame), ("action", "tensor(int64)", [1])]
        assert outputs == [("next_frame", "tensor(float)", frame)]
    else:
        assert inputs == [
            ("frame", "tensor(float)", frame),
            ("action", "tensor(int64)", [1]),
            ("hidden", "tensor(float)", hidden),
        ]
        assert outputs == [
            ("next_frame", "tensor(float)", frame),
            ("hidden_out", "tensor(float)", hidden),
        ]
    entry = next(m for m in exported["manifest"]["models"] if m["name"] == name)
    assert entry["hidden_shape"] == (None if name == "cnn" else hidden)
    assert entry["context_frames"] == exported["models"][name].context_frames


@pytest.mark.parametrize("name", MODELS)
def test_onnx_rollout_matches_model_rollout(exported, name):
    model = exported["models"][name]
    session = open_session(exported["out"] / "models" / f"{name}.onnx")
    rng = np.random.default_rng(0)
    context = rng.random((4, 3, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    actions = rng.integers(NUM_ACTIONS, size=3 + 10)
    with torch.no_grad():
        expected = rollout(model, torch.from_numpy(context)[None], torch.from_numpy(actions)[None])
    got = onnx_rollout(session, context, actions, model.context_frames)
    assert got.shape == (10, 3, FRAME_SIZE, FRAME_SIZE)
    assert got.min() >= 0.0 and got.max() <= 1.0
    assert np.abs(got - expected[0].clamp(0.0, 1.0).numpy()).max() < 1e-5
    # The rollout depends on the actions, so this compares more than a constant picture.
    other = onnx_rollout(session, context, (actions + 1) % NUM_ACTIONS, model.context_frames)
    assert np.abs(got - other).max() > 1e-3


def test_all_zero_hidden_is_the_same_as_no_hidden(exported):
    model = exported["models"]["convgru"]
    session = open_session(exported["out"] / "models" / "convgru.onnx")
    rng = np.random.default_rng(1)
    frame = rng.random((1, 3, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    action = np.array([2], dtype=np.int64)
    with torch.no_grad():
        expected_frame, expected_hidden = model(
            torch.from_numpy(frame), torch.from_numpy(action), None
        )
    got_frame, got_hidden = session.run(
        None,
        {"frame": frame, "action": action, "hidden": np.zeros((1, 96, 16, 16), np.float32)},
    )
    assert np.abs(got_frame - expected_frame.clamp(0.0, 1.0).numpy()).max() < 1e-5
    assert np.abs(got_hidden - expected_hidden.numpy()).max() < 1e-5
    assert np.abs(got_hidden).max() > 1e-3


def test_episode_sheets_decode_to_the_stored_frames(exported):
    manifest = exported["manifest"]
    for entry in manifest["episodes"]:
        episode = exported["episodes"][entry["id"]]
        frames = read_sheet(exported["out"] / entry["sheet"], entry["frames"])
        assert frames.dtype == np.uint8
        assert np.array_equal(frames, episode.frames)
        assert entry["actions"] == episode.actions.tolist()
        assert entry["frames"] == len(entry["actions"]) + 1


def test_start_frames_leave_room_to_play(exported):
    for entry in exported["manifest"]["episodes"]:
        assert 1 <= len(entry["starts"]) <= 4
        for start in entry["starts"]:
            assert 3 <= start <= entry["frames"] - 1 - 30
    # A flat stretch is never preferred over a stretch with a moving, structured view.
    frames = make_episode(0, 80).frames.copy()
    frames[:40] = 128
    starts, scores = pick_starts(frames)
    assert min(starts) >= 40 and scores == sorted(scores, reverse=True)


def test_fixture_is_evaluate_on_the_rollouts_the_page_plays(exported):
    fixture = exported["manifest"]["fixture"]
    episodes = exported_episodes(exported)
    assert fixture["interval"] == 10 and fixture["first_frame"] == 3
    assert fixture["rollouts"] == sum((len(ep) - 3) // 10 for ep in episodes)
    assert fixture["repeat_last_frame"] == evaluate_repeat_last_frame(episodes, stride=10)["steps"]
    for name in MODELS:
        result = evaluate(exported["models"][name], episodes, stride=10)
        assert result["rollouts"] == fixture["rollouts"]
        assert fixture[name] == result["steps"]


@pytest.mark.parametrize("name", MODELS)
def test_fixture_matches_playing_the_onnx_files_like_the_page(exported, name):
    """Walk each episode as the page does and compare the mean error with the fixture."""
    fixture = exported["manifest"]["fixture"]
    model = exported["models"][name]
    session = open_session(exported["out"] / "models" / f"{name}.onnx")
    errors, reference = [], []
    for episode in exported_episodes(exported):
        real = episode.frames.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        t = fixture["first_frame"]
        while t + fixture["interval"] <= len(episode):
            predicted = onnx_rollout(
                session,
                real[t - 3 : t + 1],
                episode.actions[t - 3 : t + fixture["interval"]],
                model.context_frames,
            )
            target = real[t + 1 : t + 1 + fixture["interval"]]
            errors.append(np.abs(predicted - target).mean(axis=(1, 2, 3)))
            reference.append(np.abs(real[t][None] - target).mean(axis=(1, 2, 3)))
            t += fixture["interval"]
    assert len(errors) == fixture["rollouts"]
    errors, reference = np.mean(errors, axis=0), np.mean(reference, axis=0)
    for step in (1, 5, 10):
        assert errors[step - 1] == pytest.approx(fixture[name][str(step)]["mae"], abs=1e-6)
        assert reference[step - 1] == pytest.approx(
            fixture["repeat_last_frame"][str(step)]["mae"], abs=1e-6
        )


def test_metrics_are_evaluate_on_the_whole_split(exported):
    metrics = exported["manifest"]["metrics"]
    episodes = list(exported["episodes"].values())
    assert metrics["split"] == "test" and metrics["episodes"] == len(episodes)
    assert metrics["repeat_last_frame"] == evaluate_repeat_last_frame(episodes)["steps"]
    for name in MODELS:
        result = evaluate(exported["models"][name], episodes)
        assert metrics["rollouts"] == result["rollouts"]
        assert metrics[name] == result["steps"]


def test_export_fails_when_onnx_and_pytorch_disagree(tmp_path, monkeypatch):
    # No two float implementations agree to better than this, so the check must trip.
    monkeypatch.setattr(export_demo, "PARITY_TOLERANCE", -1.0)
    with pytest.raises(RuntimeError, match="differs from model.rollout"):
        export_random_demo(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_export_onnx_reports_the_hidden_shape(tmp_path):
    assert export_onnx(build_model("cnn"), tmp_path / "cnn.onnx") is None
    assert export_onnx(build_model("convgru"), tmp_path / "convgru.onnx") == [1, 96, 16, 16]
