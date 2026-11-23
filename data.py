"""Episode file format and datasets shared by collect.py, train.py, and evaluate.py.

One episode is one `.npz` file:

    frames      uint8   (T + 1, 64, 64, 3)  RGB, frames[t + 1] is the result of actions[t]
    actions     int64   (T,)                index into ACTION_NAMES
    positions   float32 (T + 1, 3)          player x, y, z for each frame
    angles      float32 (T + 1,)            player viewing direction in degrees for each frame
    episode_id  int64   ()

Episodes live in `<data_dir>/<split>/episode_<id>.npz`, so a split is a set of whole episodes.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ACTION_NAMES = ("forward", "turn_left", "turn_right")
NUM_ACTIONS = len(ACTION_NAMES)
FRAME_SIZE = 64
SPLITS = ("train", "val", "test")
SPLIT_RATIO = (42, 8, 10)


@dataclass
class Episode:
    episode_id: int
    frames: np.ndarray
    actions: np.ndarray
    positions: np.ndarray
    angles: np.ndarray

    def __len__(self):
        return len(self.actions)


def split_counts(num_episodes):
    """Episodes per split in the ratio 42:8:10 (exactly 42/8/10 for 60 episodes)."""
    total = sum(SPLIT_RATIO)
    exact = [num_episodes * r / total for r in SPLIT_RATIO]
    counts = [int(x) for x in exact]
    by_remainder = sorted(range(len(exact)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in by_remainder[: num_episodes - sum(counts)]:
        counts[i] += 1
    return counts


def assign_splits(num_episodes):
    """Split name for each episode id: the first block is train, then val, then test."""
    return [split for split, n in zip(SPLITS, split_counts(num_episodes)) for _ in range(n)]


def save_episode(data_dir, split, episode):
    path = Path(data_dir) / split / f"episode_{episode.episode_id:04d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=episode.frames,
        actions=episode.actions,
        positions=episode.positions,
        angles=episode.angles,
        episode_id=np.int64(episode.episode_id),
    )
    return path


def load_episodes(data_dir, split):
    paths = sorted((Path(data_dir) / split).glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No episodes found in {Path(data_dir) / split}. Run collect.py first."
        )
    episodes = []
    for path in paths:
        with np.load(path) as f:
            episodes.append(
                Episode(
                    episode_id=int(f["episode_id"]),
                    frames=f["frames"],
                    actions=f["actions"],
                    positions=f["positions"],
                    angles=f["angles"],
                )
            )
    return episodes


class WindowDataset(Dataset):
    """Windows cut from episodes: observed frames, actions, and the frames to predict.

    Item i is a window around some frame t of one episode:

        context  uint8 (C, 3, 64, 64)   frames t - C + 1 .. t, the last one is the current frame
        actions  int64 (C - 1 + K,)     the action taken at each context frame, then K - 1 more
        targets  uint8 (K, 3, 64, 64)   frames t + 1 .. t + K

    With C = 1 and K = 1 this is the plain (current frame, action, next frame) example.
    Windows never cross an episode boundary. Episodes that are too short give no windows.
    """

    def __init__(self, episodes, context_frames=1, rollout_steps=1):
        self.context_frames = context_frames
        self.rollout_steps = rollout_steps
        frames = np.concatenate([ep.frames for ep in episodes])
        self.frames = torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 3, 1, 2)))

        # Actions are padded to one per frame so that frame and action indices line up.
        actions, current = [], []
        offset = 0
        for ep in episodes:
            actions.append(np.append(ep.actions, 0))
            t = np.arange(context_frames - 1, len(ep) - rollout_steps + 1)
            current.append(offset + t)
            offset += len(ep) + 1
        self.actions = torch.from_numpy(np.concatenate(actions))
        self.current = torch.from_numpy(np.concatenate(current))

    def __len__(self):
        return len(self.current)

    def __getitem__(self, i):
        t = int(self.current[i])
        first = t - self.context_frames + 1
        last = t + self.rollout_steps
        return self.frames[first : t + 1], self.actions[first:last], self.frames[t + 1 : last + 1]


def to_float(frames):
    """uint8 frames -> float32 in [0, 1]."""
    return frames.float() / 255.0
