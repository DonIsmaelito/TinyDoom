"""Collect ViZDoom My Way Home episodes and save them as train, val, and test sets.

    python collect.py                      # 60 episodes -> data/{train,val,test}, split 42/8/10
    python collect.py --episodes 120       # more data, same 42:8:10 ratio
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import vizdoom as vzd
from PIL import Image

from data import (
    ACTION_NAMES,
    FRAME_SIZE,
    NUM_ACTIONS,
    SPLITS,
    Episode,
    assign_splits,
    save_episode,
)

FORWARD, TURN_LEFT, TURN_RIGHT = range(NUM_ACTIONS)
BUTTONS = (vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT)
POSE_VARIABLES = (
    vzd.GameVariable.POSITION_X,
    vzd.GameVariable.POSITION_Y,
    vzd.GameVariable.POSITION_Z,
    vzd.GameVariable.ANGLE,
)

# The policy picks one segment at a time. Lengths are in actions (inclusive ranges).
SEGMENT_PROBS = {"random": 0.40, "forward": 0.35, "turn": 0.25}
SEGMENT_LENGTHS = {"random": (2, 6), "forward": (3, 10), "turn": (2, 6)}

# A forward step that moves the player less than this many map units counts as blocked.
# An unobstructed forward step covers about 7 units or more.
BLOCKED_DISTANCE = 1.0


class MixedPolicy:
    """Random actions mixed with simple movement sequences.

    Each segment is either a burst of uniformly random actions, a straight run forward,
    or a repeated turn in one direction. When a forward step is blocked by a wall, the rest
    of the segment is replaced by a turn so the player does not keep walking into the wall.
    """

    def __init__(self, rng):
        self.rng = rng
        self.queue = []

    def reset(self):
        self.queue = []

    def next_action(self):
        if not self.queue:
            kind = self.rng.choice(list(SEGMENT_PROBS), p=list(SEGMENT_PROBS.values()))
            self.queue = self._segment(kind)
        return self.queue.pop()

    def blocked(self):
        self.queue = self._segment("turn")

    def _segment(self, kind):
        low, high = SEGMENT_LENGTHS[kind]
        length = int(self.rng.integers(low, high + 1))
        if kind == "random":
            return [int(a) for a in self.rng.integers(NUM_ACTIONS, size=length)]
        if kind == "forward":
            return [FORWARD] * length
        return [int(self.rng.choice([TURN_LEFT, TURN_RIGHT]))] * length


def make_game(seed):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "my_way_home.cfg"))
    game.set_available_buttons(list(BUTTONS))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_seed(seed)
    game.init()
    return game


def read_frame(game):
    screen = game.get_state().screen_buffer
    image = Image.fromarray(screen).resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.BOX)
    return np.asarray(image, dtype=np.uint8)


def read_pose(game):
    return [game.get_game_variable(v) for v in POSE_VARIABLES]


def collect_episode(game, policy, episode_id, max_actions, frame_skip):
    """Play one episode. Stops after max_actions or when the game ends the episode."""
    game.new_episode()
    policy.reset()
    frames = [read_frame(game)]
    poses = [read_pose(game)]
    actions = []
    for _ in range(max_actions):
        action = policy.next_action()
        buttons = [0] * NUM_ACTIONS
        buttons[action] = 1
        game.make_action(buttons, frame_skip)
        if game.is_episode_finished():
            # No frame exists after the terminal action, so that action is not recorded.
            break
        actions.append(action)
        frames.append(read_frame(game))
        poses.append(read_pose(game))
        moved = np.hypot(poses[-1][0] - poses[-2][0], poses[-1][1] - poses[-2][1])
        if action == FORWARD and moved < BLOCKED_DISTANCE:
            policy.blocked()
    poses = np.asarray(poses, dtype=np.float32)
    return Episode(
        episode_id=episode_id,
        frames=np.stack(frames),
        actions=np.asarray(actions, dtype=np.int64),
        positions=poses[:, :3],
        angles=poses[:, 3],
    )


def episode_fingerprint(episode):
    """Hash actions and frames so two episodes can be compared for duplicates."""
    import hashlib

    h = hashlib.sha256()
    h.update(episode.actions.tobytes())
    h.update(episode.frames.tobytes())
    return h.hexdigest()


def load_fingerprints(data_dir):
    from data import load_episodes

    fingerprints = {}
    for split in SPLITS:
        if not (Path(data_dir) / split).exists():
            continue
        for episode in load_episodes(data_dir, split):
            fingerprints[episode_fingerprint(episode)] = (split, episode.episode_id)
    return fingerprints


def write_metadata(data_dir, seed, frame_skip, max_actions, episode_counts, transitions):
    data_dir = Path(data_dir)
    metadata = {
        "scenario": "my_way_home",
        "actions": list(ACTION_NAMES),
        "frame_skip": frame_skip,
        "frame_size": FRAME_SIZE,
        "max_actions": max_actions,
        "seed": seed,
        "vizdoom_version": vzd.__version__,
        "episodes": episode_counts,
        "transitions": transitions,
    }
    (data_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def count_on_disk(data_dir):
    """Episode and transition counts from saved files."""
    from data import load_episodes

    episodes = {split: 0 for split in SPLITS}
    transitions = {split: 0 for split in SPLITS}
    for split in SPLITS:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            continue
        for episode in load_episodes(data_dir, split):
            episodes[split] += 1
            transitions[split] += len(episode)
    return episodes, transitions


def collect(data_dir, num_episodes=60, max_actions=100, frame_skip=4, seed=0):
    """Collect num_episodes episodes into data_dir/{train,val,test} and write metadata.json."""
    data_dir = Path(data_dir)
    split_of = assign_splits(num_episodes)
    rng = np.random.default_rng(seed)
    policy = MixedPolicy(rng)
    game = make_game(seed)
    transitions = dict.fromkeys(SPLITS, 0)
    try:
        for episode_id, split in enumerate(split_of):
            episode = collect_episode(game, policy, episode_id, max_actions, frame_skip)
            while len(episode) == 0:
                episode = collect_episode(game, policy, episode_id, max_actions, frame_skip)
            save_episode(data_dir, split, episode)
            transitions[split] += len(episode)
            print(f"episode {episode_id:4d}  {split:5s}  {len(episode):3d} actions")
    finally:
        game.close()

    return write_metadata(
        data_dir,
        seed,
        frame_skip,
        max_actions,
        {split: split_of.count(split) for split in SPLITS},
        transitions,
    )


def collect_train_append(
    data_dir,
    num_train,
    start_episode_id=60,
    max_actions=100,
    frame_skip=4,
    seed=0,
):
    """Append num_train episodes to train/, continuing the policy from start_episode_id.

    Replays episodes 0 .. start_episode_id - 1 without saving so the RNG matches a full run,
    then saves new training episodes with ids start_episode_id .. start_episode_id + num_train - 1.
    Val and test on disk are left unchanged.
    """
    data_dir = Path(data_dir)
    known = load_fingerprints(data_dir)
    rng = np.random.default_rng(seed)
    policy = MixedPolicy(rng)
    game = make_game(seed)
    try:
        for episode_id in range(start_episode_id):
            episode = collect_episode(game, policy, episode_id, max_actions, frame_skip)
            while len(episode) == 0:
                episode = collect_episode(game, policy, episode_id, max_actions, frame_skip)

        for episode_id in range(start_episode_id, start_episode_id + num_train):
            episode = collect_episode(game, policy, episode_id, max_actions, frame_skip)
            while len(episode) == 0:
                episode = collect_episode(game, policy, episode_id, max_actions, frame_skip)
            fp = episode_fingerprint(episode)
            if fp in known:
                other = known[fp]
                raise RuntimeError(
                    f"episode {episode_id} duplicates existing episode {other[1]} in {other[0]}"
                )
            known[fp] = ("train", episode_id)
            save_episode(data_dir, "train", episode)
            print(f"episode {episode_id:4d}  train  {len(episode):3d} actions")
    finally:
        game.close()

    episode_counts, transitions = count_on_disk(data_dir)
    return write_metadata(data_dir, seed, frame_skip, max_actions, episode_counts, transitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--max-actions", type=int, default=100, help="actions per episode")
    parser.add_argument("--frame-skip", type=int, default=4, help="game ticks per action")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="replace existing episodes")
    parser.add_argument(
        "--train-only",
        type=int,
        metavar="N",
        help="append N training episodes to an existing dataset (val and test are untouched)",
    )
    parser.add_argument(
        "--start-episode-id",
        type=int,
        default=60,
        help="first episode id when appending train-only data (default 60)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.train_only is not None:
        if not (data_dir / "train").exists():
            raise SystemExit(f"{data_dir}/train does not exist. Collect or copy a base dataset first.")
        metadata = collect_train_append(
            data_dir,
            args.train_only,
            start_episode_id=args.start_episode_id,
            max_actions=args.max_actions,
            frame_skip=args.frame_skip,
            seed=args.seed,
        )
    else:
        existing = [s for s in SPLITS if any((data_dir / s).glob("episode_*.npz"))]
        if existing and not args.overwrite:
            raise SystemExit(f"{data_dir} already contains episodes. Pass --overwrite to replace them.")
        for split in existing:
            shutil.rmtree(data_dir / split)

        metadata = collect(data_dir, args.episodes, args.max_actions, args.frame_skip, args.seed)
    for split in SPLITS:
        print(
            f"{split:5s}  {metadata['episodes'][split]:3d} episodes  "
            f"{metadata['transitions'][split]:5d} transitions"
        )
    print(f"saved to {data_dir}/")


if __name__ == "__main__":
    main()
