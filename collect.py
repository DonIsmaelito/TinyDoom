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

    metadata = {
        "scenario": "my_way_home",
        "actions": list(ACTION_NAMES),
        "frame_skip": frame_skip,
        "frame_size": FRAME_SIZE,
        "max_actions": max_actions,
        "seed": seed,
        "vizdoom_version": vzd.__version__,
        "episodes": {split: split_of.count(split) for split in SPLITS},
        "transitions": transitions,
    }
    (data_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--max-actions", type=int, default=100, help="actions per episode")
    parser.add_argument("--frame-skip", type=int, default=4, help="game ticks per action")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="replace existing episodes")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
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
