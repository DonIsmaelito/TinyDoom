# TinyDoom

A small world model that predicts Doom gameplay frames from player actions. It collects
gameplay from ViZDoom, trains a model to predict the next frame from the current frame and
action, and measures how well the model predicts 1, 5, and 10 frames ahead when it only sees
its own predictions.

There are two models, trained on the same data with the same loss:

- `cnn`: an action-conditioned CNN that sees only the current frame.
- `convgru`: the same network with a ConvGRU memory, which first observes four frames.

## Setup

Tested with Python 3.12 on Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Linux this installs the CUDA build of PyTorch. On a machine without a GPU you can install
the smaller CPU build instead:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

ViZDoom runs without a display, so no window or X server is needed.

## Run the pipeline

```bash
# 1. Collect 60 episodes into data/train (42), data/val (8), data/test (10)
python collect.py

# 2. Train the CNN baseline, then evaluate it on the test episodes
python train.py --model cnn
python evaluate.py --checkpoint checkpoints/cnn_best.pt

# 3. Train the ConvGRU model, then evaluate it
python train.py --model convgru
python evaluate.py --checkpoint checkpoints/convgru_best.pt

# 4. Compare both on the same rollouts
python evaluate.py --checkpoint checkpoints/cnn_best.pt checkpoints/convgru_best.pt
```

Every script has `--help`. Training and evaluation use a GPU when PyTorch finds one and the
CPU otherwise. Pass `--device cpu`, `--device cuda`, or `--device mps` to choose.

Run the tests with `pytest`. They use small synthetic episodes and take a few seconds. One
test collects real episodes and is skipped if ViZDoom is not installed.

## Data collection (`collect.py`)

| Setting | Value |
| --- | --- |
| Scenario | ViZDoom My Way Home |
| Actions | 0 forward, 1 turn left, 2 turn right |
| Frame skip | 4 game ticks per action |
| Frames | rendered at 320 x 240, resized to 64 x 64 RGB |
| Episodes | 60 by default (`--episodes`), up to 100 actions each (`--max-actions`) |
| Split | whole episodes, 42 train / 8 val / 10 test |

The policy mixes random actions with simple movement sequences. It repeatedly picks one of
three segments: a burst of 2 to 6 random actions (40% of segments), a straight run forward
for 3 to 10 actions (35%), or 2 to 6 turns in one direction (25%). If a forward step is
blocked by a wall, the rest of the segment is replaced by a turn.

An episode ends after 100 actions, or earlier if the game ends it. Episodes are numbered in
collection order: the first 42 are train, the next 8 are val, the last 10 are test. With a
different `--episodes` the same 42:8:10 ratio is used. `--seed` makes a run repeatable.

Each episode is one file, `data/<split>/episode_<id>.npz`:

| Array | Shape | Meaning |
| --- | --- | --- |
| `frames` | `(T + 1, 64, 64, 3)` uint8 | `frames[t + 1]` is the result of `actions[t]` |
| `actions` | `(T,)` int64 | action index |
| `positions` | `(T + 1, 3)` float32 | player x, y, z at each frame |
| `angles` | `(T + 1,)` float32 | viewing direction in degrees at each frame |
| `episode_id` | scalar | episode number |

Position and angle are stored but not given to the models. `data/metadata.json` records the
settings and the number of episodes and transitions per split.

## Models (`model.py`)

Both models share one U-Net style encoder and decoder and predict a change that is added to
the input frame (next frame = current frame + predicted change).

```
64 x 64   conv 3->32, conv 32->32   ------skip------>   conv 96->32, conv 32->32, conv 32->3
             | max pool                                    ^ upsample
32 x 32   conv 32->64, conv 64->64  ------skip------>   conv 160->64, conv 64->64
             | max pool                                    ^ upsample
16 x 16   conv 64->96  ->  + action embedding (3 x 96)  ->  core (96 -> 96)
```

All convolutions are 3 x 3 with ReLU, except the last one, which has no activation. There is
no normalization and no pretrained weights.

| Model | Core | Sees before predicting | Trainable parameters |
| --- | --- | --- | --- |
| `cnn` | one 3 x 3 convolution | the current frame and action | 371,235 |
| `convgru` | ConvGRU cell, 96 hidden channels, 3 x 3 gates | four frames with their actions | 786,147 |

The ConvGRU keeps a hidden state (96 x 16 x 16) across frames. It observes four real frames
with their actions, then predicts future frames while updating the hidden state from the
actions and its own generated frames. `python model.py` prints both parameter counts.

## Training (`train.py`)

- Loss: L1 between predicted and actual frames. Optimizer: Adam, learning rate 0.0003,
  batch size 32.
- Each training example is one next-frame prediction. The CNN gets the current frame and
  action. The ConvGRU gets the current frame and the three before it, with their actions.
- After every epoch the loss on the validation episodes is measured. The checkpoint is
  written to `checkpoints/<model>_best.pt` only when that loss is the lowest so far.
- `--epochs` sets the training length (default 50).
- `--rollout-steps N` (default 1) makes each example predict N frames in a row from the
  model's own predictions instead of one. Use the same value for both models when comparing.

## Evaluation (`evaluate.py`)

Each rollout starts at a real frame and predicts 10 frames from the recorded actions alone,
feeding the model's predictions back in as input. The script reports the mean MAE (pixel
values in 0 to 1) and PSNR (dB) of the predicted frame at steps 1, 5, and 10.

Both models are scored on exactly the same rollouts. A rollout needs three real frames before
its start frame so that the ConvGRU can observe four, and the CNN starts from the last of
those four. Every start frame in the split that leaves room for 10 steps is used. Predictions
are clamped to the range 0 to 1.

After the models, the script prints the same metrics for a reference that needs no model: it
repeats the last real frame for all 10 steps. A model is only useful where it beats this. With
`--out`, each model's entry also carries these numbers under `repeat_last_frame`.

## Files

| File | Purpose |
| --- | --- |
| `collect.py` | runs ViZDoom and saves episodes into train, val, and test sets |
| `data.py` | episode file format, the split rule, and the dataset used for training and evaluation |
| `model.py` | both models and the shared rollout function |
| `train.py` | training loop, saves the best checkpoint by validation loss |
| `evaluate.py` | rollout evaluation, MAE and PSNR at 1, 5, and 10 steps |
| `test_pipeline.py` | pytest checks |
