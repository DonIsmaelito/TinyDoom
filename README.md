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

Run the tests with `pytest`. They use small synthetic episodes. The pipeline tests take a few
seconds and the demo tests about a minute. One test collects real episodes and is skipped if
ViZDoom is not installed. The demo page tests are skipped if Playwright is not installed.

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

## Demo page (`export_demo.py`, `demo/`)

The demo is a web page that plays test episodes and shows three pictures side by side: the
real game, the CNN's prediction, and the ConvGRU's prediction. Both models run in the browser.
Nothing is sent to a server.

```bash
# 1. Export both trained models and a few test episodes into demo/
python export_demo.py

# 2. Serve the page on this computer and open http://localhost:8000
python -m http.server --directory demo 8000
```

The page has to be served: opening `demo/index.html` as a file does not work, because the
browser will not let a page loaded from a file read the models. Once the page is loaded it
needs no network at all.

`export_demo.py` writes these into `demo/`:

| File | Contents |
| --- | --- |
| `models/cnn.onnx`, `models/convgru.onnx` | one prediction step of each model, output clamped to 0 to 1 |
| `episodes/episode_<id>.png` | all frames of a test episode in one lossless image |
| `manifest.json` | the list of models and episodes, the recorded actions, good start frames, and the `evaluate.py` numbers the page is checked against |
| `clip.webp` | a recording of the same comparison, shown while the models load and if they cannot run |

The export stops with an error if a 10-step rollout through an ONNX file differs from
`model.rollout()` by more than 0.00001 on any pixel value. By default it picks the 4 test
episodes with the most to look at (`--episodes`, or name them with `--episode-ids`). Many
frames in this map are one flat wall, so the export also picks start frames where the view has
large areas of different color and keeps changing.

On the page:

- **Play, Pause, Step, Reset.** Space, S, and R do the same. Reset goes back to the start
  frame of the chosen episode and clears the chart.
- **Speed**: 1, 2, 4, or 8.75 steps a second. 8.75 is the speed of the game.
- **Show the models a real frame**: every 1, 5, 10, or 30 steps, or never again. Between real
  frames each model predicts from its own last prediction. When the ConvGRU is shown a real
  frame, its memory is emptied and it observes that frame and the three before it.
- **Episode**: the exported episodes and their start frames.
- **The chart** plots each model's mean absolute error against the number of steps since the
  models last saw a real frame, measured as the page plays. The dashed line is the reference
  that repeats the last real frame.
- A paused page plays again by itself after 20 seconds without input. After two minutes it
  also returns to the default speed and real-frame setting.

The address can change the defaults, for example
`http://localhost:8000/?interval=5&speed=8.75&idle=60`. `idle=0` turns off playing by itself,
and `autoplay=0` starts the page paused.

The page runs the models with ONNX Runtime Web on one WebAssembly thread. Its files are stored
in `demo/vendor/` (about 15 MB), so the page loads nothing from other sites.

### Demo tests

`pytest` also runs `test_demo.py` and `test_demo_page.py`. Both use untrained models of the
real size and synthetic episodes.

- `test_demo.py` checks the export: the ONNX files against `model.rollout()`, the input and
  output names, shapes, and types, that the episode images decode to the stored frames, and
  that the numbers in `manifest.json` are what `evaluate.py` computes.
- `test_demo_page.py` opens the page in headless Chrome. It needs `pip install playwright`
  and Chrome (or `playwright install chromium`), and is skipped without them. It checks that
  the page's error at 1, 5, and 10 steps equals the exported `evaluate.py` numbers, that
  Reset gives the same pictures again, that no request leaves `localhost`, that both models
  plus drawing take less than the 114 ms the game takes per step, and that the clip appears
  when the models cannot load.

## Files

| File | Purpose |
| --- | --- |
| `collect.py` | runs ViZDoom and saves episodes into train, val, and test sets |
| `data.py` | episode file format, the split rule, and the dataset used for training and evaluation |
| `model.py` | both models and the shared rollout function |
| `train.py` | training loop, saves the best checkpoint by validation loss |
| `evaluate.py` | rollout evaluation, MAE and PSNR at 1, 5, and 10 steps |
| `test_pipeline.py` | pytest checks of the pipeline |
| `export_demo.py` | exports the trained models, test episodes, and check numbers for the demo page |
| `demo/` | the demo page: `index.html`, `app.js`, `style.css`, and ONNX Runtime Web in `vendor/` |
| `test_demo.py` | pytest checks of the export |
| `test_demo_page.py` | pytest checks of the page in headless Chrome |
