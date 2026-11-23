"""World models: the action-conditioned CNN baseline and the ConvGRU memory model.

    python model.py        # print the parameter count of each model

Both models share one encoder and decoder and one interface:

    next_frame, hidden = model(frame, action, hidden)

`hidden` is the memory carried between frames. The CNN has none, so its hidden is always None.
`rollout` below runs either model and is what train.py and evaluate.py both use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import FRAME_SIZE, NUM_ACTIONS


def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


def double_conv(in_channels, out_channels):
    return nn.Sequential(conv3x3(in_channels, out_channels), conv3x3(out_channels, out_channels))


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class WorldModel(nn.Module):
    """U-Net style encoder and decoder shared by both models.

    Resolution   Encoder                    Decoder
    64 x 64      double_conv 3 -> 32   -->  double_conv (64 + 32) -> 32, conv 32 -> 3
    32 x 32      double_conv 32 -> 64  -->  double_conv (96 + 64) -> 64
    16 x 16      conv 64 -> 96, + action embedding (96), then the subclass's `core` (96 -> 96)

    Downsampling is 2x2 max pooling, upsampling is bilinear, `-->` are skip connections.
    The decoder output is a change that is added to the input frame.
    """

    # Real frames the model needs to see before it predicts.
    context_frames = 1

    def __init__(self, num_actions=NUM_ACTIONS, channels=(32, 64, 96)):
        super().__init__()
        c1, c2, c3 = channels
        self.enc1 = double_conv(3, c1)
        self.enc2 = double_conv(c1, c2)
        self.enc3 = conv3x3(c2, c3)
        self.action_embedding = nn.Embedding(num_actions, c3)
        self.dec2 = double_conv(c3 + c2, c2)
        self.dec1 = double_conv(c2 + c1, c1)
        self.out = nn.Conv2d(c1, 3, kernel_size=3, padding=1)

    def core(self, features, hidden):
        """Bottleneck step: (encoder features + action, hidden) -> (decoder input, new hidden)."""
        raise NotImplementedError

    def encode(self, frame, action, hidden):
        e1 = self.enc1(frame)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        z, hidden = self.core(e3 + self.action_embedding(action)[:, :, None, None], hidden)
        return z, hidden, (e1, e2)

    def decode(self, frame, z, skips):
        e1, e2 = skips
        d2 = self.dec2(torch.cat([upsample(z), e2], dim=1))
        d1 = self.dec1(torch.cat([upsample(d2), e1], dim=1))
        return frame + self.out(d1)

    def forward(self, frame, action, hidden=None):
        """frame: float (B, 3, H, W) in [0, 1]. action: long (B,). Returns (next frame, hidden)."""
        z, hidden, skips = self.encode(frame, action, hidden)
        return self.decode(frame, z, skips), hidden

    def observe(self, frame, action, hidden=None):
        """Update the hidden state from a frame and action without predicting a frame."""
        return self.encode(frame, action, hidden)[1]


class ActionConditionedUNet(WorldModel):
    """Model 1: the CNN baseline. Sees the current frame and action only. 371,235 parameters."""

    def __init__(self, num_actions=NUM_ACTIONS, channels=(32, 64, 96)):
        super().__init__(num_actions, channels)
        self.fuse = conv3x3(channels[-1], channels[-1])

    def core(self, features, hidden):
        return self.fuse(features), None


class ConvGRUCell(nn.Module):
    """GRU whose gates are 3x3 convolutions, so the hidden state is a feature map."""

    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1
        )
        self.candidate = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1
        )

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0], self.hidden_channels, *x.shape[2:])
        reset, update = torch.sigmoid(self.gates(torch.cat([x, hidden], dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * hidden], dim=1)))
        return (1 - update) * hidden + update * candidate


class ConvGRUWorldModel(WorldModel):
    """Model 2: the same encoder and decoder with a ConvGRU between them.

    The ConvGRU replaces the baseline's `fuse` convolution. Its hidden state (96 x 16 x 16) is
    carried from frame to frame and is what the decoder reads. 786,147 parameters.
    """

    context_frames = 4

    def __init__(self, num_actions=NUM_ACTIONS, channels=(32, 64, 96)):
        super().__init__(num_actions, channels)
        self.gru = ConvGRUCell(channels[-1], channels[-1])

    def core(self, features, hidden):
        hidden = self.gru(features, hidden)
        return hidden, hidden


def rollout(model, context, actions):
    """Observe real frames, then predict future frames from the model's own predictions.

    context: float (B, C, 3, H, W) in [0, 1], C >= model.context_frames consecutive real frames.
    actions: long (B, C - 1 + K), the action taken at each context frame, then K - 1 more.
    Returns the K predicted frames, float (B, K, 3, H, W).

    The model observes its last `context_frames` context frames with their actions (earlier
    ones are ignored, so the CNN starts from the last real frame alone). Each prediction is
    clamped to [0, 1] before it is fed back in as the next input. The returned frames are not
    clamped so that the training loss has a gradient everywhere.
    """
    skip = context.shape[1] - model.context_frames
    context, actions = context[:, skip:], actions[:, skip:]
    observed = context.shape[1] - 1

    hidden = None
    for t in range(observed):
        hidden = model.observe(context[:, t], actions[:, t], hidden)

    frame = context[:, -1]
    predictions = []
    for k in range(observed, actions.shape[1]):
        prediction, hidden = model(frame, actions[:, k], hidden)
        predictions.append(prediction)
        frame = prediction.clamp(0.0, 1.0)
    return torch.stack(predictions, dim=1)


MODELS = {"cnn": ActionConditionedUNet, "convgru": ConvGRUWorldModel}


def build_model(name, **kwargs):
    return MODELS[name](**kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in MODELS:
        model = build_model(name)
        context = torch.rand(2, model.context_frames, 3, FRAME_SIZE, FRAME_SIZE)
        actions = torch.randint(NUM_ACTIONS, (2, model.context_frames))
        shape = tuple(rollout(model, context, actions).shape)
        print(
            f"{name}: {count_parameters(model):,} trainable parameters, "
            f"observes {model.context_frames} frame(s), rollout output {shape}"
        )
