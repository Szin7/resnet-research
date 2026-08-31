"""Conventional CNN baselines -- a reference point, not a parameter-matched control.

``SmallCNN`` is a textbook conv-pool-FC design: three blocks of (conv-BN-ReLU x2 -> max pool) with
32 / 64 / 128 channels, then a two-layer classifier with dropout. ``SmallCNNDeep`` is the same
skeleton with four conv layers per block (~12 conv layers). Both deliberately keep the pre-ResNet
choices (max pooling, an FC head, dropout) so the contrast with the ResNet family is visible.

The scientifically controlled comparison in this project is plain-k vs resnet-k, not these.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from resnet_research.models.weight_init import initialize_weights

_STAGE_CHANNELS = (32, 64, 128)


def _conv_block(in_channels: int, out_channels: int, num_convs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    channels = in_channels
    for _ in range(num_convs):
        layers += [
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        channels = out_channels
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SmallCNN(nn.Module):
    """Six conv layers (2 per stage) + a 2-layer classifier. ~0.81M parameters."""

    convs_per_stage = 2

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        in_channels = 3
        stages = []
        for out_channels in _STAGE_CHANNELS:
            stages.append(_conv_block(in_channels, out_channels, self.convs_per_stage))
            in_channels = out_channels
        self.features = nn.Sequential(*stages)

        # three 2x max pools take 32x32 -> 4x4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_STAGE_CHANNELS[-1] * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        initialize_weights(self)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


class SmallCNNDeep(SmallCNN):
    """Same design as :class:`SmallCNN` with four conv layers per stage (~12 conv layers)."""

    convs_per_stage = 4
