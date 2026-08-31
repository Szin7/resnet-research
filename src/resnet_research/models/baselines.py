"""Conventional CNN baselines -- a reference point, not a parameter-matched control.

``SmallCNN`` is a textbook conv-pool-FC design (three conv blocks with max pooling, then two fully
connected layers with dropout). ``SmallCNNDeep`` stretches it to ~12 conv layers. Both deliberately
keep the pre-ResNet design choices (max pooling, FC head, dropout) so the contrast with the ResNet
family is visible.

Implemented and tested in Phase 6.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class SmallCNN(nn.Module):
    """Three conv blocks (32 / 64 / 128 ch, each: conv-bn-relu-conv-bn-relu-maxpool),
    then Linear(128*4*4, 256) -> ReLU -> Dropout(0.5) -> Linear(256, num_classes)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        raise NotImplementedError("SmallCNN: Phase 6")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("SmallCNN.forward: Phase 6")


class SmallCNNDeep(nn.Module):
    """As :class:`SmallCNN` but with four conv layers per block (~12 conv layers total)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        raise NotImplementedError("SmallCNNDeep: Phase 6")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("SmallCNNDeep.forward: Phase 6")
