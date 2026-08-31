"""CIFAR ResNet / plain network -- ``6n + 2`` weighted layers (paper Section 4.2).

Stem 3x3 conv (16 ch) -> three stages of ``n`` basic blocks at 16 / 32 / 64 channels and
32 / 16 / 8 spatial size -> global average pool -> linear classifier.

``CifarResNet(depth, residual=False)`` is the parameter-matched plain network (shortcut option A
adds no parameters, so plain-k and resnet-k have identical counts -- see
docs/project_specification.md section 4.4 for the expected values asserted in tests).

Implemented and tested in Phase 8.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

# Expected total parameter counts, shortcut option A. Asserted in tests/test_resnet.py.
EXPECTED_PARAM_COUNTS = {
    20: 269_722,
    32: 464_154,
    44: 658_586,
    56: 853_018,
    110: 1_727_962,
}


def depth_to_n(depth: int) -> int:
    """``n = (depth - 2) / 6``; raises if ``depth`` is not a valid ``6n + 2``."""
    raise NotImplementedError("depth_to_n: Phase 8")


class CifarResNet(nn.Module):
    """The CIFAR residual/plain network.

    Args:
        depth: total weighted layers, must be ``6n + 2``.
        num_classes: classifier outputs (10 for CIFAR-10).
        shortcut: shortcut option "A" / "B" / "C" (ignored when ``residual`` is False except that
            option affects the plain net's downsampling path too -- documented in Phase 8).
        use_bn: include BatchNorm.
        residual: add shortcuts (True = ResNet, False = plain).
        zero_init_residual: see :class:`~resnet_research.models.blocks.BasicBlock`.
    """

    def __init__(
        self,
        depth: int,
        num_classes: int = 10,
        *,
        shortcut: str = "A",
        use_bn: bool = True,
        residual: bool = True,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        raise NotImplementedError("CifarResNet: Phase 8")

    def _make_stage(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        raise NotImplementedError("CifarResNet._make_stage: Phase 8")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("CifarResNet.forward: Phase 8")


def cifar_resnet(depth: int, **kwargs) -> CifarResNet:
    """Residual CIFAR network of the given depth."""
    raise NotImplementedError("cifar_resnet: Phase 8")


def cifar_plain(depth: int, **kwargs) -> CifarResNet:
    """Plain CIFAR network of the given depth (``residual=False``)."""
    raise NotImplementedError("cifar_plain: Phase 8")
