"""Residual building blocks and shortcut connections.

The basic block follows the paper: two 3x3 conv layers, BatchNorm after each, ReLU after the first
conv and again *after* the residual addition. The ``residual`` flag turns the additive shortcut on
(ResNet) or off (plain net) while leaving every layer otherwise identical -- this is what makes the
plain-vs-residual comparison controlled.

Implemented and tested in Phase 7.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

# Shortcut options, matching docs/project_specification.md section 4.2:
#   "A" -- parameter-free: subsample by stride, zero-pad the extra channels
#   "B" -- 1x1 conv projection only where dimensions change
#   "C" -- 1x1 conv projection on every shortcut
SHORTCUT_OPTIONS = ("A", "B", "C")


class ZeroPadDownsample(nn.Module):
    """Option-A shortcut for a dimension change: take every ``stride``-th pixel and zero-pad
    the channel dimension from ``in_channels`` to ``out_channels``. No learnable parameters."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        raise NotImplementedError("ZeroPadDownsample: Phase 7")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("ZeroPadDownsample.forward: Phase 7")


def make_shortcut(
    in_channels: int,
    out_channels: int,
    stride: int,
    option: str,
    use_bn: bool,
) -> nn.Module:
    """Return the shortcut branch for a block.

    When ``in_channels == out_channels`` and ``stride == 1`` this is ``nn.Identity`` for all
    options. Otherwise it is a zero-pad downsample (A) or a 1x1 conv + BN (B/C). Option C also
    returns a 1x1 conv for the same-shape case.
    """
    raise NotImplementedError("make_shortcut: Phase 7")


class BasicBlock(nn.Module):
    """Two-layer residual block (paper Figure 2).

    Args:
        in_channels: channels of the block input.
        out_channels: channels of the block output.
        stride: stride of the first conv; 2 at the first block of stages 2 and 3.
        shortcut: one of :data:`SHORTCUT_OPTIONS`.
        use_bn: include BatchNorm (disabled for the Exp 7 ablation).
        residual: add the shortcut (True = ResNet, False = plain net).
        zero_init_residual: initialise the last BN's weight to 0 so the block starts as identity
            (post-paper trick, off by default).
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        *,
        shortcut: str = "A",
        use_bn: bool = True,
        residual: bool = True,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        raise NotImplementedError("BasicBlock: Phase 7")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("BasicBlock.forward: Phase 7")
