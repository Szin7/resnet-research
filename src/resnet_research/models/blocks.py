"""Residual building blocks and shortcut connections.

The basic block follows the paper: two 3x3 conv layers, BatchNorm after each, ReLU after the first
conv and again *after* the residual addition. The ``residual`` flag adds the shortcut (ResNet) or
omits it (plain net) while leaving every layer otherwise identical -- with shortcut option A this
makes plain-k and resnet-k exactly parameter-matched.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Shortcut options, matching docs/project_specification.md section 4.2:
#   "A" -- parameter-free: subsample by stride, zero-pad the extra channels
#   "B" -- 1x1 conv projection only where dimensions change
#   "C" -- 1x1 conv projection on every shortcut
SHORTCUT_OPTIONS = ("A", "B", "C")


class ZeroPadDownsample(nn.Module):
    """Option-A shortcut for a dimension change (paper Section 3.2, option A).

    Subsamples spatially by taking every ``stride``-th pixel and zero-pads the channel dimension
    from ``in_channels`` to ``out_channels``, split evenly front/back like the reference
    implementation. No learnable parameters.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if out_channels < in_channels:
            raise ValueError("zero-pad shortcut cannot reduce channel count")
        self.stride = stride
        pad_total = out_channels - in_channels
        self.pad_front = pad_total // 2
        self.pad_back = pad_total - self.pad_front

    def forward(self, x: Tensor) -> Tensor:
        if self.stride > 1:
            x = x[:, :, :: self.stride, :: self.stride]
        if self.pad_front or self.pad_back:
            # F.pad channel padding: (W_left, W_right, H_top, H_bottom, C_front, C_back)
            x = F.pad(x, (0, 0, 0, 0, self.pad_front, self.pad_back))
        return x


def make_shortcut(
    in_channels: int,
    out_channels: int,
    stride: int,
    option: str,
    use_bn: bool,
) -> nn.Module:
    """Return the shortcut branch for a block.

    When the block keeps its shape (``stride == 1`` and channels unchanged) this is ``nn.Identity``
    for options A and B, and a 1x1 conv for option C. When the block changes shape it is a
    parameter-free zero-pad downsample (A) or a strided 1x1 conv + BN (B and C).
    """
    if option not in SHORTCUT_OPTIONS:
        raise ValueError(f"shortcut option {option!r} not in {SHORTCUT_OPTIONS}")

    changes_shape = stride != 1 or in_channels != out_channels
    if not changes_shape and option != "C":
        return nn.Identity()

    if option == "A":
        return ZeroPadDownsample(in_channels, out_channels, stride)

    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=not use_bn)
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    """Two-layer residual block (paper Figure 2).

    Args:
        in_channels: channels of the block input.
        out_channels: channels of the block output.
        stride: stride of the first conv; 2 at the first block of stages 2 and 3.
        shortcut: one of :data:`SHORTCUT_OPTIONS` (only used when ``residual`` is True).
        use_bn: include BatchNorm. When False the conv layers get a bias instead, since there is no
            BN bias to absorb -- this is the conventional "no BatchNorm" network (Experiment 7).
        residual: add the shortcut (True = ResNet, False = plain net). A plain block handles a
            dimension change through its strided first conv alone, with no shortcut.
        zero_init_residual: initialise the second BN's weight to 0 so the block starts as the
            identity mapping (post-paper trick, off by default).
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
        conv_bias = not use_bn

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=conv_bias
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=conv_bias
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.residual = residual
        self.shortcut: nn.Module | None = (
            make_shortcut(in_channels, out_channels, stride, shortcut, use_bn) if residual else None
        )

        if zero_init_residual and use_bn:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return self.relu(out)
