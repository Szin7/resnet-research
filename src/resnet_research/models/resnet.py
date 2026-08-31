"""CIFAR ResNet / plain network -- ``6n + 2`` weighted layers (paper Section 4.2).

Stem 3x3 conv (16 ch) -> three stages of ``n`` basic blocks at 16 / 32 / 64 channels and
32 / 16 / 8 spatial size -> global average pool -> linear classifier.

``CifarResNet(depth, residual=False)`` is the plain network: identical layers, no additive
shortcut. With shortcut option A (parameter-free) plain-k and resnet-k have identical parameter
counts -- see :data:`EXPECTED_PARAM_COUNTS`, asserted in tests against the paper's table.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from resnet_research.models.blocks import BasicBlock
from resnet_research.models.weight_init import initialize_weights

# Total parameter counts, shortcut option A. Match the paper's Table 6 to the digit.
EXPECTED_PARAM_COUNTS = {
    20: 269_722,
    32: 464_154,
    44: 658_586,
    56: 853_018,
    110: 1_727_962,
}

_STAGE_CHANNELS = (16, 32, 64)


def depth_to_n(depth: int) -> int:
    """``n = (depth - 2) / 6``; raises ``ValueError`` if ``depth`` is not a valid ``6n + 2``."""
    if not isinstance(depth, int) or depth < 8 or (depth - 2) % 6 != 0:
        raise ValueError(f"depth {depth!r} is not a valid 6n+2 CIFAR ResNet depth (8, 14, 20, ...)")
    return (depth - 2) // 6


class CifarResNet(nn.Module):
    """The CIFAR residual / plain network.

    Args:
        depth: total weighted layers, must be ``6n + 2``.
        num_classes: classifier outputs (10 for CIFAR-10).
        shortcut: shortcut option "A" / "B" / "C" (only meaningful when ``residual`` is True).
        use_bn: include BatchNorm.
        residual: add shortcuts (True = ResNet, False = plain).
        zero_init_residual: zero every block's second BN weight so blocks start as identity.
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
        n = depth_to_n(depth)
        self.depth = depth
        self.residual = residual

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=not use_bn),
            nn.BatchNorm2d(16) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        block_kwargs = dict(shortcut=shortcut, use_bn=use_bn, residual=residual)
        self.stage1 = self._make_stage(16, 16, n, stride=1, **block_kwargs)
        self.stage2 = self._make_stage(16, 32, n, stride=2, **block_kwargs)
        self.stage3 = self._make_stage(32, 64, n, stride=2, **block_kwargs)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

        initialize_weights(self)
        if zero_init_residual and use_bn:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.zeros_(module.bn2.weight)

    def _make_stage(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int, **block_kwargs
    ) -> nn.Sequential:
        # first block does any downsampling / channel change; the rest keep the shape
        blocks = [BasicBlock(in_channels, out_channels, stride=stride, **block_kwargs)]
        for _ in range(num_blocks - 1):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1, **block_kwargs))
        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


def cifar_resnet(depth: int, **kwargs) -> CifarResNet:
    """Residual CIFAR network of the given depth."""
    return CifarResNet(depth, residual=True, **kwargs)


def cifar_plain(depth: int, **kwargs) -> CifarResNet:
    """Plain CIFAR network of the given depth (no additive shortcut)."""
    return CifarResNet(depth, residual=False, **kwargs)
