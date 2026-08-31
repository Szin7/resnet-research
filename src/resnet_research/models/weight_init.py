"""Shared weight initialization (spec section 4.3).

He / MSRA initialization for conv layers, standard BatchNorm init. Used by both the ResNet family
and the baselines so the comparison is not skewed by initialization.
"""

from __future__ import annotations

import torch.nn as nn


def initialize_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            # weight keeps PyTorch's default (kaiming_uniform); only the bias is zeroed
            if module.bias is not None:
                nn.init.zeros_(module.bias)
