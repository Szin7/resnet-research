"""Model definitions: residual / plain CIFAR networks and the conventional baselines.

Nothing here uses ``torchvision.models`` or pretrained weights -- the architecture is built from
``nn.Conv2d`` / ``nn.BatchNorm2d`` / ``nn.Linear`` and friends only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn


def build_model(model_cfg: Mapping[str, Any]) -> nn.Module:
    """Construct a model from the ``model`` block of a resolved config.

    Dispatches on ``model_cfg['name']``:

    - ``resnet``          -> :class:`~resnet_research.models.resnet.CifarResNet` (residual=True)
    - ``plain``           -> :class:`~resnet_research.models.resnet.CifarResNet` (residual=False)
    - ``small_cnn``       -> :class:`~resnet_research.models.baselines.SmallCNN`
    - ``small_cnn_deep``  -> :class:`~resnet_research.models.baselines.SmallCNNDeep`

    Implemented in Phase 6 (baselines) and Phases 7-8 (residual family).
    """
    raise NotImplementedError("build_model: Phase 6-8")
