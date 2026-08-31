"""Model definitions: residual / plain CIFAR networks and the conventional baselines.

Nothing here uses ``torchvision.models`` or pretrained weights -- the architecture is built from
``nn.Conv2d`` / ``nn.BatchNorm2d`` / ``nn.Linear`` and friends only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from resnet_research.models.weight_init import initialize_weights

__all__ = ["build_model", "initialize_weights"]


def build_model(model_cfg: Mapping[str, Any]) -> nn.Module:
    """Construct a model from the ``model`` block of a resolved config.

    Dispatches on ``model_cfg['name']``:

    - ``resnet``          -> ``CifarResNet(..., residual=True)``
    - ``plain``           -> ``CifarResNet(..., residual=False)``
    - ``small_cnn``       -> ``SmallCNN``
    - ``small_cnn_deep``  -> ``SmallCNNDeep``
    """
    name = model_cfg["name"]
    num_classes = model_cfg.get("num_classes", 10)

    if name == "small_cnn":
        from resnet_research.models.baselines import SmallCNN

        return SmallCNN(num_classes=num_classes)
    if name == "small_cnn_deep":
        from resnet_research.models.baselines import SmallCNNDeep

        return SmallCNNDeep(num_classes=num_classes)
    if name in ("resnet", "plain"):
        from resnet_research.models.resnet import CifarResNet

        return CifarResNet(
            depth=model_cfg["depth"],
            num_classes=num_classes,
            shortcut=model_cfg.get("shortcut", "A"),
            use_bn=model_cfg.get("use_bn", True),
            residual=(name == "resnet"),
            zero_init_residual=model_cfg.get("zero_init_residual", False),
        )

    raise ValueError(f"unknown model name {name!r}")
