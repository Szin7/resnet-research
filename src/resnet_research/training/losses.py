"""Loss construction.

Cross-entropy on logits (``nn.CrossEntropyLoss``). ``label_smoothing`` is read from the config and
defaults to 0.0 (faithful to the paper). Kept as its own module so an experiment can swap the loss
without touching the trainer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn


def build_loss(cfg: Mapping[str, Any]) -> nn.Module:
    label_smoothing = float(cfg.get("training", {}).get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
