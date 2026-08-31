"""Loss construction.

Just cross-entropy on logits (``nn.CrossEntropyLoss``); ``label_smoothing`` is read from the config
and defaults to 0.0 (faithful). Kept as its own module so an experiment can swap it without
touching the trainer.

Implemented in Phase 10.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn


def build_loss(cfg: Mapping[str, Any]) -> nn.Module:
    raise NotImplementedError("build_loss: Phase 10")
