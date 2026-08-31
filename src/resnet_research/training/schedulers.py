"""Optimizer and learning-rate schedule construction.

Optimizer: SGD with momentum and weight decay (the paper's recipe). Schedules: ``step`` (x0.1 at
configured milestones), ``cosine`` (anneal to ~0 over training), ``constant``. Optional linear
warm-up for the resnet-110 stretch runs.

Implemented and tested in Phase 10.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

SCHEDULERS = ("step", "cosine", "constant")


def build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> Optimizer:
    """SGD from the ``training`` config block."""
    raise NotImplementedError("build_optimizer: Phase 10")


def build_scheduler(
    optimizer: Optimizer,
    cfg: Mapping[str, Any],
    *,
    steps_per_epoch: int,
) -> LRScheduler:
    """LR schedule from the ``training`` config block. ``steps_per_epoch`` lets cosine and warm-up
    be defined per step rather than per epoch."""
    raise NotImplementedError("build_scheduler: Phase 10")
