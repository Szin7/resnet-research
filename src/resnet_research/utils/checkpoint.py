"""Checkpoint save / load.

A checkpoint is a dict: ``model`` state, optionally ``optimizer`` / ``scheduler`` state (for
resume), ``epoch``, ``metrics``, and the resolved ``config``. ``best.pt`` stores weights only;
``final.pt`` also stores optimizer + scheduler so a dropped Colab session can resume.

Implemented and tested in Phase 10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    raise NotImplementedError("save_checkpoint: Phase 10")


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint, restoring state into any objects passed. Returns the raw dict."""
    raise NotImplementedError("load_checkpoint: Phase 10")
