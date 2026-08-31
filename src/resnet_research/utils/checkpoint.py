"""Checkpoint save / load.

A checkpoint is a dict: ``model`` state, optionally ``optimizer`` / ``scheduler`` state (for
resume), ``epoch``, ``metrics``, and the resolved ``config``. ``best.pt`` stores weights only;
``final.pt`` also stores optimizer + scheduler so a dropped Colab session can resume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics
    if config is not None:
        payload["config"] = config
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint, restoring state into any objects passed. Returns the raw dict.

    ``weights_only=False`` because we save and load our own files (they carry the config dict) --
    never point this at an untrusted checkpoint.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload
