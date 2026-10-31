"""Checkpoint save / load.

A checkpoint is a dict: ``model`` state, optionally ``optimizer`` / ``scheduler`` state (for
resume), ``epoch``, ``metrics``, and the resolved ``config``. ``best.pt`` stores weights only;
``final.pt`` also stores optimizer + scheduler so a dropped Colab session can resume.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
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
    scaler: Any | None = None,
    save_rng_state: bool = False,
    data_generator: torch.Generator | None = None,
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
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if save_rng_state:
        payload["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "data_generator": data_generator.get_state() if data_generator is not None else None,
        }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
    scaler: Any | None = None,
    restore_rng: bool = False,
    data_generator: torch.Generator | None = None,
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
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and "rng_state" in payload:
        state = payload["rng_state"]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["torch"].cpu())
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        if data_generator is not None and state.get("data_generator") is not None:
            data_generator.set_state(state["data_generator"].cpu())
    return payload
