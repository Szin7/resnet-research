"""Optimizer and learning-rate schedule construction.

Optimizer: SGD with momentum and weight decay (the paper's recipe), weight decay applied uniformly
to all parameters (the paper does not exclude BN / bias -- that is a later refinement).

Schedules are stepped once per epoch:
- ``step``     -- x``step_gamma`` at each milestone epoch
- ``cosine``   -- anneal to ~0 over the training length
- ``constant`` -- fixed LR

``warmup: resnet110`` prepends a one-epoch linear warm-up from 0.1x the base LR. This is a
simplification of the paper's adaptive warm-up ("train at 0.01 until training error < 80%, then
0.1"); the deviation is noted in the spec.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    SequentialLR,
)

SCHEDULERS = ("step", "cosine", "constant")


def build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> Optimizer:
    t = cfg["training"]
    if t["optimizer"] != "sgd":
        raise ValueError(f"unsupported optimizer {t['optimizer']!r} (sgd only)")
    return SGD(
        model.parameters(),
        lr=t["lr"],
        momentum=t["momentum"],
        weight_decay=t["weight_decay"],
        nesterov=t.get("nesterov", False),
    )


def _build_main_scheduler(optimizer: Optimizer, name: str, total_epochs: int, t: Mapping) -> LRScheduler:
    if name == "step":
        return MultiStepLR(optimizer, milestones=list(t["step_milestones"]), gamma=t["step_gamma"])
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_epochs)
    if name == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    raise ValueError(f"scheduler {name!r} not in {SCHEDULERS}")


def build_scheduler(optimizer: Optimizer, cfg: Mapping[str, Any]) -> LRScheduler:
    """Per-epoch LR scheduler for the whole run (call ``.step()`` once per epoch)."""
    t = cfg["training"]
    name = t["scheduler"]
    epochs = t["epochs"]
    warmup_epochs = 1 if t.get("warmup") == "resnet110" else 0

    if warmup_epochs == 0:
        return _build_main_scheduler(optimizer, name, epochs, t)

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    main = _build_main_scheduler(optimizer, name, epochs - warmup_epochs, t)
    return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])
