"""Metric functions.

Definitions are frozen in docs/project_specification.md section 6. Per-class precision / recall / F1
are derived from the confusion matrix (one-vs-rest), not a separate pass.

Implemented and tested in Phases 10 (accuracy) and 17 (per-class).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for one batch: mean of ``argmax(logits) == targets``."""
    raise NotImplementedError("accuracy: Phase 10")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Return ``{'loss': ..., 'accuracy': ...}`` over a loader (model set to eval, no grad)."""
    raise NotImplementedError("evaluate_model: Phase 10")


def per_class_metrics(
    confusion: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """One-vs-rest precision / recall / F1 / support per class from a confusion matrix
    (rows = true, cols = predicted)."""
    raise NotImplementedError("per_class_metrics: Phase 17")


def epochs_to_threshold(train_accuracies: Sequence[float], threshold: float) -> int | None:
    """First epoch index (1-based) whose training accuracy >= ``threshold``, or None."""
    raise NotImplementedError("epochs_to_threshold: Phase 12")
