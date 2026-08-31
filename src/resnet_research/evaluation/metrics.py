"""Metric functions.

Definitions are frozen in docs/project_specification.md section 6. Per-class precision / recall / F1
are derived from the confusion matrix (one-vs-rest), not a separate pass.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for one batch."""
    return (logits.argmax(dim=1) == targets).float().mean().item()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """Mean loss and top-1 accuracy over a loader (model set to eval, no grad)."""
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, targets).item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def per_class_metrics(
    confusion: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """One-vs-rest precision / recall / F1 / support per class from a confusion matrix
    (rows = true, cols = predicted)."""
    confusion = np.asarray(confusion, dtype=float)
    result: dict[str, dict[str, float]] = {}
    for i, name in enumerate(class_names):
        tp = confusion[i, i]
        support = confusion[i, :].sum()
        predicted = confusion[:, i].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        result[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }
    return result


def epochs_to_threshold(train_accuracies: Sequence[float], threshold: float) -> int | None:
    """First epoch index (1-based) whose training accuracy >= ``threshold``, or None."""
    for epoch, acc in enumerate(train_accuracies, start=1):
        if acc >= threshold:
            return epoch
    return None
