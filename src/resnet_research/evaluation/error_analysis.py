"""Misclassification analysis (Experiment 10 -> docs/error_analysis.md).

Collects the images the best model gets wrong, ranks the most-confused class pairs from the
confusion matrix, and lays out an example gallery. Observations in the write-up must come from these
outputs -- no assumed "cat vs dog is hard" without the numbers.

Implemented in Phase 17.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class Misclassified:
    image: torch.Tensor          # normalized CHW tensor as fed to the model
    true_label: int
    predicted_label: int
    confidence: float            # softmax probability of the predicted class


@torch.no_grad()
def collect_misclassified(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    limit: int | None = None,
) -> list[Misclassified]:
    raise NotImplementedError("collect_misclassified: Phase 17")


def most_confused_pairs(
    cm: np.ndarray,
    class_names: Sequence[str],
    *,
    top: int = 5,
) -> list[tuple[str, str, int]]:
    """Return the ``top`` (true, predicted, count) off-diagonal entries, largest first."""
    raise NotImplementedError("most_confused_pairs: Phase 17")


def plot_misclassification_gallery(
    items: Sequence[Misclassified],
    class_names: Sequence[str],
    *,
    n: int = 24,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    path: Path | None = None,
) -> None:
    """Grid of ``n`` misclassified images titled ``true -> predicted (conf)``. Un-normalizes for
    display when ``mean`` / ``std`` are given."""
    raise NotImplementedError("plot_misclassification_gallery: Phase 17")
