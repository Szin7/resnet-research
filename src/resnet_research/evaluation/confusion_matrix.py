"""Confusion matrix computation and plotting (Experiment 10).

Convention: ``cm[i, j]`` = number of test images with true label ``i`` predicted as ``j``.

Implemented in Phase 17.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> np.ndarray:
    raise NotImplementedError("compute_confusion_matrix: Phase 17")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    *,
    normalize: bool = True,
    path: Path | None = None,
    title: str | None = None,
) -> None:
    """Render the matrix as a heatmap with axis labels and a colorbar. Saves to ``path`` if given."""
    raise NotImplementedError("plot_confusion_matrix: Phase 17")
