"""Confusion matrix computation and plotting (Experiment 10).

Convention: ``cm[i, j]`` = number of test images with true label ``i`` predicted as ``j``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> np.ndarray:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, targets in loader:
        preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
        for true_label, pred_label in zip(targets.numpy(), preds):
            cm[true_label, pred_label] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    *,
    normalize: bool = True,
    path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    counts = np.asarray(cm)
    data = counts.astype(float)
    if normalize:
        data = data / data.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="predicted", ylabel="true",
           title=title or f"confusion matrix{' (row-normalized)' if normalize else ''}")

    threshold = 0.5 if normalize else counts.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{data[i, j]:.2f}" if normalize else str(int(counts[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] > threshold else "black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def most_confused_pairs(
    cm: np.ndarray,
    class_names: Sequence[str],
    *,
    top: int = 5,
) -> list[tuple[str, str, int]]:
    """The ``top`` largest off-diagonal (true, predicted, count) entries, largest first."""
    cm = np.asarray(cm)
    entries = [
        (class_names[i], class_names[j], int(cm[i, j]))
        for i in range(len(class_names))
        for j in range(len(class_names))
        if i != j and cm[i, j] > 0
    ]
    return sorted(entries, key=lambda e: e[2], reverse=True)[:top]
