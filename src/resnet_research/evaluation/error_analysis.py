"""Test-set misclassification sampling and a visual error gallery."""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from resnet_research.evaluation.confusion_matrix import most_confused_pairs


@dataclass
class Misclassified:
    image: torch.Tensor
    true_label: int
    predicted_label: int
    confidence: float


@torch.no_grad()
def collect_misclassified(model: nn.Module, loader: DataLoader, device: torch.device,
                          *, limit: int | None = None) -> list[Misclassified]:
    if limit is not None and limit < 1:
        return []
    previous = model.training
    model.eval()
    errors = []
    try:
        for images, labels in loader:
            confidence, prediction = model(images.to(device)).softmax(dim=1).max(dim=1)
            for index in (prediction.cpu() != labels).nonzero(as_tuple=True)[0].tolist():
                errors.append(Misclassified(images[index].detach().cpu(), int(labels[index]),
                                            int(prediction[index]), float(confidence[index])))
                if limit is not None and len(errors) >= limit:
                    return errors
    finally:
        model.train(previous)
    return errors


def plot_misclassification_gallery(items: Sequence[Misclassified], class_names: Sequence[str],
                                   *, n: int = 24, mean: Sequence[float] | None = None,
                                   std: Sequence[float] | None = None,
                                   path: Path | None = None) -> None:
    if n < 1:
        raise ValueError("n must be positive")
    chosen = list(items[:n])
    columns = min(6, max(1, len(chosen)))
    rows = max(1, (len(chosen) + columns - 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.4, rows * 2.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, entry in zip(axes.flat, chosen):
        image = entry.image.float().clone()
        if mean is not None and std is not None:
            image = image * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
        ax.imshow(image.clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(f"{class_names[entry.true_label]} -> {class_names[entry.predicted_label]}\nconfidence {entry.confidence:.2f}", fontsize=8)
    if not chosen:
        axes[0, 0].text(0.5, 0.5, "No mistakes in this sample", ha="center", va="center")
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
