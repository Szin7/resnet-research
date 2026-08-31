"""The training / validation loop.

Responsibilities: run the epoch loop, track train/val loss and accuracy, step the scheduler, save
the best (by val accuracy) and final checkpoints, write per-epoch rows to ``metrics.csv`` and a
final ``summary.json``, support resume-from-checkpoint (Colab sessions drop), and optionally record
per-layer gradient norms when ``logging.grad_flow`` is set.

Kept deliberately small -- no callback framework, no trainer subclasses. Implemented in Phase 10.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        cfg: Mapping[str, Any],
        device: torch.device,
        out_dir: Path,
    ) -> None:
        raise NotImplementedError("Trainer: Phase 10")

    def fit(self) -> dict[str, Any]:
        """Run the full schedule. Returns the summary dict that is also written to summary.json."""
        raise NotImplementedError("Trainer.fit: Phase 10")

    def _train_epoch(self, epoch: int) -> EpochMetrics:
        raise NotImplementedError("Trainer._train_epoch: Phase 10")

    @torch.no_grad()
    def _validate(self) -> EpochMetrics:
        raise NotImplementedError("Trainer._validate: Phase 10")

    def _save(self, name: str, *, epoch: int, val_acc: float, include_optimizer: bool = False) -> None:
        raise NotImplementedError("Trainer._save: Phase 10")

    def _maybe_resume(self) -> int:
        """If a ``final.pt`` exists in ``out_dir``, restore state and return the next epoch."""
        raise NotImplementedError("Trainer._maybe_resume: Phase 10")
