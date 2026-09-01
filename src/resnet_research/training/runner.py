"""Run one training job from a resolved config.

Shared by ``scripts/train.py`` (single run) and ``scripts/run_experiment.py`` (a matrix of runs),
so the two entry points cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def resolve_device(choice: str = "auto") -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def run_training(
    cfg: dict[str, Any],
    *,
    device: torch.device,
    out_dir: str | Path,
    num_workers: int = 2,
    subset_size: int | None = None,
) -> dict[str, Any]:
    """Seed, build the data + model, run the trainer, return its summary dict."""
    from torch.utils.data import DataLoader, Subset

    from resnet_research.data.dataset import build_dataloaders
    from resnet_research.models import build_model
    from resnet_research.training.trainer import Trainer
    from resnet_research.utils.seed import set_seed

    set_seed(cfg["seed"])
    train_loader, val_loader, _ = build_dataloaders(cfg, num_workers=num_workers)

    if subset_size:
        batch_size = cfg["training"]["batch_size"]
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(subset_size)),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(max(1, subset_size // 4))),
            batch_size=batch_size, num_workers=num_workers,
        )

    model = build_model(cfg["model"])
    trainer = Trainer(model, train_loader, val_loader, cfg=cfg, device=device, out_dir=out_dir)
    return trainer.fit()
