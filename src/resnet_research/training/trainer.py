"""The training / validation loop.

Responsibilities: the epoch loop, train/val loss and accuracy, per-epoch scheduler step, best-by-val
and final checkpoints, per-epoch ``metrics.csv`` + final ``summary.json``, and resume-from-checkpoint
(Colab sessions drop). Deliberately small -- no callback framework, no trainer subclasses.

Test accuracy is *not* computed here: model selection uses validation only, and the test set is
evaluated once afterwards by ``scripts/evaluate.py``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from resnet_research.evaluation.metrics import epochs_to_threshold
from resnet_research.training.losses import build_loss
from resnet_research.training.schedulers import build_optimizer, build_scheduler
from resnet_research.utils.checkpoint import load_checkpoint, save_checkpoint
from resnet_research.utils.logging import MetricsWriter, get_logger, write_run_metadata


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
        out_dir: str | Path,
    ) -> None:
        self.cfg = dict(cfg)
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.out_dir = Path(out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        training = self.cfg["training"]
        self.epochs = training["epochs"]
        self.criterion = build_loss(self.cfg)
        self.optimizer = build_optimizer(self.model, self.cfg)
        self.scheduler = build_scheduler(self.optimizer, self.cfg)

        self.use_amp = bool(training.get("amp", False)) and device.type == "cuda"
        self.scaler = GradScaler(device.type, enabled=self.use_amp)

        self.logger = get_logger()
        self.metrics = MetricsWriter(self.out_dir)
        self.history: list[dict[str, Any]] = []
        self.best_val_acc = -1.0
        self.start_epoch = 1

        self._validate_existing_config()
        write_run_metadata(self.out_dir, self.cfg)

    def _validate_existing_config(self) -> None:
        """Fail if a results directory belongs to another training setup."""
        import yaml
        path = self.out_dir / "config.yaml"
        if not path.exists():
            return
        previous = yaml.safe_load(path.read_text())
        if previous == self.cfg:
            return
        # A longer training run may continue when the original schedule is unchanged.
        # Cosine schedules depend on the planned total epochs, so exclude them.
        old_train = dict(previous.get("training", {}))
        new_train = dict(self.cfg.get("training", {}))
        old_epochs = old_train.pop("epochs", None)
        new_epochs = new_train.pop("epochs", None)
        previous["training"] = old_train
        current = dict(self.cfg)
        current["training"] = new_train
        if (previous == current and old_epochs is not None and new_epochs is not None
                and new_epochs >= old_epochs and new_train.get("scheduler") in ("step", "constant")):
            return
        raise ValueError(f"{self.out_dir} has a different configuration; choose another run name.")

    # -- public ---------------------------------------------------------------

    def fit(self) -> dict[str, Any]:
        self.start_epoch = self._maybe_resume()
        for epoch in range(self.start_epoch, self.epochs + 1):
            lr = self.optimizer.param_groups[0]["lr"]
            started = time.perf_counter()
            train = self._train_epoch()
            val = self._validate()
            self.scheduler.step()
            elapsed = time.perf_counter() - started

            row = {
                "epoch": epoch,
                "train_loss": round(train.loss, 6),
                "train_acc": round(train.accuracy, 6),
                "val_loss": round(val.loss, 6),
                "val_acc": round(val.accuracy, 6),
                "lr": round(lr, 8),
                "epoch_seconds": round(elapsed, 2),
            }
            self.history.append(row)
            self.metrics.append_epoch(row)
            self.logger.info(
                f"epoch {epoch:3d}/{self.epochs}  "
                f"train {train.accuracy:.4f}/{train.loss:.4f}  "
                f"val {val.accuracy:.4f}/{val.loss:.4f}  lr {lr:.4g}  {elapsed:.1f}s"
            )

            if val.accuracy > self.best_val_acc:
                self.best_val_acc = val.accuracy
                if self.cfg["logging"].get("save_best", True):
                    self._save("best.pt", epoch=epoch, val_acc=val.accuracy, with_optimizer=False)
            if self.cfg["logging"].get("save_final", True):
                self._save("final.pt", epoch=epoch, val_acc=val.accuracy, with_optimizer=True)

        summary = self._build_summary()
        self.metrics.write_summary(summary)
        return summary

    # -- epoch loops --------------------------------------------------------

    def _train_epoch(self) -> EpochMetrics:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
        return EpochMetrics(running_loss / total, correct / total)

    @torch.no_grad()
    def _validate(self) -> EpochMetrics:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            running_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
        return EpochMetrics(running_loss / total, correct / total)

    # -- checkpointing -----------------------------------------------------

    def _save(self, name: str, *, epoch: int, val_acc: float, with_optimizer: bool) -> None:
        save_checkpoint(
            self.ckpt_dir / name,
            model=self.model,
            optimizer=self.optimizer if with_optimizer else None,
            scheduler=self.scheduler if with_optimizer else None,
            epoch=epoch,
            metrics={"val_acc": val_acc},
            config=self.cfg if name == "final.pt" else None,
            scaler=self.scaler if with_optimizer else None,
            save_rng_state=with_optimizer,
            data_generator=self.train_loader.generator if with_optimizer else None,
        )

    def _maybe_resume(self) -> int:
        final = self.ckpt_dir / "final.pt"
        if not final.exists():
            return 1
        payload = load_checkpoint(
            final,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
            scaler=self.scaler,
            restore_rng=True,
            data_generator=self.train_loader.generator,
        )
        done_epoch = int(payload["epoch"])
        self.metrics.truncate_after(done_epoch)
        self.history = self.metrics.read_rows()
        self.best_val_acc = self._recover_best_val_acc()
        self.logger.info(f"resuming from epoch {done_epoch + 1}")
        return done_epoch + 1

    def _recover_best_val_acc(self) -> float:
        best = self.ckpt_dir / "best.pt"
        if best.exists():
            return float(load_checkpoint(best)["metrics"]["val_acc"])
        return max((r["val_acc"] for r in self.history), default=-1.0)

    # -- summary ----------------------------------------------------------

    def _build_summary(self) -> dict[str, Any]:
        train_accs = [r["train_acc"] for r in self.history]
        best_row = max(self.history, key=lambda r: r["val_acc"], default={})
        final_row = self.history[-1] if self.history else {}
        return {
            "run_name": self.out_dir.name,
            "final_epoch": final_row.get("epoch"),
            "best_val_acc": self.best_val_acc,
            "best_epoch": best_row.get("epoch"),
            "final_train_acc": final_row.get("train_acc"),
            "final_val_acc": final_row.get("val_acc"),
            "epochs_to_60pct_train": epochs_to_threshold(train_accs, 0.60),
            "epochs_to_80pct_train": epochs_to_threshold(train_accs, 0.80),
            "total_seconds": round(sum(r["epoch_seconds"] for r in self.history), 1),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
        }
