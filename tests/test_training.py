"""Training-loop sanity checks (Phase 10).

The key check for the whole project: a small model must be able to overfit a tiny synthetic
dataset. If it cannot, forward/backward/optimizer wiring is broken and no experiment is trustworthy.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from resnet_research.models.resnet import cifar_resnet
from resnet_research.training.schedulers import build_optimizer, build_scheduler
from resnet_research.training.trainer import Trainer
from resnet_research.utils.seed import set_seed

CPU = torch.device("cpu")


def _no_checkpoints(cfg):
    cfg["logging"].update({"save_best": False, "save_final": False})
    return cfg


# These tests drive the LR scheduler in isolation to inspect its values; the real Trainer
# always steps the optimizer first (see Trainer._train_epoch -> Trainer.fit).
_scheduler_only = pytest.mark.filterwarnings(
    "ignore:Detected call of `lr_scheduler.step",
)


def test_model_overfits_tiny_dataset(synthetic_dataset, tiny_train_config, tmp_path):
    set_seed(0)
    loader = DataLoader(synthetic_dataset, batch_size=32, shuffle=True)
    cfg = _no_checkpoints(tiny_train_config(epochs=40, lr=0.1, weight_decay=0.0))
    trainer = Trainer(cifar_resnet(8), loader, loader, cfg=cfg, device=CPU, out_dir=tmp_path)
    trainer.fit()
    assert trainer.history[-1]["train_acc"] >= 0.9


def test_training_loss_decreases(synthetic_dataset, tiny_train_config, tmp_path):
    set_seed(0)
    loader = DataLoader(synthetic_dataset, batch_size=32, shuffle=True)
    cfg = _no_checkpoints(tiny_train_config(epochs=8, lr=0.1))
    trainer = Trainer(cifar_resnet(8), loader, loader, cfg=cfg, device=CPU, out_dir=tmp_path)
    trainer.fit()
    assert trainer.history[-1]["train_loss"] < trainer.history[0]["train_loss"]


def test_gradients_are_finite_and_nonzero(tiny_batch):
    model = cifar_resnet(8)
    images, targets = tiny_batch
    torch.nn.functional.cross_entropy(model(images), targets).backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


@_scheduler_only
def test_step_scheduler_drops_lr_at_milestones(tiny_train_config):
    cfg = tiny_train_config(epochs=5, lr=0.1, scheduler="step", step_milestones=[2, 4], step_gamma=0.1)
    opt = build_optimizer(cifar_resnet(8), cfg)
    scheduler = build_scheduler(opt, cfg)
    seen = []
    for _ in range(5):
        seen.append(round(opt.param_groups[0]["lr"], 6))
        scheduler.step()
    assert seen == [0.1, 0.1, 0.01, 0.01, 0.001]


@_scheduler_only
def test_cosine_scheduler_anneals_monotonically(tiny_train_config):
    cfg = tiny_train_config(epochs=10, lr=0.1, scheduler="cosine")
    opt = build_optimizer(cifar_resnet(8), cfg)
    scheduler = build_scheduler(opt, cfg)
    seen = []
    for _ in range(10):
        seen.append(opt.param_groups[0]["lr"])
        scheduler.step()
    assert seen[0] == 0.1
    assert seen[-1] < 0.02
    assert all(a >= b - 1e-9 for a, b in zip(seen, seen[1:]))


@_scheduler_only
def test_resnet110_warmup_starts_low_then_rises(tiny_train_config):
    cfg = tiny_train_config(
        epochs=6, lr=0.1, scheduler="step", step_milestones=[4], step_gamma=0.1, warmup="resnet110"
    )
    opt = build_optimizer(cifar_resnet(8), cfg)
    scheduler = build_scheduler(opt, cfg)
    seen = []
    for _ in range(6):
        seen.append(opt.param_groups[0]["lr"])
        scheduler.step()
    assert seen[0] < 0.02          # warm-up epoch
    assert seen[1] == 0.1          # back to base LR


def test_resume_continues_from_checkpoint(synthetic_dataset, tiny_train_config, tmp_path):
    set_seed(0)
    loader = DataLoader(synthetic_dataset, batch_size=32, shuffle=True)
    Trainer(
        cifar_resnet(8), loader, loader,
        cfg=tiny_train_config(epochs=2, lr=0.1), device=CPU, out_dir=tmp_path,
    ).fit()
    assert (tmp_path / "checkpoints" / "final.pt").exists()

    resumed = Trainer(
        cifar_resnet(8), loader, loader,
        cfg=tiny_train_config(epochs=4, lr=0.1), device=CPU, out_dir=tmp_path,
    )
    summary = resumed.fit()
    assert resumed.start_epoch == 3
    assert summary["final_epoch"] == 4
    metrics_lines = (tmp_path / "metrics.csv").read_text().strip().splitlines()
    assert len(metrics_lines) == 1 + 4  # header + one row per epoch, no duplicates


def test_run_writes_metadata_and_summary(synthetic_dataset, tiny_train_config, tmp_path):
    import json

    loader = DataLoader(synthetic_dataset, batch_size=32)
    Trainer(
        cifar_resnet(8), loader, loader,
        cfg=tiny_train_config(epochs=1, lr=0.1), device=CPU, out_dir=tmp_path,
    ).fit()
    for name in ("config.yaml", "env.json", "metrics.csv", "summary.json"):
        assert (tmp_path / name).exists(), name
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["final_epoch"] == 1
    assert summary["parameters"] == sum(p.numel() for p in cifar_resnet(8).parameters())
