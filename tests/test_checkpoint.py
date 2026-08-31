"""Checkpoint save / load (Phase 10)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from resnet_research.models.resnet import cifar_resnet
from resnet_research.training.trainer import Trainer
from resnet_research.utils.checkpoint import load_checkpoint, save_checkpoint

CPU = torch.device("cpu")


def test_save_then_load_restores_weights_exactly(tmp_path):
    torch.manual_seed(0)
    original = cifar_resnet(8)
    save_checkpoint(tmp_path / "c.pt", model=original, epoch=7)

    torch.manual_seed(1)
    restored = cifar_resnet(8)
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    payload = load_checkpoint(tmp_path / "c.pt", model=restored)

    assert any(not torch.equal(before[k], restored.state_dict()[k]) for k in before)
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    assert payload["epoch"] == 7


def test_optimizer_state_survives_roundtrip(tmp_path):
    model = cifar_resnet(8)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.randn(4, 3, 32, 32)).sum().backward()
    opt.step()  # creates momentum buffers
    save_checkpoint(tmp_path / "c.pt", model=model, optimizer=opt, epoch=1)

    opt2 = torch.optim.SGD(cifar_resnet(8).parameters(), lr=0.1, momentum=0.9)
    load_checkpoint(tmp_path / "c.pt", optimizer=opt2)

    state1, state2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert state1.keys() == state2.keys()
    for key in state1:
        assert torch.equal(state1[key]["momentum_buffer"], state2[key]["momentum_buffer"])


def test_load_returns_metrics_and_config(tmp_path):
    save_checkpoint(
        tmp_path / "c.pt", model=cifar_resnet(8), epoch=3,
        metrics={"val_acc": 0.5}, config={"seed": 1},
    )
    payload = load_checkpoint(tmp_path / "c.pt")
    assert payload["metrics"]["val_acc"] == 0.5
    assert payload["config"]["seed"] == 1


def test_best_is_weights_only_final_has_optimizer(synthetic_dataset, tiny_train_config, tmp_path):
    loader = DataLoader(synthetic_dataset, batch_size=32)
    Trainer(
        cifar_resnet(8), loader, loader,
        cfg=tiny_train_config(epochs=1, lr=0.1), device=CPU, out_dir=tmp_path,
    ).fit()

    best = torch.load(tmp_path / "checkpoints" / "best.pt", weights_only=False)
    final = torch.load(tmp_path / "checkpoints" / "final.pt", weights_only=False)
    assert "optimizer" not in best and "scheduler" not in best
    assert "optimizer" in final and "scheduler" in final and "config" in final
