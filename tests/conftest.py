"""Shared fixtures.

Tests run on CPU and must be fast (no dataset download, no real training). Fixtures provide a tiny
synthetic dataset and batch so forward/backward/optimizer wiring can be checked cheaply.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def tiny_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """8 random CIFAR-shaped images and labels."""
    g = torch.Generator().manual_seed(0)
    images = torch.randn(8, 3, 32, 32, generator=g)
    labels = torch.randint(0, 10, (8,), generator=g)
    return images, labels


@pytest.fixture
def synthetic_dataset() -> TensorDataset:
    """64 random images / labels -- small enough for an overfitting sanity check."""
    g = torch.Generator().manual_seed(1)
    images = torch.randn(64, 3, 32, 32, generator=g)
    labels = torch.randint(0, 10, (64,), generator=g)
    return TensorDataset(images, labels)


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    return d


@pytest.fixture
def tiny_train_config():
    """Factory for a fast, valid config: a depth-8 ResNet, 3 epochs, batch 32, AMP off.
    Pass ``training`` keys to override, e.g. ``tiny_train_config(epochs=40, lr=0.1)``.
    """
    from resnet_research.utils.config import load_config

    repo = Path(__file__).resolve().parents[1]

    def _make(**training):
        cfg = load_config(
            [repo / "configs" / "models" / "resnet_20.yaml"],
            base=repo / "configs" / "default.yaml",
        )
        cfg["model"]["depth"] = 8
        cfg["training"].update({"epochs": 3, "batch_size": 32, "amp": False})
        cfg["training"].update(training)
        return cfg

    return _make


@pytest.fixture(scope="session")
def cifar_root() -> str:
    """Path to a local CIFAR-10 download, or skip the test.

    Tests never download (keeps the suite fast and offline). Run
    ``python scripts/prepare_data.py`` once to populate ``data/``.
    """
    root = Path("data")
    if not (root / "cifar-10-batches-py").exists():
        pytest.skip("CIFAR-10 not present; run: python scripts/prepare_data.py")
    return str(root)
