"""Shared fixtures.

Tests run on CPU and must be fast (no dataset download, no real training). Fixtures provide a tiny
synthetic dataset and batch so forward/backward/optimizer wiring can be checked cheaply.
"""

from __future__ import annotations

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
