"""Seeding and reproducibility (Phase 5)."""

from __future__ import annotations

import numpy as np
import torch

from resnet_research.utils.seed import make_generator, seed_worker, set_seed


def test_same_seed_same_initial_weights():
    set_seed(1)
    a = torch.nn.Conv2d(3, 16, 3)
    set_seed(1)
    b = torch.nn.Conv2d(3, 16, 3)
    assert torch.equal(a.weight, b.weight)
    assert torch.equal(a.bias, b.bias)


def test_different_seed_different_weights():
    set_seed(1)
    a = torch.nn.Conv2d(3, 16, 3)
    set_seed(2)
    b = torch.nn.Conv2d(3, 16, 3)
    assert not torch.equal(a.weight, b.weight)


def test_set_seed_makes_sampling_reproducible():
    set_seed(7)
    first = torch.randperm(128)
    set_seed(7)
    second = torch.randperm(128)
    assert torch.equal(first, second)


def test_make_generator_is_reproducible():
    g1, g2 = make_generator(3), make_generator(3)
    assert torch.equal(
        torch.randperm(50, generator=g1),
        torch.randperm(50, generator=g2),
    )


def test_seed_worker_makes_numpy_reproducible():
    torch.manual_seed(123)
    seed_worker(0)
    first = np.random.rand(4)
    torch.manual_seed(123)
    seed_worker(0)
    second = np.random.rand(4)
    assert np.allclose(first, second)
