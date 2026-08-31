"""Seeding and reproducibility (Phase 5)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="seed utils implemented in Phase 5")


def test_same_seed_same_initial_weights():
    """set_seed(1); build model twice -> identical initial parameters."""


def test_same_seed_same_first_batch_loss(synthetic_dataset):
    """Two runs with seed 1 produce the same loss on the first training batch (CPU)."""


def test_different_seed_different_weights():
    """set_seed(1) vs set_seed(2) -> different initial parameters."""


def test_seed_worker_is_deterministic():
    """seed_worker(0) sets a reproducible per-worker numpy/random seed."""
