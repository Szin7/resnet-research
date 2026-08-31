"""Checkpoint save / load (Phase 10)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="checkpoint I/O implemented in Phase 10")


def test_save_then_load_restores_weights_exactly(run_dir):
    """Weights loaded from a checkpoint match the originals bit-for-bit."""


def test_load_restores_optimizer_state(run_dir):
    """Optimizer momentum buffers survive a save/load round trip."""


def test_best_checkpoint_is_weights_only(run_dir):
    """best.pt has no optimizer/scheduler state; final.pt has both."""


def test_load_checkpoint_returns_epoch_and_metrics(run_dir):
    """The returned dict carries epoch and the metrics recorded at save time."""
