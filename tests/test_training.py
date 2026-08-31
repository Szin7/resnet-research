"""Training-loop sanity checks (Phase 10).

The key sanity check for the whole project: a small model must be able to overfit a tiny synthetic
dataset. If it cannot, forward/backward/optimizer wiring is broken and no experiment is trustworthy.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="training loop implemented in Phase 10")


def test_model_overfits_tiny_dataset(synthetic_dataset):
    """cifar_resnet(20) trained for ~50 steps on 64 samples reaches < 0.1 training loss."""


def test_gradients_are_nonzero_after_backward(tiny_batch):
    """After one backward pass, every trainable parameter has a finite, not-all-zero .grad."""


def test_loss_decreases_over_first_epoch(synthetic_dataset):
    """Mean training loss at the end of epoch 1 < loss at the first step."""


def test_step_scheduler_drops_lr_at_milestones():
    """build_scheduler('step', milestones=[2, 3], gamma=0.1): lr is 0.1 -> 0.01 -> 0.001."""


def test_amp_forced_off_when_grad_flow_enabled():
    """validate_config flips training.amp to False (with a warning) when logging.grad_flow is True."""


def test_resume_continues_from_saved_epoch(run_dir, synthetic_dataset):
    """Interrupting after epoch 2 and re-running resumes at epoch 3 with restored optimizer state."""
