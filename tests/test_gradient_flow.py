"""Gradient-flow recorder (Phase 14)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="gradient-flow tools implemented in Phase 14")


def test_probe_returns_norm_per_conv_layer(tiny_batch):
    """probe_gradient_flow(cifar_resnet(20), batch) returns one finite norm per conv/linear weight."""


def test_recorder_snapshots_accumulate():
    """GradientFlowRecorder.snapshot called at 3 steps -> records() has 3 entries."""


def test_norms_are_positive_and_finite(tiny_batch):
    """All recorded gradient norms are > 0 and not NaN/inf at initialization."""
