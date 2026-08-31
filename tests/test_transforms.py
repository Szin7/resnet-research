"""Augmentation pipelines (Phase 5)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="transforms implemented in Phase 5")


def test_standard_train_transform_output_shape():
    """standard train transform: PIL 32x32 -> tensor (3, 32, 32) after pad+crop."""


def test_minimal_transform_has_no_random_ops():
    """minimal train transform == eval transform (ToTensor + Normalize only)."""


def test_standard_transform_is_stochastic():
    """Two applications of the standard train transform to one image usually differ (crop/flip)."""


def test_normalize_uses_passed_stats():
    """build_transforms passes the given mean/std through to the Normalize op."""
