"""SmallCNN baselines (Phase 6)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="baselines implemented in Phase 6")


def test_small_cnn_output_shape(tiny_batch):
    """SmallCNN()(images) -> (N, 10)."""


def test_small_cnn_deep_output_shape(tiny_batch):
    """SmallCNNDeep()(images) -> (N, 10)."""


def test_small_cnn_deep_has_more_conv_layers_than_small_cnn():
    """SmallCNNDeep has ~12 conv layers vs SmallCNN's 6."""


def test_forward_is_finite(tiny_batch):
    """Both baselines produce finite logits for a random batch."""
