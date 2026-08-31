"""CIFAR-10 pipeline (Phase 5).

These tests need the dataset. They are skipped now and, once implemented, will be marked to skip
automatically when CIFAR-10 is not present locally (no download in CI).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="data pipeline implemented in Phase 5")


def test_split_sizes():
    """load_cifar10(val_size=5000) -> train 45000, val 5000, test 10000."""


def test_split_is_deterministic_across_calls():
    """Same split_seed -> identical train/val index partition on two separate calls."""


def test_image_shape_and_dtype():
    """A training sample is a float tensor of shape (3, 32, 32)."""


def test_labels_in_range():
    """All labels are integers in [0, 9]."""


def test_normalization_stats_close_to_known_values():
    """channel_statistics(train) ~ (0.4914, 0.4822, 0.4465) / (0.2470, 0.2435, 0.2616) within 1e-2."""


def test_eval_transform_is_deterministic():
    """Applying the eval transform to the same image twice gives identical tensors."""


def test_train_and_test_do_not_overlap():
    """No index leakage between the 45k train split and the 10k test set."""
