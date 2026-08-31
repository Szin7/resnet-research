"""CifarResNet structure and parameter counts (Phase 8)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="CifarResNet implemented in Phase 8")

# Cross-check against resnet.EXPECTED_PARAM_COUNTS and spec section 4.4.
EXPECTED = {20: 269_722, 32: 464_154, 44: 658_586, 56: 853_018, 110: 1_727_962}


@pytest.mark.parametrize("depth,expected", sorted(EXPECTED.items()))
def test_parameter_count_matches_paper(depth, expected):
    """cifar_resnet(depth) total parameter count == expected (shortcut option A)."""


@pytest.mark.parametrize("depth", sorted(EXPECTED))
def test_plain_and_resnet_have_equal_parameter_count(depth):
    """Option A adds no parameters, so cifar_plain(depth) == cifar_resnet(depth) in param count."""


def test_forward_output_shape(tiny_batch):
    """cifar_resnet(20)(images) -> (N, 10)."""


def test_feature_map_sizes_follow_stages():
    """Activations are 16x32x32 -> 32x16x16 -> 64x8x8 across the three stages."""


@pytest.mark.parametrize("bad_depth", [21, 30, 100])
def test_invalid_depth_raises(bad_depth):
    """Depth that is not 6n+2 raises ValueError."""


def test_no_torchvision_models_import():
    """resnet_research.models does not import torchvision.models (implementation is from scratch)."""
