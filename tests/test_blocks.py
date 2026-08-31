"""BasicBlock and shortcut behaviour (Phase 7)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="BasicBlock implemented in Phase 7")


def test_basic_block_preserves_shape_when_stride_1():
    """in=out=16, stride 1 -> output shape == input shape (N, 16, 32, 32)."""


def test_basic_block_downsamples_on_stride_2():
    """in=16, out=32, stride 2 -> output (N, 32, 16, 16)."""


def test_identity_shortcut_has_no_parameters():
    """make_shortcut(16, 16, 1, 'A', True) is nn.Identity / zero params."""


def test_option_a_shortcut_is_parameter_free_on_dimension_change():
    """make_shortcut(16, 32, 2, 'A', True) has zero learnable parameters (subsample + zero-pad)."""


def test_option_b_shortcut_is_a_1x1_conv_on_dimension_change():
    """make_shortcut(16, 32, 2, 'B', True) contains a Conv2d(16, 32, 1, stride=2) and a BN."""


def test_residual_flag_changes_output_but_not_parameter_count():
    """BasicBlock(..., residual=True) vs residual=False: same param count (option A), outputs differ
    for the same input and weights."""


def test_forward_runs_without_nan(tiny_batch):
    """A stack of BasicBlocks produces finite outputs for a random batch."""
