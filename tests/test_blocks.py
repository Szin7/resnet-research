"""BasicBlock and shortcut behaviour (Phase 7)."""

from __future__ import annotations

import pytest
import torch

from resnet_research.models.blocks import BasicBlock, ZeroPadDownsample, make_shortcut


def test_basic_block_preserves_shape_when_stride_1():
    block = BasicBlock(16, 16, stride=1).eval()
    x = torch.randn(4, 16, 32, 32)
    assert block(x).shape == (4, 16, 32, 32)


def test_basic_block_downsamples_on_stride_2():
    block = BasicBlock(16, 32, stride=2).eval()
    x = torch.randn(4, 16, 32, 32)
    assert block(x).shape == (4, 32, 16, 16)


def test_identity_shortcut_has_no_parameters():
    sc = make_shortcut(16, 16, 1, "A", use_bn=True)
    assert isinstance(sc, torch.nn.Identity)


def test_option_a_shortcut_is_parameter_free_on_dimension_change():
    sc = make_shortcut(16, 32, 2, "A", use_bn=True)
    assert isinstance(sc, ZeroPadDownsample)
    assert sum(p.numel() for p in sc.parameters()) == 0


def test_option_a_subsamples_and_zero_pads():
    x = torch.randn(4, 16, 32, 32)
    out = make_shortcut(16, 32, 2, "A", use_bn=True)(x)
    assert out.shape == (4, 32, 16, 16)
    assert torch.all(out[:, :8] == 0)          # front zero-pad
    assert torch.all(out[:, 24:] == 0)         # back zero-pad
    assert torch.equal(out[:, 8:24], x[:, :, ::2, ::2])  # real channels, subsampled


def test_option_b_projection_only_on_dimension_change():
    within_stage = make_shortcut(32, 32, 1, "B", use_bn=True)
    transition = make_shortcut(16, 32, 2, "B", use_bn=True)
    assert isinstance(within_stage, torch.nn.Identity)
    assert any(isinstance(m, torch.nn.Conv2d) for m in transition.modules())
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in transition.modules())
    assert transition(torch.randn(4, 16, 32, 32)).shape == (4, 32, 16, 16)


def test_option_c_projects_every_shortcut():
    within_stage = make_shortcut(32, 32, 1, "C", use_bn=True)
    assert any(isinstance(m, torch.nn.Conv2d) for m in within_stage.modules())


def test_residual_flag_changes_output_but_not_parameter_count():
    torch.manual_seed(0)
    res = BasicBlock(16, 16, residual=True, shortcut="A")
    torch.manual_seed(0)
    plain = BasicBlock(16, 16, residual=False, shortcut="A")
    assert sum(p.numel() for p in res.parameters()) == sum(p.numel() for p in plain.parameters())

    res.eval()
    plain.eval()
    x = torch.randn(4, 16, 32, 32)
    assert not torch.allclose(res(x), plain(x))


def test_no_batchnorm_gives_conv_bias():
    block = BasicBlock(16, 16, use_bn=False)
    assert block.conv1.bias is not None
    assert isinstance(block.bn1, torch.nn.Identity)


def test_batchnorm_means_no_conv_bias():
    block = BasicBlock(16, 16, use_bn=True)
    assert block.conv1.bias is None


def test_zero_init_residual_starts_as_identity():
    block = BasicBlock(16, 16, residual=True, shortcut="A", zero_init_residual=True).eval()
    x = torch.randn(4, 16, 32, 32)
    # the F(x) branch ends in a BN with weight 0 -> outputs 0 -> block output is relu(x)
    assert torch.allclose(block(x), torch.relu(x), atol=1e-6)


def test_deep_stack_forward_is_finite():
    stack = torch.nn.Sequential(*[BasicBlock(16, 16) for _ in range(12)]).eval()
    assert torch.isfinite(stack(torch.randn(2, 16, 32, 32))).all()


def test_invalid_shortcut_option_raises():
    with pytest.raises(ValueError):
        BasicBlock(16, 16, shortcut="Z")
