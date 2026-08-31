"""CifarResNet structure and parameter counts (Phase 8)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from resnet_research import models
from resnet_research.models import build_model
from resnet_research.models.resnet import (
    EXPECTED_PARAM_COUNTS,
    CifarResNet,
    cifar_plain,
    cifar_resnet,
)


def _count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("depth,expected", sorted(EXPECTED_PARAM_COUNTS.items()))
def test_parameter_count_matches_paper(depth, expected):
    assert _count(cifar_resnet(depth)) == expected


@pytest.mark.parametrize("depth", sorted(EXPECTED_PARAM_COUNTS))
def test_plain_and_resnet_have_equal_parameter_count(depth):
    assert _count(cifar_resnet(depth)) == _count(cifar_plain(depth))


def test_forward_output_shape(tiny_batch):
    images, _ = tiny_batch
    assert cifar_resnet(20).eval()(images).shape == (8, 10)


def test_feature_map_sizes_follow_stages():
    model = cifar_resnet(20).eval()
    x = torch.randn(2, 3, 32, 32)
    x = model.stem(x)
    assert x.shape == (2, 16, 32, 32)
    x = model.stage1(x)
    assert x.shape == (2, 16, 32, 32)
    x = model.stage2(x)
    assert x.shape == (2, 32, 16, 16)
    x = model.stage3(x)
    assert x.shape == (2, 64, 8, 8)


def test_weighted_layer_count_is_6n_plus_2():
    model = cifar_resnet(20)
    weighted = sum(
        isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)) for m in model.modules()
    )
    assert weighted == 20  # option A adds no conv layers


@pytest.mark.parametrize("bad_depth", [21, 30, 100])
def test_invalid_depth_raises(bad_depth):
    with pytest.raises(ValueError):
        cifar_resnet(bad_depth)


def test_projection_option_b_adds_expected_parameters():
    base = _count(cifar_resnet(56, shortcut="A"))
    proj = _count(cifar_resnet(56, shortcut="B"))
    assert proj == base + 2752  # 1x1 conv + BN at the two stage transitions


def test_build_model_dispatches_resnet_family():
    resnet = build_model({"name": "resnet", "depth": 20})
    plain = build_model({"name": "plain", "depth": 20})
    assert isinstance(resnet, CifarResNet) and resnet.residual is True
    assert isinstance(plain, CifarResNet) and plain.residual is False


def test_gradients_reach_every_parameter(tiny_batch):
    images, labels = tiny_batch
    model = cifar_resnet(20)
    model(images).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing


def test_models_are_implemented_from_scratch():
    model_dir = Path(models.__file__).parent
    for source in model_dir.glob("*.py"):
        for line in source.read_text().splitlines():
            stripped = line.strip()
            assert not stripped.startswith(("import torchvision", "from torchvision")), (
                f"{source.name} imports torchvision: {stripped!r}"
            )
