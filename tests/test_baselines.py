"""SmallCNN baselines (Phase 6)."""

from __future__ import annotations

import pytest
import torch

from resnet_research.models import build_model
from resnet_research.models.baselines import SmallCNN, SmallCNNDeep


def _num_conv_layers(model: torch.nn.Module) -> int:
    return sum(isinstance(m, torch.nn.Conv2d) for m in model.modules())


def test_small_cnn_output_shape(tiny_batch):
    images, _ = tiny_batch
    assert SmallCNN()(images).shape == (8, 10)


def test_small_cnn_deep_output_shape(tiny_batch):
    images, _ = tiny_batch
    assert SmallCNNDeep()(images).shape == (8, 10)


def test_conv_layer_counts():
    assert _num_conv_layers(SmallCNN()) == 6
    assert _num_conv_layers(SmallCNNDeep()) == 12


def test_forward_is_finite(tiny_batch):
    images, _ = tiny_batch
    for model in (SmallCNN(), SmallCNNDeep()):
        model.eval()
        assert torch.isfinite(model(images)).all()


def test_parameter_counts_are_in_expected_range():
    # framing models; exact values recorded for the benchmark table
    assert 800_000 <= sum(p.numel() for p in SmallCNN().parameters()) <= 850_000
    assert 1_150_000 <= sum(p.numel() for p in SmallCNNDeep().parameters()) <= 1_250_000


def test_build_model_dispatches_baselines():
    assert isinstance(build_model({"name": "small_cnn"}), SmallCNN)
    assert isinstance(build_model({"name": "small_cnn_deep"}), SmallCNNDeep)


def test_build_model_unknown_name_raises():
    with pytest.raises(ValueError):
        build_model({"name": "vgg"})
