"""Augmentation pipelines (Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from resnet_research.data.transforms import build_transforms

MEAN = (0.5, 0.5, 0.5)
STD = (0.25, 0.25, 0.25)


def _random_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))


def test_standard_train_transform_output_shape():
    train_tf, _ = build_transforms("standard", MEAN, STD)
    out = train_tf(_random_image())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)


def test_minimal_transform_is_identical_for_train_and_eval():
    train_tf, eval_tf = build_transforms("minimal", MEAN, STD)
    assert train_tf is eval_tf


def test_standard_transform_is_stochastic():
    train_tf, _ = build_transforms("standard", MEAN, STD)
    image = _random_image()
    outputs = [train_tf(image) for _ in range(10)]
    distinct = {o.numpy().tobytes() for o in outputs}
    assert len(distinct) > 1


def test_eval_transform_is_deterministic():
    _, eval_tf = build_transforms("standard", MEAN, STD)
    image = _random_image()
    assert torch.equal(eval_tf(image), eval_tf(image))


def test_normalize_uses_passed_stats():
    _, eval_tf = build_transforms("standard", MEAN, STD)
    black = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    out = eval_tf(black)
    # ToTensor -> 0, then Normalize -> (0 - mean) / std
    expected = -torch.tensor(MEAN).view(3, 1, 1) / torch.tensor(STD).view(3, 1, 1)
    assert torch.allclose(out, expected.expand(3, 32, 32))


def test_unknown_augmentation_raises():
    with pytest.raises(ValueError):
        build_transforms("heavy", MEAN, STD)
