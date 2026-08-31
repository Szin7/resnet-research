"""Config loading, merging, validation (Phase 5)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="config utils implemented in Phase 5")


def test_fragment_overrides_default():
    """Merging configs/models/resnet_20.yaml onto default sets model.depth = 20, keeps the rest."""


def test_dotted_override_applies():
    """apply_overrides(cfg, ['training.lr=0.05', 'seed=3']) sets both, parsed as float / int."""


def test_invalid_depth_rejected():
    """validate_config raises ValueError for model.depth = 21 with name in {resnet, plain}."""


def test_unknown_scheduler_rejected():
    """validate_config raises for training.scheduler = 'triangular'."""


def test_amp_disabled_without_batchnorm():
    """validate_config forces training.amp = False when model.use_bn is False."""


def test_resolve_run_name_default():
    """resolve_run_name for resnet/depth 56/seed 1 -> 'resnet-56_seed1'."""


def test_every_model_config_loads_and_validates():
    """Each file in configs/models/ merges onto default and passes validate_config."""
