"""Config loading, merging, validation (Phase 5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from resnet_research.utils.config import (
    apply_overrides,
    deep_merge,
    dump_config,
    load_config,
    resolve_run_name,
    validate_config,
)

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "configs" / "models"
DEFAULT = REPO / "configs" / "default.yaml"


def test_fragment_overrides_default():
    cfg = load_config([MODELS / "resnet_20.yaml"], base=DEFAULT)
    assert cfg["model"]["name"] == "resnet"
    assert cfg["model"]["depth"] == 20
    assert cfg["training"]["epochs"] == 80  # untouched, from default


def test_dotted_override_applies_with_types():
    cfg = load_config(
        [MODELS / "resnet_20.yaml"], overrides=["training.lr=0.05", "seed=3"], base=DEFAULT
    )
    assert cfg["training"]["lr"] == 0.05
    assert isinstance(cfg["training"]["lr"], float)
    assert cfg["seed"] == 3


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"y": 3}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 3}}
    assert base == {"a": {"x": 1, "y": 2}}


def test_apply_overrides_rejects_bad_format():
    with pytest.raises(ValueError):
        apply_overrides({}, ["not_a_pair"])


def test_invalid_depth_rejected():
    cfg = load_config([MODELS / "resnet_20.yaml"], base=DEFAULT)
    cfg["model"]["depth"] = 21
    with pytest.raises(ValueError, match="6n"):
        validate_config(cfg)


def test_unknown_scheduler_rejected():
    cfg = load_config([MODELS / "resnet_20.yaml"], base=DEFAULT)
    cfg["training"]["scheduler"] = "triangular"
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_amp_disabled_without_batchnorm():
    with pytest.warns(UserWarning, match="AMP disabled"):
        cfg = load_config([MODELS / "resnet_20.yaml"], overrides=["model.use_bn=false"], base=DEFAULT)
    assert cfg["training"]["amp"] is False


def test_amp_disabled_with_grad_flow():
    with pytest.warns(UserWarning, match="AMP disabled"):
        cfg = load_config(
            [MODELS / "resnet_20.yaml"], overrides=["logging.grad_flow=true"], base=DEFAULT
        )
    assert cfg["training"]["amp"] is False


@pytest.mark.parametrize(
    "fragment,expected",
    [
        ("resnet_56.yaml", "resnet-56_seed1"),
        ("plain_56.yaml", "plain-56_seed1"),
        ("small_cnn.yaml", "small_cnn_seed1"),
    ],
)
def test_resolve_run_name_default(fragment, expected):
    cfg = load_config([MODELS / fragment], base=DEFAULT)
    assert resolve_run_name(cfg) == expected


def test_dump_config_roundtrip(tmp_path):
    import yaml

    cfg = load_config([MODELS / "resnet_20.yaml"], base=DEFAULT)
    path = tmp_path / "config.yaml"
    dump_config(cfg, path)
    assert yaml.safe_load(path.read_text()) == cfg


@pytest.mark.parametrize("fragment", sorted(p.name for p in MODELS.glob("*.yaml")))
def test_every_model_config_loads_and_validates(fragment):
    load_config([MODELS / fragment], base=DEFAULT)  # raises on any problem
