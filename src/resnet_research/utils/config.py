"""Config loading, merging and validation (spec section 9).

A resolved config is ``configs/default.yaml`` with model / schedule / experiment fragments merged on
top (left to right), then dotted ``key=value`` overrides applied. ``normalize_config`` then applies
the AMP interlocks and ``validate_config`` enforces the spec's constraints.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

_MODEL_NAMES = ("resnet", "plain", "small_cnn", "small_cnn_deep")
_RESNET_FAMILY = ("resnet", "plain")
_SHORTCUTS = ("A", "B", "C")
_SCHEDULERS = ("step", "cosine", "constant")
_AUGMENTATIONS = ("standard", "minimal")
_NORMALIZERS = ("per_channel", "none")


def _find_default_config() -> Path:
    """Locate ``configs/default.yaml`` relative to the source tree, falling back to the cwd."""
    from_source = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
    if from_source.exists():
        return from_source
    from_cwd = Path.cwd() / "configs" / "default.yaml"
    if from_cwd.exists():
        return from_cwd
    raise FileNotFoundError("could not locate configs/default.yaml; pass base= explicitly")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``: nested dicts are merged, everything else is
    replaced. Neither input is mutated."""
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_overrides(cfg: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` strings. Values are parsed as YAML scalars, so ``lr=0.05`` becomes a
    float, ``seed=3`` an int, ``training.nesterov=true`` a bool."""
    result = copy.deepcopy(dict(cfg))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        value = yaml.safe_load(raw)
        node = result
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot set {dotted!r}: {key!r} is not a mapping")
        node[keys[-1]] = value
    return result


def normalize_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the AMP interlocks from spec section 9: mixed precision is turned off when gradient-flow
    logging is on or when BatchNorm is disabled, so those measurements stay clean fp32. Returns a
    new dict and warns if it changed anything."""
    result = copy.deepcopy(dict(cfg))
    training = result.setdefault("training", {})
    if training.get("amp"):
        reasons = []
        if result.get("logging", {}).get("grad_flow"):
            reasons.append("logging.grad_flow is set")
        if not result.get("model", {}).get("use_bn", True):
            reasons.append("model.use_bn is false")
        if reasons:
            training["amp"] = False
            warnings.warn(f"AMP disabled ({'; '.join(reasons)})", stacklevel=2)
    return result


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` with a specific message on any invalid field or combination."""
    model = cfg["model"]
    name = model["name"]
    if name not in _MODEL_NAMES:
        raise ValueError(f"model.name {name!r} not in {_MODEL_NAMES}")

    if name in _RESNET_FAMILY:
        depth = model["depth"]
        if not isinstance(depth, int) or depth < 8 or (depth - 2) % 6 != 0:
            raise ValueError(f"model.depth {depth!r} is not a valid 6n+2 depth (20, 32, 44, ...)")
        if model.get("shortcut", "A") not in _SHORTCUTS:
            raise ValueError(f"model.shortcut {model.get('shortcut')!r} not in {_SHORTCUTS}")

    data = cfg["data"]
    if data["augment"] not in _AUGMENTATIONS:
        raise ValueError(f"data.augment {data['augment']!r} not in {_AUGMENTATIONS}")
    if data["normalize"] not in _NORMALIZERS:
        raise ValueError(f"data.normalize {data['normalize']!r} not in {_NORMALIZERS}")

    training = cfg["training"]
    if training["optimizer"] != "sgd":
        raise ValueError(f"training.optimizer {training['optimizer']!r} not supported (sgd only)")
    if training["scheduler"] not in _SCHEDULERS:
        raise ValueError(f"training.scheduler {training['scheduler']!r} not in {_SCHEDULERS}")
    if training["scheduler"] == "step" and not training.get("step_milestones"):
        raise ValueError("training.scheduler 'step' requires training.step_milestones")


def load_config(
    fragments: Iterable[str | Path],
    *,
    overrides: Iterable[str] | None = None,
    base: str | Path | None = None,
) -> dict[str, Any]:
    """Merge fragments onto the default config, apply overrides and interlocks, validate, return."""
    cfg = _read_yaml(base if base is not None else _find_default_config())
    for fragment in fragments:
        cfg = deep_merge(cfg, _read_yaml(fragment))
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    cfg = normalize_config(cfg)
    validate_config(cfg)
    return cfg


def resolve_run_name(cfg: Mapping[str, Any]) -> str:
    """Default run directory name, e.g. ``resnet-56_seed1`` or ``small_cnn_seed1``."""
    if cfg["logging"].get("run_name"):
        return str(cfg["logging"]["run_name"])
    model = cfg["model"]
    if model["name"] in _RESNET_FAMILY:
        stem = f"{model['name']}-{model['depth']}"
    else:
        stem = model["name"]
    return f"{stem}_seed{cfg['seed']}"


def dump_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    """Write the resolved config to ``path`` as YAML, key order preserved."""
    Path(path).write_text(yaml.safe_dump(dict(cfg), default_flow_style=False, sort_keys=False))
