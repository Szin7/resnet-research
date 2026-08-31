"""Config loading and validation (spec section 9).

A resolved config is ``configs/default.yaml`` with model / schedule / experiment fragments merged on
top (left to right), then dotted ``key=value`` overrides applied. ``validate_config`` enforces the
constraints from the spec (valid ``6n + 2`` depth, known enum values, step schedule needs
milestones, AMP forced off when grad_flow or no-BN, ...).

Implemented and tested in Phase 5.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (dicts merged, other values replaced)."""
    raise NotImplementedError("deep_merge: Phase 5")


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` strings; values parsed as YAML scalars."""
    raise NotImplementedError("apply_overrides: Phase 5")


def load_config(
    fragments: Iterable[str | Path],
    *,
    overrides: Iterable[str] | None = None,
    base: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    raise NotImplementedError("load_config: Phase 5")


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` with a specific message on any invalid field or combination."""
    raise NotImplementedError("validate_config: Phase 5")


def resolve_run_name(cfg: Mapping[str, Any]) -> str:
    """Default run directory name, e.g. ``resnet-56_seed1``."""
    raise NotImplementedError("resolve_run_name: Phase 5")


def dump_config(cfg: Mapping[str, Any], path: Path) -> None:
    raise NotImplementedError("dump_config: Phase 5")
