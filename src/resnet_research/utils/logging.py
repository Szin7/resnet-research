"""Run logging: console logger, per-epoch metrics CSV, summary + environment JSON.

Each run directory gets: ``metrics.csv`` (one row per epoch), ``summary.json`` (final numbers),
``env.json`` (versions, GPU, platform, pip-freeze hash). See spec section 10 for the exact fields.

Implemented in Phase 10.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def get_logger(name: str = "resnet_research") -> logging.Logger:
    """Module logger with a single stream handler at INFO."""
    raise NotImplementedError("get_logger: Phase 10")


class MetricsWriter:
    """Appends epoch rows to ``<out_dir>/metrics.csv`` and writes ``<out_dir>/summary.json``."""

    def __init__(self, out_dir: Path) -> None:
        raise NotImplementedError("MetricsWriter: Phase 10")

    def append_epoch(self, row: dict[str, Any]) -> None:
        raise NotImplementedError("MetricsWriter.append_epoch: Phase 10")

    def write_summary(self, summary: dict[str, Any]) -> None:
        raise NotImplementedError("MetricsWriter.write_summary: Phase 10")


def capture_environment() -> dict[str, Any]:
    """Collect the ``env.json`` payload: python / torch / torchvision / CUDA / cuDNN versions,
    GPU name, platform, and a hash of ``pip freeze``."""
    raise NotImplementedError("capture_environment: Phase 10")
