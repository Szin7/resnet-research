"""Run logging: console logger, per-epoch metrics CSV, summary + environment JSON.

Each run directory gets ``metrics.csv`` (one row per epoch), ``summary.json`` (final numbers) and
``env.json`` (versions, GPU, platform, git commit). See spec section 10 for the fields.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

_LOGGER_NAME = "resnet_research"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class MetricsWriter:
    """Appends epoch rows to ``<out_dir>/metrics.csv`` and writes ``<out_dir>/summary.json``."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "metrics.csv"
        self._fieldnames: list[str] | None = None

    def append_epoch(self, row: dict[str, Any]) -> None:
        write_header = not self.csv_path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.csv_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def read_rows(self) -> list[dict[str, Any]]:
        if not self.csv_path.exists():
            return []
        with open(self.csv_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [_coerce_numbers(r) for r in rows]

    def truncate_after(self, epoch: int) -> None:
        """Drop any rows with ``epoch`` greater than the given value (used when resuming)."""
        rows = self.read_rows()
        kept = [r for r in rows if r["epoch"] <= epoch]
        if len(kept) == len(rows):
            return
        self.csv_path.unlink(missing_ok=True)
        self._fieldnames = None
        for row in kept:
            self.append_epoch(row)

    def write_summary(self, summary: dict[str, Any]) -> None:
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _coerce_numbers(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def capture_environment() -> dict[str, Any]:
    """Collect the ``env.json`` payload."""
    import torch
    import torchvision

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        info["git_commit"] = None
    return info


def write_run_metadata(out_dir: str | Path, config: dict[str, Any]) -> None:
    """Write ``config.yaml`` and ``env.json`` into a run directory."""
    from resnet_research.utils.config import dump_config

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, out_dir / "config.yaml")
    (out_dir / "env.json").write_text(json.dumps(capture_environment(), indent=2))
