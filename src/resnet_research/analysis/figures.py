"""Figures from logged run metrics.

Pure post-processing: reads ``results/runs/<name>/{config.yaml,metrics.csv,summary.json}`` and draws
plots. Never trains, never invents numbers. Each figure answers a question from an experiment README.

New figures are added to ``FIGURE_REGISTRY``; ``generate`` looks them up by name.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402


def _coerce(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


@dataclass
class RunData:
    name: str
    config: dict
    metrics: list[dict]
    summary: dict

    @property
    def family(self) -> str:
        return self.config["model"]["name"]

    @property
    def depth(self) -> int | None:
        return self.config["model"].get("depth")

    @property
    def final_train_error(self) -> float:
        return 100.0 * (1.0 - self.metrics[-1]["train_acc"])


def load_run(run_dir: str | Path) -> RunData:
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    with open(run_dir / "metrics.csv", newline="") as handle:
        metrics = [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    return RunData(run_dir.name, config, metrics, summary)


def load_runs(names: list[str], runs_dir: str | Path = "results/runs") -> list[RunData]:
    return [load_run(Path(runs_dir) / name) for name in names]


# -- individual figures ------------------------------------------------------

_FAMILY_STYLE = {"plain": ("--", "tab:red"), "resnet": ("-", "tab:blue")}


def _depth_series(runs: list[RunData], value: Callable[[RunData], float]):
    for family, (line, color) in _FAMILY_STYLE.items():
        points = sorted((r.depth, value(r)) for r in runs if r.family == family and r.depth)
        if points:
            xs, ys = zip(*points)
            yield family, line, color, xs, ys


def depth_vs_val_accuracy(runs: list[RunData]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    for family, line, color, xs, ys in _depth_series(runs, lambda r: r.summary["best_val_acc"] * 100):
        ax.plot(xs, ys, line, color=color, marker="o", label=family)
    ax.set(xlabel="depth (layers)", ylabel="best validation accuracy (%)",
           title="Validation accuracy vs depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def depth_vs_train_error(runs: list[RunData]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    for family, line, color, xs, ys in _depth_series(runs, lambda r: r.final_train_error):
        ax.plot(xs, ys, line, color=color, marker="o", label=family)
    ax.set(xlabel="depth (layers)", ylabel="final training error (%)",
           title="Training error vs depth (degradation check)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def training_loss_curves(runs: list[RunData]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in sorted(runs, key=lambda r: (r.family, r.depth or 0)):
        line, _ = _FAMILY_STYLE.get(run.family, ("-", None))
        epochs = [m["epoch"] for m in run.metrics]
        losses = [m["train_loss"] for m in run.metrics]
        ax.plot(epochs, losses, line, label=run.name, alpha=0.9)
    ax.set(xlabel="epoch", ylabel="training loss", title="Training loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    return fig


def val_accuracy_curves(runs: list[RunData]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in sorted(runs, key=lambda r: (r.family, r.depth or 0)):
        line, _ = _FAMILY_STYLE.get(run.family, ("-", None))
        epochs = [m["epoch"] for m in run.metrics]
        acc = [m["val_acc"] * 100 for m in run.metrics]
        ax.plot(epochs, acc, line, label=run.name, alpha=0.9)
    ax.set(xlabel="epoch", ylabel="validation accuracy (%)", title="Validation accuracy")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    return fig


FIGURE_REGISTRY: dict[str, Callable[[list[RunData]], plt.Figure]] = {
    "depth_vs_val_accuracy": depth_vs_val_accuracy,
    "depth_vs_train_error": depth_vs_train_error,
    "training_loss_curves": training_loss_curves,
    "val_accuracy_curves": val_accuracy_curves,
}


def generate(
    figure_names: list[str],
    run_names: list[str],
    *,
    runs_dir: str | Path = "results/runs",
    out_dir: str | Path = "results/figures",
) -> list[Path]:
    """Draw the named figures from whichever of ``run_names`` have finished. Returns the paths written."""
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unknown = [n for n in figure_names if n not in FIGURE_REGISTRY]
    if unknown:
        raise KeyError(f"unknown figures {unknown}; available: {sorted(FIGURE_REGISTRY)}")

    finished = [n for n in run_names if (runs_dir / n / "summary.json").exists()]
    if not finished:
        return []
    runs = load_runs(finished, runs_dir=runs_dir)

    written: list[Path] = []
    for name in figure_names:
        fig = FIGURE_REGISTRY[name](runs)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written
