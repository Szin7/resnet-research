"""Experiment orchestration (Phase 12)."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts import run_experiment

DEPTH_EXPERIMENT = Path("experiments/01_depth/experiment.yaml")


def test_plan_resolves_all_eight_runs():
    spec = yaml.safe_load(DEPTH_EXPERIMENT.read_text())
    planned = run_experiment._plan(spec, data_root=None)
    names = {name for name, _ in planned}
    assert len(planned) == 8
    assert all(any(n.startswith(prefix + "_") for n in names) for prefix in
               ("plain-20_seed1", "plain-56_seed1", "resnet-20_seed1", "resnet-56_seed1"))
    for _, cfg in planned:
        assert cfg["training"]["epochs"] == 80  # standard schedule merged in
        assert cfg["training"]["scheduler"] == "step"


def test_plan_applies_data_root_override():
    spec = yaml.safe_load(DEPTH_EXPERIMENT.read_text())
    planned = run_experiment._plan(spec, data_root="/content/drive/MyDrive/cifar")
    assert all(cfg["data"]["root"] == "/content/drive/MyDrive/cifar" for _, cfg in planned)


def test_dry_run_prints_every_run(capsys):
    run_experiment.main(["--experiment", str(DEPTH_EXPERIMENT), "--dry-run"])
    out = capsys.readouterr().out
    for depth in (20, 32, 44, 56):
        assert f"plain-{depth}_seed1_" in out
        assert f"resnet-{depth}_seed1_" in out
