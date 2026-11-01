"""Offline checks for repeat-seed reporting. No GPU or CIFAR files needed."""

from pathlib import Path

import pytest

from resnet_research.utils.config import load_config, resolve_run_name


def test_seed2_only_experiment_has_two_distinct_model_runs():
    import yaml

    spec = yaml.safe_load(
        Path("experiments/10_depth56_seed_repeats/seed2_only.yaml").read_text()
    )
    assert len(spec["runs"]) == 2
    assert [entry["config"] for entry in spec["runs"]] == [
        "configs/models/plain_56.yaml",
        "configs/models/resnet_56.yaml",
    ]
    for entry in spec["runs"]:
        assert entry["overrides"] == ["seed=2"]

    cfgs = [
        load_config([entry["config"], spec["schedule"]], overrides=entry["overrides"])
        for entry in spec["runs"]
    ]
    assert len({resolve_run_name(cfg) for cfg in cfgs}) == 2
    assert all(cfg["seed"] == 2 and cfg["training"]["epochs"] == 80 for cfg in cfgs)


def test_repeat_report_rejects_missing_finished_runs(tmp_path):
    from scripts.report_seed_repeats import collect

    with pytest.raises(FileNotFoundError, match="missing"):
        collect(tmp_path, seeds=(1, 2))
