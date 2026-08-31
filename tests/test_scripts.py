"""Script-level integration (needs CIFAR-10; skipped otherwise)."""

from __future__ import annotations

import json

from scripts import train


def test_train_script_runs_end_to_end(cifar_root, tmp_path):
    train.main(
        [
            "--config", "configs/models/resnet_20.yaml",
            "--schedule", "configs/schedules/smoke.yaml",
            "--subset-size", "200",
            "--device", "cpu",
            "--num-workers", "0",
            "--set", f"logging.out_dir={tmp_path.as_posix()}", "training.epochs=2",
            "data.root=" + cifar_root,
        ]
    )
    run_dir = tmp_path / "resnet-20_seed1"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["final_epoch"] == 2
    assert summary["parameters"] == 269_722
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "env.json").exists()
