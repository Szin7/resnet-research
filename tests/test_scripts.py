"""Script-level integration (needs CIFAR-10; skipped otherwise)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import evaluate, run_experiment, train

MINI_EXPERIMENT = Path(__file__).parent / "fixtures" / "mini_experiment.yaml"


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


def test_experiment_then_evaluate_end_to_end(cifar_root, tmp_path):
    runs_dir = tmp_path / "runs"
    figures_dir = tmp_path / "figures"
    common = [
        "--experiment", str(MINI_EXPERIMENT),
        "--runs-dir", str(runs_dir),
        "--figures-dir", str(figures_dir),
        "--device", "cpu", "--num-workers", "0",
    ]
    run_experiment.main([*common, "--subset-size", "120", "--data-root", cifar_root])

    assert (runs_dir / "resnet-20_seed1" / "summary.json").exists()
    assert (runs_dir / "plain-20_seed1" / "summary.json").exists()
    assert (figures_dir / "training_loss_curves.png").exists()

    # second call skips finished runs and just rebuilds figures
    run_experiment.main([*common, "--figures-only"])

    evaluate.main(
        ["--run", str(runs_dir / "resnet-20_seed1"), "--device", "cpu",
         "--num-workers", "0", "--data-root", cifar_root, "--subset-size", "120"]
    )
    test_metrics = json.loads((runs_dir / "resnet-20_seed1" / "test_metrics.json").read_text())
    assert "final.pt" in test_metrics["test_accuracy"]
    assert len(test_metrics["confusion_matrix"]) == 10
    assert (runs_dir / "resnet-20_seed1" / "confusion_matrix.png").exists()
