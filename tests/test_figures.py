"""Figure generation from logged metrics (Phase 12)."""

from __future__ import annotations

import json

import pytest
import yaml

from resnet_research.analysis.figures import FIGURE_REGISTRY, generate, load_run


def _fake_run(runs_dir, name, *, family, depth, epochs=6, final_train_acc=0.97, best_val_acc=0.9):
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {"seed": 1, "model": {"name": family, "depth": depth},
             "training": {"epochs": epochs}, "data": {}, "logging": {}}
        )
    )
    header = "epoch,train_loss,train_acc,val_loss,val_acc,lr,epoch_seconds"
    lines = [header]
    for e in range(1, epochs + 1):
        frac = e / epochs
        lines.append(
            f"{e},{2.0 / e:.4f},{final_train_acc * frac:.4f},"
            f"{2.3 / e:.4f},{best_val_acc * frac:.4f},0.1,10.0"
        )
    (run_dir / "metrics.csv").write_text("\n".join(lines) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {"run_name": name, "final_epoch": epochs, "best_val_acc": best_val_acc,
             "final_train_acc": final_train_acc, "final_val_acc": best_val_acc}
        )
    )
    return run_dir


def test_load_run_parses_types(tmp_path):
    _fake_run(tmp_path, "resnet-20_seed1", family="resnet", depth=20)
    run = load_run(tmp_path / "resnet-20_seed1")
    assert run.family == "resnet"
    assert run.depth == 20
    assert isinstance(run.metrics[0]["train_acc"], float)
    assert run.metrics[-1]["epoch"] == 6


@pytest.mark.parametrize("figure", sorted(FIGURE_REGISTRY))
def test_every_registered_figure_renders(tmp_path, figure):
    runs_dir = tmp_path / "runs"
    for depth in (20, 44):
        _fake_run(runs_dir, f"plain-{depth}_seed1", family="plain", depth=depth)
        _fake_run(runs_dir, f"resnet-{depth}_seed1", family="resnet", depth=depth)
    names = [f"{fam}-{d}_seed1" for fam in ("plain", "resnet") for d in (20, 44)]
    written = generate([figure], names, runs_dir=runs_dir, out_dir=tmp_path / "figs")
    assert len(written) == 1
    assert written[0].exists() and written[0].stat().st_size > 0


def test_generate_skips_unfinished_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    _fake_run(runs_dir, "resnet-20_seed1", family="resnet", depth=20)
    written = generate(
        ["depth_vs_val_accuracy"],
        ["resnet-20_seed1", "resnet-56_seed1"],  # second has no summary.json
        runs_dir=runs_dir,
        out_dir=tmp_path / "figs",
    )
    assert len(written) == 1


def test_unknown_figure_name_raises(tmp_path):
    with pytest.raises(KeyError):
        generate(["not_a_figure"], [], runs_dir=tmp_path, out_dir=tmp_path)
