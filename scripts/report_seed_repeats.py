"""Summarize the finished 56-layer seed repeats without retraining anything.

Expects three complete runs for each model, including test_metrics.json
from scripts/evaluate.py. The study uses seed 1 from the depth sweep and
seeds 2/3 from experiments/10_depth56_seed_repeats.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from resnet_research.utils.config import load_config, resolve_run_name


MODELS = ("plain_56", "resnet_56")
SEEDS = (1, 2, 3)


def collect(runs_dir: Path) -> list[dict]:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            cfg = load_config(
                [f"configs/models/{model}.yaml", "configs/schedules/standard.yaml"],
                overrides=[f"seed={seed}"],
            )
            run = runs_dir / resolve_run_name(cfg)
            required = ("summary.json", "test_metrics.json", "metrics.csv")
            missing = [name for name in required if not (run / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{run}: missing {', '.join(missing)}; finish training and evaluation first"
                )

            summary = json.loads((run / "summary.json").read_text())
            test = json.loads((run / "test_metrics.json").read_text())
            with (run / "metrics.csv").open(newline="") as handle:
                epochs = list(csv.DictReader(handle))

            if not epochs or int(epochs[-1]["epoch"]) != 80 or summary["final_epoch"] != 80:
                raise ValueError(f"{run}: expected a complete 80-epoch run")

            accuracy = test.get("test_accuracy", {}).get("best.pt")
            if accuracy is None:
                raise ValueError(f"{run}: no best.pt test accuracy")

            rows.append({
                "model": model.replace("_", "-"),
                "seed": seed,
                "final_train_error_pct": round(100 * (1 - float(epochs[-1]["train_acc"])), 4),
                "best_val_accuracy_pct": round(100 * float(summary["best_val_acc"]), 4),
                "test_accuracy_pct": round(100 * float(accuracy), 4),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--output", default="results/tables/depth56_seed_repeats.csv")
    args = parser.parse_args()

    rows = collect(Path(args.runs_dir))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Individual measurements saved: {target}")
    for model in ("plain-56", "resnet-56"):
        samples = [r for r in rows if r["model"] == model]
        train = [r["final_train_error_pct"] for r in samples]
        test = [r["test_accuracy_pct"] for r in samples]
        print(
            f"{model}: training error {statistics.mean(train):.3f}% "
            f"(sample SD {statistics.stdev(train):.3f}); "
            f"test accuracy {statistics.mean(test):.3f}% "
            f"(sample SD {statistics.stdev(test):.3f}); n=3"
        )


if __name__ == "__main__":
    main()
