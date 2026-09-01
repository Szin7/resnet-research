"""Regenerate an experiment's figures from logged run metrics.

Pure post-processing -- never trains, never invents numbers.

Example:
    python scripts/make_figures.py --experiment experiments/01_depth/experiment.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", required=True, help="path to an experiments/*/experiment.yaml")
    p.add_argument("--runs-dir", default="results/runs")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from resnet_research.analysis.figures import generate
    from resnet_research.utils.config import load_config, resolve_run_name

    exp_path = Path(args.experiment)
    spec = yaml.safe_load(exp_path.read_text())
    schedule = spec.get("schedule", "configs/schedules/standard.yaml")

    run_names = []
    for entry in spec["runs"]:
        cfg = load_config([entry["config"], schedule], overrides=entry.get("overrides", []))
        run_names.append(entry.get("name") or resolve_run_name(cfg))

    written = generate(
        spec.get("figures", []), run_names,
        runs_dir=args.runs_dir, out_dir=exp_path.parent / "figures",
    )
    if not written:
        print("no finished runs yet -- nothing to plot")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
