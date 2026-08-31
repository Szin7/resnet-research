"""Run every training run listed in an experiment YAML, then regenerate its figures/tables.

Skips runs whose results/runs/<run_name>/summary.json already exists (so a dropped Colab session
resumes at the experiment level too).

Example:
    python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml

Implemented in Phase 12.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, help="path to an experiments/*/experiment.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--dry-run", action="store_true", help="print the resolved runs and exit")
    p.add_argument("--figures-only", action="store_true", help="skip training, just rebuild outputs")
    return p


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    raise NotImplementedError("scripts/run_experiment.py: Phase 12")


if __name__ == "__main__":
    main()
