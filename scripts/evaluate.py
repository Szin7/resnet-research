"""Evaluate a finished run on the test set.

Loads the best-val and final checkpoints from a run directory, reports test accuracy for both, and
writes the confusion matrix + per-class metrics into the run directory.

Example:
    python scripts/evaluate.py --run results/runs/resnet-56_seed1

Implemented in Phase 10 (accuracy) / Phase 17 (confusion matrix, per-class).
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="path to results/runs/<run_name>")
    p.add_argument("--device", default="auto")
    p.add_argument("--split", default="test", choices=["val", "test"])
    return p


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    raise NotImplementedError("scripts/evaluate.py: Phase 10")


if __name__ == "__main__":
    main()
