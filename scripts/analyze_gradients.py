"""Measure per-layer gradient norms for plain-56 vs resnet-56 (Experiment 5).

Runs short instrumented training (AMP off) recording gradient norms at initialization and after a
few epochs, then plots the depth profile. Writes to experiments/05_gradient_flow/.

Example:
    python scripts/analyze_gradients.py --models plain_56 resnet_56 --epochs 5

Implemented in Phase 14.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=["plain_56", "resnet_56"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--device", default="auto")
    return p


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    raise NotImplementedError("scripts/analyze_gradients.py: Phase 14")


if __name__ == "__main__":
    main()
