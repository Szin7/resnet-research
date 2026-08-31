"""Benchmark every model: parameters, MACs, inference latency, peak memory.

Writes results/tables/benchmark.csv and .md. Uses measured values only.

Example:
    python scripts/benchmark.py --models resnet_20 resnet_56 plain_56 small_cnn

Implemented in Phase 18.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=None, help="model config stems; default = all")
    p.add_argument("--device", default="auto")
    p.add_argument("--iters", type=int, default=50)
    return p


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    raise NotImplementedError("scripts/benchmark.py: Phase 18")


if __name__ == "__main__":
    main()
