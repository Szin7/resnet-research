"""Regenerate figures and tables from logged run metrics.

Reads results/runs/*/metrics.csv and summary.json and rebuilds everything under results/figures/
and results/tables/. Pure post-processing -- never trains, never invents numbers.

Example:
    python scripts/make_figures.py --what depth plain_vs_residual

Implemented incrementally from Phase 12 on.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--what", nargs="*", default=None, help="figure groups to rebuild; default = all")
    return p


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    raise NotImplementedError("scripts/make_figures.py: Phase 12+")


if __name__ == "__main__":
    main()
