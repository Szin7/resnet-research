"""Train one model from a composed config.

Examples:
    python scripts/train.py --config configs/models/resnet_56.yaml \\
        --schedule configs/schedules/standard.yaml
    python scripts/train.py --config configs/models/plain_20.yaml \\
        --schedule configs/schedules/smoke.yaml --set seed=2

Writes results/runs/<run_name>/ with config.yaml, env.json, metrics.csv, summary.json and
checkpoints/. Implemented in Phase 10.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="model config fragment")
    p.add_argument("--schedule", default="configs/schedules/standard.yaml", help="schedule fragment")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--data-root", default=None, help="override data.root (e.g. a Drive path on Colab)")
    return p


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    raise NotImplementedError("scripts/train.py: Phase 10")


if __name__ == "__main__":
    main()
