"""Train one model from a composed config.

Examples:
    python scripts/train.py --config configs/models/resnet_56.yaml \\
        --schedule configs/schedules/standard.yaml
    python scripts/train.py --config configs/models/plain_20.yaml \\
        --schedule configs/schedules/smoke.yaml --set seed=2

Writes results/runs/<run_name>/ with config.yaml, env.json, metrics.csv, summary.json and
checkpoints/. Re-running an existing run directory resumes from checkpoints/final.pt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="model config fragment")
    p.add_argument("--schedule", default="configs/schedules/standard.yaml", help="schedule fragment")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--data-root", default=None, help="override data.root (e.g. a Drive path on Colab)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="use only the first N train / N//4 val samples (for a fast pipeline smoke test)",
    )
    return p


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from torch.utils.data import DataLoader, Subset

    from resnet_research.data.dataset import build_dataloaders
    from resnet_research.models import build_model
    from resnet_research.training.trainer import Trainer
    from resnet_research.utils.config import load_config, resolve_run_name
    from resnet_research.utils.seed import set_seed

    overrides = list(args.set)
    if args.data_root:
        overrides.append(f"data.root={args.data_root}")
    cfg = load_config([args.config, args.schedule], overrides=overrides)

    set_seed(cfg["seed"])
    device = _resolve_device(args.device)
    out_dir = Path(cfg["logging"]["out_dir"]) / resolve_run_name(cfg)

    train_loader, val_loader, _ = build_dataloaders(cfg, num_workers=args.num_workers)
    if args.subset_size:
        batch_size = cfg["training"]["batch_size"]
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(args.subset_size)),
            batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(max(1, args.subset_size // 4))),
            batch_size=batch_size, num_workers=args.num_workers,
        )
    model = build_model(cfg["model"])
    trainer = Trainer(model, train_loader, val_loader, cfg=cfg, device=device, out_dir=out_dir)
    summary = trainer.fit()

    print(json.dumps(summary, indent=2))
    print(f"\nrun written to {out_dir}")


if __name__ == "__main__":
    main()
