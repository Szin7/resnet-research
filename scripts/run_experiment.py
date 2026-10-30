"""Run every training run in an experiment YAML, then (re)build its figures.

Runs whose results/runs/<name>/summary.json already exists are skipped, so a dropped Colab session
resumes at the experiment level. Individual runs also resume from their own final.pt.

Example:
    python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml
    python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DEFAULT_SCHEDULE = "configs/schedules/standard.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", required=True, help="path to an experiments/*/experiment.yaml")
    p.add_argument("--runs-dir", default="results/runs")
    p.add_argument("--figures-dir", default=None, help="default: <experiment>/figures")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-root", default=None, help="override data.root (Drive path on Colab)")
    p.add_argument("--subset-size", type=int, default=None, help="smoke-test with a data subset")
    p.add_argument("--dry-run", action="store_true", help="print the planned runs and exit")
    p.add_argument("--figures-only", action="store_true", help="skip training, just rebuild figures")
    return p


def _plan(spec: dict, data_root: str | None):
    from resnet_research.utils.config import load_config, resolve_run_name

    schedule = spec.get("schedule", DEFAULT_SCHEDULE)
    planned = []
    for entry in spec["runs"]:
        overrides = list(entry.get("overrides", []))
        if data_root:
            overrides.append(f"data.root={data_root}")
        cfg = load_config([entry["config"], schedule], overrides=overrides)
        name = entry.get("name") or resolve_run_name(cfg)
        planned.append((name, cfg))
    return planned


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from resnet_research.analysis.figures import generate
    from resnet_research.training.runner import resolve_device, run_training
    from resnet_research.utils.logging import get_logger

    exp_path = Path(args.experiment)
    spec = yaml.safe_load(exp_path.read_text())
    logger = get_logger()
    runs_dir = Path(args.runs_dir)

    planned = _plan(spec, args.data_root)
    if args.subset_size is not None:
        if args.subset_size < 1:
            raise ValueError("subset-size must be positive")
        planned = [(f"{name}_subset{args.subset_size}",
                    {**cfg, "data": {**cfg["data"], "debug_subset_size": args.subset_size}})
                   for name, cfg in planned]

    if args.dry_run:
        for name, cfg in planned:
            t = cfg["training"]
            print(f"{name:20s}  {cfg['model']}  {t['scheduler']} x{t['epochs']} lr={t['lr']}")
        return

    if not args.figures_only:
        device = resolve_device(args.device)
        for i, (name, cfg) in enumerate(planned, 1):
            out_dir = runs_dir / name
            if (out_dir / "summary.json").exists():
                path = out_dir / "config.yaml"
                if not path.exists() or yaml.safe_load(path.read_text()) != cfg:
                    raise ValueError(f"Cannot skip {name}: existing run has different settings")
                logger.info(f"[{i}/{len(planned)}] skip {name} (completed)")
                continue
            logger.info(f"[{i}/{len(planned)}] training {name}")
            run_training(
                cfg, device=device, out_dir=out_dir,
                num_workers=args.num_workers, subset_size=args.subset_size,
            )

    figure_names = spec.get("figures", [])
    if figure_names:
        figures_dir = Path(args.figures_dir) if args.figures_dir else exp_path.parent / "figures"
        written = generate(
            figure_names,
            [name for name, _ in planned],
            runs_dir=runs_dir,
            out_dir=figures_dir,
        )
        for path in written:
            logger.info(f"wrote {path}")


if __name__ == "__main__":
    main()
