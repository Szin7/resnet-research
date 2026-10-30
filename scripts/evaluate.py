"""Evaluate a finished run on the test set.

Loads the best-val and final checkpoints from a run directory, reports test accuracy for both,
computes the confusion matrix and per-class metrics, and writes them into the run directory
(test_metrics.json, confusion_matrix.png). The test set is only ever touched here.

Example:
    python scripts/evaluate.py --run results/runs/resnet-56_seed1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True, help="path to results/runs/<run_name>")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default=None, help="override data.root")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--subset-size", type=int, default=None, help="limit the test set (smoke test)")
    p.add_argument("--gallery", type=int, default=0, help="number of misclassified images to visualize")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    import yaml

    from resnet_research.data import CIFAR10_CLASSES
    from resnet_research.data.dataset import build_dataloaders
    from resnet_research.evaluation.confusion_matrix import (
        compute_confusion_matrix,
        most_confused_pairs,
        plot_confusion_matrix,
    )
    from resnet_research.evaluation.metrics import evaluate_model, per_class_metrics
    from resnet_research.models import build_model
    from resnet_research.training.runner import resolve_device
    from resnet_research.utils.checkpoint import load_checkpoint

    run_dir = Path(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    device = resolve_device(args.device)

    _, _, test_loader = build_dataloaders(cfg, num_workers=args.num_workers)
    if args.subset_size:
        from torch.utils.data import DataLoader, Subset

        test_loader = DataLoader(
            Subset(test_loader.dataset, range(args.subset_size)),
            batch_size=cfg["training"]["batch_size"], num_workers=args.num_workers,
        )

    results: dict = {"run": run_dir.name, "test_accuracy": {}}
    best_model = None
    for checkpoint in ("best.pt", "final.pt"):
        path = run_dir / "checkpoints" / checkpoint
        if not path.exists():
            continue
        model = build_model(cfg["model"]).to(device)
        load_checkpoint(path, model=model, map_location=device)
        metrics = evaluate_model(model, test_loader, device)
        results["test_accuracy"][checkpoint] = metrics["accuracy"]
        results.setdefault("test_loss", {})[checkpoint] = metrics["loss"]
        if checkpoint == "best.pt" or best_model is None:
            best_model = model

    cm = compute_confusion_matrix(best_model, test_loader, device, num_classes=len(CIFAR10_CLASSES))
    results["confusion_matrix"] = cm.tolist()
    results["per_class"] = per_class_metrics(cm, CIFAR10_CLASSES)
    results["most_confused_pairs"] = most_confused_pairs(cm, CIFAR10_CLASSES, top=5)
    if args.gallery > 0:
        from resnet_research.evaluation.error_analysis import (
            collect_misclassified, plot_misclassification_gallery,
        )
        mistakes = collect_misclassified(best_model, test_loader, device, limit=args.gallery)
        plot_misclassification_gallery(
            mistakes, CIFAR10_CLASSES, n=args.gallery,
            path=run_dir / "misclassified_gallery.png",
        )
        results["gallery_count"] = len(mistakes)

    (run_dir / "test_metrics.json").write_text(json.dumps(results, indent=2))
    plot_confusion_matrix(
        cm, CIFAR10_CLASSES, normalize=True,
        path=run_dir / "confusion_matrix.png", title=f"{run_dir.name} — test confusion",
    )

    print(json.dumps({k: results[k] for k in ("run", "test_accuracy", "most_confused_pairs")}, indent=2))
    print(f"\nwritten to {run_dir}/test_metrics.json and confusion_matrix.png")


if __name__ == "__main__":
    main()
