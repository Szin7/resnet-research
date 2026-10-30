"""Benchmark selected CIFAR network configurations and write measured results."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["plain_20", "resnet_20", "plain_56", "resnet_56"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=[1, 128])
    parser.add_argument("--output", default="results/tables/benchmark.csv")
    args = parser.parse_args(argv)
    if not args.models or not args.batch_sizes:
        parser.error("Choose at least one model and batch size")

    from resnet_research.analysis.benchmark import (
        count_macs, count_parameters, measure_latency, measure_peak_memory,
    )
    from resnet_research.models import build_model
    from resnet_research.training.runner import resolve_device
    from resnet_research.utils.config import load_config

    device = resolve_device(args.device)
    rows = []
    for name in args.models:
        cfg = load_config([str(Path("configs/models") / (name + ".yaml"))])
        model = build_model(cfg["model"]).to(device)
        total, trainable = count_parameters(model)
        macs = count_macs(model)
        times = measure_latency(model, device=device, batch_sizes=tuple(args.batch_sizes),
                                warmup=args.warmup, iters=args.iters)
        peak = measure_peak_memory(model, device=device, batch_size=min(32, max(args.batch_sizes)))
        for batch, timing in times.items():
            rows.append({"model": name, "device": str(device), "batch_size": batch,
                         "parameters": total, "trainable_parameters": trainable,
                         "macs_per_image": macs, "latency_mean_ms": round(timing["mean_ms"], 5),
                         "latency_std_ms": round(timing["std_ms"], 5),
                         "peak_train_memory_bytes_bs32": peak})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = list(rows[0])
    target.with_suffix(".md").write_text(
        "# Model benchmark\n\nMACs exclude normalization, activations and pooling.\n\n"
        + "| " + " | ".join(columns) + " |\n|"
        + "|".join(["---"] * len(columns)) + "|\n"
        + "".join("| " + " | ".join(str(row[key]) for key in columns) + " |\n" for row in rows)
    )
    print(f"Measured benchmark written to {target}")


if __name__ == "__main__":
    main()
