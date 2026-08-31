"""Download CIFAR-10 and pre-compute the normalization statistics for a split seed.

Run once before training (and once on Colab, pointing --root at a Drive path so the ~170 MB
download is not repeated every session).

    python scripts/prepare_data.py --root data --split-seed 42
"""

from __future__ import annotations

import argparse

from resnet_research.data.dataset import load_cifar10


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data", help="download directory")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-size", type=int, default=5000)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_ds, val_ds, test_ds = load_cifar10(
        args.root, val_size=args.val_size, split_seed=args.split_seed, download=True
    )
    print(f"CIFAR-10 ready in {args.root!r}")
    print(f"  train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")
    print(f"  normalization stats cached to {args.root}/cifar10_stats_split{args.split_seed}.json")


if __name__ == "__main__":
    main()
