"""CIFAR-10 datasets and dataloaders.

The train/val split is a random 45,000 / 5,000 partition of the official 50,000-image training set,
seeded by ``split_seed`` (default 42) so it is identical across every run and on every machine
(NumPy's PCG64 generator is used for cross-version stability). The official 10,000-image test set is
never touched during training or model selection.

Per-channel normalization constants are computed from the 45k training split only (not val, not
test) and cached to ``<root>/cifar10_stats_split<seed>.json``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from resnet_research.data.transforms import build_transforms
from resnet_research.utils.seed import make_generator, seed_worker

TRAIN_TOTAL = 50_000
TEST_TOTAL = 10_000


def _split_indices(n_total: int, val_size: int, split_seed: int) -> tuple[list[int], list[int]]:
    """Deterministic (train_idx, val_idx) partition of ``range(n_total)``."""
    if not 0 < val_size < n_total:
        raise ValueError(f"val_size {val_size} out of range for {n_total} samples")
    perm = np.random.default_rng(split_seed).permutation(n_total)
    train_idx = perm[: n_total - val_size].tolist()
    val_idx = perm[n_total - val_size :].tolist()
    return train_idx, val_idx


def channel_statistics(dataset: Dataset) -> tuple[list[float], list[float]]:
    """Per-channel (mean, std) over a dataset that yields ``(CHW float tensor in [0, 1], label)``.
    Accumulates in float64 -- a float32 sum over ~46M values per channel loses precision."""
    loader = DataLoader(dataset, batch_size=512, num_workers=0)
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0
    for images, _ in loader:
        images = images.double()
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += (images**2).sum(dim=(0, 2, 3))
        n_pixels += images.shape[0] * images.shape[2] * images.shape[3]
    mean = channel_sum / n_pixels
    std = (channel_sq_sum / n_pixels - mean**2).clamp_min(0).sqrt()
    return mean.tolist(), std.tolist()


def _resolve_stats(
    root: str, train_subset: Dataset, split_seed: int, normalize: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if normalize == "none":
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    if normalize != "per_channel":
        raise ValueError(f"unknown normalize {normalize!r}")

    cache_path = Path(root) / f"cifar10_stats_split{split_seed}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return tuple(cached["mean"]), tuple(cached["std"])

    mean, std = channel_statistics(train_subset)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"mean": mean, "std": std, "split_seed": split_seed}, indent=2)
    )
    return tuple(mean), tuple(std)


def load_cifar10(
    root: str,
    *,
    val_size: int = 5000,
    split_seed: int = 42,
    augment: str = "standard",
    normalize: str = "per_channel",
    download: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """Return ``(train_ds, val_ds, test_ds)``.

    ``train_ds`` uses the training transform; ``val_ds`` and ``test_ds`` use the eval transform.
    ``val_ds`` is a disjoint subset of the same training data as ``train_ds``.
    """
    root = str(root)
    train_idx, val_idx = _split_indices(TRAIN_TOTAL, val_size, split_seed)

    # A plain to-tensor view of the training data, used only to compute normalization stats.
    stats_view = Subset(
        CIFAR10(root, train=True, download=download, transform=transforms.ToTensor()),
        train_idx,
    )
    mean, std = _resolve_stats(root, stats_view, split_seed, normalize)
    train_transform, eval_transform = build_transforms(augment, mean, std)

    # Two dataset objects over the same files with different transforms, sliced by disjoint indices.
    train_ds = Subset(CIFAR10(root, train=True, download=False, transform=train_transform), train_idx)
    val_ds = Subset(CIFAR10(root, train=True, download=False, transform=eval_transform), val_idx)
    test_ds = CIFAR10(root, train=False, download=download, transform=eval_transform)
    return train_ds, val_ds, test_ds


def build_dataloaders(
    cfg: Mapping[str, Any],
    *,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders from a resolved config. The train loader shuffles with a seeded
    generator and uses ``seed_worker`` so augmentation is reproducible."""
    data_cfg = cfg["data"]
    train_ds, val_ds, test_ds = load_cifar10(
        data_cfg["root"],
        val_size=data_cfg["val_size"],
        split_seed=cfg["split_seed"],
        augment=data_cfg["augment"],
        normalize=data_cfg["normalize"],
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=make_generator(cfg["seed"]),
    )
    eval_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return (
        train_loader,
        DataLoader(val_ds, **eval_kwargs),
        DataLoader(test_ds, **eval_kwargs),
    )
