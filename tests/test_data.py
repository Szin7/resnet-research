"""CIFAR-10 pipeline (Phase 5).

The split-index tests are pure and always run. The rest need a local CIFAR-10 download and are
skipped via the ``cifar_root`` fixture when it is absent (no download in the test suite).
"""

from __future__ import annotations

import torch

from resnet_research.data.dataset import (
    TRAIN_TOTAL,
    _split_indices,
    channel_statistics,
    load_cifar10,
)

# Known per-channel stats over the CIFAR-10 training set (spec section 3).
KNOWN_MEAN = (0.4914, 0.4822, 0.4465)
KNOWN_STD = (0.2470, 0.2435, 0.2616)


# --- pure: split logic ---------------------------------------------------------

def test_split_indices_sizes_and_partition():
    train_idx, val_idx = _split_indices(TRAIN_TOTAL, 5000, split_seed=42)
    assert len(train_idx) == 45_000
    assert len(val_idx) == 5_000
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx) | set(val_idx) == set(range(TRAIN_TOTAL))


def test_split_indices_deterministic():
    a = _split_indices(TRAIN_TOTAL, 5000, split_seed=42)
    b = _split_indices(TRAIN_TOTAL, 5000, split_seed=42)
    assert a == b


def test_split_indices_changes_with_seed():
    train_42, _ = _split_indices(TRAIN_TOTAL, 5000, split_seed=42)
    train_43, _ = _split_indices(TRAIN_TOTAL, 5000, split_seed=43)
    assert train_42 != train_43


# --- needs the dataset --------------------------------------------------------

def test_split_sizes(cifar_root):
    train_ds, val_ds, test_ds = load_cifar10(cifar_root, download=False)
    assert (len(train_ds), len(val_ds), len(test_ds)) == (45_000, 5_000, 10_000)


def test_split_is_deterministic_across_calls(cifar_root):
    a, _, _ = load_cifar10(cifar_root, download=False)
    b, _, _ = load_cifar10(cifar_root, download=False)
    assert a.indices == b.indices


def test_train_and_val_are_disjoint(cifar_root):
    train_ds, val_ds, _ = load_cifar10(cifar_root, download=False)
    assert set(train_ds.indices).isdisjoint(val_ds.indices)


def test_sample_shape_and_dtype(cifar_root):
    train_ds, _, _ = load_cifar10(cifar_root, download=False)
    image, label = train_ds[0]
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert 0 <= label <= 9


def test_eval_transform_is_deterministic(cifar_root):
    _, val_ds, _ = load_cifar10(cifar_root, download=False)
    first, _ = val_ds[0]
    second, _ = val_ds[0]
    assert torch.equal(first, second)


def test_channel_stats_close_to_known_values(cifar_root):
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    stats_view = CIFAR10(cifar_root, train=True, download=False, transform=transforms.ToTensor())
    mean, std = channel_statistics(stats_view)
    assert torch.allclose(torch.tensor(mean), torch.tensor(KNOWN_MEAN), atol=1e-2)
    assert torch.allclose(torch.tensor(std), torch.tensor(KNOWN_STD), atol=1e-2)
