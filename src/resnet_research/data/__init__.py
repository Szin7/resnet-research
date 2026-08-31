"""CIFAR-10 loading, the frozen 45k/5k split, and the augmentation pipelines."""

from __future__ import annotations

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Per-channel standardization -- sanity-check targets. The values actually used by a run are
# computed from that run's 45k training split and recorded in env.json (see spec section 3).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
