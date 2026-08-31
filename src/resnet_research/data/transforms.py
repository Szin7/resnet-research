"""Augmentation / preprocessing pipelines.

``standard`` -- pad 4 (reflect) -> random crop 32 -> random horizontal flip -> to tensor ->
normalize. This matches the paper's crop+flip; the paper zero-pads, we reflect-pad (a minor
deviation, recorded in the spec).

``minimal`` -- to tensor -> normalize only. The control condition for Experiment 9; the train and
eval transforms are identical.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from torchvision import transforms

AUGMENTATIONS = ("standard", "minimal")


def build_transforms(
    augment: str,
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[Callable, Callable]:
    """Return ``(train_transform, eval_transform)`` for the named augmentation setting."""
    if augment not in AUGMENTATIONS:
        raise ValueError(f"unknown augmentation {augment!r}, expected one of {AUGMENTATIONS}")

    normalize = transforms.Normalize(mean, std)
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if augment == "minimal":
        return eval_transform, eval_transform

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform
