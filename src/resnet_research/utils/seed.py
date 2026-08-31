"""Reproducibility helpers (spec section 10).

``set_seed`` seeds Python / NumPy / torch (CPU + CUDA) and, when ``deterministic`` is set, enables
``cudnn.deterministic``, disables ``cudnn.benchmark``, and calls
``torch.use_deterministic_algorithms(True, warn_only=True)``. Full bitwise determinism on GPU is not
guaranteed -- run-to-run variation is quantified with repeated seeds.

Implemented and tested in Phase 5.
"""

from __future__ import annotations

import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    raise NotImplementedError("set_seed: Phase 5")


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoader: derive each worker's seed from the base seed so
    augmentation is reproducible."""
    raise NotImplementedError("seed_worker: Phase 5")


def make_generator(seed: int) -> torch.Generator:
    """A ``torch.Generator`` seeded for DataLoader shuffling."""
    raise NotImplementedError("make_generator: Phase 5")
