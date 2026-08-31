"""Reproducibility helpers (spec section 10).

``set_seed`` seeds Python / NumPy / torch (CPU + CUDA) and, when ``deterministic`` is set, asks
cuDNN and torch to avoid nondeterministic kernels. Full bitwise determinism on GPU is *not*
guaranteed -- run-to-run variation is quantified separately with repeated seeds.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Needed for deterministic CUDA matmuls; harmless on CPU. Must be set before the first
        # CUDA call, so we set it here rather than relying on the shell environment.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only: a few ops have no deterministic implementation; we prefer a warning over a
        # hard crash mid-experiment.
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoader.

    Each worker process inherits a base seed from the DataLoader's ``generator``; we derive the
    NumPy and Python RNG seeds from it so augmentation is reproducible across runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A CPU ``torch.Generator`` seeded for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
