"""Computational benchmarking (Phase 18 -> results/tables/benchmark.*).

Parameter counts are computed directly. MACs come from ``torchinfo`` (recorded as the tool used).
Latency and peak memory are measured on the run device with warm-up. All values reported are
measured, never estimated.

Implemented in Phase 18.
"""

from __future__ import annotations

import torch
from torch import nn


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts."""
    raise NotImplementedError("count_parameters: Phase 9")


def count_macs(model: nn.Module, *, input_size: tuple[int, ...] = (1, 3, 32, 32)) -> int:
    """Multiply-accumulates for one forward pass, via ``torchinfo``."""
    raise NotImplementedError("count_macs: Phase 18")


def measure_latency(
    model: nn.Module,
    *,
    device: torch.device,
    batch_sizes: tuple[int, ...] = (1, 128),
    warmup: int = 10,
    iters: int = 50,
) -> dict[int, dict[str, float]]:
    """Mean / std forward-pass wall-clock per batch size (no grad, after warm-up)."""
    raise NotImplementedError("measure_latency: Phase 18")


def measure_peak_memory(
    model: nn.Module,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> int | None:
    """Peak CUDA memory (bytes) for one training step, or None on CPU."""
    raise NotImplementedError("measure_peak_memory: Phase 18")


def benchmark(model: nn.Module, *, device: torch.device) -> dict:
    """Run all of the above and return a single dict for the results table."""
    raise NotImplementedError("benchmark: Phase 18")
