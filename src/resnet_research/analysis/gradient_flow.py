"""Per-layer gradient-norm measurement (Experiment 5 -> docs/gradient_analysis.md).

Method (spec section 6): after ``loss.backward()`` and before ``optimizer.step()``, record the L2
norm of ``weight.grad`` for every Conv2d / Linear layer, on a fixed probe batch, in fp32 (AMP off).
Compare the depth profile for plain-56 vs resnet-56.

Interpretation must stay careful: per docs/mathematics.md section 5, the paper found plain-net
gradients are *not* vanishing, so a flat plain-net profile is a valid and informative result.

Implemented in Phase 14.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn


class GradientFlowRecorder:
    """Reads ``.grad`` off named parameters at chosen training steps.

    Args:
        model: the network being trained.
        param_filter: predicate on parameter name; defaults to conv/linear weights only.
    """

    def __init__(self, model: nn.Module, *, param_filter: Callable[[str], bool] | None = None) -> None:
        raise NotImplementedError("GradientFlowRecorder: Phase 14")

    def snapshot(self, step: int) -> dict[str, float]:
        """Record and return ``{layer_name: grad_l2_norm}`` for the current ``.grad`` state."""
        raise NotImplementedError("GradientFlowRecorder.snapshot: Phase 14")

    def records(self) -> list[dict]:
        raise NotImplementedError("GradientFlowRecorder.records: Phase 14")

    def save(self, path: Path) -> None:
        """Write all snapshots to ``grad_flow.json``."""
        raise NotImplementedError("GradientFlowRecorder.save: Phase 14")


def probe_gradient_flow(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> dict[str, float]:
    """One forward + backward on ``batch`` (no optimizer step); returns per-layer gradient norms.
    Used for the at-initialization measurement."""
    raise NotImplementedError("probe_gradient_flow: Phase 14")
