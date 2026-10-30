"""Measure per-layer weight gradients on a fixed batch."""
from __future__ import annotations
import json
from collections.abc import Callable
from pathlib import Path
import torch
from torch import nn


def _tracked_weights(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            yield (f"{name}.weight" if name else "weight"), module.weight


class GradientFlowRecorder:
    def __init__(self, model: nn.Module, *, param_filter: Callable[[str], bool] | None = None):
        self.model = model
        self.param_filter = param_filter or (lambda name: True)
        self._records: list[dict] = []

    def snapshot(self, step: int) -> dict[str, float]:
        norms = {}
        for name, weight in _tracked_weights(self.model):
            if not self.param_filter(name):
                continue
            if weight.grad is None:
                raise RuntimeError(f"No gradient for {name}; call backward() first")
            norm = float(torch.linalg.vector_norm(weight.grad.detach().float()).item())
            if not torch.isfinite(torch.tensor(norm)):
                raise ValueError(f"Non-finite gradient for {name}")
            norms[name] = norm
        self._records.append({"step": int(step), "norms": norms})
        return norms

    def records(self) -> list[dict]:
        return list(self._records)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._records, indent=2))


def probe_gradient_flow(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> dict[str, float]:
    """Diagnostic backward pass; restore gradients and BN statistics afterwards."""
    model.to(device)
    previous_mode = model.training
    previous_gradients = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    bn_state = [(m, m.running_mean.detach().clone(), m.running_var.detach().clone(),
                 m.num_batches_tracked.detach().clone())
                for m in model.modules()
                if isinstance(m, nn.BatchNorm2d) and m.track_running_stats]
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        images, targets = (part.to(device) for part in batch)
        with torch.enable_grad():
            loss = (loss_fn or nn.CrossEntropyLoss())(model(images.float()), targets)
            loss.backward()
        return GradientFlowRecorder(model).snapshot(0)
    finally:
        for layer, mean, var, steps in bn_state:
            layer.running_mean.copy_(mean)
            layer.running_var.copy_(var)
            layer.num_batches_tracked.copy_(steps)
        for parameter, gradient in zip(model.parameters(), previous_gradients):
            parameter.grad = gradient
        model.train(previous_mode)
