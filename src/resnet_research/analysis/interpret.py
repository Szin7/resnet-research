"""Feature maps and Grad-CAM for qualitative model inspection."""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def extract_feature_maps(model: nn.Module, image: torch.Tensor, *, layers: Sequence[str]) -> dict[str, torch.Tensor]:
    modules = dict(model.named_modules())
    unknown = set(layers) - modules.keys()
    if unknown:
        raise KeyError(f"Unknown layer(s): {sorted(unknown)}")
    captured = {}
    handles = [modules[name].register_forward_hook(
        lambda module, args, out, key=name: captured.__setitem__(key, out.detach().cpu())
    ) for name in layers]
    was_training = model.training
    try:
        model.eval()
        device = next(model.parameters()).device
        batch = image.unsqueeze(0) if image.ndim == 3 else image
        model(batch.to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return captured


def grad_cam(model: nn.Module, image: torch.Tensor, *, target_layer: str, target_class: int | None = None) -> np.ndarray:
    modules = dict(model.named_modules())
    if target_layer not in modules:
        raise KeyError(f"Unknown layer: {target_layer}")
    activations, gradients = [], []
    def capture(module, args, out):
        if out.ndim != 4:
            raise ValueError("Grad-CAM needs a spatial activation map")
        activations.append(out)
        out.register_hook(lambda grad: gradients.append(grad))
    handle = modules[target_layer].register_forward_hook(capture)
    previous_mode = model.training
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        device = next(model.parameters()).device
        batch = (image.unsqueeze(0) if image.ndim == 3 else image).to(device)
        if batch.shape[0] != 1:
            raise ValueError("Grad-CAM expects one image")
        with torch.enable_grad():
            logits = model(batch)
            predicted = logits.argmax(1).item() if target_class is None else target_class
            logits[0, predicted].backward()
        weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
        heat = F.relu((weights * activations[-1]).sum(dim=1, keepdim=True))
        heat = F.interpolate(heat, size=batch.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        heat = (heat - heat.min()) / (heat.max() - heat.min()).clamp_min(1e-12)
        return heat.detach().cpu().numpy().astype(np.float32)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(previous_mode)
