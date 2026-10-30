"""Measure parameter count, MACs, latency, and peak GPU training memory."""
from __future__ import annotations
import statistics
import time
import torch
from torch import nn


def count_parameters(model: nn.Module) -> tuple[int, int]:
    params = list(model.parameters())
    return sum(p.numel() for p in params), sum(p.numel() for p in params if p.requires_grad)


def count_macs(model: nn.Module, *, input_size: tuple[int, ...] = (1, 3, 32, 32)) -> int:
    total = 0
    hooks = []
    def conv_hook(layer, inputs, output):
        nonlocal total
        n, channels, h, w = output.shape
        total += n * channels * h * w * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
    def linear_hook(layer, inputs, output):
        nonlocal total
        total += output.numel() * layer.in_features
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))
    device = next(model.parameters()).device
    previous = model.training
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(input_size, device=device))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(previous)
    return int(total)


def measure_latency(model: nn.Module, *, device: torch.device,
                    batch_sizes: tuple[int, ...] = (1, 128),
                    warmup: int = 10, iters: int = 50) -> dict[int, dict[str, float]]:
    if warmup < 0 or iters < 1 or any(size < 1 for size in batch_sizes):
        raise ValueError("warmup >= 0, iters > 0, and batch sizes > 0 required")
    model.to(device)
    previous = model.training
    model.eval()
    results = {}
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    try:
        with torch.inference_mode():
            for size in batch_sizes:
                data = torch.randn(size, 3, 32, 32, device=device)
                for _ in range(warmup):
                    model(data)
                synchronize()
                measurements = []
                for _ in range(iters):
                    synchronize()
                    start = time.perf_counter()
                    model(data)
                    synchronize()
                    measurements.append((time.perf_counter() - start) * 1000)
                results[size] = {"mean_ms": statistics.mean(measurements),
                                 "std_ms": statistics.stdev(measurements) if len(measurements) > 1 else 0.0}
    finally:
        model.train(previous)
    return results


def measure_peak_memory(model: nn.Module, *, device: torch.device, batch_size: int = 128) -> int | None:
    if device.type != "cuda":
        return None
    model.to(device)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    previous = model.training
    try:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        torch.cuda.reset_peak_memory_stats(device)
        images = torch.randn(batch_size, 3, 32, 32, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(images), labels).backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))
    finally:
        model.load_state_dict(original)
        model.zero_grad(set_to_none=True)
        model.train(previous)


def benchmark(model: nn.Module, *, device: torch.device) -> dict:
    model.to(device)
    parameters, trainable = count_parameters(model)
    return {"parameters": parameters, "trainable_parameters": trainable,
            "macs_per_image": count_macs(model),
            "latency": measure_latency(model, device=device),
            "peak_train_memory_bytes": measure_peak_memory(model, device=device),
            "device": str(device)}
