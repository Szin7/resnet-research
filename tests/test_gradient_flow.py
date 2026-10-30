"""Check gradient instrumentation on a small synthetic batch."""
import json
import numpy as np
import torch
from resnet_research.analysis.gradient_flow import GradientFlowRecorder, probe_gradient_flow
from resnet_research.models.resnet import cifar_resnet


def test_probe_returns_norm_per_weighted_layer(tiny_batch):
    model = cifar_resnet(8)
    values = probe_gradient_flow(model, tiny_batch, device=torch.device("cpu"))
    expected = sum(isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)) for m in model.modules())
    assert len(values) == expected
    assert all(np.isfinite(value) for value in values.values())


def test_recorder_snapshots_accumulate(tmp_path):
    model = cifar_resnet(8)
    model(torch.randn(8, 3, 32, 32)).sum().backward()
    recorder = GradientFlowRecorder(model)
    for step in range(3):
        recorder.snapshot(step)
    assert len(recorder.records()) == 3
    path = tmp_path / "norms.json"
    recorder.save(path)
    assert len(json.loads(path.read_text())) == 3
