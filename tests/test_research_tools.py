"""CPU checks for supplementary research tools."""
import numpy as np
import torch
from resnet_research.analysis.benchmark import count_macs, count_parameters, measure_latency, measure_peak_memory
from resnet_research.analysis.interpret import extract_feature_maps, grad_cam
from resnet_research.evaluation.error_analysis import collect_misclassified, plot_misclassification_gallery
from resnet_research.models.resnet import cifar_resnet


def test_model_benchmark_cpu():
    model = cifar_resnet(8)
    total, trainable = count_parameters(model)
    assert total == trainable and total > 0
    assert count_macs(model) > 0
    assert measure_peak_memory(model, device=torch.device("cpu")) is None
    timing = measure_latency(model, device=torch.device("cpu"), batch_sizes=(1,), warmup=0, iters=2)
    assert timing[1]["mean_ms"] > 0


def test_feature_maps_and_gradcam():
    model = cifar_resnet(8)
    image = torch.randn(3, 32, 32)
    features = extract_feature_maps(model, image, layers=["stage3"])
    assert features["stage3"].shape[-2:] == (8, 8)
    heat = grad_cam(model, image, target_layer="stage3")
    assert heat.shape == (32, 32) and np.isfinite(heat).all()


def test_misclassification_gallery(tmp_path):
    images = torch.rand(6, 3, 32, 32)
    labels = torch.full((6,), 9)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=2)
    class PredictZero(torch.nn.Module):
        def forward(self, x):
            result = torch.zeros(x.size(0), 10, device=x.device)
            result[:, 0] = 10
            return result
    records = collect_misclassified(PredictZero(), loader, torch.device("cpu"), limit=2)
    assert len(records) == 2
    target = tmp_path / "gallery.png"
    plot_misclassification_gallery(records, [str(i) for i in range(10)], path=target)
    assert target.exists()
