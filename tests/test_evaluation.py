"""Confusion matrix, per-class metrics, evaluation (Phases 10 / 12 / 17)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from resnet_research.evaluation.confusion_matrix import (
    compute_confusion_matrix,
    most_confused_pairs,
    plot_confusion_matrix,
)
from resnet_research.evaluation.metrics import epochs_to_threshold, evaluate_model, per_class_metrics
from resnet_research.models.resnet import cifar_resnet


def test_per_class_metrics_from_confusion():
    cm = np.array([[8, 2], [1, 9]])
    metrics = per_class_metrics(cm, ("a", "b"))
    assert metrics["a"]["recall"] == 8 / 10
    assert metrics["a"]["precision"] == 8 / 9
    assert metrics["a"]["support"] == 10
    assert 0 < metrics["a"]["f1"] < 1


def test_most_confused_pairs_ordering():
    cm = np.array([[10, 3, 1], [0, 10, 5], [2, 0, 10]])
    pairs = most_confused_pairs(cm, ("a", "b", "c"), top=2)
    assert pairs == [("b", "c", 5), ("a", "b", 3)]


def test_compute_confusion_matrix_totals():
    images = torch.randn(24, 3, 32, 32)
    labels = torch.randint(0, 10, (24,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    cm = compute_confusion_matrix(cifar_resnet(8), loader, torch.device("cpu"))
    assert cm.shape == (10, 10)
    assert cm.sum() == 24


def test_evaluate_model_returns_loss_and_accuracy():
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    result = evaluate_model(cifar_resnet(8), loader, torch.device("cpu"))
    assert set(result) == {"loss", "accuracy"}
    assert 0.0 <= result["accuracy"] <= 1.0


def test_plot_confusion_matrix_writes_file(tmp_path):
    cm = np.eye(10, dtype=int) * 5 + 1
    plot_confusion_matrix(cm, tuple("abcdefghij"), path=tmp_path / "cm.png")
    assert (tmp_path / "cm.png").stat().st_size > 0


def test_epochs_to_threshold():
    assert epochs_to_threshold([0.2, 0.5, 0.65, 0.9], 0.6) == 3
    assert epochs_to_threshold([0.2, 0.3], 0.6) is None
