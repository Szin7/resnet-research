# Limitations

This document distinguishes what was measured from what is planned.

- **Seed coverage:** Eight models were trained with seed 1. The 56-layer plain/residual pair was repeated using seeds 2 and 3; the other depths were not repeated. Three seeds still provide only a limited estimate of variability. Differences of a few hundredths of a percent should not be ranked.
- **Dataset scope:** CIFAR-10 (32×32 images), not ImageNet; results are about these particular models and settings.
- **Schedule:** 80 epochs, with learning-rate milestones at 40 and 60, rather than the original paper's longer training run. No exact paper numerical reproduction is claimed.
- **Implementation deviations:** Per-channel standardization and reflected-padding crop differ from the paper's preprocessing. Comparisons are valid within this implementation; direct comparison to paper percentages requires care.
- **Device:** The original sweep used Colab Tesla T4 with PyTorch 2.11.0+cu130 and Python 3.13.16. Seed-2 ResNet, the final 15 epochs of seed-2 Plain-56, and both seed-3 models ran on Kaggle Tesla T4 with PyTorch 2.11.0+cu128. Checkpoint resumption across environments does not guarantee bitwise trajectory identity.
- **Gradient diagnostic:** Five-epoch analysis trained fresh models and examined a fixed probe batch; it does not measure the completed 80-epoch checkpoints or prove the cause of depth degradation.
- **Latency / memory:** Small iteration counts; hardware- and batch-sensitive numbers. Memory is an allocated-memory peak for a synthetic training step.
- **Planned ablations:** Further seeds and additional shortcut choices, BatchNorm, optimization and augmentation configurations have not yet published measurements. Some no-BatchNorm runs change learning rate, confounding the comparison.
- **Not claimed:** State-of-the-art accuracy, exact replication of the original paper, definitive proof of a single gradient mechanism, or production readiness.
