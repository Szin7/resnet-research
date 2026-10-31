# Limitations

This document distinguishes what was measured from what is planned.

- **Single-seed depth sweep:** Eight complete models used seed 1 and a fixed split seed 42; no independent-seed uncertainty interval has yet been published. Differences of a few hundredths of a percent should not be ranked.
- **Dataset scope:** CIFAR-10 (32×32 images), not ImageNet; results are about these particular models and settings.
- **Schedule:** 80 epochs, with learning-rate milestones at 40 and 60, rather than the original paper's longer training run. No exact paper numerical reproduction is claimed.
- **Implementation deviations:** Per-channel standardization and reflected-padding crop differ from the paper's preprocessing. Comparisons are valid within this implementation; direct comparison to paper percentages requires care.
- **Device:** Single Colab Tesla T4, mixed precision in the main sweep; PyTorch 2.11.0+cu130, Python 3.13.16. Mid-run checkpoint resumption does not guarantee bitwise trajectory identity.
- **Gradient diagnostic:** Five-epoch analysis trained fresh models and examined a fixed probe batch; it does not measure the completed 80-epoch checkpoints or prove the cause of depth degradation.
- **Latency / memory:** Small iteration counts; hardware- and batch-sensitive numbers. Memory is an allocated-memory peak for a synthetic training step.
- **Planned ablations:** Additional seeds, shortcut choices, BatchNorm, optimization and augmentation have configurations but not yet published measurements. Some no-BatchNorm runs change learning rate, confounding the comparison.
- **Not claimed:** State-of-the-art accuracy, exact replication of the original paper, definitive proof of a single gradient mechanism, or production readiness.
