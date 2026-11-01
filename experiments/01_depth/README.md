# Depth comparison — completed

The main experiment compares plain and residual CIFAR networks at 20, 32, 44 and 56 layers. All eight models finished 80 epochs with seed 1. The models at each depth have equal numbers of learned parameters.

| Layers | Plain final train error | ResNet final train error | Plain test accuracy | ResNet test accuracy |
|---:|---:|---:|---:|---:|
| 20 | 4.187% | 2.740% | 89.41% | 91.04% |
| 32 | 5.380% | 1.556% | 88.77% | 91.73% |
| 44 | 7.836% | 1.264% | 87.07% | 91.26% |
| 56 | 19.518% | 1.036% | 79.43% | 91.68% |

Training error rose as the plain models got deeper, but fell for the ResNets. The depth-56 comparison shows the clearest difference. See [the full results](../../docs/results.md).

## Reproduce the setup

```bash
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --dry-run
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --device cuda
```

The data split, optimizer, augmentation and schedule are shared across the eight runs. The runner saves per-epoch logs and checkpoints; completed runs are skipped if the configuration matches. The original training-history figures can be regenerated with `scripts/make_figures.py` from the run logs.

**Limitations:** One seed and an 80-epoch schedule on CIFAR-10. These results support the degradation pattern in this setup, not a statistically repeated estimate.
