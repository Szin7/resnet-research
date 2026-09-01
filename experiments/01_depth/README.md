# Experiment 1 — Depth

## Research question
Does increasing depth always improve optimization and validation performance? Do deeper *plain*
networks show higher *training* error (an optimization failure), as opposed to just worse test error
(overfitting)?

## Hypothesis
- **H1:** plain-net test accuracy plateaus then declines across depth 20 → 32 → 44 → 56, and
  plain-56 shows higher training error than a shallower plain net.
- **H1':** ResNet test accuracy improves or holds flat across the same range with no training-error
  degradation.
- Fallback if degradation is not seen at 56: extend to plain-74 / plain-110 and report that it did
  not appear at 56 under the 80-epoch schedule.

## Setup
- Runs: the core sweep — plain-{20,32,44,56} and resnet-{20,32,44,56}, shortcut option A.
- Fixed: standard 80-epoch schedule, seed 1, `standard` augmentation, BN on.
- Varied: depth, and residual on/off.
- Metrics: final train / val / test accuracy and loss, training-loss curves, epochs-to-{60,80}% train accuracy, parameter count.

## How to run

```bash
# ~3 h on a Colab T4; finished runs are skipped so it is safe to re-run after a disconnect
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --device cuda

# then evaluate each run on the held-out test set
for run in results/runs/plain-* results/runs/resnet-*; do
    python scripts/evaluate.py --run "$run" --device cuda
done
```

Or use `notebooks/colab_train.ipynb`. Figures land in `experiments/01_depth/figures/`; the run
config, environment, and per-epoch metrics for each model land in `results/runs/<name>/`.

## Results
_Pending._

## Interpretation
_Pending._

## Limitations
_Pending — at minimum: single seed (variance runs quantify noise separately); 80-epoch schedule may
under-train the deepest models; CIFAR-10 only._
