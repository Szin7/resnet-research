# Depth-56 seed repeats

The original comparison used seed 1 at all four depths. I repeated Plain-56 and ResNet-56 with seeds 2 and 3 to check whether the training gap persisted.

All four additional runs are **complete (80 epochs)** and have been evaluated on the official CIFAR-10 test set.

| Network | Seed 2 final training error | Seed 2 best-validation checkpoint test accuracy |
|---|---:|---:|
| Plain-56 | 14.8067% | 82.65% |
| ResNet-56 | 0.9378% | 91.90% |

Seed 3 reached **15.0200% training error / 82.62% test accuracy** for Plain-56 and **1.3111% / 91.43%** for ResNet-56.

The matched seed-2 accuracy difference was 9.25 percentage points in favor of the ResNet. Seed 1 showed a difference of 12.25 points. Individual results, three-seed means and caveats are in the [three-seed report](../../docs/depth56_seed_comparison.md) and [CSV](../../results/tables/depth56_seed_repeats.csv).

## How the runs were completed

Plain-56 seed 2 began on Colab and stopped after epoch 65 because GPU access ran out. I moved the saved checkpoint to Kaggle and finished epochs 66–80. ResNet-56 seed 2 and both seed-3 models were trained entirely on Kaggle. The complete checkpoints, logs, and evaluation outputs were backed up separately from GitHub.

## Running the same setup

The smaller configuration runs seed 2 for both models, without starting seed 3:

```bash
python scripts/run_experiment.py --experiment experiments/10_depth56_seed_repeats/seed2_only.yaml --dry-run
python scripts/run_experiment.py --experiment experiments/10_depth56_seed_repeats/seed2_only.yaml --device cuda
```

The full [experiment.yaml](experiment.yaml) also includes **seed 3**, which has now completed. The runner skips completed configurations and resumes from a matching saved `final.pt`. For a resumed run across machines, the data-root path must be updated in the local copy of the saved configuration; keep the other settings identical.

To regenerate the three-seed summary with **all six completed run folders available locally**:

```bash
python scripts/report_seed_repeats.py --seeds 1 2 3 --runs-dir results/runs
```

This report uses `test_metrics.json` from the best-validation checkpoint and the last epoch's `metrics.csv` row. It does not train models. The present report is limited to three seeds; more repeats would be needed for reliable uncertainty estimates.
