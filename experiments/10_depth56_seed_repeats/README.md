# 56-layer seed repeats

The first depth sweep used training seed 1. Its 56-layer results showed a large training-error gap between the plain and residual networks. This follow-up checks how much that result changes with the training seed.

The existing seed-1 runs are retained. This experiment schedules **four new runs**: plain-56 and ResNet-56 with seeds 2 and 3. It uses the same model configurations, data split (seed 42), 80-epoch schedule, augmentation and optimizer settings.

## Run

From the repository root, review the planned configurations:

```bash
python scripts/run_experiment.py --experiment experiments/10_depth56_seed_repeats/experiment.yaml --dry-run
```

On Colab, use the same Google Drive dataset and run directories as the first experiment:

```bash
python scripts/run_experiment.py \
  --experiment experiments/10_depth56_seed_repeats/experiment.yaml \
  --device cuda --num-workers 2 \
  --data-root /content/drive/MyDrive/resnet-research/data \
  --runs-dir /content/drive/MyDrive/resnet-research/results/runs
```

Completed runs are skipped and interrupted runs use their saved final checkpoints. Don't rename or remove the original seed-1 directories.

After training, evaluate each **new** completed run on CIFAR-10's official test split using `scripts/evaluate.py`. Then run `scripts/report_seed_repeats.py` to summarize seeds 1, 2 and 3 for each model.

## What to report

Compare final training error first; that is the measurement most directly related to the degradation question. Report mean and sample standard deviation across the three seeds, with the individual values alongside them. Test accuracy should also be summarized, but a three-seed sample is still limited.

**Status:** prepared only. No new seed-2 or seed-3 measurements are included in this directory.
