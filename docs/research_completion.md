# Project status

I finished the main CIFAR-10 study with eight networks: plain and residual models at 20, 32, 44 and 56 layers. All eight completed 80 epochs with seed 1.

The main finding is in training error. Plain-56 finished at 19.518% while ResNet-56 finished at 1.036%, despite having the same 853,018 learned parameters. Their held-out test accuracies were 79.43% and 91.68% respectively. The [results table](../results/tables/depth_experiment_summary.csv) and [research report](research_report.md) contain the other depths.

## What is in the repository

- CIFAR ResNet and parameter-matched plain model code
- Training, evaluation, checkpoint resume and analysis scripts
- The fixed experiment configurations and offline tests
- Measured accuracy tables, two summary figures, gradient and benchmark summaries
- Design notes, limitations and the research report

The four original training-history PNGs and full per-epoch histories are saved in the research archive on Google Drive. Large model checkpoints and dataset files are not committed.

## What happened during training

An original ResNet-56 run was interrupted by a Colab GPU limit after epoch 31. The checkpoint loader initially failed to restore the saved CUDA RNG state; that issue was fixed and training resumed from epoch 32. The model completed all 80 epochs.

A later Plain-56 seed-2 run stopped at epoch 65 on Colab. I transferred the checkpoint and metrics to Kaggle, resumed from epoch 66, and completed 80 epochs. I also trained ResNet-56 seed 2 and both depth-56 seed-3 models for 80 epochs on Kaggle. All four new runs were evaluated on the held-out test set and backed up outside the repository. See the [three-seed comparison](depth56_seed_comparison.md).

## Quick verification

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --dry-run
```

No GPU is needed to preview the experiment matrix or run the offline tests. To reproduce the published measurements, the full 80-epoch runs and held-out evaluations must be performed using the documented setup.

The depth-56 comparison now has three completed training seeds for both models. The other depths were not repeated; BatchNorm, optimizer, augmentation and shortcut-variant experiments remain possible follow-ups.
