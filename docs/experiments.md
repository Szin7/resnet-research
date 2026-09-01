# Experiments — index and how to reproduce

_Status: outline. Each row links to the experiment's protocol; results are filled from Phase 12 on._

| # | Folder | Question | Status |
|---|--------|----------|--------|
| 1 | `experiments/01_depth/` | Does depth always help? Plain-net training-error degradation? | infra ready, runs pending |
| 2 | `experiments/02_plain_vs_residual/` | Shortcut effect at matched depth/params | pending |
| 3 | `experiments/03_residual_ablation/` | Remove the shortcut | pending |
| 4 | `experiments/04_identity_vs_projection/` | Shortcut type A/B/C | pending |
| 5 | `experiments/05_gradient_flow/` | Per-layer gradient norms | pending |
| 6 | `experiments/06_ablation_grid/` | residual × depth × BN | pending |
| 7 | `experiments/07_batchnorm/` | BatchNorm's contribution | pending |
| 8 | `experiments/08_optimization/` | LR schedule / momentum | pending |
| 9 | `experiments/09_augmentation/` | Augmentation effect | pending |
| 10 | `docs/error_analysis.md` | Error analysis of the best model | pending |

Reproduction:

```bash
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --device cuda
python scripts/evaluate.py --run results/runs/resnet-56_seed1 --device cuda
python scripts/make_figures.py --experiment experiments/01_depth/experiment.yaml
```

Finished runs (those with a `summary.json`) are skipped on re-run. Every run's resolved config,
environment, and per-epoch metrics are saved under `results/runs/<run_name>/`; figures land in
`experiments/<n>_<name>/figures/`. `notebooks/colab_train.ipynb` wraps all of this for Colab.
