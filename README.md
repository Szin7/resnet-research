# ResNet depth study on CIFAR-10

This repository investigates a result from He et al.'s ResNet paper: a deeper plain convolutional network can be *harder to train* than a shallower one. I used the same CIFAR-style architecture at four depths, with and without residual shortcuts, to compare training behavior under one consistent setup.

The primary experiment is finished: eight networks, each trained for 80 epochs on CIFAR-10. The matched plain and ResNet models have the same learnable parameter counts. Source code, model configurations, evaluation scripts, and results are included here.

## What the runs showed

| Depth | Plain train error | ResNet train error | Plain test accuracy | ResNet test accuracy |
|---:|---:|---:|---:|---:|
| 20 | 4.187% | 2.740% | 89.41% | 91.04% |
| 32 | 5.380% | 1.556% | 88.77% | 91.73% |
| 44 | 7.836% | 1.264% | 87.07% | 91.26% |
| 56 | 19.518% | 1.036% | 79.43% | 91.68% |

For the plain networks, training error increased with depth. The residual networks showed the opposite pattern. Plain-56 and ResNet-56 each have **853,018 parameters**, but their final training errors differ by **18.482 percentage points**.

These results come from one training seed. They show the degradation pattern in this setup, not an uncertainty estimate across repeated experiments.

See the [full results](docs/results.md), [research report](docs/research_report.md), and [raw summary table](results/tables/depth_experiment_summary.csv).

## Implementation

- CIFAR ResNet with 20, 32, 44, or 56 weighted layers (the `6n + 2` layout).
- Three stages with 16, 32, and 64 channels.
- Plain and residual variants use the same convolutional layers. Residual option A adds parameter-free shortcuts.
- CIFAR-10 split: 45,000 training images, 5,000 validation images, and the untouched 10,000-image test set.
- SGD with momentum, batch size 128, 80 epochs, and learning-rate drops at epochs 40 and 60.
- Checkpoints, per-epoch CSV logs, and run configuration files saved for each model.

The [architecture notes](docs/architecture.md) describe the blocks and shortcuts in more detail. The 80-epoch schedule and some preprocessing choices differ from the original paper; this is a controlled comparison rather than an exact reproduction of its reported numbers.

## Running the project

Requires Python 3.10+ and PyTorch. A CUDA GPU is useful for the full experiment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
python scripts/prepare_data.py --root data
```

Check the eight planned runs first:

```bash
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --dry-run
```

Train the experiment:

```bash
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --device cuda
```

The runner skips runs with matching completed summaries and resumes an incomplete run from `checkpoints/final.pt`. Outputs go to `results/runs/` unless a different `--runs-dir` is supplied. In Colab, use [the notebook](notebooks/colab_train.ipynb), which writes checkpoints to Google Drive.

To evaluate a run:

```bash
python scripts/evaluate.py --run results/runs/<run-directory> --device cuda
```

## Other analysis

The repository also has scripts to compare layerwise gradient norms, benchmark model computation and latency, and examine class-level errors. The gradient experiment uses separately initialized models trained for five epochs; it is not a gradient trace from the completed 80-epoch checkpoints.

Experiment configurations for shortcut types, BatchNorm, optimizer settings and augmentation are available under [`experiments/`](experiments/), but those additional full-length comparisons have not been reported as completed. See the [experiment guide](docs/experiments.md) and [limitations](docs/limitations.md).

## Project layout

```text
configs/        Model and training settings
src/            Architecture, data, training and analysis code
scripts/        Commands for training, evaluation and diagnostics
experiments/    Experiment definitions
notebooks/      Colab runner
tests/          Automated checks
docs/           Design notes and research write-up
results/        Published result tables
```

## References

- He, Zhang, Ren and Sun, *Deep Residual Learning for Image Recognition*, CVPR 2016.
- He, Zhang, Ren and Sun, *Identity Mappings in Deep Residual Networks*, ECCV 2016.
- Ioffe and Szegedy, *Batch Normalization*, ICML 2015.

MIT License.
