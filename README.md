# ResNet from Scratch — CIFAR-10 Research Study

This project examines a question raised by the original ResNet paper: why can adding layers to a plain convolutional network make it harder to optimize, and what changes when those layers have identity shortcuts?

The study implements CIFAR-sized residual networks directly in PyTorch. A plain-network switch removes the shortcut while preserving the weighted layers, making the two families comparable at the same depth. The main experiments use CIFAR-10, a fixed training/validation split, and controlled optimizer settings.

**Current state:** the primary eight-model CIFAR-10 depth sweep (plain and residual networks, depths 20/32/44/56) has finished 80 epochs per model and been evaluated on the held-out test set. Measured results are in [docs/results.md](docs/results.md) and [the summary CSV](results/tables/depth_experiment_summary.csv). Additional-seed replication and further ablations remain pending. At depth 56 the plain network had 19.518% final training error versus 1.036% for its parameter-matched residual counterpart; held-out test accuracy was 79.43% versus 91.68%. These are single-seed findings, not confidence intervals.

## Research question

How do residual shortcuts affect training behavior and generalization as convolutional networks become deeper?

The first comparison uses plain and residual networks with 20, 32, 44, and 56 weighted layers. Further experiments test shortcut variants, BatchNorm, optimization settings, and augmentation. The experiment descriptions document the hypotheses and limitations before results are collected.

## Architecture

The CIFAR ResNet follows the 6n + 2 arrangement described by He et al. Three stages use 16, 32, and 64 channels, with spatial resolutions 32x32, 16x16, and 8x8. Each basic block contains two 3x3 convolutions. The main comparison uses parameter-free option-A shortcuts, keeping plain and residual parameter counts equal.

Implementation: [models](src/resnet_research/models/) · [architecture notes](docs/architecture.md)

## Data and training

- CIFAR-10: 45,000 training images, 5,000 validation images, and the original 10,000 test images.
- The validation split is fixed using seed 42. Normalization statistics use the training split only.
- Standard augmentation uses random crop and horizontal flip. Training uses SGD with momentum and a documented learning-rate schedule.
- Every completed run records its resolved configuration, environment, per-epoch metrics, and checkpoints. Its directory identity includes a hash of its research settings so different ablations do not overwrite each other.
- Model selection uses validation accuracy; test evaluation is a separate step.

## Setup

Python 3.10+ is required. A CUDA GPU is recommended for the full comparison.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
python scripts/prepare_data.py --root data
```

## Run the first study

Start by reviewing the eight configurations without training:

```bash
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --dry-run
```

Run the full comparison on a CUDA-capable machine:

```bash
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --device cuda
```

The generated run folders appear under `results/runs/`. Evaluate each completed run with:

```bash
python scripts/evaluate.py --run results/runs/<run-directory> --device cuda
```

The experiment runner skips finished runs and resumes unfinished runs when a compatible checkpoint is available. For a quick wiring check, pass `--subset-size 120` and use the smoke schedule. Subset results are not research evidence.

## Experiments

| Study | Configuration | What it examines |
|---|---|---|
| Depth sweep | `experiments/01_depth/` | Training error as depth increases |
| Plain vs residual | `experiments/02_plain_vs_residual/` | Effect of additive shortcuts |
| Shortcut removal | `experiments/03_residual_ablation/` | Controlled removal of a skip connection |
| Shortcut choice | `experiments/04_identity_vs_projection/` | Identity, projection at transitions, projection everywhere |
| Gradient flow | `experiments/05_gradient_flow/` | Weight-gradient norms on a fixed probe batch |
| Ablation grid | `experiments/06_ablation_grid/` | Residual connections, depth, BatchNorm |
| BatchNorm | `experiments/07_batchnorm/` | Normalization and learning-rate adjustment |
| Optimization | `experiments/08_optimization/` | Schedule and momentum |
| Augmentation | `experiments/09_augmentation/` | Standard augmentation versus none |

Each experiment YAML can be reviewed using `scripts/run_experiment.py --experiment <path> --dry-run`. Gradient measurements use `scripts/analyze_gradients.py`; computational measurements use `scripts/benchmark.py`.

## Understanding the outputs

The central plots compare training error and validation accuracy across depth, alongside epoch-by-epoch training curves. The project also supports confusion matrices, per-class statistics, misclassified-image samples, gradient profiles, and model-compute benchmarks.

See [results](docs/results.md), [experiment protocols](docs/experiments.md), [research report](docs/research_report.md), and [known limitations](docs/limitations.md).

**Research integrity:** The aim is not to obtain a particular accuracy value. An experiment that contradicts its original hypothesis still belongs in the report. Differences from the original paper's training recipe are documented and any reported measurements must come from actual runs.

## Repository structure

```text
configs/       Model and training configurations
src/           Models, data, training, evaluation, diagnostics
scripts/       Training and analysis entry points
experiments/   Research protocols and experiment matrices
notebooks/     Colab runner
tests/         Unit and integration checks
docs/          Paper notes, architecture, methodology, reporting
results/       Metrics and figures produced by completed experiments
```

## References

- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- K. He, X. Zhang, S. Ren, J. Sun. *Identity Mappings in Deep Residual Networks.* ECCV 2016.
- S. Ioffe, C. Szegedy. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML 2015.

Released under the MIT License.
