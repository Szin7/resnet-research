# ResNet From Scratch: A Research Reproduction

A from-scratch PyTorch reproduction and experimental study of the residual-learning experiments in
**He, Zhang, Ren, Sun — _Deep Residual Learning for Image Recognition_** (CVPR 2016,
[arXiv:1512.03385](https://arxiv.org/abs/1512.03385)), run on CIFAR-10 under a single-GPU / Google
Colab compute budget.

> **Status: in development.** The implementation and experiments are being built in phases (see
> `docs/`). **No experimental results are reported yet.** Every section marked _pending_ will be
> filled with measured numbers only — never estimates.

```mermaid
flowchart LR
    A[Research paper] --> B[From-scratch implementation]
    B --> C[Plain CNN baselines]
    C --> D[Controlled experiments]
    D --> E[Shortcut ablation]
    E --> F[Gradient-flow analysis]
    F --> G[Error analysis]
    G --> H[Conclusions]
```

## Motivation

The ResNet paper makes a specific, testable claim: past a certain depth, plain convolutional
networks become *harder to optimize* (higher training error, not just overfitting), and adding an
identity shortcut around each pair of layers fixes this. This project reproduces that claim at a
scale that fits a laptop/Colab, implements the architecture by hand, and runs controlled experiments
to see how far the claim holds and what the shortcut actually contributes.

## Research question

> How do residual (skip) connections affect the optimization and generalization of deep
> convolutional networks on CIFAR-10, and does their benefit grow with depth?

Sub-questions and the experiments that address them are listed in
[docs/project_specification.md](docs/project_specification.md).

## Paper

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition.*
CVPR 2016. Reading notes: [docs/paper_notes.md](docs/paper_notes.md).

This is a **lightweight reproduction of the CIFAR-10 experiments (paper Section 4.2)**, not the
ImageNet experiments. Differences from the original are tabulated in
[docs/project_specification.md](docs/project_specification.md) §11.

## What I implemented

By hand (no `torchvision.models`, no pretrained weights):

- [ ] CIFAR ResNet / plain networks, `6n + 2` layers (Phase 7–8)
- [ ] Basic residual block with identity (option A) and projection (options B/C) shortcuts (Phase 7)
- [ ] `SmallCNN` baselines (Phase 6)
- [ ] CIFAR-10 pipeline with the frozen 45k/5k split and the paper's augmentation (Phase 5)
- [ ] Training loop, LR schedules, checkpointing, seeded runs (Phase 10)
- [ ] Per-layer gradient-norm measurement (Phase 14)
- [ ] Confusion matrix + per-class + misclassification analysis (Phase 17)
- [ ] Parameter / MAC / latency / memory benchmarking (Phase 18)

## Architecture

Three-stage CIFAR ResNet (16 → 32 → 64 channels over 32² → 16² → 8² feature maps), basic blocks,
stride-2 downsampling, global average pooling, linear classifier. Full spec and per-model parameter
counts: [docs/project_specification.md](docs/project_specification.md) §4. Explanation and diagrams:
`docs/architecture.md` (pending).

## Dataset

CIFAR-10 (50k train / 10k test; we hold out 5k of train for validation with a fixed split seed).
Downloaded on demand via `torchvision`. Per-channel standardization computed from the training
split.

## Experimental setup

Frozen in [docs/project_specification.md](docs/project_specification.md): "standard" 80-epoch
schedule, SGD (momentum 0.9, weight decay 1e-4), batch 128, base LR 0.1 with step decay at epochs
40 and 60. Runs on Colab (T4) with mixed precision, except gradient-flow and BatchNorm experiments
which run in fp32.

## Baselines

`SmallCNN` (conventional conv-pool-FC design) and `SmallCNN-deep` provide a non-residual reference
point. The scientifically controlled comparison is the parameter-matched **plain-_k_ vs resnet-_k_**
pairs (identical depth, width, and parameter count under shortcut option A).

## Experiments

| # | Question | Status |
|---|----------|--------|
| 1 | Does depth always help? Do plain nets show training-error degradation? | pending |
| 2 | Plain vs residual at matched depth and parameters | pending |
| 3 | Remove the shortcut — what changes? | pending |
| 4 | Identity vs projection shortcuts | pending |
| 5 | Per-layer gradient flow, plain vs residual | pending |
| 6 | Ablation grid: residual × depth × BatchNorm | pending |
| 7 | BatchNorm's contribution | pending |
| 8 | LR schedule / momentum sensitivity | pending |
| 9 | Data augmentation effect | pending |
| 10 | Error analysis of the best model | pending |

## Results

_Pending._

## Gradient-flow analysis

_Pending._ Method: `docs/mathematics.md` §5.

## Error analysis

_Pending._ Will produce `docs/error_analysis.md`.

## Limitations

Tracked in `docs/limitations.md` as they are identified. Headline constraints: CIFAR-10 not ImageNet,
depths ≤ 56 (110 optional), an 80-epoch schedule rather than the paper's ~182, single-GPU compute.

## Reproducibility

Every run writes its resolved config, environment (`env.json`), per-epoch `metrics.csv`, and
`summary.json` to `results/runs/<run_name>/`. Seeds are set for Python / NumPy / torch; determinism
flags are enabled where practical. Run-to-run variation on GPU is quantified with repeated seeds.
Details: [docs/project_specification.md](docs/project_specification.md) §10.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

_Scripts under `scripts/` are stubs until Phase 10._ Intended entry points:

```bash
python scripts/train.py --config configs/models/resnet_56.yaml --schedule configs/schedules/standard.yaml
python scripts/evaluate.py --run results/runs/resnet-56_seed1
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml
```

## Repository layout

```
configs/       composable YAML fragments (default + per-model + per-schedule)
src/resnet_research/
  models/      blocks, resnet, baselines
  data/        cifar-10 loading, transforms
  training/    trainer, schedulers, losses
  evaluation/  metrics, confusion matrix, error analysis
  analysis/    gradient flow, benchmarking, interpretability
  utils/       seed, checkpoint, config, logging
scripts/       thin CLI entry points
experiments/   one folder per experiment: protocol + results + figures
results/       committed metrics, figures, tables (checkpoints are gitignored)
tests/         unit tests + sanity checks
docs/          paper notes, math, spec, architecture, analyses, interview prep
```

## References

- He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- He et al. *Identity Mappings in Deep Residual Networks.* ECCV 2016.
- Ioffe & Szegedy. *Batch Normalization.* ICML 2015.
- He et al. *Delving Deep into Rectifiers.* ICCV 2015.

Full list with context in [docs/mathematics.md](docs/mathematics.md).
