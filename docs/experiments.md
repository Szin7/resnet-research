# Running and interpreting the experiments

The main depth comparison is finished. Eight networks were trained for 80 epochs: plain and residual versions at depths 20, 32, 44 and 56. Results are in [results.md](results.md).

## How the comparisons are controlled

The CIFAR-10 train/validation split is fixed at 45,000/5,000 images, with seed 42. The primary sweep uses model seed 1 and the same optimizer and training schedule. Plain/residual pairs at the same depth have equal learnable parameter counts.

For the degradation question, I compare training error before looking at validation or test accuracy. A model can have worse test accuracy for several reasons; higher *training* error in a deeper plain model is the more relevant observation here. The 10,000 test images are reserved for the final evaluation.

## Available studies

| Folder | Comparison | Status |
|---|---|---|
| `01_depth` | Plain vs residual networks at four depths | Complete: 8 full runs |
| `02_plain_vs_residual` | Same depth, shortcuts on or off | Covered by main runs; separate presentation pending |
| `03_residual_ablation` | Shortcut removed at 56 layers | Configured; original plain/residual runs available |
| `04_identity_vs_projection` | Shortcut options A, B and C | Additional runs required |
| `05_gradient_flow` | Gradient norms during a short training probe | Five-epoch diagnostic completed |
| `06_ablation_grid` | Depth, shortcuts and BatchNorm | Additional runs required |
| `07_batchnorm` | With and without BatchNorm | Additional runs required; LR also changes |
| `08_optimization` | Learning-rate schedule and momentum | Additional runs required |
| `09_augmentation` | Standard vs minimal augmentation | Additional runs required |

These are experiment definitions, not a claim that every planned study has been run.

## Commands

Review a planned experiment before starting a GPU session:

```bash
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --dry-run
```

Run the main experiment:

```bash
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --device cuda
```

The runner skips completed runs with matching configurations and resumes unfinished ones from saved checkpoints. Run folders and configurations are kept separate so new ablations do not overwrite the primary results.

The gradient diagnostic trains fresh models for five epochs:

```bash
python scripts/analyze_gradients.py --models plain_56 resnet_56 --epochs 5 --device cuda
```

Benchmark the selected model architectures:

```bash
python scripts/benchmark.py --models plain_20 resnet_20 plain_56 resnet_56 --device cuda
```

Any additional study should report the full settings, seed, final training error, validation results, held-out test results and limitations. See [the limitations note](limitations.md) for points that still need independent repeats.
