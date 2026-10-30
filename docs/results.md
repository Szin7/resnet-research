# Experimental results

**No completed full CIFAR-10 measurements are published in this repository yet.**

Plots and summaries should be generated from actual run artifacts under `results/runs/`. Smoke tests and synthetic test fixtures are not evidence of model accuracy on CIFAR-10.

## Primary comparison

| Network | Depth | Train error | Validation accuracy | Held-out test accuracy | Seed |
|---|---:|---:|---:|---:|---:|
| Plain | 20 | pending | pending | pending | 1 |
| ResNet | 20 | pending | pending | pending | 1 |
| Plain | 32 | pending | pending | pending | 1 |
| ResNet | 32 | pending | pending | pending | 1 |
| Plain | 44 | pending | pending | pending | 1 |
| ResNet | 44 | pending | pending | pending | 1 |
| Plain | 56 | pending | pending | pending | 1 |
| ResNet | 56 | pending | pending | pending | 1 |

The central question is whether deeper **plain** networks show higher *training error*, and whether additive shortcuts change that trend.

## Additional analyses

- Gradient norm profiles for plain-56 and ResNet-56 are generated with `scripts/analyze_gradients.py`.
- Parameter counts, MACs, inference latency and peak memory are generated with `scripts/benchmark.py`.
- Error statistics, a confusion matrix and an optional misclassification gallery are produced using `scripts/evaluate.py`.
- Variance across independent training seeds must be quantified before making claims about small improvements.

## Reporting rules

Use the saved configuration and environment metadata to identify each run. Report train and validation curves before drawing conclusions about optimization. Keep final test accuracy separate from model-selection metrics. Distinguish deviations from the original paper's schedule and preprocessing.

No values should be entered from memory or estimated from the paper's figures.
