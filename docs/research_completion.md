# Research completion notes

The project implements CIFAR-sized residual networks and their parameter-matched plain-network controls. The next step is to run and evaluate the comparisons, not to assume the expected findings.

## Current status

- Model code, data preprocessing, training, checkpointing and evaluation are implemented.
- Training state and experiment directory handling have been strengthened.
- The depth comparison is defined in `experiments/01_depth/experiment.yaml`.
- No measured CIFAR-10 performance numbers are included in this branch until the experiments run.

## Quick checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --dry-run
```

## First experiment

Use a CUDA environment for the full comparison:

```bash
python scripts/prepare_data.py --root data
python scripts/run_experiment.py --experiment experiments/01_depth/experiment.yaml --device cuda
```

On completion, evaluate each run with `scripts/evaluate.py --run <run-directory>` and record measured training loss, validation accuracy and final test accuracy. A comparison is only ready for the research report after its metrics and configuration have been checked.

## Method

Compare plain and residual networks at matched depth under the same data split, initialization policy, optimization recipe and seed. Explain any deviations from the original paper in the results. If a hypothesis is not supported by the measured runs, report the result rather than changing the hypothesis afterwards.
