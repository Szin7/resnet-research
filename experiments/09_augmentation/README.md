# Experiment 9 — Data augmentation

## Research question
How much of the test-accuracy result is due to augmentation rather than architecture?

## Hypothesis
**H9:** standard augmentation (pad + crop + flip) narrows the train/test gap and improves test
accuracy by ~1–3 points versus no augmentation, while slowing training-accuracy growth slightly.

## Setup
- Runs: resnet-20 with `standard` augmentation (from the core sweep) and with `minimal`.
- Fixed: model, standard 80-epoch schedule, seed 1.
- Varied: augmentation pipeline.
- Metrics: final train vs test accuracy (the gap), test accuracy, curves.

## Results
_Pending._

## Interpretation
_Pending._

## Limitations
_Pending — one model; augmentation interacts with schedule length._
