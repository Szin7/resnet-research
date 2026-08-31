# Experiment 8 — Optimization

## Research question
How sensitive are the results to the learning-rate schedule and to momentum?

## Hypothesis
**H8:** a decaying schedule (step or cosine) beats a constant LR at the same initial value by
> 2 points; cosine ≈ step at 80 epochs; momentum speeds early convergence.

## Setup
- Runs: resnet-20 with {step (baseline, from core sweep), cosine, constant} schedules and one run
  with momentum 0.
- Fixed: model, data, 80 epochs, base LR 0.1.
- Varied: schedule; momentum.
- Metrics: loss curves, final test accuracy, epochs-to-{60,80}% train accuracy.

## Results
_Pending._

## Interpretation
_Pending._

## Limitations
_Pending — one model, one seed; conclusions may not transfer to the deeper models._
