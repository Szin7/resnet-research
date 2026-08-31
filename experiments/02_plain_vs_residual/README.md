# Experiment 2 — Plain vs Residual

## Research question
At matched depth and parameter count, how does adding the residual shortcut change convergence and
final performance?

## Hypothesis
**H2:** at depth ≥ 32 with identical parameter count, the residual model reaches a given training
loss in fewer epochs and ends with higher test accuracy; at depth 20 the gap is small (< 1 point).

## Setup
- Runs: reuse the core sweep (plain-k vs resnet-k for k in {20,32,44,56}).
- Fixed: depth, parameter count (option A), standard schedule, seed 1.
- Varied: additive shortcut present / absent.
- Metrics: train and val loss curves overlaid by family, final test accuracy, epochs-to-threshold.

## Results
_Pending._

## Interpretation
_Pending._

## Limitations
_Pending._
