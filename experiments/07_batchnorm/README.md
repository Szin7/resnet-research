# Experiment 7 — BatchNorm

## Research question
How much of the training stability attributed to the residual design actually comes from
BatchNorm?

## Hypothesis
**H7:** without BN, LR 0.1 is unstable; a lower LR is needed, convergence is slower, and final
accuracy is lower. BN's clearest observable effect is enabling a larger stable learning rate.
Removing BN also changes regularization, so the comparison is not perfectly clean.

## Setup
- Runs: resnet-20 and plain-20, each with BN (from the core sweep) and without (**AMP off**), plus a
  short LR sweep to find the largest stable rate for the no-BN models.
- Fixed: depth 20, data, schedule length.
- Varied: BN on/off, and the LR change it forces.
- Metrics: largest stable LR, convergence curves, final test accuracy.

## Results
_Pending._

## Interpretation
_Pending._

## Limitations
_Pending — no-BN runs require an LR change, confounding the comparison; stated plainly._
