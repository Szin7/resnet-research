# Experiment 4 — Identity vs projection shortcut

## Research question
When is a projection shortcut mechanically necessary, and does the choice of shortcut type affect
accuracy?

## Hypothesis
**H4:** options B and C land within ~1 point of option A; using projections everywhere (C) adds
parameters and compute without a reliable accuracy gain. Projections are *required* only where
channel count or spatial size changes (stage 2 and stage 3 entry blocks).

## Setup
- Runs: resnet-56 with shortcut A (from the core sweep), B, and C.
- Fixed: depth 56, standard schedule, seed 1.
- Varied: shortcut option.
- Metrics: parameter count, MACs, test accuracy, training time; a tensor-shape walkthrough of both
  stage transitions.

## Results
_Pending._

## Interpretation
_Pending._

## Limitations
_Pending — option C's extra parameters make the comparison not perfectly capacity-matched; documented._
