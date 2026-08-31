# Experiment 5 — Gradient flow

## Research question
Do residual connections change how gradient magnitude is distributed across layers, early in
training?

## Hypothesis
**H5:** the per-layer gradient-norm profile is flatter across depth for resnet-56 than for plain-56.
**Careful caveat (docs/mathematics.md §5):** with BatchNorm, plain-56 gradients are likely *not*
vanishing — the paper measured healthy norms in plain nets. If the profiles look similar, that
supports the view that the degradation problem is about the optimization landscape, not gradient
magnitude, and that is the result we report.

## Setup
- Runs: short instrumented training of plain-56 and resnet-56 (~5 epochs, **AMP off**, fp32),
  recording the L2 norm of every conv/linear weight gradient on a fixed probe batch at
  initialization, epoch 1, and epoch 5.
- Fixed: schedule, data, probe batch, seed.
- Varied: family.
- Metrics: per-layer gradient norm vs depth; aggregate mean / median / max across layers; evolution
  over the three checkpoints. Supplement: residual-branch output magnitudes (paper Fig. 7 analogue).

## Results
_Pending._

## Interpretation
_Pending._

## Limitations
_Pending — one depth, one seed, first 5 epochs only; weight-gradient norms, not input-gradient norms
(a second view may be added)._
