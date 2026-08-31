# Gradient-flow analysis

_Status: outline. Filled in Phase 14 from Experiment 5 measurements._

Method is in [mathematics.md](mathematics.md) §5. Protocol is in
`experiments/05_gradient_flow/README.md`.

Planned contents:

- What is measured: L2 norm of each conv/linear weight gradient, on a fixed probe batch, fp32, at
  initialization / epoch 1 / epoch 5, for plain-56 and resnet-56.
- Figure: gradient norm vs layer depth, plain vs residual, one line per checkpoint.
- Figure: aggregate (mean/median/max across layers) over training.
- Supplement: residual-branch output magnitudes vs depth (paper Fig. 7 analogue).
- Interpretation, with the §5 caveats stated up front: if plain-56 gradients are not vanishing, the
  conclusion is that degradation is a landscape/conditioning problem, not a gradient-magnitude one.
- What cannot be concluded from this measurement.
