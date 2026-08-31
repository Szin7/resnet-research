# Experiment 3 — Remove the shortcut

## Research question
Is the additive shortcut itself the causal factor behind the residual network's behaviour, with
everything else held identical?

## Hypothesis
**H3:** removing only the additive skip from resnet-56 (yielding an architecturally identical
plain-56) increases final training loss and decreases test accuracy. Because nothing else changes,
this isolates the shortcut. A null result (no difference) would itself be a notable finding.

## Setup
- Runs: resnet-56 and plain-56 from the core sweep, plus one confirmation run of resnet-56 with the
  additive term forced to zero at runtime (should match plain-56 within noise).
- Fixed: architecture, parameter count, standard schedule, seed 1.
- Varied: the `out = out + shortcut(x)` term only.
- Metrics: Δ final training loss, Δ test accuracy, side-by-side loss curves.

## Results
_Pending._

## Interpretation
_Pending._

## Limitations
_Pending — one depth, one seed; "identical except the add" depends on the option-A design._
