# Experiment 6 — Ablation grid

## Research question
How do residual connections, depth, and BatchNorm combine to determine trainability and test
accuracy?

## Hypothesis
**H6:** ordering by test accuracy is deep-residual ≥ shallow-residual ≈ shallow-plain > deep-plain
(assuming H1 holds). BatchNorm removal hurts most at greater depth.

## Setup
- Runs: mostly reuse (resnet/plain × {20, 56} from the core sweep); add plain-56 without BN if not
  already covered by Experiment 7.
- Fixed: standard schedule (LR retuned where BN is removed — noted), seed 1.
- Varied: residual on/off, depth {20, 56}, BN on/off.
- Metrics: a single structured table — configuration → test accuracy, final train accuracy,
  epochs-to-80% train accuracy.

## Results
_Pending._

## Interpretation
_Pending._

## Limitations
_Pending — removing BN forces an LR change, so that row varies two things; called out explicitly._
