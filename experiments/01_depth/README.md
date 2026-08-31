# Experiment 1 — Depth

## Research question
Does increasing depth always improve optimization and validation performance? Do deeper *plain*
networks show higher *training* error (an optimization failure), as opposed to just worse test error
(overfitting)?

## Hypothesis
- **H1:** plain-net test accuracy plateaus then declines across depth 20 → 32 → 44 → 56, and
  plain-56 shows higher training error than a shallower plain net.
- **H1':** ResNet test accuracy improves or holds flat across the same range with no training-error
  degradation.
- Fallback if degradation is not seen at 56: extend to plain-74 / plain-110 and report that it did
  not appear at 56 under the 80-epoch schedule.

## Setup
- Runs: the core sweep — plain-{20,32,44,56} and resnet-{20,32,44,56}, shortcut option A.
- Fixed: standard 80-epoch schedule, seed 1, `standard` augmentation, BN on.
- Varied: depth, and residual on/off.
- Metrics: final train / val / test accuracy and loss, training-loss curves, epochs-to-{60,80}% train accuracy, parameter count.

## Results
_Pending._

## Interpretation
_Pending._

## Limitations
_Pending — at minimum: single seed (variance runs quantify noise separately); 80-epoch schedule may
under-train the deepest models; CIFAR-10 only._
