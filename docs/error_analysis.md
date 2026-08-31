# Error analysis

_Status: outline. Filled in Phase 17 from Experiment 10, using the best model's test predictions._

Planned contents:

- Confusion matrix (raw counts + row-normalized figure).
- Per-class precision / recall / F1 / support table.
- The most-confused class pairs, ranked from the confusion matrix (not assumed in advance).
- Gallery of ≥ 24 misclassified images labelled `true → predicted (confidence)`.
- Discussion of plausible reasons, grounded in the examples — e.g. only claim "cat/dog is the
  hardest pair" if the numbers show it.
- Whether errors are high-confidence (a calibration concern) or low-confidence (genuine ambiguity).
- Limitations: single model, single seed, CIFAR-10's 32×32 resolution limits what is separable.
