# Gradient-flow study

## Research question

Does the residual shortcut change the distribution of layerwise weight-gradient norms early in training?

A fixed, minimally augmented CIFAR-10 probe batch is passed through plain and residual models of the same depth. Gradients are recorded at initialization, epoch 1 and the final instrumented epoch. The probe is run in full precision.

## Procedure

```bash
python scripts/analyze_gradients.py --models plain_56 resnet_56 --epochs 5 --device cuda
```

Output files under `experiments/05_gradient_flow/` include `gradient_norms.json` and `gradient_norms.png`.

## Interpretation

Gradient norms are diagnostic measurements, not a direct proof of vanishing gradients. The original ResNet paper observed that deep plain networks can have healthy gradient magnitudes despite optimization difficulties. A similar observation here would be consistent with that distinction.

**Status: awaiting measured runs.** The analysis implementation is available; no numerical conclusions have been recorded.
