# Plain and residual networks — comparison from the depth sweep

This comparison uses the same eight finished runs as [Experiment 1](../01_depth/README.md); it does not require another training session. The only architectural difference in each matched pair is the addition of the parameter-free option-A shortcut.

| Layers | Plain test accuracy | ResNet test accuracy | Difference |
|---:|---:|---:|---:|
| 20 | 89.41% | 91.04% | +1.63 points |
| 32 | 88.77% | 91.73% | +2.96 points |
| 44 | 87.07% | 91.26% | +4.19 points |
| 56 | 79.43% | 91.68% | +12.25 points |

The test-accuracy gap increases with depth in these runs. The training-error comparison is more important for the degradation question: at 56 layers, it was 19.518% for the plain model and 1.036% for the residual one.

The training curves and the complete table are covered in [the main results](../../docs/results.md). This is one seed, so the gap should be checked with repeat runs before treating its size as stable.
