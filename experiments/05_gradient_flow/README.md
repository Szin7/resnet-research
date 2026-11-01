# Early gradient norms — completed diagnostic

I compared Plain-56 and ResNet-56 in a separate five-epoch run, using a fixed probe batch and recording L2 norms of convolutional and linear weight gradients. This is not a measurement taken from the saved 80-epoch models.

| Probe | Plain-56 median norm | ResNet-56 median norm |
|---|---:|---:|
| Initialization | 6.711 | 0.454 |
| Epoch 1 | 0.075 | 0.038 |
| Epoch 5 | 0.141 | 0.166 |

The plain network showed a particularly large gradient outlier at initialization. The layerwise distribution changed during training, and by epoch 5 the medians were relatively close.

The measurements were saved to Google Drive. The published [gradient table](../../results/tables/gradient_norm_summary.csv) and [analysis notes](../../docs/gradient_analysis.md) explain their limits. Gradient size alone does not identify the cause of depth degradation.
