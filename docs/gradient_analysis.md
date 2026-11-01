# Gradient-flow check

I ran a short gradient experiment to compare how the plain and residual 56-layer models behave early in training. This was a separate five-epoch run, not a probe of the finished 80-epoch checkpoints.

The script records the L2 norm of each convolutional or linear layer's weight gradient on a fixed batch. It takes snapshots at initialization, after epoch 1 and after epoch 5, with mixed precision disabled.

```bash
python scripts/analyze_gradients.py --models plain_56 resnet_56 --epochs 5 --device cuda
```

The output was saved in the Google Drive research archive as `gradient_norms.json` and `gradient_norms.png`. A small [summary table](../results/tables/gradient_norm_summary.csv) is also published here.

## What I observed

| Probe point | Plain-56 median norm | ResNet-56 median norm |
|---|---:|---:|
| Initialization | 6.711 | 0.454 |
| Epoch 1 | 0.075 | 0.038 |
| Epoch 5 | 0.141 | 0.166 |

At initialization the plain network had a much larger outlier in its layer-weight gradients. By epoch 5, the median norms were closer. The distributions change substantially during those first few epochs.

I would not use these numbers alone to explain the training degradation. Gradient magnitude depends on which layer is measured, the loss and the state of training. A nonzero gradient does not mean a network is easy to optimize. The full-depth training error comparison remains the main result of the project.

See [results.md](results.md) for the completed 80-epoch study and its limitations.
