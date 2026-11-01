# Depth-56 repeat: seeds 1 and 2

After the original depth sweep, I repeated the 56-layer comparison with a second training seed. I wanted to check whether the large gap between the plain CNN and ResNet-56 appeared again when the initialization and data-order randomness changed.

Both networks have 853,018 learned parameters. They used the same CIFAR-10 45,000/5,000 train/validation split (split seed 42), standard data augmentation, SGD with momentum, and 80 epochs with learning-rate drops at epochs 40 and 60. For both seeds, the official 10,000-image test set was evaluated using the **best-validation checkpoint**.

## Individual results

| Network | Seed | Final train error | Best validation accuracy | Test accuracy |
|---|---:|---:|---:|---:|
| Plain-56 | 1 | 19.518% | 79.38% | 79.43% |
| Plain-56 | 2 | 14.8067% | 83.34% | 82.65% |
| ResNet-56 | 1 | 1.036% | 92.52% | 91.68% |
| ResNet-56 | 2 | 0.9378% | 92.58% | 91.90% |

The [CSV table](../results/tables/depth56_seed_repeats.csv) contains these values. Seed-1 values are from the completed depth-sweep summary. Seed-2 values are from the final Kaggle/Colab training summaries and held-out test-evaluation output. The seed-2 full training histories and checkpoints were archived separately, rather than committed to GitHub.

## Comparing the two seeds

| Measurement | Plain-56, mean (sample SD) | ResNet-56, mean (sample SD) |
|---|---:|---:|
| Final training error | 17.162% (3.331 pp) | 0.987% (0.069 pp) |
| Test accuracy | 81.04% (2.277 pp) | 91.79% (0.156 pp) |

The test-accuracy advantage of the ResNet was **12.25 percentage points** at seed 1 and **9.25 points** at seed 2. The average matched difference was **10.75 points**.

The final training-error difference was **18.482 percentage points** at seed 1 and **13.8689 points** at seed 2. The same direction of the degradation result appeared in both runs.

Two seeds are enough to show that this was not an observation from only one initialization, but **not** enough to estimate the full variability reliably or claim statistical significance. I have kept the individual measurements visible rather than presenting the means on their own.

## Running the second seed

Plain-56 seed 2 started on Colab, but GPU quota stopped the run after epoch 65. I copied its saved model, optimizer, scheduler, RNG state, configuration and metrics into Kaggle. Once the dataset was prepared with the original split seed, I updated only the data-root path and resumed from epoch 66. Training completed at epoch 80 and I evaluated the saved checkpoints.

ResNet-56 seed 2 was trained from epoch 1 through 80 on a Kaggle Tesla T4. The recorded software environment for the Kaggle session was PyTorch 2.11.0+cu128; the original Colab environment used a different CUDA build. The change in runtime is worth noting: a resumed run on different hardware/software is not guaranteed to follow the identical floating-point trajectory it would have followed without interruption.

The configurations for running the seed-2 pair are in [experiments/10_depth56_seed_repeats](../experiments/10_depth56_seed_repeats/). The script [report_seed_repeats.py](../scripts/report_seed_repeats.py) can regenerate a comparison with `--seeds 1 2` when all four complete run folders are present in one `--runs-dir` directory. The published CSV contains the values from the recorded outputs; it is not a replacement for keeping the complete run folders.

## Conclusion

Adding residual connections still made the 56-layer network substantially easier to train under both tested seeds. The amount of improvement varied, especially for the plain model. This is a useful follow-up to the original eight-model study, not a complete multi-seed replication across all four depths.
