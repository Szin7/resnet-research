# Depth and residual shortcuts on CIFAR-10

## Question

The original ResNet paper showed that adding more layers to a plain network does not always make it easier to fit the training data. The question here is whether I can observe the same behavior in a smaller CIFAR-10 experiment, and what changes when the convolutional blocks include residual shortcuts.

## Setup

I trained four plain networks and four residual networks at depths 20, 32, 44 and 56. The corresponding pairs use the same convolutional architecture and number of learned parameters. The residual versions use parameter-free option-A shortcuts.

All eight runs used seed 1, a fixed 45,000/5,000 training/validation split (split seed 42), SGD with momentum, batch size 128, and an 80-epoch schedule with learning-rate drops at epochs 40 and 60. The official 10,000 CIFAR-10 test images were not used for selecting checkpoints. Test accuracy below is from each run's best-validation checkpoint.

This is not an exact numerical reproduction of the paper. The training schedule is shorter, and some preprocessing choices differ.

## Results

| Layers | Plain final train error | ResNet final train error | Plain test accuracy | ResNet test accuracy |
|---:|---:|---:|---:|---:|
| 20 | 4.187% | 2.740% | 89.41% | 91.04% |
| 32 | 5.380% | 1.556% | 88.77% | 91.73% |
| 44 | 7.836% | 1.264% | 87.07% | 91.26% |
| 56 | 19.518% | 1.036% | 79.43% | 91.68% |

The plain models became harder to fit as depth increased: final training error grew from 4.187% at 20 layers to 19.518% at 56. For ResNets, it fell from 2.740% to 1.036%. At 56 layers, both models have 853,018 parameters, so the result is not explained by one model simply having more learned weights.

The 56-layer test accuracy gap is 12.25 percentage points. ResNet-32 had the highest test score of this one-seed sweep (91.73%), only 0.05 points above ResNet-56. That difference is too small to interpret as a reliable ranking without repeat runs.

The test confusion matrices show a related pattern. Cat recall was 61.9% for Plain-56 and 81.8% for ResNet-56. Cat/dog confusions remained common in both models, but were more frequent in Plain-56.

## What the measurements suggest

The main evidence is the *training error*, not the test accuracy alone. Deeper plain networks fitted the training set less well under the same training recipe; deeper ResNets did not. This is the degradation behavior the experiment was designed to examine.

It does not establish a single cause. A separate five-epoch gradient-norm probe showed substantially different layer-gradient scales, especially near initialization, but those models were trained separately. I cannot directly attribute the completed 80-epoch results to one gradient mechanism from that diagnostic.

The compute benchmark likewise measures architectural costs, not model quality. Plain and option-A residual pairs share the same parameter count and convolution/linear MAC count; measured latency varies with the GPU environment and timing noise.

## Problems encountered

The initial Colab session hit its GPU usage limit during ResNet-56. The run had completed 31 epochs and had saved a checkpoint to Drive. When GPU access returned, loading the saved CUDA random state failed because PyTorch required a CPU ByteTensor. I corrected the checkpoint loader, resumed at epoch 32, and finished all 80 epochs. The fix is now part of the repository.

This matters for reproducibility: the checkpoint retained the optimizer, scheduler and model state, but recovering a GPU run does not guarantee bit-for-bit identical training to an uninterrupted session.

## Limits and next experiments

These are single-seed results on CIFAR-10. More seeds are needed to estimate variation. The 80-epoch schedule, dataset size, mixed precision and preprocessing differences also limit direct comparison to the paper's numbers.

The experiment configurations include shortcut variants, BatchNorm changes, learning-rate schedules and augmentation changes. Those full comparisons still need to be run. In the no-BatchNorm setup, learning rate also changes; that experiment cannot isolate the effect of BatchNorm alone.

## Conclusion

In these eight CIFAR-10 runs, residual shortcuts made the deeper networks much easier to train under otherwise matched settings. The clearest result was at 56 layers, where training error dropped from 19.518% for the plain network to 1.036% for ResNet-56. That is strong evidence of depth degradation in this particular experiment, while broader claims require additional runs and controls.

For the detailed table, see [results.md](results.md); for the full caveats, see [limitations.md](limitations.md).
