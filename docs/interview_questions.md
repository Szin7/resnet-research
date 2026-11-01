# Questions I prepared for the project review

These are the questions I would expect about the implementation and the experiments. The answers are based on this repository's design and measured results.

## What problem was I studying?

The degradation problem: making a plain network deeper can lead to *higher training error*, even when the deeper architecture should have enough capacity to represent the same function. That is different from overfitting, where training error is low but test performance is poor.

## What changes between Plain-56 and ResNet-56?

Both use the same basic convolutional layers, normalization and classifier. The ResNet blocks add the shortcut signal to the block output before the final activation. With shortcut option A, the shortcut adds no learned projection parameters. Both depth-56 models have 853,018 parameters.

## What did the eight runs show?

Plain-network final training error increased from 4.187% at 20 layers to 19.518% at 56. The corresponding residual-network error decreased from 2.740% to 1.036%. At depth 56, the residual model also achieved higher test accuracy: 91.68% versus 79.43%.

## Why focus on training error?

The research question is about optimization. A lower test accuracy can be caused by generalization problems, but *worse training error* in the deeper plain model is evidence that the training process struggled to fit the training data.

## How did I keep the comparison fair?

I used the same CIFAR-10 split, 80-epoch schedule, optimizer, augmentation and initial seed for the depth sweep. At each depth the plain and residual models share the same number of learned parameters. I selected checkpoints using validation accuracy and evaluated on the official test set separately.

## Did I reproduce the original ResNet paper exactly?

No. I used its CIFAR-style architecture but a shorter 80-epoch schedule, a Colab T4, and slightly different preprocessing. I describe this as a controlled study inspired by the paper, not a numerical reproduction of every result.

## What did the gradient experiment establish?

It showed how per-layer weight-gradient magnitudes changed in two separate five-epoch probe runs. It did *not* prove that one particular gradient problem caused the 80-epoch results. Those are separate experiments.

## What went wrong during training?

Colab GPU access ran out during ResNet-56. When resuming, the saved CUDA random-number-generator state could not be restored until I fixed the checkpoint loader to pass CPU byte tensors to the CUDA RNG restore function. I resumed from epoch 32 and completed that run.

## What would I improve with more compute?

I would repeat Plain-56 and ResNet-56 with additional training seeds, report the variation in final training error, and then run a more focused shortcut or normalization ablation. The current findings use one completed seed.

For architecture details, see [architecture.md](architecture.md); for exact measurements, see [results.md](results.md).
