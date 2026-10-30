# Research report

**Status: implementation and experimental protocols prepared; findings pending training.**

## Research question

How does the addition of residual connections affect optimization and generalization in CIFAR-sized convolutional networks as depth increases?

## Motivation

The original ResNet study showed that simply making a plain network deeper did not reliably improve training. Residual shortcuts were proposed to ease optimization. This project tests that comparison on CIFAR-10 with a smaller and explicitly documented compute budget.

## Methods

Custom PyTorch implementations cover plain and residual CIFAR networks with the same weighted layers. For the central comparison, parameter-free shortcut option A keeps the total number of learnable parameters equal across corresponding models. The fixed data split uses 45,000 training images and 5,000 validation images; the official test set is reserved for the final assessment.

The standard schedule has 80 epochs with SGD, momentum and two learning-rate drops. This differs from the longer schedule used in the paper. Controlled experiment matrices additionally vary shortcuts, BatchNorm, optimization and image augmentation.

## Measurements

Training error and loss are the main evidence for the depth-degradation question. Validation accuracy provides a separate measure of generalization. Final test evaluation, confusion matrices, seed variation, gradient-norm profiles and inference benchmarks provide additional evidence.

## Results

Pending measured experimental runs. No numerical claims have been inserted.

## Discussion and limitations

Interpretation must distinguish optimization difficulty from overfitting, and avoid treating gradient magnitude alone as a complete explanation. Experiments with changes to both BatchNorm and learning rate cannot isolate BatchNorm's independent effect.

The study uses CIFAR-10 and a shorter schedule rather than a complete ImageNet or paper-faithful reproduction. Model outputs and figures will be referenced here after the experiments have run.

## Conclusion

Pending the results. The code establishes an experimental framework, not yet an empirical conclusion.
