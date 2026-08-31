# Interview questions and answers

_Status: outline. Written in Phase 21, after the experiments, so the research answers cite real
results from this project. Answers must match the actual implementation._

Planned sections (from the master plan §39):

## CNN fundamentals
convolution; multiple filters; stride and padding; receptive field; why ReLU.

## ResNet
the degradation problem; residual learning; why shortcuts help (carefully — see
[mathematics.md](mathematics.md) §5); `F(x) + x`; when a projection shortcut is needed; why
mismatched dimensions cannot just be added; why BatchNorm; where the activation goes.

## Optimization
vanishing / exploding gradients; how a shortcut affects gradient propagation; why deeper ≠ better;
weight decay; momentum; learning-rate scheduling.

## Research (answers cite this project's results)
how this differs from the paper; why CIFAR-10; why a smaller model; the baseline; what the ablation
showed; what happened when the shortcut was removed; what happened to gradient flow; what surprised
me; the limitations.

## Implementation
the residual block; the projection shortcut; the training loop; the evaluation pipeline; how
gradient norms were measured; how reproducibility was handled.
