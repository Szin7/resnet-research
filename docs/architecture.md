# Model architecture

The main models use the three-stage CIFAR ResNet introduced in *Deep Residual Learning for Image Recognition*. They operate directly on 32x32 RGB images and use a 6n + 2 weighted-layer layout.

## Network structure

```text
RGB image (3 x 32 x 32)
  |
  +-- 3x3 stem convolution, BN, ReLU --> 16 x 32 x 32
  |
  +-- Stage 1: n blocks, 16 channels --> 16 x 32 x 32
  |
  +-- Stage 2: n blocks, 32 channels --> 32 x 16 x 16
  |
  +-- Stage 3: n blocks, 64 channels --> 64 x 8 x 8
  |
  +-- Global average pool --> 64
  +-- Linear classifier --> 10 classes
```

Each basic block contains a 3x3 convolution, BatchNorm, ReLU, another 3x3 convolution, and another BatchNorm layer. If the residual flag is enabled, the shortcut input is added before the final ReLU.

```text
x ------------ shortcut -----------+
|                                   |
conv 3x3 -> BN -> ReLU -> conv -> BN + -> ReLU
```

The plain-network variant uses the same convolutional branch without the addition.

## Why the shortcut is useful

A residual block learns a transformation F(x) which is added to the input: y = F(x) + x, assuming the tensor dimensions match. The additive path allows the network to represent an identity-like transformation more easily. This does not mean that gradient magnitudes alone explain the entire optimization benefit.

At each stage transition, spatial resolution halves and channel count doubles. Option A uses strided sampling and zero padding to match shapes without learned parameters. Option B adds a 1x1 projection only when dimensions change. Option C projects every shortcut.

## Depth and parameter matching

Depth 20 corresponds to n=3 basic blocks per stage. Depth 32 uses n=5; depth 44 uses n=7; depth 56 uses n=9. ResNet-110 uses n=18.

For the central experiment, the plain and residual networks use option A, so their weighted-layer counts and learnable parameter counts are identical. This isolates the influence of the shortcut more carefully than comparing networks with unrelated architectures.

The model implementations are in `src/resnet_research/models/blocks.py` and `resnet.py`. Tests check both tensor sizes and parameter counts against the expected CIFAR configurations.
