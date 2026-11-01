# How I describe this project

**Project:** ResNet depth study on CIFAR-10 (PyTorch)

Short version for a resume:

> Implemented CIFAR-style plain CNNs and ResNets in PyTorch, trained eight 20–56-layer models under matched settings, and compared optimization behavior using training curves and held-out CIFAR-10 evaluations.

When there is room for one result:

> At 56 layers, the residual model reached 1.036% final training error and 91.68% test accuracy, versus 19.518% and 79.43% for the parameter-matched plain model (single-seed experiment).

## Talking about it in an interview

I would explain the result using training error first: deeper plain networks fitted the training data worse, while deeper residual networks did not show that pattern. I would then explain the option-A shortcut, the matching parameter counts, and why a single training seed limits the conclusion.

The engineering side is also relevant: I wrote the experiment configurations and training/evaluation workflow, saved model and optimizer checkpoints to Drive, and recovered a Colab run after a CUDA RNG-state restore error.

Avoid presenting this as an exact reproduction of the original paper's accuracy figures or as a multi-seed study. The additional seed-2 run is unfinished.
