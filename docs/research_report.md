# Research report — depth degradation and residual connections on CIFAR-10

## Research question

For matched CIFAR-sized convolutional architectures, does adding depth make plain networks harder to train, and do residual identity shortcuts change this behavior?

## Experimental design

Eight models were trained using the same 80-epoch SGD/momentum setup, seed 1, fixed split seed 42, standard augmentation, and a 45,000-training/5,000-validation partition of CIFAR-10. The original 10,000-image test split was reserved for final evaluation. Plain and shortcut-option-A residual networks at depths 20, 32, 44 and 56 share exactly the same number of learned parameters at each depth. Learning-rate reductions occur at epochs 40 and 60.

The original ResNet paper used a longer training recipe; this is a controlled research study, not a numerically exact paper reproduction. Full source implementation and configuration are in this repository.

## Results

| Depth | Plain training error (%) | ResNet training error (%) | Plain test accuracy (%) | ResNet test accuracy (%) |
|---:|---:|---:|---:|---:|
| 20 | 4.187 | 2.740 | 89.41 | 91.04 |
| 32 | 5.380 | 1.556 | 88.77 | 91.73 |
| 44 | 7.836 | 1.264 | 87.07 | 91.26 |
| 56 | 19.518 | 1.036 | 79.43 | 91.68 |

All eight models completed 80 epochs and were tested using their best-validation checkpoints. The main result is in **training error**: making the plain network deeper from 20 to 56 layers raises final training error by 15.331 percentage points, whereas residual-network training error falls by 1.704 points. At depth 56, the plain/residual gap in final training error is 18.482 points and in test accuracy is 12.25 points. Both depth-56 models contain 853,018 parameters.

The best measured test score, 91.73% for ResNet-32, is only 0.05 percentage points above ResNet-56 (91.68%); a one-seed experiment cannot establish which generalizes better on average. See [complete measured results](results.md) and [CSV](../results/tables/depth_experiment_summary.csv).

## Additional observations

For Plain-56, cat class recall was 61.9%, compared with 81.8% for ResNet-56. Cat/dog confusion was prominent in both, and markedly larger in the plain model. A separate five-epoch gradient probe on newly initialized plain-56/resnet-56 models and an eight-model Tesla T4 benchmark were completed; these provide context but do not substitute for a controlled mechanism study. Parameter counts and per-image convolution/linear MACs are matched by depth. Raw diagnostic and benchmark artifacts are preserved in the owner's Drive research archive.

## Interpretation

These results support an optimization-related degradation pattern in this setting: increased plain-network depth was accompanied by **worse final training performance**, while increased residual-network depth was not. Poorer training performance cannot be explained solely by overfitting, which ordinarily concerns a widening gap between training and unseen-data performance. The observed results are compatible with the motivating claim of the ResNet architecture but do not isolate every contributing mechanism.

## Limitations and next experiments

The sweep uses one seed, CIFAR-10 rather than ImageNet, a shorter 80-epoch schedule, and minor preprocessing deviations. Optimizer trajectories and measured timing depend on hardware/software; one run of each model does not supply confidence intervals. Gradient profiles came from a separate five-epoch experiment, not the saved 80-epoch checkpoints. Additional-seed replication and the proposed shortcut, BatchNorm, optimization and augmentation ablations are still planned. In particular, the no-BatchNorm protocol also changes learning rate, so it cannot isolate BatchNorm alone.

## Conclusion

Under equal-parameter, matched-depth comparisons on CIFAR-10, additive residual shortcuts correlated with substantially better trainability and held-out accuracy as depth increased to 56 layers. The evidence supports the depth-degradation hypothesis in the measured experimental setup; further random seeds and controlled ablations are required to quantify reproducibility and isolate mechanisms.
