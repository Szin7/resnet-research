# Experimental results — CIFAR-10 depth study

The primary eight-model depth sweep completed 80 epochs per model, with seed 1 and a fixed 45,000/5,000 train/validation split (split seed 42). Results below are measured, not estimates. Test accuracy is calculated using the best-validation checkpoint for each run.

| Depth | Plain final training error | ResNet final training error | Plain best validation | ResNet best validation | Plain test | ResNet test | Parameters (each) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 4.187% | 2.740% | 89.74% | 91.24% | 89.41% | 91.04% | 269,722 |
| 32 | 5.380% | 1.556% | 89.36% | 92.32% | 88.77% | 91.73% | 464,154 |
| 44 | 7.836% | 1.264% | 87.80% | 92.10% | 87.07% | 91.26% | 658,586 |
| 56 | 19.518% | 1.036% | 79.38% | 92.52% | 79.43% | 91.68% | 853,018 |

Source: [measured run table](../results/tables/depth_experiment_summary.csv), exported from eight completed run histories, summaries, and held-out test evaluations. Units are percentage points for differences; the improvement at depth 56 is 12.25 points in test accuracy and 18.482 points in final training error.

## Main finding

The plain family's **training** error rises with depth (4.187% to 19.518%); the residual family's falls (2.740% to 1.036%). Since the paired models have identical learnable parameter counts, this is consistent with optimization-related *depth degradation* for these training configurations, not simply deeper models having too few parameters. The test-set trend supports the practical benefit of residual shortcuts here but does not independently diagnose optimization.

## Accuracy and prediction errors

ResNet-32 attains the highest measured test accuracy in this single-seed sweep (91.73%), narrowly ahead of ResNet-56 (91.68%). Do not interpret this 0.05-point gap as a reliable ranking without repeat seeds.

For Plain-56, the lowest class recall is **cat** (61.9%); its most common confusions include dog→cat (170) and cat→dog (158). For ResNet-56, cat recall improves to 81.8%, with the same prominent confusions at lower counts: cat→dog (76), dog→cat (69). These counts come from the best-validation checkpoint and the 10,000-image test set.

## Gradient diagnostic

A separate five-epoch probe, starting new models from initialization, recorded convolution/linear layer weight-gradient L2 norms. At epoch 5, median per-layer norms were about 0.141 (Plain-56) and 0.166 (ResNet-56); the initial snapshot showed a much more extreme outlier in the plain network's first layer. The diagnostic is suggestive, **not** proof that the 80-epoch result was caused by a specific vanishing/exploding-gradient mechanism. Norms also depend on layer scale and loss dynamics.

## Compute measurements

On a Colab Tesla T4 GPU, measured parameter counts and per-image multiply–accumulate counts were identical for the plain and option-A residual network at each depth: 40.55M, 68.86M, 97.17M and 125.49M MACs for 20/32/44/56 layers respectively. The eight-model benchmark ran 20 timing iterations with 5 warmups, batch sizes 1 and 32. These timings are environment-dependent and should not be used as a hardware-independent speed ranking. The reported peak memory values describe the benchmark's 32-image synthetic training step, not entire original training runs.

## Figures and reproducibility

Original measured plots were generated and backed up to Google Drive under `resnet-research/results/figures/`; they are not embedded in this Git repository yet. They are named `depth_vs_train_error.png`, `depth_vs_val_accuracy.png`, `training_loss_curves.png`, `val_accuracy_curves.png`. Each run's full metrics, configuration, and evaluation results are held with the owner; the small summary CSV above is published for inspection.

## Limitations

These are eight complete **single-seed** runs with an 80-epoch schedule, not a statistical replication of the original longer CIFAR experiment. Differences in augmentation, normalization, schedules and platform matter. Additional seeds, shortcut variants and controlled ablations remain unrun or unpublished; do not present them as completed research findings. The held-out test set was used only after training for final evaluation.
