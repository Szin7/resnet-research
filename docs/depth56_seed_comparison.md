# Depth-56 repeat: three training seeds

After finishing the first eight models, I wanted to make sure the big Plain-56 vs ResNet-56 gap wasn't just a result of one random seed. So I trained the two 56-layer models again with seeds 2 and 3, keeping the same settings.

The plain and residual 56-layer networks each have **853,018 learnable parameters**. Each run used CIFAR-10, a fixed 45,000/5,000 train/validation split (split seed 42), the same data augmentation, SGD with momentum, and 80 epochs with learning-rate reductions after epochs 40 and 60. Checkpoints were selected on validation accuracy, and the official 10,000-image test set was evaluated with the **best-validation checkpoint**. The seed-1 results belong to the original depth sweep.

## Per-seed results

| Model | Training seed | Final training error | Best validation accuracy | Test accuracy |
|---|---:|---:|---:|---:|
| Plain-56 | 1 | 19.518% | 79.38% | 79.43% |
| Plain-56 | 2 | 14.8067% | 83.34% | 82.65% |
| Plain-56 | 3 | 15.0200% | 82.52% | 82.62% |
| ResNet-56 | 1 | 1.0360% | 92.52% | 91.68% |
| ResNet-56 | 2 | 0.9378% | 92.58% | 91.90% |
| ResNet-56 | 3 | 1.3111% | 91.92% | 91.43% |

The [measurement table](../results/tables/depth56_seed_repeats.csv) records the published results. Seed-1 values are from the original depth sweep; seed-2 and seed-3 measurements come from completed Kaggle/Colab training logs, summary JSON and held-out test evaluation outputs. Full checkpoint and training archives were backed up separately; they are not stored in Git.

## What I found across the three runs

All figures below use **three independent training seeds** per architecture. The standard deviations are sample standard deviations (denominator n − 1) and describe differences between these three observed runs, not confidence intervals.

| Metric | Plain-56 mean (sample SD) | ResNet-56 mean (sample SD) |
|---|---:|---:|
| Final training error | 16.448% (2.661 pp) | 1.095% (0.194 pp) |
| Test accuracy | 81.567% (1.850 pp) | 91.670% (0.235 pp) |

ResNet-56 beat Plain-56 on test accuracy by **12.25, 9.25 and 8.81 percentage points** for seeds 1, 2 and 3 respectively, averaging **10.103 points**. Its lower final training error corresponded to gaps of **18.482, 13.869 and 13.709 points**, averaging **15.353 points**.

The training error is the important part of this experiment. Even on its training images, Plain-56 made far more mistakes than ResNet-56 in every seed. That is the degradation pattern I was looking for. Higher test accuracy is useful too, but it doesn't tell me exactly *why* the optimization improved.

## A problem I ran into while training

The original depth sweep (including seed 1) ran on Colab. Plain-56 seed 2 reached epoch 65 on Colab before the GPU quota was exhausted. I transferred its full saved state—weights, optimizer, scheduler, random state and metrics—to Kaggle, changed only the dataset root in the saved configuration, and resumed training from epoch 66. The final 15 epochs were completed on Kaggle. ResNet-56 seed 2, Plain-56 seed 3 and ResNet-56 seed 3 were trained from epoch 1 to 80 on Kaggle using a Tesla T4.

The Kaggle session used PyTorch 2.11.0+cu128, while the original Colab environment used a different CUDA build. Resuming a checkpoint across environments preserves training state but does not necessarily reproduce a bit-for-bit identical uninterrupted run. The original training settings and data split were retained, and all reported results were measured after the runs completed.

## Recreating the summary without training again

When the **six complete run directories** (two models × three seeds) are all restored under one `results/runs/` directory, the existing script can regenerate the CSV and print per-model mean/sample standard deviation:

```bash
python scripts/report_seed_repeats.py --seeds 1 2 3 --runs-dir results/runs
```

This only reads `summary.json`, `metrics.csv` and `test_metrics.json`. It does **not** train models. A Kaggle session containing only seeds 2 and 3 cannot regenerate the complete report until the seed-1 run folders are restored.

## What I can and can't conclude

Residual connections were associated with much lower training error and higher test accuracy in **all three** matched 56-layer comparisons. The plain network varied more between seeds. This strengthens the original single-seed observation, but three seeds remain a small sample and are insufficient to make strong statistical-significance or generalization claims.

The 20-, 32- and 44-layer models were only tested with seed 1. The training schedule and some preprocessing choices differ from the original ResNet paper. These findings are a controlled study within this implementation, not an exact numerical reproduction of the published paper or proof of one particular gradient mechanism.

In this setup, adding the parameter-free residual shortcuts helped the 56-layer network fit the training data much better, and the test accuracy was higher in all three seeds. That's the result I wanted to check. I would still need more seeds and other settings before claiming it always happens.
