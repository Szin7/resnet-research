# Classification errors on CIFAR-10

The evaluation script saves a confusion matrix and per-class precision, recall and F1 for each evaluated run. The matrix uses actual labels on rows and predicted labels on columns.

For the finished seed-1 depth sweep, I used the best-validation checkpoint for test evaluation. Here are two observations from the 56-layer models:

| Test-set measurement | Plain-56 | ResNet-56 |
|---|---:|---:|
| Overall accuracy | 79.43% | 91.68% |
| Cat recall | 61.9% | 81.8% |
| Cats classified as dogs | 158 | 76 |
| Dogs classified as cats | 170 | 69 |

The cat/dog mix-up is noticeable in both models, but it is much more common in Plain-56. It is also worth checking the other classes rather than relying only on the overall accuracy score.

## Looking at a saved run

```bash
python scripts/evaluate.py --run results/runs/<run-directory> --device cuda --gallery 24
```

The evaluation command produces the confusion matrix and, when requested, a gallery of misclassified images. The test set is for reporting final results, not for tuning the training settings after the fact.

The full 10,000-image test set and completed run metrics are kept outside the repository; the summary results are in [results.md](results.md).

## What the numbers do not show

These are observations from one training seed. They do not show that residual connections specifically solve cat/dog confusion, and they cannot tell us which visual features caused individual mistakes. Reviewing a misclassification gallery is useful for finding patterns, but it is still a descriptive check.
