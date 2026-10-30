# Classification error analysis

The evaluation script computes a confusion matrix, per-class precision, recall and F1, and a ranked list of confused class pairs.

Run it after completing and validating a model:

```bash
python scripts/evaluate.py --run results/runs/<run-directory> --device cuda --gallery 24
```

The optional gallery displays examples that the model classified incorrectly, with the predicted label and confidence. Images are visualized for diagnostic inspection only and are not used to adjust the model after test evaluation.

Questions for the written discussion:

- Which classes have the lowest recall?
- Are most errors concentrated in a few class pairs?
- Does the confusion pattern change between plain and residual networks?
- Do improvements in aggregate accuracy hide worse outcomes for particular classes?

**Status: no measured error analysis is available yet.**
