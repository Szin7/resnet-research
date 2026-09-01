# Notebooks

## `colab_train.ipynb`

The way the training matrix is run. On a Colab T4 GPU it:

1. clones the repo, `pip install -e .`
2. mounts Google Drive and symlinks `data/` and `results/` there, so a dropped session loses
   nothing
3. downloads CIFAR-10 into Drive (once)
4. runs an experiment (`experiments/*/experiment.yaml`) — finished runs are skipped, so re-running
   after a disconnect resumes where it left off
5. evaluates each finished run on the test set
6. displays the figures
7. optionally commits `results/` + figures back to the repo (needs a GitHub token in a Colab secret)

Set `REPO_URL`, `BRANCH`, and `EXPERIMENT` in the second cell before running.

## `analysis.ipynb`

_Pending._ Exploratory analysis of the logged metrics once the experiments have run; its narrative
mirrors `docs/research_report.md`. Figures in the report come from `scripts/make_figures.py`, not
from this notebook.
