# Results

What is committed here:

- `runs/<run_name>/` — per run: `config.yaml`, `env.json`, `metrics.csv`, `summary.json`,
  `grad_flow.json` (when enabled). **Checkpoints are gitignored** (`*.pt`).
- `figures/` — final plots used in the report and README.
- `tables/` — result tables (CSV + Markdown).

Everything here is regenerated from logged metrics by `scripts/make_figures.py`. No numbers are
entered by hand.
