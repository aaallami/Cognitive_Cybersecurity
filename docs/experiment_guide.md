# Experiment Guide

## Running the executed benchmark

```bash
python experiments/run_baseline_benchmark.py
```

This is the script backing notebooks 01-02. It:

1. Loads data (synthetic by default; real dataset if `path=` is set inside
   the script or via Hydra override).
2. Cleans/encodes/scales it and produces stratified train/val/test splits.
3. Trains 7 supervised + 3 unsupervised baselines.
4. Computes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, train/inference
   time, and peak memory for every model.
5. Writes tables to `tables/*.{csv,xlsx,md,tex}` and figures to
   `figures/*.{png,svg,pdf}` at 300 dpi.
6. Writes `outputs/run_manifest.json` summarizing the run (seed, dataset
   metadata, best model).

## Reproducibility

- All randomness is seeded (`RANDOM_STATE = 42` by default, threaded through
  data generation, splitting, and every model constructor).
- `configs/config.yaml` and `configs/models.yaml` define the Hydra-style
  configuration surface (dataset params, split ratios, model hyperparameters,
  output settings). The current scripts read defaults directly from
  `src/cognitivecyber/`; wiring `@hydra.main` onto `experiments/run_baseline_benchmark.py`
  is a small, documented next step (see the `TODO(hydra)` marker in that file)
  once you're ready to sweep hyperparameters from the CLI.
- Re-running `pytest tests/ -v` after any pipeline change is the fastest way
  to confirm nothing broke.

## Extending to more notebooks

`notebooks/03` through `notebooks/12` in this repository are **scoped
planning notebooks**, not executed benchmarks: each documents the intended
content, the concrete inputs/outputs contract, and exactly what
infrastructure (data feeds, GPU, API keys) is needed to execute it for
real, per `docs/architecture.md`. Turning a planning notebook into an
executed one means:

1. Supplying the missing input (e.g. a cached MITRE ATT&CK STIX bundle for
   notebook 04, an LLM API key for notebook 05, a GPU + `requirements-deep.txt`
   for notebooks 06-08).
2. Writing the corresponding `src/cognitivecyber/<module>/` code (following
   the same pattern as `data/`, `models/`, `evaluation/`).
3. Adding tests under `tests/` for the new module.
4. Running `jupyter nbconvert --to notebook --execute --inplace <nb>.ipynb`
   and confirming zero error cells, exactly as was done for 01 and 02.

## Statistical significance testing

For comparing two models' held-out metrics across repeated runs (different
seeds), use `scipy.stats`:

```python
from scipy import stats

# Paired t-test across per-seed F1 scores for two models
stats.ttest_rel(model_a_f1_scores, model_b_f1_scores)

# Wilcoxon signed-rank test (non-parametric alternative)
stats.wilcoxon(model_a_f1_scores, model_b_f1_scores)
```

A worked example (multi-seed benchmark -> paired tests -> effect size ->
Bonferroni/Holm correction across all pairwise model comparisons) belongs
in `notebooks/11_Statistical_Analysis.ipynb` once you've run
`experiments/run_baseline_benchmark.py` across multiple seeds and collected
per-seed results (a `for seed in [1,2,3,4,5]:` sweep around the existing
script is the natural way to generate that input).
