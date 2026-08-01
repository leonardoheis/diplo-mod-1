# Notebook 05 — W&B Logging + Visual Critical Analysis

**Goal:** Extend the just-built `notebooks/05-evaluation-comparison.ipynb` so (1) every comparison artifact (metrics table, plots, SHAP output) logs to Weights & Biases, and (2) the "critical analysis" section is backed by charts, not prose alone.

**Architecture:** Notebook 05 gains a new "Step 7 — Comparison charts" (4 focused code cells) inserted between the existing SHAP step (Step 6) and the critical-analysis markdown, which is renumbered to Step 8 so its prose can reference the charts by name. A single `wandb` run spans the whole notebook — `wandb.init()` right after data loading, `wandb.log(...)` added to every step that already produces a table/figure plus the four new chart cells, `wandb.finish()` at the end of Step 7. Same `WANDB_ENABLED` opt-in guard notebooks 03/04 already use.

**Tech Stack:** No new dependencies — `wandb`, `os`, `datetime`, `dotenv` (all already used by notebooks 03/04, just not yet imported into 05); `pandas.cut` for score-bucketing (stdlib-adjacent, already a project dependency).

## Context

Notebook 05 was just built (this session) with a metrics table, a predicted-vs-actual plot, fresh side-by-side SHAP, and a written critical-analysis markdown answering the assignment brief's five questions. Two gaps surfaced on review:

1. **No W&B logging at all.** Notebooks 03/04 log every training run; notebook 05, the final comparison, currently logs nothing — there's no permanent record of the comparison artifacts alongside the training runs that produced them.
2. **"Critical analysis" is prose-only**, except for the SHAP plots (which already answer "most relevant features" visually). The other four brief questions — best model + why, did the NN beat XGBoost, overfitting signs, what to improve — are currently asserted in markdown text with no supporting chart.

Both are additive changes to the notebook built this session; no changes to `src/` are needed — every new chart is computed from data already loaded/predicted in Steps 1-4.

## Design

### New Step 7 — Comparison charts (4 cells, each a focused chart)

All four reuse data already in memory from earlier steps — no new model calls, no new data loading.

1. **Overall scorecard** — 3 subplots side by side (RMSE, MAE, R²), each a 2-bar chart (XGBoost vs. NN) on the test split. Source: `comparison_df` (already built in Step 4).
2. **Overfitting bars** — 2 subplots (RMSE, R²), each a grouped bar chart with train/val/test on the x-axis, one bar pair (XGBoost, NN) per split. Source: `comparison_df`.
3. **Error by score bucket** — reuses `POINTS_BINS = [79, 85, 88, 91, 94, 100]` from `src/diplo_mod_1/constants.py` (the exact bucketing `DataSplitter` already uses for stratified splitting, via `pd.cut(y, bins=POINTS_BINS)` in `src/diplo_mod_1/preprocessing/splitter.py`). Computes per-bucket RMSE for each model from the real Step 3 predictions (`xgb_pred`, `nn_pred`, `y_test`), grouped bar chart, buckets on the x-axis.
4. **Residual overlay** — one histogram, `y_test - xgb_pred` and `y_test - nn_pred` plotted on the same axes with distinct colors + alpha transparency, shared crimson dashed zero-line (matches the existing single-model residual-plot convention from notebooks 03/04, extended to two overlaid series).

### Renumbering

Existing Step 7 (critical-analysis markdown) becomes Step 8. Its prose is unchanged in content but should reference the new charts by name where relevant (e.g. "see the scorecard above" / "see the overfitting bars above").

### W&B logging

New imports needed in notebook 05's imports cell (currently has none of these): `os`, `datetime`/`timezone`, `wandb`, `from dotenv import load_dotenv`. Add the same `WANDB_ENABLED = os.environ.get("WANDB_ENABLED", "false").lower() == "true"` flag and `load_dotenv(override=True)` call notebooks 03/04 use.

- **Init**, added right after Step 1 (data loading) — `xgb_history`/`nn_history` (and their `best_run_id`s) aren't loaded until Step 4, so `config=` at init time carries only what Step 1 already knows (test-set shape); the run IDs are logged as part of Step 4's own `wandb.log` call instead:
  ```python
  if WANDB_ENABLED:
      wandb.init(
          project=os.environ.get("WANDB_PROJECT", "diplo-mod-1"),
          name=f"comparison-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
          group="model-comparison",
          job_type="evaluation",
          config={"n_test_rows": int(y_test.shape[0])},
      )
  ```
- **Step 4** (metrics table): `if WANDB_ENABLED: wandb.log({"comparison_table": wandb.Table(dataframe=comparison_df), "xgboost_run_id": xgb_history.best_run_id, "nn_run_id": nn_history.best_run_id})`.
- **Step 5** (predicted-vs-actual): `if WANDB_ENABLED: wandb.log({"predicted_vs_actual": wandb.Image(fig)})`.
- **Step 6** (SHAP): `if WANDB_ENABLED: wandb.log({"shap_xgboost": wandb.Image(...), "shap_xgboost_table": wandb.Table(dataframe=xgb_shap_table), "shap_nn": wandb.Image(...), "shap_nn_table": wandb.Table(dataframe=nn_shap_table)})`.
- **Step 7** (new charts): one `wandb.log({...})` call per chart — keys `"scorecard"`, `"overfitting_bars"`, `"error_by_points_bucket"`, `"residual_overlay"`.
- **`wandb.finish()`** at the very end of Step 7, guarded the same way.

Every logging call wrapped in `if WANDB_ENABLED:` — nothing logs unless the user opts in via `.env`, matching the standing project convention (`poe check`/`nbtest` never trigger a run).

## Files touched

- `notebooks/05-evaluation-comparison.ipynb` — imports cell extended (wandb/os/datetime/dotenv); `wandb.init`/`wandb.log`/`wandb.finish` added to Steps 1(or 4)/4/5/6; new Step 7 (4 chart cells); existing Step 7 renumbered to Step 8.
- No `src/` changes — everything reuses existing constants (`POINTS_BINS`) and data already computed in the notebook.

## Verification

- `uv run poe lint && uv run poe typecheck` — must pass clean (nbqa against the notebook).
- Every cell keeps `execution_count: null` / empty `outputs` — edit via `nbformat`, never execute.
- Actually running the notebook (confirming the real charts and, if `WANDB_ENABLED=true`, the W&B run) is the user's to do.
