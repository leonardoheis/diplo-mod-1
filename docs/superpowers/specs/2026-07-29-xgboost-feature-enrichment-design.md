# XGBoost feature enrichment — design

## Context

Four separate Optuna hyperparameter searches (`configs/xgboost_tuning*.json`, notebook 03) have all converged to test R² ~0.71-0.72, RMSE ~1.61-1.64 — a ~2% relative spread that's within noise, not a real difference between configs. Hyperparameter tuning has plateaued: it's no longer the lever that moves this model. The goal is to push test R²/RMSE meaningfully past that plateau, which requires new predictive *signal*, not more search over the same 40 tabular columns.

Two concrete gaps exist in the current feature set:
1. `taster_strictness` is already computed by `TabularEncoder` (`src/diplo_mod_1/preprocessing/encoders.py`) but only wired into the NN export (notebook 02, Step 5) — XGBoost's export (Step 4) never asks for it. This is a bug of omission, not a design choice.
2. `province` (state/region, e.g. "California") is loaded but never used. `region_1` gets target-encoded but not frequency-encoded, unlike `winery`/`variety` which get both. Absolute `log_price` is used, but nothing captures price *relative to* what's typical for that grape variety.
3. `description` (free text) only reaches XGBoost via `description_length` and 25 curated keyword flags — the full TF-IDF matrix (2000 terms) already built for the NN dataset (`data/processed/nn/X_txt_*.npz`) is never given to XGBoost.

## Approach

Three parts, done **sequentially** so results are attributable to a specific change rather than a bundled unknown:

### Part 1 — Enable `taster_strictness` for XGBoost

`notebooks/02-preprocessing.ipynb`, Step 4 (XGBoost export cell): change
```python
x_xgb["train"], feature_names_xgb, groups_xgb = encoder.fit_transform(train_df, y_splits["train"])
x_xgb["val"], _, _ = encoder.transform(val_df)
x_xgb["test"], _, _ = encoder.transform(test_df)
```
to pass `include_strictness=True` on all three calls, matching the NN branch (Step 5) exactly. No other code changes — `TabularEncoder` already supports this. `data/processed/xgboost/` regenerates with 41 columns instead of 40.

### Part 2 — New tabular features

All in `src/diplo_mod_1/preprocessing/`, following existing patterns exactly (no new abstractions):

- **`columns.py`**: add `("province", "province_avg_points")` to `TARGET_ENCODE_COLS`; add `("region_1", "region1_freq")` to `FREQ_COLS`.
- **`feature_engineer.py`**: add `price_vs_variety` — `log_price` minus that row's variety's mean `log_price` (computed train-only in `.fit()`, applied in `.transform()`, same lifecycle as `median_vintage_`). Captures "priced above/below typical for its variety" rather than absolute price.
- **`config.py`** (`FeatureEngineerArtifacts`): add `variety_avg_log_price: dict[str, float]` — the per-variety mean `log_price` lookup fitted in `.fit()`, persisted/inspectable the same way `median_vintage` already is.

Existing tests (`tests/preprocessing/test_feature_engineer.py`, `test_encoders.py`) get new cases for each addition, mirroring the existing `has_designation`/`is_luxury` test style. `data/processed/xgboost/` regenerates again, now with 44 columns (41 + province + region1_freq + price_vs_variety).

**After parts 1+2**: re-run notebook 03 with the existing `configs/xgboost_tuning*.json` files unchanged (same search spaces) against the enriched dataset. Compare against the current 4 runs in Step 8's table — this alone tells us whether richer tabular signal moves test R², before text is even in play.

### Part 3 — TF-IDF text augmentation

Restores the sparse-input work done and then reverted earlier this session:

- `src/diplo_mod_1/domain/predictor.py`: `WineScorePredictor.predict`/`.fit` accept `FeatureMatrix = np.ndarray | scipy.sparse.spmatrix` instead of `np.ndarray` only.
- `src/diplo_mod_1/domain/evaluate.py`: `evaluate_predictor`'s `splits` type accepts `FeatureMatrix` per split.
- `src/diplo_mod_1/training/xgboost_tuner.py`: `XGBoostTuner.tune()`/`.fit_best()` accept `FeatureMatrix` for `X_train`/`X_val`.
- Each gets a test confirming a `scipy.sparse.csr_matrix` input works end-to-end (mirrors the tests written and reverted earlier).

`notebooks/03-train-baseline-xgboost.ipynb`, new steps after the existing persist step:
- Load `data/processed/nn/X_txt_{train,val,test}.npz` + `tfidf_vectorizer.joblib` (for feature names).
- `scipy.sparse.hstack` each split's (now 44-column) tabular block with its 2000-column TF-IDF block.
- Re-run `XGBoostTuner` on the combined matrix, same config file as whatever's active via `XGBOOST_TUNING_CONFIG`, full `ModelRegistry`/W&B logging exactly like every other run, tagged `"<config>+text"` in `reports/xgboost_metrics.json` so Step 8's comparison table shows it distinctly.
- Feature importance / residual-analysis steps re-run against the combined model, with TF-IDF vocabulary appended to the feature-name list for the importance plot.

## Data flow

```
notebook 02 (Step 4, XGBoost export)
  DataCleaner → FeatureEngineer (+ price_vs_variety) → DataSplitter
    → TabularEncoder.fit_transform(..., include_strictness=True)   [parts 1+2]
    → data/processed/xgboost/{X,y}_{train,val,test}.npy  (44 cols)

notebook 03 (existing Steps 1-9, unchanged)
  load 44-col X/y → XGBoostTuner → ModelRegistry → reports/xgboost_metrics.json

notebook 03 (new steps, part 3)
  44-col X  +  data/processed/nn/X_txt_*.npz (2000-col TF-IDF, sparse)
    → scipy.sparse.hstack → 2044-col sparse X
    → XGBoostTuner (sparse-aware) → ModelRegistry (tagged "+text")
    → reports/xgboost_metrics.json (comparable row, same schema)
```

## Testing

- `tests/preprocessing/test_feature_engineer.py`: new cases for `price_vs_variety` (correct value, train-only fit, no leakage).
- `tests/preprocessing/test_encoders.py` / `test_config.py`: new cases for `province_avg_points`, `region1_freq`, updated `FeatureEngineerArtifacts` shape.
- `tests/domain/test_evaluate.py`, `tests/training/test_xgboost_tuner.py`: sparse-input cases (restored from the earlier reverted work).
- `uv run poe lint` / `poe typecheck` / `poe test` after each part — no full notebook execution (stays the user's, per standing instruction).

## Out of scope (explicitly, not forgotten)

- A less-naive `ModelRegistry.best_run()` selection criterion (confidence intervals / statistical significance instead of raw lowest-RMSE) — flagged by the user as a real gap, tracked for its own separate design/plan later.
- Any change to notebook 04 (NN) or notebook 05 (comparison) — this design only touches notebooks 02 and 03.
- Pruning/retention policy for `models/*.joblib` accumulation — still an accepted, unaddressed tradeoff from the earlier `ModelRegistry` design.
