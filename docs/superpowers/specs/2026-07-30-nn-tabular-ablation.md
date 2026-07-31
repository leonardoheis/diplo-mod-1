# Notebook 04B — Tabular-Only NN Ablation

**Goal:** Isolate how much the 2000-column TF-IDF block is actually buying the PyTorch NN (notebook 04) versus the 44 engineered tabular columns alone, without touching notebook 04 itself.

**Architecture:** Same `WineScorePredictorNet`/`NNTuner`/`NNTuningConfig` pipeline as notebook 04, pointed at a 44-column `X` instead of the concatenated 2044-column one. No source code changes — every class involved already derives `input_dim` from `X_train.shape[1]`.

**Tech Stack:** Same as notebook 04 (PyTorch, Optuna, `configs/nn_training.json` unchanged), minus scipy.sparse/TF-IDF/SHAP, which aren't needed here.

## Context

Notebook 04 (full 2044-column input) has been built, reviewed, and run once. Real results:

| | train | val | test |
|---|---|---|---|
| NN (2044 cols, tuned) | RMSE 1.399 / R² 0.788 | RMSE 1.592 / R² 0.726 | RMSE 1.595 / R² 0.725 |
| XGBoost (best, for reference) | RMSE 1.060 / R² 0.878 | RMSE 1.434 / R² 0.778 | RMSE 1.445 / R² 0.775 |

SHAP on the NN showed it leans heavily on the 44 tabular aggregates (`winery_avg_points`, `description_length`, `price_vs_variety`, ...) and barely uses individual TF-IDF terms — unlike XGBoost, whose top SHAP features are almost all TF-IDF unigrams. Before spending more tuning budget on the full model (wider architecture search, more trials, LR scheduler — deferred to a future pass on notebook 04 itself), a cheap, isolated diagnostic answers whether the TF-IDF block is worth its cost at all.

**Decision:** run the full tuning pipeline — `configs/nn_training.json` unchanged (10 trials, same search space, same `max_epochs`/`early_stopping_patience`) — not just a single baseline fit, so the comparison stays tuned-vs-tuned against the numbers above. Expected to run well under the ~29 min the full run took, since 44 columns means no large sparse-to-dense TF-IDF block and a much smaller first `Linear` layer.

## Global Constraints

- No changes to `src/`, `configs/nn_training.json`, `notebooks/04-train-nn-pytorch.ipynb`, `CLAUDE.md`, or `README.md` — this is additive only.
- Build the notebook via `nbformat` (or careful direct `.ipynb` JSON editing) — never execute cells or run `jupyter nbconvert --execute`.
- `uv run poe lint && uv run poe typecheck` must pass clean against the new notebook (nbqa).
- Every code cell must have `execution_count: null` and empty `outputs` — this notebook is for the user to run, not this session.
- `RANDOM_STATE`/`CONFIGS`/`MODELS`/`PROCESSED`/`REPORTS` from `diplo_mod_1.constants` — reuse, don't hardcode.

## Notebook: `notebooks/04b-nn-tabular-ablation.ipynb`

New notebook, not part of the 00→06 report sequence in `CLAUDE.md`'s notebook table — a side diagnostic, not a report section. Lean version of notebook 04: skips the training-curve plot, residual analysis, SHAP, and cross-run comparison table, since none of those are needed to answer "does dropping TF-IDF hurt much?"

### Cell 1 — Imports

Same core imports as notebook 04's imports cell, minus what's TF-IDF/SHAP-specific (no `joblib`, `shap`, `sparse`, or `from torch import nn`), plus `torch` itself (needed for the manual persistence step) and `diplo_mod_1.training.config`'s `RunRecord`/`TuningHistory`:

```python
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import torch
import wandb
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from diplo_mod_1.constants import CONFIGS, MODELS, PROCESSED, RANDOM_STATE, REPORTS
from diplo_mod_1.schemas.evaluation import evaluate_predictor
from diplo_mod_1.training import NNTuner, NNTuningConfig, WineScorePredictorNet, detect_torch_device
from diplo_mod_1.training.config import RunRecord, TuningHistory

load_dotenv(override=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
WANDB_ENABLED = os.environ.get("WANDB_ENABLED", "false").lower() == "true"
```

### Cell 2 — Load tabular-only data

Reuse `data/processed/nn/X_tab_*.npy` / `y_*.npy` directly, skip `X_txt_*.npz` and `tfidf_vectorizer.joblib` entirely:

```python
nn_dir = PROCESSED / "nn"
X = {s: np.load(nn_dir / f"X_tab_{s}.npy") for s in ("train", "val", "test")}
y = {s: np.load(nn_dir / f"y_{s}.npy") for s in ("train", "val", "test")}

feature_meta = json.loads((nn_dir / "feature_names.json").read_text(encoding="utf-8"))
feature_names = feature_meta["feature_names"]

print(f"Tabular-only feature count: {len(feature_names)}")
print(f"X_train shape: {X['train'].shape}")
```

`X["train"]` is already a plain dense `np.ndarray` — no `sparse.hstack`/`csr_matrix` wrapping needed. Expect `(83180, 44)`.

### Cell 3 — Baseline model

Same shape as notebook 04's baseline step (`hidden_sizes=[128, 64]`, `input_dim=X["train"].shape[1]` → 44), same W&B guard/logging pattern, but `group="nn-tabular-ablation"` (not `"nn-baseline"`) so it's distinguishable in W&B if `WANDB_ENABLED`.

### Cell 4 — Optuna tuning

Identical to notebook 04's tuning step: `NNTuner(NNTuningConfig.from_json(CONFIGS / "nn_training.json"))`, unchanged config file.

### Cell 5 — Final refit

`tuner.fit_best(X["train"], y["train"], X["val"], y["val"], study, callbacks=epoch_callbacks)` — uses the `callbacks` parameter added to `fit_best` in the final-review fix round.

### Cell 6 — Evaluate

Identical call: `evaluate_predictor(best_model, splits, model_type="neural_net")`, same per-split print loop. Same `ModelMetrics`/`EvaluationResult` schema as notebook 04's output — this is the "same metrics."

### Cell 7 — Persist (to separate files, NOT via `NNModelRegistry`)

**Design decision:** `NNModelRegistry.save_run` unconditionally overwrites `models_dir / "nn_best.pt"` (hardcoded in `nn_registry.py`) whenever its own history's best run changes — with a single ablation run in its own history, that's every time. Passing the same `MODELS` dir would clobber the real NN's `models/nn_best.pt` pointer that notebook 05 will use. A subdirectory (e.g. `models/nn_tabular/`) doesn't work either — `.gitignore`'s `models/*` + `!models/*.pt` negation only re-includes files directly under `models/`, not nested subdirectories.

**Fix:** skip `NNModelRegistry.save_run`; manually build the same `RunRecord`/`TuningHistory` objects (already model-agnostic, no changes needed) and write to separate, top-level files:

```python
run_id = f"nn_tabular_ablation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
model_filename = "nn_tabular_ablation.pt"
torch.save(
    {
        "state_dict": best_model.model_.state_dict(),
        "input_dim": best_model.input_dim,
        "hidden_sizes": best_model.hidden_sizes,
        "dropout": best_model.dropout,
    },
    MODELS / model_filename,
)
record = RunRecord(
    run_id=run_id,
    tuning_config="nn_training.json",
    model_filename=model_filename,
    best_params=study.best_params,
    metrics=result.metrics,
)
history = TuningHistory(runs=[record], best_run_id=run_id)
(REPORTS / "nn_tabular_ablation_metrics.json").write_text(history.model_dump_json(indent=2), encoding="utf-8")
print(f"Model saved to {MODELS / model_filename}")
```

`reports/nn_metrics.json` and `models/nn_best.pt` (the real NN's artifacts) are never touched.

### Cell 8 — Quick comparison print

Read back the full-feature NN's current best from `reports/nn_metrics.json` (`TuningHistory.model_validate_json`, same pattern notebook 04's comparison step uses) and print both test RMSE/R² side by side. Guard against `reports/nn_metrics.json` not existing yet (skip the cell's comparison gracefully) so the notebook doesn't hard-fail if run before notebook 04:

```python
metrics_path = REPORTS / "nn_metrics.json"
if metrics_path.exists():
    full_history = TuningHistory.model_validate_json(metrics_path.read_text(encoding="utf-8"))
    full_best = next(r for r in full_history.runs if r.run_id == full_history.best_run_id)
    full_test = next(m for m in full_best.metrics if m.split == "test")
    ablation_test = next(m for m in result.metrics if m.split == "test")
    print(f"Full NN   (2044 cols) test RMSE={full_test.rmse:.4f}  R2={full_test.r2:.4f}")
    print(f"Tabular NN  (44 cols) test RMSE={ablation_test.rmse:.4f}  R2={ablation_test.r2:.4f}")
else:
    print("reports/nn_metrics.json not found yet — run notebook 04 first for a full comparison.")
```

### Cell 9 — Short markdown note

2-3 sentences (not a full design-notes section): states the notebook's purpose as an ablation isolating the tabular-only signal, and that results feed the decision on whether/how to invest further in the TF-IDF block for notebook 04's next iteration.

## Files touched

- New: `notebooks/04b-nn-tabular-ablation.ipynb`
- New artifacts (written when the user runs it): `reports/nn_tabular_ablation_metrics.json`, `models/nn_tabular_ablation.pt`
- No changes to `src/`, `configs/nn_training.json`, `notebooks/04-train-nn-pytorch.ipynb`, `CLAUDE.md`, or `README.md`.

## Verification

- `uv run poe lint && uv run poe typecheck` after writing the notebook — must pass clean.
- Every code cell has `execution_count: null` and empty `outputs`.
- Actually running it (confirming real timing and real metrics) is the user's to do.
