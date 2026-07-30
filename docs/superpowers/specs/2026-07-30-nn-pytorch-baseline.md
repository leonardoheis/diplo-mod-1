# PyTorch Neural Network (notebook 04) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build notebook 04 — a PyTorch MLP regressor predicting wine `points`, trained on the same 2044-column (44 tabular + 2000 TF-IDF) feature set XGBoost already uses, with Optuna tuning, versioned checkpoints, and GPU auto-detection — structurally mirroring notebook 03 so notebook 05 can compare them.

**Architecture:** Single feed-forward MLP on the concatenated sparse+dense input (no two-branch network). Config-driven `src/` classes (mirrors `XGBoostTuner`/`ModelRegistry`), thin notebook orchestration. Device detection consolidated into one shared module rather than scattered per-library.

**Tech Stack:** PyTorch (CUDA on Windows, MPS on Mac, CPU fallback), Optuna, Pydantic (config schemas), scipy.sparse, existing `evaluate_predictor`/`ModelMetrics`/`EvaluationResult`/`WineScorePredictor` from `domain`/`schemas`.

## Global Constraints

- Python 3.10+ (project floor) — no syntax/APIs requiring 3.11+.
- Per-task commits ARE authorized for this subagent-driven-development run, on the existing `feature/nn-pytorch-baseline` branch only (explicitly confirmed with the user — this is a scoped exception to `CLAUDE.md`'s general no-commit rule, needed because the review tooling diffs commits). **Never push or open a PR** — that still requires separate explicit instruction.
- After each task: `uv run poe lint`, `uv run poe typecheck`, `uv run poe test` — all three, not a subset.
- No full notebook execution (`poe nbtest`, `poe check`, `jupyter nbconvert --execute`) — that stays the user's to run.
- `WineScorePredictor` protocol (`src/diplo_mod_1/domain/predictor.py`) is the contract: `predict(self, X: FeatureMatrix) -> np.ndarray`, `fit(self, X: FeatureMatrix, y: np.ndarray) -> Self`. `FeatureMatrix = np.ndarray | sparse.spmatrix | sparse.sparray`.
- `RANDOM_STATE` (from `src/diplo_mod_1/constants.py`) is the project-wide seed constant — reuse it, don't hardcode `42` again.
- All new hyperparameters live in `configs/*.json`, loaded via a Pydantic `from_json(path)` classmethod — never hardcode a search space in the notebook (matches `XGBoostTuningConfig`'s pattern).

---

### Task 1: Fix the torch CUDA build (Windows only)

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `torch.cuda.is_available() == True` on this Windows machine after `uv sync`, with no change to what the Mac install resolves to.

- [ ] **Step 1: Confirm the current torch build is CPU-only**

Run: `uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"`
Expected: `2.11.0+cpu None` (confirms the problem before fixing it).

- [ ] **Step 2: Add the platform-conditional CUDA source**

In `pyproject.toml`, add (the `cu128` tag matches the currently-resolved torch version `2.11.0` exactly — verified `torch-2.11.0+cu128-cp310-cp310-win_amd64.whl` exists on PyTorch's index):

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu128", marker = "sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

- [ ] **Step 3: Sync and verify**

Run: `uv sync`
Run: `uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
Expected: version string ends in `+cu128`, `torch.version.cuda` is a real CUDA version string (not `None`), `torch.cuda.is_available()` is `True`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: install CUDA-enabled torch build for Windows"
```

---

### Task 2: Consolidate device detection into `training/device.py`

Moves the existing XGBoost-only `detect_device()` out of `xgboost_tuner.py` (it's misplaced now that a second, PyTorch-specific detector is being added) into one shared module, and adds the new PyTorch one alongside it.

**Files:**
- Create: `src/diplo_mod_1/training/device.py`
- Modify: `src/diplo_mod_1/training/xgboost_tuner.py` (remove `detect_device`, import `detect_xgboost_device` from `device.py` instead)
- Modify: `src/diplo_mod_1/training/__init__.py` (export `detect_xgboost_device`, `detect_torch_device` instead of `detect_device`)
- Modify: `notebooks/03-train-baseline-xgboost.ipynb` — imports cell and Step 2 baseline cell: `detect_device` → `detect_xgboost_device`
- Modify: `tests/training/test_xgboost_tuner.py` — remove `test_detect_device_returns_cpu_or_cuda` and its import
- Create: `tests/training/test_device.py`

**Interfaces:**
- Produces: `detect_xgboost_device() -> str` (returns `"cuda"` or `"cpu"`), `detect_torch_device() -> str` (returns `"cuda"`, `"mps"`, or `"cpu"`). Both `@lru_cache(maxsize=1)`.

- [ ] **Step 1: Write the failing tests**

`tests/training/test_device.py`:

```python
"""Tests for device auto-detection (XGBoost's CUDA probe, PyTorch's CUDA/MPS/CPU)."""

from diplo_mod_1.training.device import detect_torch_device, detect_xgboost_device


def test_detect_xgboost_device_returns_cpu_or_cuda() -> None:
    assert detect_xgboost_device() in {"cpu", "cuda"}


def test_detect_torch_device_returns_valid_string() -> None:
    assert detect_torch_device() in {"cpu", "cuda", "mps"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/training/test_device.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'diplo_mod_1.training.device'`

- [ ] **Step 3: Create `device.py` with both detectors**

```python
"""Device auto-detection for XGBoost (CUDA-or-CPU) and PyTorch (CUDA/MPS/CPU).

Kept in one module rather than split by library — "which accelerator is
available" is a single cross-cutting concern, not something that belongs
inside a tuner class named after one specific model type.
"""

from functools import lru_cache

import numpy as np
from xgboost import XGBRegressor


@lru_cache(maxsize=1)
def detect_xgboost_device() -> str:
    """CUDA if XGBoost can actually train on it here, else CPU.

    XGBoost has no Apple Silicon/MPS backend — CUDA is the only GPU path.
    Probes XGBoost directly rather than trusting ``torch.cuda.is_available()``:
    XGBoost bundles its own CUDA runtime independent of torch's, so a
    CPU-only torch build says nothing about whether XGBoost's CUDA works.
    Cached — the throwaway fit only runs once per process.
    """
    try:
        XGBRegressor(tree_method="hist", device="cuda", n_estimators=1).fit(
            np.zeros((2, 1), dtype=np.float32), np.zeros(2, dtype=np.float32)
        )
        device = "cuda"
    except Exception:
        device = "cpu"

    print(f"XGBoost device: {device}")
    return device


@lru_cache(maxsize=1)
def detect_torch_device() -> str:
    """CUDA, then Apple Silicon MPS, then CPU — whichever this machine has.

    Unlike XGBoost, torch's own ``torch.cuda.is_available()``/
    ``torch.backends.mps.is_available()`` are trustworthy here — there's no
    separate bundled-runtime mismatch to work around, torch is the only
    library involved.
    """
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"PyTorch device: {device}")
    return device
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/training/test_device.py -v`
Expected: PASS (2 tests). On this machine, `detect_xgboost_device()` should print `XGBoost device: cuda` and `detect_torch_device()` should print `PyTorch device: cuda` (after Task 1's fix).

- [ ] **Step 5: Update `xgboost_tuner.py` to use the moved function**

In `src/diplo_mod_1/training/xgboost_tuner.py`:
- Remove the `detect_device()` function definition and its `from functools import lru_cache` import (no longer used in this file).
- Add `from diplo_mod_1.training.device import detect_xgboost_device`.
- In `_make_model`, change `device=detect_device()` to `device=detect_xgboost_device()`.

- [ ] **Step 6: Update `tests/training/test_xgboost_tuner.py`**

Remove the `test_detect_device_returns_cpu_or_cuda` test and its `detect_device` import (now covered by `test_device.py`):

```python
from diplo_mod_1.training.config import XGBoostSearchSpace, XGBoostTuningConfig
from diplo_mod_1.training.xgboost_tuner import XGBoostTuner
```

(no longer imports `detect_device`)

- [ ] **Step 7: Update `training/__init__.py`**

```python
"""XGBoost hyperparameter tuning package."""

from diplo_mod_1.training.config import (
    RunRecord,
    TuningHistory,
    XGBoostSearchSpace,
    XGBoostTuningConfig,
)
from diplo_mod_1.training.device import detect_torch_device, detect_xgboost_device
from diplo_mod_1.training.registry import ModelRegistry
from diplo_mod_1.training.xgboost_tuner import XGBoostTuner

__all__ = [
    "ModelRegistry",
    "RunRecord",
    "TuningHistory",
    "XGBoostSearchSpace",
    "XGBoostTuner",
    "XGBoostTuningConfig",
    "detect_torch_device",
    "detect_xgboost_device",
]
```

- [ ] **Step 8: Update notebook 03**

In `notebooks/03-train-baseline-xgboost.ipynb`'s imports cell: change `detect_device` to `detect_xgboost_device` in the `from diplo_mod_1.training import (...)` block. In the Step 2 baseline cell: change `device=detect_device()` to `device=detect_xgboost_device()`.

- [ ] **Step 9: Full verification**

Run: `uv run poe lint && uv run poe typecheck && uv run poe test`
Expected: all pass, no references to the old `detect_device` name remain (search for it: `grep -rn "detect_device" src tests notebooks` should only show `detect_xgboost_device`/`detect_torch_device`).

- [ ] **Step 10: Commit**

```bash
git add src/diplo_mod_1/training/device.py src/diplo_mod_1/training/xgboost_tuner.py src/diplo_mod_1/training/__init__.py tests/training/test_device.py tests/training/test_xgboost_tuner.py notebooks/03-train-baseline-xgboost.ipynb
git commit -m "refactor: consolidate device detection into training/device.py"
```

---

### Task 3: NN config schema (`nn_config.py` + `configs/nn_training.json`)

**Files:**
- Create: `src/diplo_mod_1/training/nn_config.py`
- Create: `configs/nn_training.json`
- Test: `tests/training/test_nn_config.py`

**Interfaces:**
- Produces: `NNSearchSpace` (Pydantic model: `architecture: list[str]`, `dropout: tuple[float, float]`, `learning_rate: tuple[float, float]`, `weight_decay: tuple[float, float]`, `batch_size: list[int]`), `NNTuningConfig` (Pydantic model: `n_trials: int`, `max_epochs: int`, `early_stopping_patience: int`, `random_state: int`, `search_space: NNSearchSpace`, classmethod `from_json(path: Path) -> NNTuningConfig`).

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for NN tuning config."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from diplo_mod_1.training.nn_config import NNSearchSpace, NNTuningConfig


def test_default_config_valid() -> None:
    config = NNTuningConfig()
    assert config.n_trials >= 1
    assert config.max_epochs >= 1


def test_inverted_bounds_raise() -> None:
    with pytest.raises(ValidationError):
        NNSearchSpace(dropout=(0.5, 0.1))


def test_from_json_round_trips(tmp_path: Path) -> None:
    original = NNTuningConfig(n_trials=5, search_space=NNSearchSpace(batch_size=[32, 64]))
    path = tmp_path / "config.json"
    path.write_text(original.model_dump_json(), encoding="utf-8")

    loaded = NNTuningConfig.from_json(path)

    assert loaded == original


def test_repo_config_file_is_valid() -> None:
    path = Path(__file__).parents[2] / "configs" / "nn_training.json"
    config = NNTuningConfig.from_json(path)
    assert config.n_trials >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/training/test_nn_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'diplo_mod_1.training.nn_config'`

- [ ] **Step 3: Write `nn_config.py`**

```python
"""Pydantic configuration for PyTorch NN hyperparameter tuning."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from diplo_mod_1.constants import RANDOM_STATE


class NNSearchSpace(BaseModel):
    """Optuna suggestion ranges/choices for each NN hyperparameter.

    ``architecture`` encodes hidden-layer sizes as underscore-joined strings
    (e.g. ``"256_64"`` -> ``[256, 64]``) — a categorical choice among a few
    fixed depths, since tuning variable-depth architectures directly in
    Optuna is awkward.
    """

    architecture: list[str] = ["128_64", "256_64", "512_128_32"]
    dropout: tuple[float, float] = (0.1, 0.5)
    learning_rate: tuple[float, float] = (1e-4, 1e-2)
    weight_decay: tuple[float, float] = (1e-6, 1e-2)
    batch_size: list[int] = [64, 128, 256]

    @field_validator("dropout", "learning_rate", "weight_decay")
    @classmethod
    def bounds_ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] > v[1]:
            raise ValueError(f"lower bound {v[0]} exceeds upper bound {v[1]}")
        return v


class NNTuningConfig(BaseModel):
    """Hyperparameters for NNTuner.

    Usage::

        # defaults
        config = NNTuningConfig()

        # load from JSON file
        config = NNTuningConfig.from_json(Path("configs/nn_training.json"))
    """

    n_trials: int = Field(default=30, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    random_state: int = RANDOM_STATE
    search_space: NNSearchSpace = Field(default_factory=NNSearchSpace)

    @classmethod
    def from_json(cls, path: Path) -> "NNTuningConfig":
        """Load config from a JSON file (same shape as ``model_dump()``)."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Create the default config file**

`configs/nn_training.json` — deliberately starts small (per grilling-session discussion: notebook 03's TF-IDF trials took 2.5+ min each on this same data; better to get one full pass working and see real per-epoch timing before committing to a bigger search):

```json
{
  "n_trials": 10,
  "max_epochs": 30,
  "early_stopping_patience": 5,
  "random_state": 42,
  "search_space": {
    "architecture": ["128_64", "256_64", "512_128_32"],
    "dropout": [0.1, 0.5],
    "learning_rate": [0.0001, 0.01],
    "weight_decay": [0.000001, 0.01],
    "batch_size": [64, 128, 256]
  }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/training/test_nn_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/diplo_mod_1/training/nn_config.py configs/nn_training.json tests/training/test_nn_config.py
git commit -m "feat: add NN tuning config schema and default config"
```

---

### Task 4: `WineScoreNet` (the network) + `SparseTabularDataset`

**Files:**
- Create: `src/diplo_mod_1/training/nn_model.py`
- Test: `tests/training/test_nn_model.py`

**Interfaces:**
- Consumes: `FeatureMatrix` from `diplo_mod_1.domain.predictor`.
- Produces: `WineScoreNet(nn.Module)` — `__init__(self, input_dim: int, hidden_sizes: list[int], dropout: float = 0.2)`, `forward(self, x: torch.Tensor) -> torch.Tensor` (shape `(batch,)`). `SparseTabularDataset(Dataset)` — `__init__(self, X: FeatureMatrix, y: np.ndarray | None = None)`, `__len__(self) -> int`, `__getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for WineScoreNet and SparseTabularDataset."""

import numpy as np
import torch
from scipy import sparse

from diplo_mod_1.training.nn_model import SparseTabularDataset, WineScoreNet


def test_wine_score_net_forward_shape() -> None:
    net = WineScoreNet(input_dim=10, hidden_sizes=[8, 4], dropout=0.1)
    x = torch.randn(5, 10)
    out = net(x)
    assert out.shape == (5,)


def test_wine_score_net_handles_single_hidden_layer() -> None:
    net = WineScoreNet(input_dim=6, hidden_sizes=[4], dropout=0.0)
    out = net(torch.randn(3, 6))
    assert out.shape == (3,)


def test_dataset_len_matches_row_count() -> None:
    X = np.zeros((7, 4), dtype=np.float32)
    y = np.arange(7, dtype=np.float32)
    ds = SparseTabularDataset(X, y)
    assert len(ds) == 7


def test_dataset_getitem_dense_input() -> None:
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    ds = SparseTabularDataset(X, y)
    features, target = ds[1]
    assert features.shape == (4,)
    assert torch.equal(features, torch.tensor(X[1]))
    assert target.item() == 20.0


def test_dataset_getitem_sparse_input_densifies_one_row() -> None:
    X = sparse.csr_matrix(np.eye(5, dtype=np.float32))
    y = np.arange(5, dtype=np.float32)
    ds = SparseTabularDataset(X, y)
    features, target = ds[2]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (5,)
    assert features[2].item() == 1.0
    assert target.item() == 2.0


def test_dataset_without_targets_returns_placeholder() -> None:
    X = np.zeros((2, 3), dtype=np.float32)
    ds = SparseTabularDataset(X)
    features, target = ds[0]
    assert features.shape == (3,)
    assert target.item() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/training/test_nn_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'diplo_mod_1.training.nn_model'`

- [ ] **Step 3: Write `nn_model.py`**

```python
"""WineScoreNet — the PyTorch regression MLP, and its dense/sparse Dataset."""

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import Dataset

from diplo_mod_1.domain.predictor import FeatureMatrix


class WineScoreNet(nn.Module):
    """Feed-forward MLP regressor: ``[Linear -> BatchNorm1d -> ReLU -> Dropout]``
    per hidden layer, then a single ``Linear(*, 1)`` output.

    Trained on the full concatenated feature matrix (tabular + TF-IDF, 2044
    columns) — same "just concatenate and let the model learn" approach that
    already worked for XGBoost, not a two-branch architecture.
    """

    def __init__(self, input_dim: int, hidden_sizes: list[int], dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class SparseTabularDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Wraps a dense-or-sparse feature matrix for PyTorch's DataLoader.

    Densifies one row at a time in ``__getitem__`` rather than the whole
    matrix up front — the TF-IDF block alone would be ~665MB dense despite
    being ~1% non-zero; this keeps memory proportional to batch size instead.
    """

    def __init__(self, X: FeatureMatrix, y: np.ndarray | None = None) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.X[idx]
        if sparse.issparse(row):
            row = row.toarray()
        row = np.asarray(row, dtype=np.float32).ravel()
        features = torch.from_numpy(row)
        target = (
            torch.tensor(self.y[idx], dtype=torch.float32)
            if self.y is not None
            else torch.tensor(0.0)
        )
        return features, target
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/training/test_nn_model.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/diplo_mod_1/training/nn_model.py tests/training/test_nn_model.py
git commit -m "feat: add WineScoreNet and SparseTabularDataset"
```

---

### Task 5: `WineScorePredictorNet` (sklearn-style training/inference wrapper)

**Files:**
- Modify: `src/diplo_mod_1/training/nn_model.py` (append to the file created in Task 4)
- Test: `tests/training/test_nn_predictor.py`

**Interfaces:**
- Consumes: `WineScoreNet`, `SparseTabularDataset` (Task 4), `detect_torch_device` (Task 2), `FeatureMatrix`.
- Produces: `WineScorePredictorNet` implementing `WineScorePredictor` — `__init__(self, input_dim: int, hidden_sizes: list[int], dropout: float = 0.2, learning_rate: float = 1e-3, weight_decay: float = 0.0, batch_size: int = 128, max_epochs: int = 100, early_stopping_patience: int = 10, random_state: int = 42, device: str | None = None)`; `fit(self, X: FeatureMatrix, y: np.ndarray, *, X_val: FeatureMatrix | None = None, y_val: np.ndarray | None = None, callbacks: list[Callable[[int, float, float], None]] | None = None) -> "WineScorePredictorNet"`; `predict(self, X: FeatureMatrix) -> np.ndarray`. After `fit`, exposes `self.train_losses_: list[float]`, `self.val_losses_: list[float]`, `self.best_epoch_: int`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for WineScorePredictorNet."""

import numpy as np

from diplo_mod_1.domain.predictor import WineScorePredictor
from diplo_mod_1.training.nn_model import WineScorePredictorNet

_FAST_KWARGS = dict(
    input_dim=5,
    hidden_sizes=[4],
    max_epochs=5,
    early_stopping_patience=2,
    batch_size=8,
    device="cpu",
)


def test_implements_wine_score_predictor_protocol() -> None:
    model = WineScorePredictorNet(**_FAST_KWARGS)
    assert isinstance(model, WineScorePredictor)


def test_fit_returns_self(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    model = WineScorePredictorNet(**_FAST_KWARGS)
    result = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert result is model


def test_fit_tracks_loss_curves(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    model = WineScorePredictorNet(**_FAST_KWARGS)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert len(model.train_losses_) >= 1
    assert len(model.val_losses_) == len(model.train_losses_)
    assert model.best_epoch_ >= 0


def test_predict_returns_correct_shape(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    model = WineScorePredictorNet(**_FAST_KWARGS)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    preds = model.predict(X_val)
    assert preds.shape == y_val.shape


def test_fit_without_validation_set_still_works(regression_split) -> None:
    X_train, y_train, _, _ = regression_split
    model = WineScorePredictorNet(**_FAST_KWARGS)
    model.fit(X_train, y_train)
    assert len(model.train_losses_) == _FAST_KWARGS["max_epochs"]  # no early stop without val


def test_fit_invokes_callback(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    model = WineScorePredictorNet(**_FAST_KWARGS)
    calls: list[int] = []
    model.fit(
        X_train, y_train, X_val=X_val, y_val=y_val,
        callbacks=[lambda epoch, train_loss, val_loss: calls.append(epoch)],
    )
    assert calls == list(range(len(calls)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/training/test_nn_predictor.py -v`
Expected: FAIL — `ImportError: cannot import name 'WineScorePredictorNet'`

- [ ] **Step 3: Append `WineScorePredictorNet` to `nn_model.py`**

```python
from collections.abc import Callable

from torch.utils.data import DataLoader

from diplo_mod_1.training.device import detect_torch_device


class WineScorePredictorNet:
    """Sklearn-style wrapper around ``WineScoreNet`` — implements
    ``WineScorePredictor`` (``domain/predictor.py``) so ``evaluate_predictor``
    can score it identically to the XGBoost model.

    ``fit``'s extra keyword-only args (``X_val``, ``y_val``, ``callbacks``)
    are additional to the Protocol's minimal ``fit(X, y)`` — fine, since
    ``@runtime_checkable`` Protocols only check method presence, not exact
    signatures.

    Usage::

        model = WineScorePredictorNet(input_dim=2044, hidden_sizes=[256, 64])
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        preds = model.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        X: FeatureMatrix,
        y: np.ndarray,
        *,
        X_val: FeatureMatrix | None = None,
        y_val: np.ndarray | None = None,
        callbacks: list[Callable[[int, float, float], None]] | None = None,
    ) -> "WineScorePredictorNet":
        torch.manual_seed(self.random_state)
        device = self.device or detect_torch_device()
        self.model_ = WineScoreNet(self.input_dim, self.hidden_sizes, self.dropout).to(device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(
            SparseTabularDataset(X, y), batch_size=self.batch_size, shuffle=True
        )
        has_val = X_val is not None and y_val is not None
        val_loader = (
            DataLoader(SparseTabularDataset(X_val, y_val), batch_size=self.batch_size)
            if has_val
            else None
        )

        self.train_losses_: list[float] = []
        self.val_losses_: list[float] = []
        self.best_epoch_ = 0
        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            self.model_.train()
            running_loss, n_samples = 0.0, 0
            for features, target in train_loader:
                features, target = features.to(device), target.to(device)
                optimizer.zero_grad()
                preds = self.model_(features)
                loss = loss_fn(preds, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * features.size(0)
                n_samples += features.size(0)
            train_loss = running_loss / n_samples
            self.train_losses_.append(train_loss)

            val_loss = train_loss
            if val_loader is not None:
                self.model_.eval()
                running_val, n_val = 0.0, 0
                with torch.no_grad():
                    for features, target in val_loader:
                        features, target = features.to(device), target.to(device)
                        running_val += loss_fn(self.model_(features), target).item() * features.size(0)
                        n_val += features.size(0)
                val_loss = running_val / n_val
            self.val_losses_.append(val_loss)

            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                self.best_epoch_ = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if has_val and epochs_without_improvement >= self.early_stopping_patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X: FeatureMatrix) -> np.ndarray:
        device = self.device or detect_torch_device()
        self.model_.to(device)
        self.model_.eval()
        loader = DataLoader(SparseTabularDataset(X), batch_size=self.batch_size)
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for features, _ in loader:
                predictions.append(self.model_(features.to(device)).cpu().numpy())
        return np.concatenate(predictions)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/training/test_nn_predictor.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Full verification**

Run: `uv run poe lint && uv run poe typecheck && uv run poe test`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/diplo_mod_1/training/nn_model.py tests/training/test_nn_predictor.py
git commit -m "feat: add WineScorePredictorNet training/inference wrapper"
```

---

### Task 6: `NNTuner` (Optuna search, mirrors `XGBoostTuner`)

**Files:**
- Create: `src/diplo_mod_1/training/nn_tuner.py`
- Test: `tests/training/test_nn_tuner.py`

**Interfaces:**
- Consumes: `NNTuningConfig`/`NNSearchSpace` (Task 3), `WineScorePredictorNet` (Task 5), `FeatureMatrix`.
- Produces: `NNTuner` — `__init__(self, config: NNTuningConfig | None = None)`, `tune(self, X_train: FeatureMatrix, y_train: np.ndarray, X_val: FeatureMatrix, y_val: np.ndarray, callbacks: list[OptunaTrialCallback] | None = None) -> optuna.Study`, `fit_best(self, X_train, y_train, X_val, y_val, study: optuna.Study) -> WineScorePredictorNet`. Same parameter order/names as `XGBoostTuner` for consistency.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for NNTuner."""

import numpy as np
import optuna

from diplo_mod_1.training.nn_config import NNSearchSpace, NNTuningConfig
from diplo_mod_1.training.nn_tuner import NNTuner

_FAST_CONFIG = NNTuningConfig(
    n_trials=2,
    max_epochs=3,
    early_stopping_patience=2,
    search_space=NNSearchSpace(architecture=["4_2"], batch_size=[8]),
)


def test_tune_returns_completed_study(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    tuner = NNTuner(_FAST_CONFIG)
    study = tuner.tune(X_train, y_train, X_val, y_val)
    assert isinstance(study, optuna.Study)
    assert len(study.trials) == 2
    assert study.best_value >= 0


def test_tune_invokes_callback(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    tuner = NNTuner(_FAST_CONFIG)
    calls: list[int] = []
    tuner.tune(X_train, y_train, X_val, y_val, callbacks=[lambda s, t: calls.append(t.number)])
    assert calls == [0, 1]


def test_fit_best_returns_fitted_model(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    tuner = NNTuner(_FAST_CONFIG)
    study = tuner.tune(X_train, y_train, X_val, y_val)
    model = tuner.fit_best(X_train, y_train, X_val, y_val, study)
    preds = model.predict(X_val)
    assert preds.shape == y_val.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/training/test_nn_tuner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'diplo_mod_1.training.nn_tuner'`

- [ ] **Step 3: Write `nn_tuner.py`**

```python
"""NNTuner — Optuna-based hyperparameter search for WineScorePredictorNet."""

from collections.abc import Callable

import numpy as np
import optuna
from sklearn.metrics import root_mean_squared_error

from diplo_mod_1.domain.predictor import FeatureMatrix
from diplo_mod_1.training.device import detect_torch_device
from diplo_mod_1.training.nn_config import NNTuningConfig
from diplo_mod_1.training.nn_model import WineScorePredictorNet

OptunaTrialCallback = Callable[[optuna.Study, optuna.trial.FrozenTrial], None]


class NNTuner:
    """Bayesian (TPE) hyperparameter search for ``WineScorePredictorNet``,
    validated by early stopping — mirrors ``XGBoostTuner``'s shape.

    Usage::

        tuner = NNTuner()
        study = tuner.tune(X_train, y_train, X_val, y_val)
        model = tuner.fit_best(X_train, y_train, X_val, y_val, study)
    """

    def __init__(self, config: NNTuningConfig | None = None) -> None:
        self.config = config or NNTuningConfig()

    def _make_model(self, input_dim: int, params: dict[str, float | int | str]) -> WineScorePredictorNet:
        hidden_sizes = [int(x) for x in params["architecture"].split("_")]
        return WineScorePredictorNet(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            max_epochs=self.config.max_epochs,
            early_stopping_patience=self.config.early_stopping_patience,
            random_state=self.config.random_state,
            device=detect_torch_device(),
        )

    def _build_objective(
        self, X_train: FeatureMatrix, y_train: np.ndarray, X_val: FeatureMatrix, y_val: np.ndarray
    ) -> Callable[[optuna.Trial], float]:
        space = self.config.search_space

        def objective(trial: optuna.Trial) -> float:
            params = {
                "architecture": trial.suggest_categorical("architecture", space.architecture),
                "dropout": trial.suggest_float("dropout", *space.dropout),
                "learning_rate": trial.suggest_float("learning_rate", *space.learning_rate, log=True),
                "weight_decay": trial.suggest_float("weight_decay", *space.weight_decay, log=True),
                "batch_size": trial.suggest_categorical("batch_size", space.batch_size),
            }
            model = self._make_model(X_train.shape[1], params)
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            return root_mean_squared_error(y_val, model.predict(X_val))

        return objective

    def tune(
        self,
        X_train: FeatureMatrix,
        y_train: np.ndarray,
        X_val: FeatureMatrix,
        y_val: np.ndarray,
        callbacks: list[OptunaTrialCallback] | None = None,
    ) -> optuna.Study:
        """Run the Optuna search; returns the completed study."""
        objective = self._build_objective(X_train, y_train, X_val, y_val)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )
        study.optimize(
            objective, n_trials=self.config.n_trials, show_progress_bar=True, callbacks=callbacks
        )
        return study

    def fit_best(
        self,
        X_train: FeatureMatrix,
        y_train: np.ndarray,
        X_val: FeatureMatrix,
        y_val: np.ndarray,
        study: optuna.Study,
    ) -> WineScorePredictorNet:
        """Refit on the winning trial's params, early-stopped against val."""
        model = self._make_model(X_train.shape[1], study.best_params)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/training/test_nn_tuner.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/diplo_mod_1/training/nn_tuner.py tests/training/test_nn_tuner.py
git commit -m "feat: add NNTuner for Optuna hyperparameter search"
```

---

### Task 7: `NNModelRegistry` (versioned checkpoints, mirrors `ModelRegistry`)

**Files:**
- Create: `src/diplo_mod_1/training/nn_registry.py`
- Test: `tests/training/test_nn_registry.py`

**Interfaces:**
- Consumes: `RunRecord`, `TuningHistory` (from `training/config.py`, already model-agnostic), `WineScorePredictorNet` (Task 5).
- Produces: `NNModelRegistry.save_run(models_dir: Path, metrics_path: Path, model: WineScorePredictorNet, run_id: str, tuning_config_name: str, best_params: dict[str, float], result: EvaluationResult) -> tuple[RunRecord, TuningHistory]` — same parameter order/names as `ModelRegistry.save_run` (deliberately not a shared/generalized class with it — the two serialization mechanics differ, `torch.save(state_dict)` vs `joblib.dump`, and forcing one abstraction for two call sites would be premature).

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for NNModelRegistry."""

from pathlib import Path

from diplo_mod_1.domain.metrics import ModelMetrics
from diplo_mod_1.schemas.evaluation import EvaluationResult
from diplo_mod_1.training.nn_model import WineScorePredictorNet
from diplo_mod_1.training.nn_registry import NNModelRegistry


def _fitted_model(seed: int) -> WineScorePredictorNet:
    import numpy as np

    model = WineScorePredictorNet(
        input_dim=3, hidden_sizes=[4], max_epochs=2, batch_size=4, device="cpu", random_state=seed
    )
    X = np.random.default_rng(seed).normal(size=(8, 3)).astype(np.float32)
    y = np.random.default_rng(seed).normal(size=8).astype(np.float32)
    model.fit(X, y)
    return model


def _result(test_rmse: float) -> EvaluationResult:
    result = EvaluationResult()
    result.add(ModelMetrics(model_type="neural_net", split="train", rmse=0.5, mae=0.4, r2=0.9))
    result.add(ModelMetrics(model_type="neural_net", split="val", rmse=1.0, mae=0.8, r2=0.8))
    result.add(ModelMetrics(model_type="neural_net", split="test", rmse=test_rmse, mae=0.8, r2=0.8))
    return result


def test_first_run_becomes_best(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    record, history = NNModelRegistry.save_run(
        tmp_path, metrics_path, _fitted_model(1), "run-a", "cfg.json", {"dropout": 0.2}, _result(1.5)
    )

    assert record.model_filename == "run-a.pt"
    assert (tmp_path / record.model_filename).exists()
    assert (tmp_path / "nn_best.pt").exists()
    assert history.best_run_id == "run-a"
    assert len(history.runs) == 1


def test_better_run_replaces_best(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    NNModelRegistry.save_run(
        tmp_path, metrics_path, _fitted_model(1), "run-a", "cfg.json", {}, _result(1.5)
    )
    _, history = NNModelRegistry.save_run(
        tmp_path, metrics_path, _fitted_model(2), "run-b", "cfg.json", {}, _result(1.0)
    )

    assert history.best_run_id == "run-b"


def test_loading_pre_existing_incompatible_file_starts_fresh(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"metrics": [{"split": "test", "rmse": 1.0}]}', encoding="utf-8")

    record, history = NNModelRegistry.save_run(
        tmp_path, metrics_path, _fitted_model(1), "run-a", "cfg.json", {}, _result(1.5)
    )

    assert len(history.runs) == 1
    assert history.runs[0].run_id == record.run_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/training/test_nn_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'diplo_mod_1.training.nn_registry'`

- [ ] **Step 3: Write `nn_registry.py`**

```python
"""NNModelRegistry — persists versioned PyTorch checkpoints and tracks the best run.

Deliberately parallel to (not sharing a base class with) ``ModelRegistry``
(``registry.py``) — XGBoost's ``joblib.dump`` of a full picklable object and
PyTorch's ``torch.save(state_dict)`` + architecture config are different
enough serialization mechanics that a shared abstraction for exactly these
two call sites would be premature generalization.
"""

import shutil
from pathlib import Path

import torch

from diplo_mod_1.schemas.evaluation import EvaluationResult
from diplo_mod_1.training.config import RunRecord, TuningHistory
from diplo_mod_1.training.nn_model import WineScorePredictorNet


class NNModelRegistry:
    """Writes one checkpoint file per tuning run and keeps a JSON history of all of them.

    Every run's ``state_dict`` (plus enough config to reconstruct the
    network) is kept as ``<run_id>.pt`` rather than overwritten, and
    ``nn_best.pt`` always points at whichever run has the lowest test-split
    RMSE on record — same contract as ``ModelRegistry``.

    Usage::

        run_record, history = NNModelRegistry.save_run(
            MODELS, REPORTS / "nn_metrics.json",
            best_model, run_id, tuning_config_name, study.best_params, result,
        )
    """

    @staticmethod
    def save_run(
        models_dir: Path,
        metrics_path: Path,
        model: WineScorePredictorNet,
        run_id: str,
        tuning_config_name: str,
        best_params: dict[str, float],
        result: EvaluationResult,
    ) -> tuple[RunRecord, TuningHistory]:
        """Save ``model``, append its run to the history, and update the best pointer."""
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        model_filename = f"{run_id}.pt"
        torch.save(
            {
                "state_dict": model.model_.state_dict(),
                "input_dim": model.input_dim,
                "hidden_sizes": model.hidden_sizes,
                "dropout": model.dropout,
            },
            models_dir / model_filename,
        )

        history = NNModelRegistry._load_history(metrics_path)
        record = RunRecord(
            run_id=run_id,
            tuning_config=tuning_config_name,
            model_filename=model_filename,
            best_params=best_params,
            metrics=result.metrics,
        )
        history.runs.append(record)

        runnable = TuningHistory(
            runs=[r for r in history.runs if (models_dir / r.model_filename).exists()]
        )
        best = runnable.best_run(split="test")
        if best is not None:
            history.best_run_id = best.run_id
            shutil.copyfile(models_dir / best.model_filename, models_dir / "nn_best.pt")

        metrics_path.write_text(history.model_dump_json(indent=2), encoding="utf-8")
        return record, history

    @staticmethod
    def _load_history(metrics_path: Path) -> TuningHistory:
        if not metrics_path.exists():
            return TuningHistory()
        try:
            return TuningHistory.model_validate_json(metrics_path.read_text(encoding="utf-8"))
        except ValueError:
            return TuningHistory()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/training/test_nn_registry.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Full verification**

Run: `uv run poe lint && uv run poe typecheck && uv run poe test`

- [ ] **Step 6: Commit**

```bash
git add src/diplo_mod_1/training/nn_registry.py tests/training/test_nn_registry.py
git commit -m "feat: add NNModelRegistry for versioned NN checkpoints"
```

---

### Task 8: Export new names, update `training/__init__.py`

**Files:**
- Modify: `src/diplo_mod_1/training/__init__.py`

**Interfaces:**
- Produces: `diplo_mod_1.training` exposes `NNModelRegistry`, `NNSearchSpace`, `NNTuner`, `NNTuningConfig`, `WineScoreNet`, `WineScorePredictorNet` alongside the existing XGBoost/device exports from Task 2.

- [ ] **Step 1: Update `__init__.py`**

```python
"""XGBoost and PyTorch NN training package."""

from diplo_mod_1.training.config import (
    RunRecord,
    TuningHistory,
    XGBoostSearchSpace,
    XGBoostTuningConfig,
)
from diplo_mod_1.training.device import detect_torch_device, detect_xgboost_device
from diplo_mod_1.training.nn_config import NNSearchSpace, NNTuningConfig
from diplo_mod_1.training.nn_model import WineScoreNet, WineScorePredictorNet
from diplo_mod_1.training.nn_registry import NNModelRegistry
from diplo_mod_1.training.nn_tuner import NNTuner
from diplo_mod_1.training.registry import ModelRegistry
from diplo_mod_1.training.xgboost_tuner import XGBoostTuner

__all__ = [
    "ModelRegistry",
    "NNModelRegistry",
    "NNSearchSpace",
    "NNTuner",
    "NNTuningConfig",
    "RunRecord",
    "TuningHistory",
    "WineScoreNet",
    "WineScorePredictorNet",
    "XGBoostSearchSpace",
    "XGBoostTuner",
    "XGBoostTuningConfig",
    "detect_torch_device",
    "detect_xgboost_device",
]
```

- [ ] **Step 2: Full verification**

Run: `uv run poe lint && uv run poe typecheck && uv run poe test`
Expected: all pass, no import cycles (this mirrors the exact shape that already works for the XGBoost exports — `nn_model.py`/`nn_tuner.py`/`nn_registry.py` only import from `domain`/`schemas`/each other, never from `training/__init__.py` itself, same one-way dependency rule already established this project).

- [ ] **Step 3: Commit**

```bash
git add src/diplo_mod_1/training/__init__.py
git commit -m "feat: export NN training classes from training package"
```

---

### Task 9: Notebook 04 — orchestration

**Files:**
- Modify: `notebooks/04-train-nn-pytorch.ipynb` (currently a 3-cell stub: title, brief, `# TODO`)

**Interfaces:**
- Consumes: everything from Tasks 1-8, plus `data/processed/nn/*` (already on disk — confirmed 44-column `X_tab_*.npy`, `X_txt_*.npz` TF-IDF, `y_*.npy`), `evaluate_predictor`/`EvaluationResult` (`schemas/evaluation.py`, already shared/no changes needed), `RANDOM_STATE`/`MODELS`/`PROCESSED`/`REPORTS`/`CONFIGS` (`constants.py`).

This task is notebook-cell wiring, not pytest-testable the same way — verify via `poe lint`/`typecheck` (which run `nbqa` against the notebook) plus a standalone `python -c` smoke check of the data-loading step, not a full run (that stays the user's).

- [ ] **Step 1: Imports cell**

Mirror notebook 03's imports cell shape exactly (same `load_dotenv(override=True)`, same `WANDB_ENABLED` guard):

```python
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import torch
import wandb
from dotenv import load_dotenv
from scipy import sparse

from diplo_mod_1.constants import CONFIGS, MODELS, PROCESSED, RANDOM_STATE, REPORTS
from diplo_mod_1.schemas.evaluation import evaluate_predictor
from diplo_mod_1.training import (
    NNModelRegistry,
    NNTuner,
    NNTuningConfig,
    TuningHistory,
    WineScoreNet,
    detect_torch_device,
)

load_dotenv(override=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

WANDB_ENABLED = os.environ.get("WANDB_ENABLED", "false").lower() == "true"
```

- [ ] **Step 2: Step 1 — Load processed dataset (tabular + TF-IDF, concatenated)**

```python
nn_dir = PROCESSED / "nn"
X_tab = {s: np.load(nn_dir / f"X_tab_{s}.npy") for s in ("train", "val", "test")}
X_txt = {s: sparse.load_npz(nn_dir / f"X_txt_{s}.npz") for s in ("train", "val", "test")}
y = {s: np.load(nn_dir / f"y_{s}.npy") for s in ("train", "val", "test")}

feature_meta = json.loads((nn_dir / "feature_names.json").read_text(encoding="utf-8"))
tabular_feature_names = feature_meta["feature_names"]

tfidf_vectorizer = joblib.load(nn_dir / "tfidf_vectorizer.joblib")
text_feature_names = list(tfidf_vectorizer.get_feature_names_out())
feature_names = tabular_feature_names + text_feature_names

X = {
    s: sparse.hstack([sparse.csr_matrix(X_tab[s]), X_txt[s]], format="csr")
    for s in ("train", "val", "test")
}

print(f"Combined feature count: {len(feature_names)}")
print(f"X_train shape: {X['train'].shape}")
```

Verify (standalone, not the notebook): `uv run python -c "..."` with the same body run from `notebooks/` as cwd — confirm it prints `Combined feature count: 2044` and `X_train shape: (83180, 2044)` (matches notebook 03's already-verified numbers exactly, since it's the same source data).

- [ ] **Step 3: Step 2 — Baseline model**

```python
nn_config_name = os.environ.get("NN_TRAINING_CONFIG", "nn_training.json")
nn_config = NNTuningConfig.from_json(CONFIGS / nn_config_name)
print(f"Loaded NN config: {nn_config_name}")

run_id = f"{Path(nn_config_name).stem}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

if WANDB_ENABLED:
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "diplo-mod-1"),
        name=f"nn-{run_id}",
        group="nn-baseline",
        job_type="hpo",
        config={"nn_config_name": nn_config_name, **nn_config.model_dump()},
    )

from diplo_mod_1.training import WineScorePredictorNet

baseline = WineScorePredictorNet(
    input_dim=X["train"].shape[1],
    hidden_sizes=[128, 64],
    max_epochs=nn_config.max_epochs,
    early_stopping_patience=nn_config.early_stopping_patience,
    random_state=RANDOM_STATE,
    device=detect_torch_device(),
)
baseline.fit(X["train"], y["train"], X_val=X["val"], y_val=y["val"])

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

baseline_pred = baseline.predict(X["val"])
print(f"Baseline val RMSE: {root_mean_squared_error(y['val'], baseline_pred):.4f}")
print(f"Baseline val MAE:  {mean_absolute_error(y['val'], baseline_pred):.4f}")
print(f"Baseline val R2:   {r2_score(y['val'], baseline_pred):.4f}")
```

(Note: move the `mean_absolute_error`/`r2_score`/`root_mean_squared_error`/`WineScorePredictorNet` imports into the Step 1 imports cell when actually writing the notebook — shown inline here only for readability in this plan.)

- [ ] **Step 4: Step 3 — Optuna tuning**

```python
def log_trial_to_wandb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    wandb.log({"trial": trial.number, "val_rmse": trial.value, **trial.params})


tuner = NNTuner(nn_config)
callbacks = [log_trial_to_wandb] if WANDB_ENABLED else None
study = tuner.tune(X["train"], y["train"], X["val"], y["val"], callbacks=callbacks)

print(f"Best val RMSE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

if WANDB_ENABLED:
    wandb.log({"best_val_rmse": study.best_value, **{f"best_{k}": v for k, v in study.best_params.items()}})
```

- [ ] **Step 5: Step 4 — Final model + per-epoch W&B logging**

```python
def log_epoch_to_wandb(epoch: int, train_loss: float, val_loss: float) -> None:
    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})


epoch_callbacks = [log_epoch_to_wandb] if WANDB_ENABLED else None
best_model = tuner._make_model(X["train"].shape[1], study.best_params)
best_model.fit(X["train"], y["train"], X_val=X["val"], y_val=y["val"], callbacks=epoch_callbacks)
print(f"Best epoch: {best_model.best_epoch_} / {len(best_model.train_losses_)} trained")
```

- [ ] **Step 6: Step 5 — Evaluate on train/val/test**

```python
splits = {"train": (X["train"], y["train"]), "val": (X["val"], y["val"]), "test": (X["test"], y["test"])}
result = evaluate_predictor(best_model, splits, model_type="neural_net")

for m in result.metrics:
    print(f"{m.split:5s}  RMSE={m.rmse:.4f}  MAE={m.mae:.4f}  R2={m.r2:.4f}")
    if WANDB_ENABLED:
        wandb.log({f"{m.split}_rmse": m.rmse, f"{m.split}_mae": m.mae, f"{m.split}_r2": m.r2})
```

- [ ] **Step 7: Step 6 — Training curves**

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(best_model.train_losses_, label="train")
ax.plot(best_model.val_losses_, label="val")
ax.axvline(best_model.best_epoch_, color="crimson", lw=1.5, linestyle="--", label="best epoch")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE loss")
ax.set_title("Training curves")
ax.legend()
plt.tight_layout()
plt.show()

if WANDB_ENABLED:
    wandb.log({"training_curves": wandb.Image(fig)})
```

- [ ] **Step 8: Step 7 — Residual analysis**

Same 3-subplot pattern as notebook 03 Step 7 (residuals vs. predicted, distribution, residuals vs. actual), computed from `best_model.predict(X["test"])` vs `y["test"]`.

- [ ] **Step 9: Step 8 — Persist**

```python
run_record, history = NNModelRegistry.save_run(
    MODELS, REPORTS / "nn_metrics.json", best_model, run_id, nn_config_name, study.best_params, result,
)
print(f"Model saved to {MODELS / run_record.model_filename}")
print(f"Best run so far: {history.best_run_id} -> models/nn_best.pt")

if WANDB_ENABLED:
    artifact = wandb.Artifact("nn_model", type="model")
    artifact.add_file(str(MODELS / run_record.model_filename))
    wandb.log_artifact(artifact)
    wandb.finish()
```

- [ ] **Step 10: Step 9 — Compare all NN runs**

Same pattern as notebook 03 Step 9, reading `reports/nn_metrics.json` via `TuningHistory.model_validate_json`.

- [ ] **Step 11: Probe `shap.GradientExplainer` standalone, before wiring it into the notebook**

Confirmed via `shap`'s `master` branch `pyproject.toml` (`'torch<=2.11.0; python_version < "3.14"'` in its test deps) that `torch==2.11.0` — the exact version this plan installs — is inside `shap`'s actively-tested range. `GradientExplainer` is the PyTorch path `shap`'s own README doesn't hedge as "preliminary" (unlike `DeepExplainer`). Still, verify against a real `WineScoreNet` before trusting it in the notebook — same discipline that caught the XGBoost `base_score` bug outside the notebook first.

Run (after Tasks 4/5 exist), from a scratch script or `python -c`:

```python
import numpy as np
import shap
import torch

from diplo_mod_1.training.nn_model import WineScoreNet

net = WineScoreNet(input_dim=10, hidden_sizes=[8, 4], dropout=0.0).eval()
background = torch.randn(20, 10)
sample = torch.randn(5, 10)

explainer = shap.GradientExplainer(net, background)
shap_values = explainer.shap_values(sample)
print(type(shap_values), np.asarray(shap_values).shape)
```

Expected: runs without error, prints a shape compatible with `(5, 10)` (possibly wrapped in a length-1 list — `shap.GradientExplainer` sometimes returns a list even for single-output regression; the notebook step below handles that). **If this fails**, stop and report back rather than silently falling back to permutation importance — the failure mode and fix (if any) need to be understood first, same as the XGBoost case.

- [ ] **Step 12: Step 10 — SHAP explainability for the NN**

Loads `models/nn_best.pt` fresh from disk (not whichever model happens to be in kernel memory), reconstructs `WineScoreNet` from the saved architecture config, runs `shap.GradientExplainer` against a background + sample drawn from `test` (densifying only those rows, not the whole sparse test matrix — same pattern as notebook 03's SHAP step and `SparseTabularDataset`). Own W&B run (`group="nn-shap"`, `job_type="explainability"`).

```python
# shap and WineScoreNet are already imported in the Step 1 imports cell —
# no local imports here, matches this project's "imports live in the first
# cell only" convention.
SHAP_SAMPLE_SIZE = 500
SHAP_BACKGROUND_SIZE = 100

checkpoint = torch.load(MODELS / "nn_best.pt", map_location="cpu", weights_only=False)
best_overall_nn = WineScoreNet(
    input_dim=checkpoint["input_dim"],
    hidden_sizes=checkpoint["hidden_sizes"],
    dropout=checkpoint["dropout"],
)
best_overall_nn.load_state_dict(checkpoint["state_dict"])
best_overall_nn.eval()

rng = np.random.default_rng(RANDOM_STATE)
n_test = X["test"].shape[0]
background_idx = rng.choice(n_test, size=SHAP_BACKGROUND_SIZE, replace=False)
sample_idx = rng.choice(n_test, size=min(SHAP_SAMPLE_SIZE, n_test), replace=False)

background_rows = X["test"][background_idx]
sample_rows = X["test"][sample_idx]
if sparse.issparse(background_rows):
    background_rows = background_rows.toarray()
    sample_rows = sample_rows.toarray()

background = torch.from_numpy(np.asarray(background_rows, dtype=np.float32))
X_sample_tensor = torch.from_numpy(np.asarray(sample_rows, dtype=np.float32))

explainer = shap.GradientExplainer(best_overall_nn, background)
shap_values = explainer.shap_values(X_sample_tensor)
if isinstance(shap_values, list):
    shap_values = shap_values[0]
shap_values = np.asarray(shap_values).reshape(len(sample_idx), -1)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:30]
shap_table = pd.DataFrame(
    {"feature": [feature_names[i] for i in top_idx], "mean_abs_shap": mean_abs_shap[top_idx]}
)
print(shap_table.head(15).to_string(index=False))

shap.summary_plot(
    shap_values, X_sample_tensor.numpy(), feature_names=feature_names, max_display=25, show=False
)
beeswarm_fig = plt.gcf()
plt.tight_layout()
plt.show()

if WANDB_ENABLED:
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "diplo-mod-1"),
        name=f"shap-nn-{history.best_run_id}",
        group="nn-shap",
        job_type="explainability",
        config={"explained_run_id": history.best_run_id, "sample_size": min(SHAP_SAMPLE_SIZE, n_test)},
    )
    wandb.log(
        {
            "shap_summary_beeswarm": wandb.Image(beeswarm_fig),
            "shap_top_features": wandb.Table(dataframe=shap_table),
        }
    )
    wandb.finish()
```

- [ ] **Step 13: Design notes markdown cell**

Same style/depth as notebook 03's Step 14 — algorithm choice justification (why an MLP on concatenated features, referencing what already worked for XGBoost), architecture, tuning process (including the deliberately small starting `n_trials`/`max_epochs` budget and why), validation strategy, persistence, and the SHAP `GradientExplainer` results (top features by mean |SHAP|, compared against XGBoost's SHAP results from notebook 03 Step 13 if both are available by the time this is written).

- [ ] **Step 14: Verify without executing the full notebook**

Run: `uv run poe lint && uv run poe typecheck` (runs `nbqa` against the new notebook cells — catches syntax/type errors without a full run).
Do NOT run `poe nbtest`/`poe check`/`jupyter nbconvert --execute` — that's the user's to do.

- [ ] **Step 15: Commit**

```bash
git add notebooks/04-train-nn-pytorch.ipynb
git commit -m "feat: build notebook 04 PyTorch NN training pipeline"
```

---

### Task 10: Update `CLAUDE.md` and `README.md`

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: `CLAUDE.md`**

Update the notebooks table's `04-train-nn-pytorch.ipynb` row if its description changed, and the device-selection gotcha (already correctly describes the intended CUDA→MPS→CPU priority for PyTorch — verify it now also correctly says `detect_torch_device()` by name instead of the placeholder description, and that XGBoost's function is referred to as `detect_xgboost_device()` after Task 2's rename).

- [ ] **Step 2: `README.md`**

- Repository structure: extend the `training/` bullet to mention the NN classes (`WineScorePredictorNet`, `NNTuner`, `NNModelRegistry`) alongside the XGBoost ones.
- "Hyperparameter tuning & experiment tracking" section: extend to mention `configs/nn_training.json` / `NN_TRAINING_CONFIG` alongside the XGBoost config pattern, and `reports/nn_metrics.json`/`models/nn_best.pt`.
- Stack section: add PyTorch's actual current setup (was already listed as "PyTorch (neural network, with Apple Silicon / MPS GPU support)" — extend to also mention CUDA now that it's real, not just MPS).
- Once real numbers exist (after the user runs the notebook), a "Current best result (NN)" section mirroring the XGBoost one — **not written now**, since it needs actual metrics from a real run; flag this as a follow-up once the user has executed notebook 04.

- [ ] **Step 3: Full verification**

Run: `uv run poe lint && uv run poe typecheck && uv run poe test`

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update CLAUDE.md and README for NN training pipeline"
```

---

## Self-Review Checklist (for whoever executes this plan)

- Every `WineScorePredictorNet.fit`/`.predict` call site uses the same keyword names (`X_val=`, `y_val=`, `callbacks=`) — check Task 5's class against Task 6's `NNTuner._make_model`/`fit_best` and Task 9's notebook cells.
- `NNModelRegistry.save_run`'s parameter order matches `ModelRegistry.save_run`'s exactly (`models_dir, metrics_path, model, run_id, tuning_config_name, best_params, result`) — Task 7.
- After Task 2, grep the whole repo for the bare string `detect_device` (not `detect_xgboost_device`/`detect_torch_device`) — should return zero matches outside this plan file itself.
- `input_dim` is always `X["train"].shape[1]` (2044), never hardcoded — Tasks 5, 6, 9.
- Task 9's Step 11 (standalone `GradientExplainer` probe) must actually pass before Step 12 (the real notebook SHAP cell) is trusted — don't skip the probe just because the code "looks right"; this is the exact failure mode that produced the undetected XGBoost `base_score` bug earlier in this project.
- If Step 11's probe fails, stop and report back rather than silently swapping in permutation importance — that fallback was explicitly rejected in favor of actually testing `GradientExplainer` first (see the grilling-session discussion this plan came from).
