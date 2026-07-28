"""Fixtures for training package tests."""

import numpy as np
import optuna
import pytest

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture
def regression_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small synthetic linear-ish regression data, train/val split."""
    rng = np.random.default_rng(42)
    n_train, n_val, n_features = 120, 40, 5
    X_train = rng.normal(size=(n_train, n_features)).astype(np.float32)
    X_val = rng.normal(size=(n_val, n_features)).astype(np.float32)
    weights = rng.normal(size=n_features)
    y_train = (X_train @ weights + rng.normal(scale=0.1, size=n_train)).astype(np.float32)
    y_val = (X_val @ weights + rng.normal(scale=0.1, size=n_val)).astype(np.float32)
    return X_train, y_train, X_val, y_val
