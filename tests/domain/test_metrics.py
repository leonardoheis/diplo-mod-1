"""Tests for ModelMetrics domain model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from diplo_mod_1.domain.metrics import ModelMetrics

_VALID = dict(model_type="xgboost", split="test", rmse=1.83, mae=1.41, r2=0.51)


def test_valid_instance_creates() -> None:
    m = ModelMetrics(**_VALID)
    assert m.rmse == pytest.approx(1.83)
    assert m.split == "test"


def test_all_splits_accepted() -> None:
    for split in ("train", "val", "test"):
        m = ModelMetrics(**{**_VALID, "split": split})
        assert m.split == split


def test_negative_rmse_raises() -> None:
    with pytest.raises(ValidationError):
        ModelMetrics(**{**_VALID, "rmse": -0.1})


def test_negative_mae_raises() -> None:
    with pytest.raises(ValidationError):
        ModelMetrics(**{**_VALID, "mae": -0.1})


def test_invalid_split_raises() -> None:
    with pytest.raises(ValidationError):
        ModelMetrics(**{**_VALID, "split": "holdout"})  # type: ignore[arg-type]


def test_invalid_model_type_raises() -> None:
    with pytest.raises(ValidationError):
        ModelMetrics(**{**_VALID, "model_type": "random_forest"})  # type: ignore[arg-type]


def test_frozen_rejects_mutation() -> None:
    m = ModelMetrics(**_VALID)
    with pytest.raises(ValidationError):
        m.rmse = 2.0  # type: ignore[misc]
