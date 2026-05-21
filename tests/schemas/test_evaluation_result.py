"""Tests for EvaluationResult and PreprocessingResult schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from diplo_mod_1.domain.metrics import ModelMetrics
from diplo_mod_1.schemas.evaluation import EvaluationResult
from diplo_mod_1.schemas.pipeline import PreprocessingResult


def _metric(**kwargs: object) -> ModelMetrics:
    defaults: dict[str, object] = dict(
        model_type="xgboost", split="test", rmse=1.83, mae=1.41, r2=0.51
    )
    return ModelMetrics(**{**defaults, **kwargs})  # type: ignore[arg-type]


# ── EvaluationResult ─────────────────────────────────────────────────────────


def test_add_appends_metric() -> None:
    result = EvaluationResult()
    m = _metric()
    result.add(m)
    assert len(result.metrics) == 1
    assert result.metrics[0] is m


def test_best_returns_lowest_rmse() -> None:
    result = EvaluationResult()
    result.add(_metric(model_type="xgboost", split="val", rmse=2.0))
    result.add(_metric(model_type="neural_net", split="val", rmse=1.5))
    best = result.best(split="val")
    assert best.model_type == "neural_net"
    assert best.rmse == pytest.approx(1.5)


def test_best_on_empty_split_raises() -> None:
    with pytest.raises(ValueError, match="No metrics"):
        EvaluationResult().best(split="test")


def test_for_model_filters_by_type() -> None:
    result = EvaluationResult()
    result.add(_metric(model_type="xgboost", split="train"))
    result.add(_metric(model_type="neural_net", split="train"))
    result.add(_metric(model_type="xgboost", split="test"))
    xgb = result.for_model("xgboost")
    assert len(xgb) == 2
    assert all(m.model_type == "xgboost" for m in xgb)


def test_for_model_returns_empty_when_none_match() -> None:
    result = EvaluationResult()
    result.add(_metric(model_type="xgboost"))
    assert result.for_model("neural_net") == []


# ── PreprocessingResult ───────────────────────────────────────────────────────


def test_preprocessing_result_valid() -> None:
    r = PreprocessingResult(
        n_rows=1000,
        xgb_features=15,
        nn_tab_features=15,
        txt_features=2000,
        split_sizes={"train": 640, "val": 160, "test": 200},
    )
    assert r.n_rows == 1000


def test_preprocessing_result_zero_rows_raises() -> None:
    with pytest.raises(ValidationError):
        PreprocessingResult(
            n_rows=0,
            xgb_features=15,
            nn_tab_features=15,
            txt_features=2000,
            split_sizes={"train": 0},
        )


def test_preprocessing_result_zero_features_raises() -> None:
    with pytest.raises(ValidationError):
        PreprocessingResult(
            n_rows=100,
            xgb_features=0,
            nn_tab_features=15,
            txt_features=2000,
            split_sizes={"train": 80},
        )
