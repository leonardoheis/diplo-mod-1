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
        tmp_path,
        metrics_path,
        _fitted_model(1),
        "run-a",
        "cfg.json",
        {"dropout": 0.2},
        _result(1.5),
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
