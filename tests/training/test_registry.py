"""Tests for ModelRegistry."""

from pathlib import Path

from diplo_mod_1.domain.metrics import ModelMetrics
from diplo_mod_1.schemas.evaluation import EvaluationResult
from diplo_mod_1.training.registry import ModelRegistry


class _FakeModel:
    """Minimal joblib-picklable stand-in for XGBRegressor — avoids fitting a real model."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


def _result(test_rmse: float) -> EvaluationResult:
    result = EvaluationResult()
    result.add(ModelMetrics(model_type="xgboost", split="train", rmse=0.5, mae=0.4, r2=0.9))
    result.add(ModelMetrics(model_type="xgboost", split="val", rmse=1.0, mae=0.8, r2=0.8))
    result.add(ModelMetrics(model_type="xgboost", split="test", rmse=test_rmse, mae=0.8, r2=0.8))
    return result


def test_first_run_becomes_best(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    record, history = ModelRegistry.save_run(
        tmp_path, metrics_path, _FakeModel("a"), "run-a", "cfg.json", {"max_depth": 5}, _result(1.5)
    )

    assert record.model_filename == "run-a.joblib"  # no doubled "xgboost_" prefix
    assert (tmp_path / record.model_filename).exists()
    assert (tmp_path / "xgboost_best.joblib").exists()
    assert history.best_run_id == "run-a"
    assert len(history.runs) == 1


def test_worse_run_does_not_replace_best(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    ModelRegistry.save_run(
        tmp_path, metrics_path, _FakeModel("a"), "run-a", "cfg.json", {}, _result(1.5)
    )
    _, history = ModelRegistry.save_run(
        tmp_path, metrics_path, _FakeModel("b"), "run-b", "cfg.json", {}, _result(2.0)
    )

    assert history.best_run_id == "run-a"
    assert (tmp_path / "xgboost_best.joblib").read_bytes() == (
        tmp_path / "run-a.joblib"
    ).read_bytes()
    assert (tmp_path / "run-b.joblib").exists()  # still kept, just not best


def test_better_run_replaces_best(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    ModelRegistry.save_run(
        tmp_path, metrics_path, _FakeModel("a"), "run-a", "cfg.json", {}, _result(1.5)
    )
    _, history = ModelRegistry.save_run(
        tmp_path, metrics_path, _FakeModel("b"), "run-b", "cfg.json", {}, _result(1.0)
    )

    assert history.best_run_id == "run-b"
    assert (tmp_path / "xgboost_best.joblib").read_bytes() == (
        tmp_path / "run-b.joblib"
    ).read_bytes()


def test_loading_pre_existing_incompatible_file_starts_fresh(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"metrics": [{"split": "test", "rmse": 1.0}]}', encoding="utf-8")

    record, history = ModelRegistry.save_run(
        tmp_path, metrics_path, _FakeModel("a"), "run-a", "cfg.json", {}, _result(1.5)
    )

    assert len(history.runs) == 1
    assert history.runs[0].run_id == record.run_id


def test_survives_history_entry_with_missing_checkpoint(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    # A run recorded but whose checkpoint file no longer exists on disk —
    # e.g. models/ was cleared, or an older bug produced an unrecoverable name.
    metrics_path.write_text(
        '{"runs": [{"run_id": "ghost", "tuning_config": "cfg.json", '
        '"model_filename": "ghost.joblib", "best_params": {}, "metrics": '
        '[{"model_type": "xgboost", "split": "test", "rmse": 0.1, "mae": 0.1, "r2": 0.9}]}], '
        '"best_run_id": "ghost"}',
        encoding="utf-8",
    )

    record, history = ModelRegistry.save_run(
        tmp_path, metrics_path, _FakeModel("a"), "run-a", "cfg.json", {}, _result(1.5)
    )

    assert history.best_run_id == "run-a"  # the only run whose file actually exists
    assert (tmp_path / "xgboost_best.joblib").read_bytes() == (
        tmp_path / "run-a.joblib"
    ).read_bytes()
    assert len(history.runs) == 2  # the ghost entry stays on record, just not "best"


def test_loading_malformed_json_starts_fresh(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("not valid json {{{", encoding="utf-8")

    record, history = ModelRegistry.save_run(
        tmp_path, metrics_path, _FakeModel("a"), "run-a", "cfg.json", {}, _result(1.5)
    )

    assert len(history.runs) == 1
    assert history.runs[0].run_id == record.run_id
