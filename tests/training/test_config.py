"""Tests for XGBoost tuning config."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from diplo_mod_1.domain.metrics import ModelMetrics
from diplo_mod_1.training.config import (
    RunRecord,
    TuningHistory,
    XGBoostSearchSpace,
    XGBoostTuningConfig,
)


def test_default_config_valid() -> None:
    config = XGBoostTuningConfig()
    assert config.n_trials == 50
    assert config.early_stopping_rounds == 50


def test_inverted_bounds_raise() -> None:
    with pytest.raises(ValidationError):
        XGBoostSearchSpace(max_depth=(8, 3))


def test_custom_search_space_overrides_default() -> None:
    space = XGBoostSearchSpace(max_depth=(2, 4))
    config = XGBoostTuningConfig(search_space=space)
    assert config.search_space.max_depth == (2, 4)


def test_from_json_round_trips(tmp_path: Path) -> None:
    original = XGBoostTuningConfig(n_trials=5, search_space=XGBoostSearchSpace(max_depth=(2, 4)))
    path = tmp_path / "config.json"
    path.write_text(original.model_dump_json(), encoding="utf-8")

    loaded = XGBoostTuningConfig.from_json(path)

    assert loaded == original


CONFIGS_DIR = Path(__file__).parents[2] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.json")), ids=lambda p: p.name)
def test_repo_config_files_are_valid(path: Path) -> None:
    """Every configs/*.json file must parse and satisfy XGBoostTuningConfig's own validation.

    Discovers files via glob rather than a hardcoded list, so a new
    configs/*.json is covered automatically without editing this test.
    Doesn't assert specific values (n_trials, bounds, ...) — those are meant
    to be tuned freely; this only guards against a malformed/invalid file.
    """
    config = XGBoostTuningConfig.from_json(path)
    assert config.n_trials >= 1


def test_best_run_returns_none_without_matching_split() -> None:
    record = RunRecord(
        run_id="run-a",
        tuning_config="cfg.json",
        model_filename="model.joblib",
        best_params={},
        metrics=[ModelMetrics(model_type="xgboost", split="train", rmse=1.0, mae=1.0, r2=0.5)],
    )
    history = TuningHistory(runs=[record])

    assert history.best_run(split="test") is None
