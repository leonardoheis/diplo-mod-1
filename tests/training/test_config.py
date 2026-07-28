"""Tests for XGBoost tuning config."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from diplo_mod_1.training.config import XGBoostSearchSpace, XGBoostTuningConfig


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


@pytest.mark.parametrize("filename", ["xgboost_tuning.json", "xgboost_tuning_wide.json"])
def test_repo_config_files_are_valid(filename: str) -> None:
    path = Path(__file__).parents[2] / "configs" / filename
    config = XGBoostTuningConfig.from_json(path)
    assert config.n_trials == 50
