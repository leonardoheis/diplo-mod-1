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
