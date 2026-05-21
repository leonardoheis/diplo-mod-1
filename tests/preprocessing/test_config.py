"""Tests for Pydantic config models and their validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from diplo_mod_1.constants import YEAR_MAX, YEAR_MIN
from diplo_mod_1.preprocessing.config import (
    DataSplitterConfig,
    FeatureEngineerConfig,
    PipelineConfig,
    TabularEncoderConfig,
    TextEncoderConfig,
)


# ── FeatureEngineerConfig ─────────────────────────────────────────────────────


def test_feature_engineer_negative_luxury_threshold_raises() -> None:
    with pytest.raises(ValidationError):
        FeatureEngineerConfig(luxury_threshold=-1.0)


def test_feature_engineer_zero_luxury_threshold_raises() -> None:
    with pytest.raises(ValidationError):
        FeatureEngineerConfig(luxury_threshold=0.0)


def test_feature_engineer_ref_year_below_minimum_raises() -> None:
    with pytest.raises(ValidationError):
        FeatureEngineerConfig(ref_year=YEAR_MIN - 1)


def test_feature_engineer_ref_year_above_maximum_raises() -> None:
    with pytest.raises(ValidationError):
        FeatureEngineerConfig(ref_year=YEAR_MAX + 1)


def test_feature_engineer_valid_defaults() -> None:
    config = FeatureEngineerConfig()
    assert config.luxury_threshold > 0
    assert YEAR_MIN <= config.ref_year <= YEAR_MAX


# ── DataSplitterConfig ────────────────────────────────────────────────────────


def test_splitter_test_plus_val_gte_one_raises() -> None:
    with pytest.raises(ValidationError):
        DataSplitterConfig(test_size=0.6, val_size=0.5)


def test_splitter_zero_test_size_raises() -> None:
    with pytest.raises(ValidationError):
        DataSplitterConfig(test_size=0.0)


def test_splitter_test_size_gte_one_raises() -> None:
    with pytest.raises(ValidationError):
        DataSplitterConfig(test_size=1.1)


def test_splitter_zero_val_size_raises() -> None:
    with pytest.raises(ValidationError):
        DataSplitterConfig(val_size=0.0)


# ── TabularEncoderConfig ──────────────────────────────────────────────────────


def test_tabular_encoder_cv_below_two_raises() -> None:
    with pytest.raises(ValidationError):
        TabularEncoderConfig(target_encoder_cv=1)


def test_tabular_encoder_cv_zero_raises() -> None:
    with pytest.raises(ValidationError):
        TabularEncoderConfig(target_encoder_cv=0)


# ── TextEncoderConfig ─────────────────────────────────────────────────────────


def test_text_encoder_zero_max_features_raises() -> None:
    with pytest.raises(ValidationError):
        TextEncoderConfig(max_features=0)


def test_text_encoder_zero_min_df_raises() -> None:
    with pytest.raises(ValidationError):
        TextEncoderConfig(min_df=0)


def test_text_encoder_inverted_ngram_range_raises() -> None:
    with pytest.raises(ValidationError):
        TextEncoderConfig(ngram_range=(3, 1))


# ── PipelineConfig ────────────────────────────────────────────────────────────


def test_pipeline_config_json_round_trip() -> None:
    config = PipelineConfig()
    loaded = PipelineConfig.model_validate_json(config.model_dump_json())
    assert loaded == config
