"""Tests for FeatureEngineer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diplo_mod_1.constants import LUXURY_THRESHOLD, REF_YEAR
from diplo_mod_1.preprocessing.config import FeatureEngineerArtifacts
from diplo_mod_1.preprocessing.feature_engineer import FeatureEngineer


def test_transform_before_fit_raises(cleaned_df: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        FeatureEngineer().transform(cleaned_df)


def test_artifacts_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        _ = FeatureEngineer().artifacts


def test_fit_returns_self(small_raw_df: pd.DataFrame) -> None:
    fe = FeatureEngineer()
    assert fe.fit(small_raw_df) is fe


def test_fit_sets_median_vintage_as_4digit_year(small_raw_df: pd.DataFrame) -> None:
    fe = FeatureEngineer()
    fe.fit(small_raw_df)
    assert isinstance(fe.median_vintage_, float)
    assert fe.median_vintage_ >= 1900


def test_transform_adds_expected_columns(featured_df: pd.DataFrame) -> None:
    expected = {
        "log_price",
        "is_luxury",
        "is_us",
        "has_designation",
        "description_length",
        "wine_age",
    }
    assert expected.issubset(set(featured_df.columns))


def test_log_price_computed_correctly(cleaned_df: pd.DataFrame, featured_df: pd.DataFrame) -> None:
    expected = np.log1p(cleaned_df["price"].to_numpy())
    np.testing.assert_allclose(featured_df["log_price"].to_numpy(), expected, rtol=1e-5)


def test_is_luxury_threshold(cleaned_df: pd.DataFrame, featured_df: pd.DataFrame) -> None:
    expected = (cleaned_df["price"] > LUXURY_THRESHOLD).astype(int)
    pd.testing.assert_series_equal(
        featured_df["is_luxury"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_is_us_flag(cleaned_df: pd.DataFrame, featured_df: pd.DataFrame) -> None:
    expected = (cleaned_df["country"] == "US").astype(int)
    pd.testing.assert_series_equal(
        featured_df["is_us"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_wine_age_non_negative(featured_df: pd.DataFrame) -> None:
    assert (featured_df["wine_age"] >= 0).all()


def test_wine_age_equals_ref_year_minus_vintage_year(featured_df: pd.DataFrame) -> None:
    expected = REF_YEAR - featured_df["vintage_year"]
    pd.testing.assert_series_equal(featured_df["wine_age"], expected, check_names=False)


def test_artifacts_returns_correct_values(small_raw_df: pd.DataFrame) -> None:
    fe = FeatureEngineer()
    fe.fit(small_raw_df)
    arts = fe.artifacts
    assert isinstance(arts, FeatureEngineerArtifacts)
    assert arts.median_vintage == fe.median_vintage_
    assert arts.luxury_threshold == LUXURY_THRESHOLD
    assert arts.ref_year == REF_YEAR
