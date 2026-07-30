"""Tests for FeatureEngineer."""

import numpy as np
import pandas as pd
import pytest

from diplo_mod_1.constants import LUXURY_THRESHOLD, REF_YEAR
from diplo_mod_1.preprocessing.columns import TASTING_FLAG_COLS, TASTING_KEYWORDS
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


def test_tasting_keyword_flags_added(featured_df: pd.DataFrame) -> None:
    assert set(TASTING_FLAG_COLS).issubset(set(featured_df.columns))
    for col in TASTING_FLAG_COLS:
        assert set(featured_df[col].unique()).issubset({0, 1})


def test_tasting_keyword_flag_matches_description_substring(featured_df: pd.DataFrame) -> None:
    term = TASTING_KEYWORDS[0]
    expected = featured_df["description"].astype(str).str.contains(term, case=False, regex=False)
    pd.testing.assert_series_equal(
        featured_df[f"has_{term}"].astype(bool).reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_artifacts_returns_correct_values(small_raw_df: pd.DataFrame) -> None:
    fe = FeatureEngineer()
    fe.fit(small_raw_df)
    arts = fe.artifacts
    assert isinstance(arts, FeatureEngineerArtifacts)
    assert arts.median_vintage == fe.median_vintage_
    assert arts.luxury_threshold == LUXURY_THRESHOLD
    assert arts.ref_year == REF_YEAR
    assert arts.variety_avg_log_price == fe.variety_avg_log_price_


def test_price_vs_variety_added(featured_df: pd.DataFrame) -> None:
    assert "price_vs_variety" in featured_df.columns
    assert featured_df["price_vs_variety"].notna().all()


def test_price_vs_variety_computed_correctly(
    small_raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, featured_df: pd.DataFrame
) -> None:
    fe = FeatureEngineer()
    fe.fit(small_raw_df)
    assert fe.variety_avg_log_price_ is not None

    variety_avg = cleaned_df["variety"].map(fe.variety_avg_log_price_)
    expected = np.log1p(cleaned_df["price"]) - variety_avg
    np.testing.assert_allclose(
        featured_df["price_vs_variety"].to_numpy(), expected.to_numpy(), rtol=1e-5
    )


def test_price_vs_variety_zero_for_average_priced_wine(small_raw_df: pd.DataFrame) -> None:
    """A wine priced exactly at its variety's average should score ~0."""
    fe = FeatureEngineer()
    fe.fit(small_raw_df)
    assert fe.variety_avg_log_price_ is not None
    variety = next(iter(fe.variety_avg_log_price_))
    avg_log_price = fe.variety_avg_log_price_[variety]

    row = small_raw_df.iloc[[0]].copy()
    row["variety"] = variety
    row["price"] = np.expm1(avg_log_price)
    row["title"] = "Test Winery 2015 Reserve"

    from diplo_mod_1.preprocessing.cleaner import DataCleaner

    cleaner = DataCleaner()
    cleaner.fit(small_raw_df)
    cleaned_row = cleaner.clean(row)
    result = fe.transform(cleaned_row)

    assert result["price_vs_variety"].iloc[0] == pytest.approx(0.0, abs=1e-4)
