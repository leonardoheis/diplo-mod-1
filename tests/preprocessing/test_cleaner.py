"""Tests for DataCleaner."""

import pandas as pd
import pytest

from diplo_mod_1.preprocessing.cleaner import DataCleaner
from diplo_mod_1.preprocessing.config import CleaningArtifacts


def test_clean_before_fit_raises(small_raw_df: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        DataCleaner().clean(small_raw_df)


def test_artifacts_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        _ = DataCleaner().artifacts


def test_fit_returns_self(small_raw_df: pd.DataFrame) -> None:
    cleaner = DataCleaner()
    assert cleaner.fit(small_raw_df) is cleaner


def test_fit_sets_global_price_median(small_raw_df: pd.DataFrame) -> None:
    cleaner = DataCleaner()
    cleaner.fit(small_raw_df)
    assert isinstance(cleaner.global_price_median_, float)
    assert cleaner.global_price_median_ > 0


def test_clean_drops_redundant_columns(cleaned_df: pd.DataFrame) -> None:
    assert "region_2" not in cleaned_df.columns
    assert "taster_twitter_handle" not in cleaned_df.columns


def test_clean_extracts_4digit_vintage_year(cleaned_df: pd.DataFrame) -> None:
    has_year = cleaned_df["vintage_missing"] == 0
    assert has_year.sum() > 0
    extracted = cleaned_df.loc[has_year, "vintage_year"]
    assert (extracted >= 1900).all()
    assert (extracted <= 2100).all()


def test_clean_vintage_missing_flag_for_titles_without_year(
    small_raw_df: pd.DataFrame, cleaned_df: pd.DataFrame
) -> None:
    no_year_mask = ~small_raw_df["title"].str.contains(r"\b(?:19|20)\d{2}\b", regex=True)
    assert (cleaned_df.loc[no_year_mask.to_numpy(), "vintage_missing"] == 1).all()


def test_clean_price_has_no_nans(cleaned_df: pd.DataFrame) -> None:
    assert cleaned_df["price"].isna().sum() == 0


def test_clean_price_missing_flag_matches_original_nans(
    small_raw_df: pd.DataFrame, cleaned_df: pd.DataFrame
) -> None:
    original_nan = small_raw_df["price"].isna().to_numpy()
    assert (cleaned_df.loc[original_nan, "price_missing"] == 1).all()
    assert (cleaned_df.loc[~original_nan, "price_missing"] == 0).all()


def test_clean_fills_categoricals_with_unknown(
    small_raw_df: pd.DataFrame, cleaned_df: pd.DataFrame
) -> None:
    for col in ["country", "region_1", "variety", "taster_name"]:
        nan_mask = small_raw_df[col].isna().to_numpy()
        if nan_mask.any():
            assert (cleaned_df.loc[nan_mask, col] == "Unknown").all()
    assert cleaned_df[["country", "region_1", "variety", "taster_name"]].isna().sum().sum() == 0


def test_artifacts_returns_cleaning_artifacts(small_raw_df: pd.DataFrame) -> None:
    cleaner = DataCleaner()
    cleaner.fit(small_raw_df)
    arts = cleaner.artifacts
    assert isinstance(arts, CleaningArtifacts)
    assert arts.global_price_median == cleaner.global_price_median_
