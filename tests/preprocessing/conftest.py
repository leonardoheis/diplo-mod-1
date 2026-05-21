"""Preprocessing-layer fixtures derived from the root small_raw_df."""

import pandas as pd
import pytest

from diplo_mod_1.preprocessing.cleaner import DataCleaner
from diplo_mod_1.preprocessing.feature_engineer import FeatureEngineer


@pytest.fixture(scope="session")
def cleaned_df(small_raw_df: pd.DataFrame) -> pd.DataFrame:
    cleaner = DataCleaner()
    cleaner.fit(small_raw_df)
    return cleaner.clean(small_raw_df)


@pytest.fixture(scope="session")
def featured_df(small_raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> pd.DataFrame:
    engineer = FeatureEngineer()
    engineer.fit(small_raw_df)
    return engineer.transform(cleaned_df)
