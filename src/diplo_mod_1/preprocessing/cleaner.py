"""Data loading, cleaning, and base feature engineering."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from diplo_mod_1.constants import RANDOM_STATE
from diplo_mod_1.preprocessing.config import CleaningArtifacts, DataCleanerConfig

_VINTAGE_RE = re.compile(r"\b(19|20)\d{2}\b")


class DataCleaner:
    """Cleans raw Wine Reviews data and engineers base features.

    Fits on the full dataset before splitting — no leakage because imputation
    targets (global price median, median vintage) are derived from the full
    corpus and applied uniformly.

    Usage::

        cleaner = DataCleaner()
        cleaner.fit(df)
        cleaned  = cleaner.clean(df)              # save as 01_cleaned.parquet
        featured = cleaner.add_features(cleaned)  # save as 02_features.parquet

        # or in one shot:
        featured = cleaner.fit_transform(df)
    """

    def __init__(self, config: DataCleanerConfig | None = None) -> None:
        self.config = config or DataCleanerConfig()
        self.global_price_median_: float | None = None
        self.median_vintage_: float | None = None

    @staticmethod
    def load(csv_path: Path) -> pd.DataFrame:
        """Load the raw CSV by path."""
        return pd.read_csv(csv_path, index_col=0)

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Learn imputation statistics from the full dataset."""
        self.global_price_median_ = float(df["price"].median())
        vintage_year = pd.to_numeric(
            df["title"].astype(str).str.extract(_VINTAGE_RE, expand=False),
            errors="coerce",
        )
        self.median_vintage_ = float(vintage_year.median())
        return self

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 1: drop columns, extract vintage, impute price and categoricals."""
        if self.global_price_median_ is None:
            raise RuntimeError("Call fit() before clean().")
        out = df.copy()
        out = out.drop(columns=["region_2", "taster_twitter_handle"], errors="ignore")

        years = out["title"].astype(str).str.extract(_VINTAGE_RE, expand=False)
        vintage_year = pd.to_numeric(years, errors="coerce")
        out["vintage_year"] = vintage_year
        out["vintage_missing"] = vintage_year.isna().astype(int)

        price_missing = out["price"].isna().astype(int)
        median_price = out.groupby(["country", "variety"], dropna=False)["price"].transform(
            "median"
        )
        out["price"] = out["price"].fillna(median_price).fillna(self.global_price_median_)
        out["price_missing"] = price_missing

        for col in ["country", "region_1", "variety", "taster_name"]:
            out[col] = out[col].fillna("Unknown")

        return out

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 2: add engineered features. Expects output of clean()."""
        if self.median_vintage_ is None:
            raise RuntimeError("Call fit() before add_features().")
        out = df.copy()
        out["log_price"] = np.log1p(out["price"])
        out["is_luxury"] = (out["price"] > self.config.luxury_threshold).astype(int)
        out["is_us"] = (out["country"] == "US").astype(int)
        out["has_designation"] = out["designation"].notna().astype(int)
        out["description_length"] = out["description"].astype(str).str.len()
        out["vintage_year"] = out["vintage_year"].fillna(self.median_vintage_)
        out["wine_age"] = self.config.ref_year - out["vintage_year"]
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: clean + add_features in one step."""
        return self.add_features(self.clean(df))

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    @property
    def artifacts(self) -> CleaningArtifacts:
        """Fitted values as a typed Pydantic model — serialisable to JSON."""
        if self.global_price_median_ is None:
            raise RuntimeError("Call fit() before accessing artifacts.")
        return CleaningArtifacts(
            luxury_threshold=self.config.luxury_threshold,
            ref_year=self.config.ref_year,
            global_price_median=self.global_price_median_,
            random_state=RANDOM_STATE,
        )
