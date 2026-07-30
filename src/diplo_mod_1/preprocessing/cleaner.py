"""Data loading and cleaning for the Wine Reviews dataset."""

from pathlib import Path

import pandas as pd

from diplo_mod_1.constants import RANDOM_STATE, VINTAGE_RE
from diplo_mod_1.preprocessing.config import CleaningArtifacts, DataCleanerConfig


class DataCleaner:
    """Cleans raw Wine Reviews data: drops redundant columns, extracts vintage,
    imputes missing price and categorical fields.

    Fits on the full dataset before splitting — no leakage because
    ``global_price_median_`` is derived from the full corpus.

    Usage::

        cleaner = DataCleaner()
        cleaner.fit(df)
        cleaned = cleaner.clean(df)   # save as 01_cleaned.parquet
    """

    def __init__(self, config: DataCleanerConfig | None = None) -> None:
        self.config = config or DataCleanerConfig()
        self.global_price_median_: float | None = None

    @staticmethod
    def load(csv_path: Path) -> pd.DataFrame:
        """Load the raw CSV by path."""
        return pd.read_csv(csv_path, index_col=0)

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Learn the global price median from the full dataset."""
        self.global_price_median_ = float(df["price"].median())
        return self

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop redundant columns, extract vintage year, impute price and categoricals."""
        if self.global_price_median_ is None:
            raise RuntimeError("Call fit() before clean().")
        out = df.copy()
        out = out.drop(columns=["region_2", "taster_twitter_handle"], errors="ignore")

        years = out["title"].astype(str).str.extract(VINTAGE_RE, expand=False)
        vintage_year = pd.to_numeric(years, errors="coerce")
        out["vintage_year"] = vintage_year
        out["vintage_missing"] = vintage_year.isna().astype(int)

        price_missing = out["price"].isna().astype(int)
        median_price = out.groupby(["country", "variety"], dropna=False)["price"].transform(
            "median"
        )
        out["price"] = out["price"].fillna(median_price).fillna(self.global_price_median_)
        out["price_missing"] = price_missing

        for col in ["country", "region_1", "variety", "taster_name", "province"]:
            out[col] = out[col].fillna("Unknown")

        return out

    @property
    def artifacts(self) -> CleaningArtifacts:
        """Fitted values as a typed Pydantic model — serialisable to JSON."""
        if self.global_price_median_ is None:
            raise RuntimeError("Call fit() before accessing artifacts.")
        return CleaningArtifacts(
            global_price_median=self.global_price_median_,
            random_state=RANDOM_STATE,
        )
