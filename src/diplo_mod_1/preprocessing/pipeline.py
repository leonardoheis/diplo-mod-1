"""End-to-end preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from diplo_mod_1.preprocessing.cleaner import DataCleaner
from diplo_mod_1.preprocessing.config import PipelineConfig
from diplo_mod_1.preprocessing.encoders import TabularEncoder, TextEncoder
from diplo_mod_1.preprocessing.exporter import DatasetExporter
from diplo_mod_1.preprocessing.splitter import DataSplitter
from diplo_mod_1.schemas.pipeline import PreprocessingResult


class _XGBoostDataset(NamedTuple):
    X_splits: dict[str, np.ndarray]
    feature_names: list[str]
    out_dir: Path


class _NNDataset(NamedTuple):
    X_tab_splits: dict[str, np.ndarray]
    feature_names: list[str]
    X_txt_splits: dict[str, sparse.csr_matrix]
    out_dir: Path


class PreprocessingPipeline:
    """Orchestrates the full preprocessing run end-to-end.

    All hyperparameters are controlled through a single ``PipelineConfig``
    object, which can be constructed from a JSON file for reproducible runs.

    Usage::

        # default config
        pipeline = PreprocessingPipeline()
        result = pipeline.run(raw_csv, interim_dir, processed_dir)

        # custom config
        config = PipelineConfig(text_encoder=TextEncoderConfig(max_features=5000))
        pipeline = PreprocessingPipeline(config)

        # load config from file
        pipeline = PreprocessingPipeline.from_json(Path("config.json"))
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.cleaner = DataCleaner(self.config.cleaner)
        self.splitter = DataSplitter(self.config.splitter)
        self.encoder = TabularEncoder(self.config.encoder)
        self.text_encoder = TextEncoder(self.config.text_encoder)
        self.exporter = DatasetExporter()

    @classmethod
    def from_json(cls, path: Path) -> "PreprocessingPipeline":
        """Load pipeline config from a JSON file."""
        return cls(PipelineConfig.model_validate_json(path.read_text(encoding="utf-8")))

    # ── private stages ────────────────────────────────────────────────────────

    def _clean_and_feature(self, raw_csv: Path, interim_dir: Path) -> pd.DataFrame:
        """Load raw CSV, clean, engineer base features; persist interim files."""
        df = DataCleaner.load(raw_csv)
        self.cleaner.fit(df)
        cleaned = self.cleaner.clean(df)
        cleaned.to_parquet(interim_dir / "01_cleaned.parquet")
        (interim_dir / "preprocessing_config.json").write_text(
            self.cleaner.artifacts.model_dump_json(indent=2), encoding="utf-8"
        )
        featured = self.cleaner.add_features(cleaned)
        featured.to_parquet(interim_dir / "02_features.parquet")
        return featured

    def _split(
        self,
        featured: pd.DataFrame,
        processed_dir: Path,
    ) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame], dict[str, np.ndarray]]:
        """Stratified split; persist index arrays; return indices, frames, labels."""
        split_idx = self.splitter.split(featured)
        np.savez(
            processed_dir / "split_indices.npz",
            train=split_idx["train"],
            val=split_idx["val"],
            test=split_idx["test"],
        )
        splits = {key: featured.iloc[idx] for key, idx in split_idx.items()}
        y_splits = {
            key: featured.iloc[idx]["points"].to_numpy(dtype=np.float32)
            for key, idx in split_idx.items()
        }
        return split_idx, splits, y_splits

    def _build_xgboost_dataset(
        self,
        splits: dict[str, pd.DataFrame],
        y_splits: dict[str, np.ndarray],
        processed_dir: Path,
    ) -> _XGBoostDataset:
        """Fit encoder on train, encode all splits, export XGBoost dataset."""
        x_xgb: dict[str, np.ndarray] = {}
        x_xgb["train"], xgb_names, xgb_groups = self.encoder.fit_transform(
            splits["train"], y_splits["train"]
        )
        for key in ("val", "test"):
            x_xgb[key], _, _ = self.encoder.transform(splits[key])
        xgb_dir = processed_dir / "xgboost"
        self.exporter.export_xgboost(xgb_dir, x_xgb, y_splits, self.encoder, xgb_names, xgb_groups)
        return _XGBoostDataset(X_splits=x_xgb, feature_names=xgb_names, out_dir=xgb_dir)

    def _build_nn_dataset(
        self,
        splits: dict[str, pd.DataFrame],
        y_splits: dict[str, np.ndarray],
        processed_dir: Path,
    ) -> _NNDataset:
        """Scale tabular features, encode text, export NN dataset.

        Requires ``self.encoder`` to already be fitted (call after
        ``_build_xgboost_dataset``).
        """
        x_nn: dict[str, np.ndarray] = {}
        x_nn["train"], nn_names, nn_groups = self.encoder.transform(splits["train"])
        for key in ("val", "test"):
            x_nn[key], _, _ = self.encoder.transform(splits[key])

        scaler = StandardScaler()
        scaler.fit(x_nn["train"][:, nn_groups["continuous"]])
        x_nn_scaled: dict[str, np.ndarray] = {}
        for key, x in x_nn.items():
            arr = x.copy()
            arr[:, nn_groups["continuous"]] = scaler.transform(x[:, nn_groups["continuous"]])
            x_nn_scaled[key] = arr

        self.text_encoder.fit(splits["train"])
        x_txt = {key: self.text_encoder.transform(df) for key, df in splits.items()}

        nn_dir = processed_dir / "nn"
        self.exporter.export_nn(
            nn_dir, x_nn_scaled, x_txt, y_splits, nn_names, nn_groups, scaler, self.text_encoder
        )
        return _NNDataset(
            X_tab_splits=x_nn_scaled, feature_names=nn_names, X_txt_splits=x_txt, out_dir=nn_dir
        )

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        raw_csv: Path,
        interim_dir: Path,
        processed_dir: Path,
    ) -> PreprocessingResult:
        interim_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        featured = self._clean_and_feature(raw_csv, interim_dir)
        split_idx, splits, y_splits = self._split(featured, processed_dir)
        xgb = self._build_xgboost_dataset(splits, y_splits, processed_dir)
        nn = self._build_nn_dataset(splits, y_splits, processed_dir)

        self.exporter.write_manifest(
            processed_dir / "dataset_manifest.json",
            xgb.out_dir,
            nn.out_dir,
            processed_dir / "split_indices.npz",
            {k: xgb.X_splits[k].shape for k in xgb.X_splits},
            {k: nn.X_tab_splits[k].shape for k in nn.X_tab_splits},
            {k: nn.X_txt_splits[k].nnz for k in nn.X_txt_splits},
        )

        return PreprocessingResult(
            n_rows=len(featured),
            xgb_features=len(xgb.feature_names),
            nn_tab_features=len(nn.feature_names),
            txt_features=nn.X_txt_splits["train"].shape[1],
            split_sizes={k: len(v) for k, v in split_idx.items()},
        )
