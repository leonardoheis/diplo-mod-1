"""End-to-end preprocessing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from diplo_mod_1.preprocessing.cleaner import DataCleaner
from diplo_mod_1.preprocessing.encoders import TabularEncoder, TextEncoder
from diplo_mod_1.preprocessing.exporter import DatasetExporter
from diplo_mod_1.preprocessing.splitter import DataSplitter


class PreprocessingPipeline:
    """Orchestrates the full preprocessing run end-to-end.

    Composes DataCleaner, DataSplitter, TabularEncoder, TextEncoder, and
    DatasetExporter into a single callable.  Each component can be replaced
    at construction time for experimentation or testing.

    Usage::

        pipeline = PreprocessingPipeline()
        summary = pipeline.run(raw_csv, interim_dir, processed_dir)
    """

    def __init__(
        self,
        cleaner: DataCleaner | None = None,
        splitter: DataSplitter | None = None,
        encoder: TabularEncoder | None = None,
        text_encoder: TextEncoder | None = None,
        exporter: DatasetExporter | None = None,
    ) -> None:
        self.cleaner = cleaner or DataCleaner()
        self.splitter = splitter or DataSplitter()
        self.encoder = encoder or TabularEncoder()
        self.text_encoder = text_encoder or TextEncoder()
        self.exporter = exporter or DatasetExporter()

    def run(
        self,
        raw_csv: Path,
        interim_dir: Path,
        processed_dir: Path,
    ) -> dict[str, Any]:
        interim_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        df = DataCleaner.load(raw_csv)
        self.cleaner.fit(df)
        cleaned = self.cleaner.clean(df)
        cleaned.to_parquet(interim_dir / "01_cleaned.parquet")
        (interim_dir / "preprocessing_config.json").write_text(
            json.dumps(self.cleaner.config, indent=2), encoding="utf-8"
        )
        featured = self.cleaner.add_features(cleaned)
        featured.to_parquet(interim_dir / "02_features.parquet")

        split_idx = self.splitter.split(featured)
        np.savez(
            processed_dir / "split_indices.npz",
            train=split_idx["train"],
            val=split_idx["val"],
            test=split_idx["test"],
        )

        train_df = featured.iloc[split_idx["train"]]
        val_df = featured.iloc[split_idx["val"]]
        test_df = featured.iloc[split_idx["test"]]
        y_splits = {
            key: featured.iloc[idx]["points"].to_numpy(dtype=np.float32)
            for key, idx in split_idx.items()
        }

        x_xgb: dict[str, np.ndarray] = {}
        x_xgb["train"], xgb_names, xgb_groups = self.encoder.fit_transform(
            train_df, y_splits["train"]
        )
        for key, part in [("val", val_df), ("test", test_df)]:
            x_xgb[key], _, _ = self.encoder.transform(part)

        xgb_dir = processed_dir / "xgboost"
        self.exporter.export_xgboost(xgb_dir, x_xgb, y_splits, self.encoder, xgb_names, xgb_groups)

        x_nn: dict[str, np.ndarray] = {}
        x_nn["train"], nn_names, nn_groups = self.encoder.transform(train_df)
        for key, part in [("val", val_df), ("test", test_df)]:
            x_nn[key], _, _ = self.encoder.transform(part)

        scaler = StandardScaler()
        scaler.fit(x_nn["train"][:, nn_groups["continuous"]])
        x_nn_scaled: dict[str, np.ndarray] = {}
        for key, x in x_nn.items():
            arr = x.copy()
            arr[:, nn_groups["continuous"]] = scaler.transform(x[:, nn_groups["continuous"]])
            x_nn_scaled[key] = arr

        self.text_encoder.fit(train_df)
        x_txt = {
            key: self.text_encoder.transform(featured.iloc[idx]) for key, idx in split_idx.items()
        }

        nn_dir = processed_dir / "nn"
        self.exporter.export_nn(
            nn_dir,
            x_nn_scaled,
            x_txt,
            y_splits,
            nn_names,
            nn_groups,
            scaler,
            self.text_encoder,
        )

        self.exporter.write_manifest(
            processed_dir / "dataset_manifest.json",
            xgb_dir,
            nn_dir,
            processed_dir / "split_indices.npz",
            {k: x_xgb[k].shape for k in x_xgb},
            {k: x_nn_scaled[k].shape for k in x_nn_scaled},
            {k: x_txt[k].nnz for k in x_txt},
        )

        return {
            "n_rows": len(featured),
            "xgb_features": len(xgb_names),
            "nn_tab_features": len(nn_names),
            "txt_features": x_txt["train"].shape[1],
        }
