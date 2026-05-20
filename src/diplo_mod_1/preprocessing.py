"""Wine Reviews preprocessing: cleaning, features, encoding, dual dataset export."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from diplo_mod_1.constants import LUXURY_THRESHOLD, POINTS_BINS, RANDOM_STATE, REF_YEAR

VINTAGE_RE = re.compile(r"\b(19|20)\d{2}\b")

TARGET_ENCODE_COLS = [
    ("taster_name", "taster_avg_points"),
    ("variety", "variety_avg_points"),
    ("country", "country_avg_points"),
    ("region_1", "region1_avg_points"),
    ("winery", "winery_avg_points"),
]
FREQ_COLS = [
    ("winery", "winery_freq"),
    ("variety", "variety_review_count"),
]
OHE_COLS: list[str] = []  # taster_name and country covered by target encoding

DIRECT_CONTINUOUS = ["log_price", "wine_age", "description_length"]
BINARY_COLS = [
    "price_missing",
    "vintage_missing",
    "is_luxury",
    "is_us",
    "has_designation",
]
TARGET_FEATURE_COLS = [name for _, name in TARGET_ENCODE_COLS]
FREQ_FEATURE_COLS = [name for _, name in FREQ_COLS]


def load_raw(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col=0)


def extract_vintage_year(title: pd.Series) -> tuple[pd.Series, pd.Series]:
    years = title.astype(str).str.extract(VINTAGE_RE, expand=False)
    vintage_year = pd.to_numeric(years, errors="coerce")
    vintage_missing = vintage_year.isna().astype(int)
    return vintage_year, vintage_missing


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    out = out.drop(columns=["region_2", "taster_twitter_handle"], errors="ignore")

    vintage_year, vintage_missing = extract_vintage_year(out["title"])
    out["vintage_year"] = vintage_year
    out["vintage_missing"] = vintage_missing

    price_missing = out["price"].isna().astype(int)
    median_price = out.groupby(["country", "variety"], dropna=False)["price"].transform("median")
    global_median = out["price"].median()
    out["price"] = out["price"].fillna(median_price).fillna(global_median)

    for col in ["country", "region_1", "variety", "taster_name"]:
        out[col] = out[col].fillna("Unknown")
    if out["variety"].isna().any():
        out["variety"] = out["variety"].fillna("Unknown")

    config = {
        "luxury_threshold": LUXURY_THRESHOLD,
        "ref_year": REF_YEAR,
        "global_price_median": float(global_median),
        "random_state": RANDOM_STATE,
    }
    out["price_missing"] = price_missing
    return out, config


def add_base_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    ref_year = config["ref_year"]
    luxury_threshold = config["luxury_threshold"]

    out["log_price"] = np.log1p(out["price"])
    out["is_luxury"] = (out["price"] > luxury_threshold).astype(int)
    out["is_us"] = (out["country"] == "US").astype(int)
    out["has_designation"] = out["designation"].notna().astype(int)
    out["description_length"] = out["description"].astype(str).str.len()

    median_vintage = out["vintage_year"].median()
    out["vintage_year"] = out["vintage_year"].fillna(median_vintage)
    out["wine_age"] = ref_year - out["vintage_year"]

    return out


def stratified_split(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    y = df["points"]
    strata = pd.cut(y, bins=POINTS_BINS, labels=False)

    idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=random_state,
        stratify=strata,
    )
    strata_tv = strata.iloc[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        random_state=random_state,
        stratify=strata_tv,
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _target_encode_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, TargetEncoder]]:
    encoders: dict[str, TargetEncoder] = {}
    train_parts: dict[str, np.ndarray] = {}
    for src_col, feat_name in TARGET_ENCODE_COLS:
        enc = TargetEncoder(cv=5, random_state=RANDOM_STATE, target_type="continuous")
        train_parts[feat_name] = enc.fit_transform(train_df[[src_col]], y_train).ravel()
        encoders[src_col] = enc

    train_target = pd.DataFrame(train_parts, index=train_df.index)
    val_target = pd.DataFrame(
        {
            feat_name: encoders[src_col].transform(val_df[[src_col]]).ravel()
            for src_col, feat_name in TARGET_ENCODE_COLS
        },
        index=val_df.index,
    )
    test_target = pd.DataFrame(
        {
            feat_name: encoders[src_col].transform(test_df[[src_col]]).ravel()
            for src_col, feat_name in TARGET_ENCODE_COLS
        },
        index=test_df.index,
    )
    return train_target, val_target, test_target, encoders


def _fit_freq_maps(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    maps: dict[str, dict[str, float]] = {}
    for src_col, feat_name in FREQ_COLS:
        counts = train_df[src_col].value_counts()
        maps[feat_name] = {k: float(np.log1p(v)) for k, v in counts.items()}
    return maps


def _apply_freq_features(df: pd.DataFrame, freq_maps: dict[str, dict[str, float]]) -> pd.DataFrame:
    parts = {}
    for src_col, feat_name in FREQ_COLS:
        mapping = freq_maps[feat_name]
        parts[feat_name] = df[src_col].map(mapping).fillna(0.0)
    return pd.DataFrame(parts, index=df.index)


def _fit_ohe(train_df: pd.DataFrame) -> OneHotEncoder:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    if OHE_COLS:
        ohe.fit(train_df[OHE_COLS])
    return ohe


def _apply_ohe(df: pd.DataFrame, ohe: OneHotEncoder) -> tuple[np.ndarray, list[str]]:
    if not OHE_COLS:
        return np.empty((len(df), 0), dtype=np.float32), []
    arr = ohe.transform(df[OHE_COLS])
    names = list(ohe.get_feature_names_out(OHE_COLS))
    return arr, names


def assemble_matrix(
    df: pd.DataFrame,
    target_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    ohe_arr: np.ndarray,
    ohe_names: list[str],
    *,
    include_strictness: bool = False,
    taster_strictness: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []

    cont_direct = df[DIRECT_CONTINUOUS].to_numpy(dtype=np.float32)
    blocks.append(cont_direct)
    names.extend(DIRECT_CONTINUOUS)

    binary = df[BINARY_COLS].to_numpy(dtype=np.float32)
    blocks.append(binary)
    names.extend(BINARY_COLS)

    target_arr = target_df[TARGET_FEATURE_COLS].to_numpy(dtype=np.float32)
    blocks.append(target_arr)
    names.extend(TARGET_FEATURE_COLS)

    if include_strictness and taster_strictness is not None:
        blocks.append(taster_strictness.reshape(-1, 1).astype(np.float32))
        names.append("taster_strictness")

    freq_arr = freq_df[FREQ_FEATURE_COLS].to_numpy(dtype=np.float32)
    blocks.append(freq_arr)
    names.extend(FREQ_FEATURE_COLS)

    blocks.append(ohe_arr.astype(np.float32))
    names.extend(ohe_names)

    x = np.hstack(blocks).astype(np.float32)

    n_cont_direct = len(DIRECT_CONTINUOUS)
    n_binary = len(BINARY_COLS)
    n_target = len(TARGET_FEATURE_COLS)
    n_strict = 1 if include_strictness else 0
    n_freq = len(FREQ_FEATURE_COLS)

    # Actual column layout:
    #   [0 .. n_cont_direct-1]                          → DIRECT_CONTINUOUS
    #   [n_cont_direct .. n_cont_direct+n_binary-1]     → BINARY_COLS  (do NOT scale)
    #   [n_cont_direct+n_binary .. +n_target-1]         → TARGET_FEATURE_COLS
    #   [... +n_strict]                                  → taster_strictness (NN only)
    #   [... +n_freq-1]                                  → FREQ_FEATURE_COLS
    #   rest                                             → OHE
    binary_start = n_cont_direct
    target_start = binary_start + n_binary
    ohe_start = target_start + n_target + n_strict + n_freq

    continuous_idx = list(range(0, n_cont_direct)) + list(range(target_start, ohe_start))
    binary_idx = list(range(binary_start, binary_start + n_binary))
    groups = {
        "continuous": continuous_idx,
        "binary": binary_idx,
        "ohe": list(range(ohe_start, len(names))),
    }
    return x, names, groups


def build_encoded_splits(
    df: pd.DataFrame,
    split_idx: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    train_df = df.iloc[split_idx["train"]]
    val_df = df.iloc[split_idx["val"]]
    test_df = df.iloc[split_idx["test"]]

    y = {
        "train": train_df["points"].to_numpy(dtype=np.float32),
        "val": val_df["points"].to_numpy(dtype=np.float32),
        "test": test_df["points"].to_numpy(dtype=np.float32),
    }

    train_target, val_target, test_target, encoders = _target_encode_splits(
        train_df, val_df, test_df, y["train"]
    )
    target_by_split = {"train": train_target, "val": val_target, "test": test_target}
    freq_maps = _fit_freq_maps(train_df)
    ohe = _fit_ohe(train_df)

    global_mean = float(y["train"].mean())
    taster_strict = {
        key: target_by_split[key]["taster_avg_points"].to_numpy() - global_mean
        for key in ("train", "val", "test")
    }

    x_xgb: dict[str, np.ndarray] = {}
    artifacts: dict[str, Any] = {
        "target_encoders": encoders,
        "target_by_split": target_by_split,
        "freq_maps": freq_maps,
        "ohe": ohe,
        "global_points_mean": global_mean,
        "taster_strictness": taster_strict,
    }

    feature_names_xgb: list[str] = []
    groups_xgb: dict[str, list[int]] = {}

    for key, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        target_part = target_by_split[key]
        freq_part = _apply_freq_features(part, freq_maps)
        ohe_arr, ohe_names = _apply_ohe(part, ohe)
        x_mat, names, groups = assemble_matrix(
            part,
            target_part,
            freq_part,
            ohe_arr,
            ohe_names,
            include_strictness=False,
        )
        x_xgb[key] = x_mat
        if key == "train":
            feature_names_xgb = names
            groups_xgb = groups

    artifacts["feature_names_xgb"] = feature_names_xgb
    artifacts["column_groups_xgb"] = groups_xgb
    artifacts["taster_strictness"] = taster_strict

    return x_xgb, y, artifacts


def build_nn_tabular(
    df: pd.DataFrame,
    split_idx: dict[str, np.ndarray],
    artifacts: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[str], dict[str, list[int]]]:
    freq_maps = artifacts["freq_maps"]
    ohe = artifacts["ohe"]
    target_by_split = artifacts["target_by_split"]

    x_nn: dict[str, np.ndarray] = {}
    feature_names: list[str] = []
    train_groups: dict[str, list[int]] = {}

    for key, indices in [
        ("train", split_idx["train"]),
        ("val", split_idx["val"]),
        ("test", split_idx["test"]),
    ]:
        part = df.iloc[indices]
        target_part = target_by_split[key]
        freq_part = _apply_freq_features(part, freq_maps)
        ohe_arr, ohe_names = _apply_ohe(part, ohe)
        x_mat, names, col_groups = assemble_matrix(
            part,
            target_part,
            freq_part,
            ohe_arr,
            ohe_names,
            include_strictness=False,
        )
        x_nn[key] = x_mat
        if key == "train":
            feature_names = names
            train_groups = col_groups

    return x_nn, feature_names, train_groups


def scale_continuous(
    x_splits: dict[str, np.ndarray],
    continuous_idx: list[int],
) -> tuple[dict[str, np.ndarray], StandardScaler]:
    scaler = StandardScaler()
    train = x_splits["train"]
    scaler.fit(train[:, continuous_idx])

    scaled: dict[str, np.ndarray] = {}
    for key, x in x_splits.items():
        out = x.copy()
        out[:, continuous_idx] = scaler.transform(x[:, continuous_idx])
        scaled[key] = out
    return scaled, scaler


def fit_tfidf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, sparse.csr_matrix], TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=5,
        stop_words="english",
    )
    vectorizer.fit(train_df["description"].astype(str))
    return {
        "train": vectorizer.transform(train_df["description"].astype(str)),
        "val": vectorizer.transform(val_df["description"].astype(str)),
        "test": vectorizer.transform(test_df["description"].astype(str)),
    }, vectorizer


def export_xgboost_dataset(
    out_dir: Path,
    x_splits: dict[str, np.ndarray],
    y_splits: dict[str, np.ndarray],
    artifacts: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        np.save(out_dir / f"X_{split}.npy", x_splits[split])
        np.save(out_dir / f"y_{split}.npy", y_splits[split])

    meta = {
        "feature_names": artifacts["feature_names_xgb"],
        "column_groups": artifacts["column_groups_xgb"],
        "scaled": False,
    }
    (out_dir / "feature_names.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    joblib.dump(artifacts["target_encoders"], out_dir / "target_encoders.joblib")
    joblib.dump(artifacts["ohe"], out_dir / "ohe.joblib")


def export_nn_dataset(
    out_dir: Path,
    x_tab_splits: dict[str, np.ndarray],
    x_txt_splits: dict[str, sparse.csr_matrix],
    y_splits: dict[str, np.ndarray],
    feature_names: list[str],
    column_groups: dict[str, list[int]],
    scaler: StandardScaler,
    vectorizer: TfidfVectorizer,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        np.save(out_dir / f"X_tab_{split}.npy", x_tab_splits[split])
        sparse.save_npz(out_dir / f"X_txt_{split}.npz", x_txt_splits[split])
        np.save(out_dir / f"y_{split}.npy", y_splits[split])

    meta = {
        "feature_names": feature_names,
        "column_groups": column_groups,
        "scaled": "continuous_only",
        "continuous_column_indices": column_groups["continuous"],
    }
    (out_dir / "feature_names.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "continuous_column_indices.json").write_text(
        json.dumps(column_groups["continuous"], indent=2),
        encoding="utf-8",
    )
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")


def write_manifest(
    path: Path,
    xgb_dir: Path,
    nn_dir: Path,
    split_path: Path,
    x_shapes: dict[str, tuple[int, ...]],
    nn_tab_shapes: dict[str, tuple[int, ...]],
    txt_nnz: dict[str, int],
) -> None:
    manifest = {
        "target": "points",
        "random_state": RANDOM_STATE,
        "split": str(split_path.name),
        "xgboost": {
            "path": "xgboost/",
            "scaled": False,
            "X_shape_train": list(x_shapes["train"]),
            "X_shape_val": list(x_shapes["val"]),
            "X_shape_test": list(x_shapes["test"]),
        },
        "nn": {
            "path": "nn/",
            "scaled": "continuous_only",
            "X_tab_shape_train": list(nn_tab_shapes["train"]),
            "X_tab_shape_val": list(nn_tab_shapes["val"]),
            "X_tab_shape_test": list(nn_tab_shapes["test"]),
            "X_txt_nnz_train": txt_nnz["train"],
            "X_txt_nnz_val": txt_nnz["val"],
            "X_txt_nnz_test": txt_nnz["test"],
        },
        "note": "Training not performed in notebook 02.",
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_pipeline(
    raw_csv: Path,
    interim_dir: Path,
    processed_dir: Path,
) -> dict[str, Any]:
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_csv)
    cleaned, config = clean_dataframe(df)
    cleaned.to_parquet(interim_dir / "01_cleaned.parquet")
    (interim_dir / "preprocessing_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    featured = add_base_features(cleaned, config)
    featured.to_parquet(interim_dir / "02_features.parquet")

    split_idx = stratified_split(featured)
    np.savez(
        processed_dir / "split_indices.npz",
        train=split_idx["train"],
        val=split_idx["val"],
        test=split_idx["test"],
    )

    x_xgb, y_splits, artifacts = build_encoded_splits(featured, split_idx)

    xgb_dir = processed_dir / "xgboost"
    export_xgboost_dataset(xgb_dir, x_xgb, y_splits, artifacts)

    x_nn_raw, nn_names, nn_groups = build_nn_tabular(featured, split_idx, artifacts)
    x_nn_scaled, scaler = scale_continuous(x_nn_raw, nn_groups["continuous"])

    train_df = featured.iloc[split_idx["train"]]
    val_df = featured.iloc[split_idx["val"]]
    test_df = featured.iloc[split_idx["test"]]
    x_txt, vectorizer = fit_tfidf(train_df, val_df, test_df)

    nn_dir = processed_dir / "nn"
    export_nn_dataset(
        nn_dir,
        x_nn_scaled,
        x_txt,
        y_splits,
        nn_names,
        nn_groups,
        scaler,
        vectorizer,
    )

    write_manifest(
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
        "xgb_features": len(artifacts["feature_names_xgb"]),
        "nn_tab_features": len(nn_names),
        "txt_features": x_txt["train"].shape[1],
    }
