"""Feature column schema for the Wine Reviews preprocessing pipeline.

This is the single source of truth for which columns receive which encoding
treatment.  Change here and the rest of the pipeline adapts automatically.
"""

# (source_column, encoded_feature_name) pairs — fit on train, CV-aware
TARGET_ENCODE_COLS: list[tuple[str, str]] = [
    ("taster_name", "taster_avg_points"),
    ("variety", "variety_avg_points"),
    ("country", "country_avg_points"),
    ("region_1", "region1_avg_points"),
    ("winery", "winery_avg_points"),
]

# (source_column, encoded_feature_name) pairs — log-frequency maps
FREQ_COLS: list[tuple[str, str]] = [
    ("winery", "winery_freq"),
    ("variety", "variety_review_count"),
]

# One-hot encoded columns (empty: taster_name and country already target-encoded)
OHE_COLS: list[str] = []

# Continuous features passed directly to the matrix (no encoding step)
DIRECT_CONTINUOUS: list[str] = ["log_price", "wine_age", "description_length"]

# Binary / flag features — excluded from scaling
BINARY_COLS: list[str] = [
    "price_missing",
    "vintage_missing",
    "is_luxury",
    "is_us",
    "has_designation",
]

# Derived name lists (convenience — do not edit directly)
TARGET_FEATURE_COLS: list[str] = [feat for _, feat in TARGET_ENCODE_COLS]
FREQ_FEATURE_COLS: list[str] = [feat for _, feat in FREQ_COLS]
