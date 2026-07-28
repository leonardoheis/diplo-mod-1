"""Project-wide constants shared across notebooks and modules."""

import re
from pathlib import Path

# ── Paths (relative to notebooks/ working directory) ─────────────────────────
RAW = Path("../data/raw")
INTERIM = Path("../data/interim")
PROCESSED = Path("../data/processed")
MODELS = Path("../models")
REPORTS = Path("../reports")
CONFIGS = Path("../configs")
PRIMARY_CSV = "winemag-data-130k-v2.csv"

# ── ML / preprocessing ──────────────────────────────────────────────────────
RANDOM_STATE = 42
REF_YEAR = 2017
LUXURY_THRESHOLD = 200.0
POINTS_MIN = 80
POINTS_MAX = 100
POINTS_BINS = [79, 85, 88, 91, 94, 100]
YEAR_MIN = 1900
YEAR_MAX = 2100
TEST_SIZE = 0.2
VAL_SIZE = 0.2
TARGET_ENCODER_CV = 5

# ── Column groups for preprocessing pipeline (see columns.py) ─────────────────
VINTAGE_RE = re.compile(r"\b((?:19|20)\d{2})\b")
