"""Project-wide constants shared across notebooks and modules."""

from pathlib import Path

from sympy import re

# ── Paths (relative to notebooks/ working directory) ─────────────────────────
RAW = Path("../data/raw")
INTERIM = Path("../data/interim")
PROCESSED = Path("../data/processed")
MODELS = Path("../models")
REPORTS = Path("../reports")
PRIMARY_CSV = "winemag-data-130k-v2.csv"

# ── ML / preprocessing ──────────────────────────────────────────────────────
RANDOM_STATE = 42
REF_YEAR = 2017
LUXURY_THRESHOLD = 200.0
POINTS_BINS = [79, 85, 88, 91, 94, 100]

# ── Column groups for preprocessing pipeline (see columns.py) ─────────────────
VINTAGE_RE = re.compile(r"\b(19|20)\d{2}\b")
