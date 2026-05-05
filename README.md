# Postgraduate Diploma in AI — Module 1: Machine Learning & Deep Learning
## Final Project

Final project for the **Machine Learning & Deep Learning** module of the diplomatura program.

## Goal

Apply the techniques covered in the program to a real dataset, going through the full cycle:

1. Exploration (EDA)
2. Preprocessing
3. Model training
4. Critical evaluation of results

## Assignment

### Dataset

Use a dataset from your own work environment (recommended) or one of the example datasets listed below.

### Models to train

At least **two models** must be trained on the same dataset for the same problem:

- **A classical ML model**: Linear/Logistic Regression, Decision Tree, XGBoost, Random Forest, or another justified choice.
- **A Neural Network**: with TensorFlow or PyTorch.

This allows a direct comparison between a classical approach and a Deep Learning one.

## Report structure

1. **Introduction** — Dataset description, target variable, and problem type (classification or regression).
2. **Exploratory Data Analysis (EDA)** — Descriptive statistics, visualizations, detection of class imbalance, outliers, and missing values.
3. **Preprocessing** — Handling nulls and outliers, categorical encoding, normalization, train/test split, balancing techniques.
4. **Training**
   - *ML model*: algorithm justification, hyperparameters and tuning.
   - *Neural Network*: architecture (layers, activations), optimizer, epochs, training curves, regularization (dropout, early stopping, etc.).
5. **Evaluation and comparison** — Key section of the project.
   - Classification: accuracy, precision, recall, F1-score, confusion matrix (mandatory for both models), ROC/AUC optional.
   - Regression: MAE, RMSE, R², predicted vs actual values plot.
   - Critical analysis: which model performed better and why? Did the neural network outperform classical ML? Is there overfitting? Which features were most relevant? What would improve with more data/time?
6. **Conclusions** — Summary of results, lessons learned, and future work.

## Example datasets

| Dataset | Type | Rows | Source |
| --- | --- | --- | --- |
| [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Classification | 7,043 | IBM / Kaggle |
| [Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45) | Classification | 303 | UCI ML Repository |
| California Housing (`sklearn.datasets.fetch_california_housing`) | Regression | 20,640 | StatLib / scikit-learn |

## Chosen dataset

**[Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews/data)** — wine reviews published in WineEnthusiast magazine, scraped by Zack Thoutt (Kaggle).

| Attribute | Value |
| --- | --- |
| File | `winemag-data-130k-v2.csv` |
| Rows | 129,971 |
| Columns | 13 |
| Problem type | Regression |
| Target variable | `points` — score assigned by the taster (80–100) |

### Columns

| Column | Type | Description |
| --- | --- | --- |
| `country` | categorical | Country where the wine was produced |
| `description` | text | Review written by the taster |
| `designation` | categorical | Vineyard sub-label within the winery |
| `points` | integer | Taster score (80–100) — **target variable** |
| `price` | float | Bottle price in USD |
| `province` | categorical | Province or state of origin |
| `region_1` | categorical | Specific wine-growing region |
| `region_2` | categorical | More specific sub-region |
| `taster_name` | categorical | Name of the WineEnthusiast reviewer |
| `taster_twitter_handle` | categorical | Twitter handle of the reviewer |
| `title` | text | Full wine label (includes vintage year) |
| `variety` | categorical | Grape variety (e.g. Pinot Noir, Chardonnay) |
| `winery` | categorical | Name of the producing winery |

> The other two files in `data/raw/` (`winemag-data_first150k.csv` and `winemag-data-130k-v2.json`) were discarded: the JSON is an exact mirror of the CSV, and the 150k is an independent scrape with only 57% row overlap and no taster columns.

## Repository structure

```
.
├── data/
│   ├── raw/             # Original, immutable data
│   ├── interim/         # Intermediate transformations
│   └── processed/       # Final, model-ready data
├── notebooks/           # Numbered notebooks (one per report section)
├── src/                 # Reusable source code (diplo_mod_1 package)
├── models/              # Trained models / checkpoints (not versioned)
├── reports/             # Final report and figures
├── pyproject.toml       # Dependencies and project metadata (uv)
├── uv.lock              # Reproducible lockfile
└── README.md
```

## Notebooks

Naming convention: `NN-section-description.ipynb`. The numeric prefix sets the execution order (Restart & Run All).

| Notebook | Report section |
| --- | --- |
| `notebooks/00-intro-dataset.ipynb` | 3.1 Introduction |
| `notebooks/01-eda.ipynb` | 3.2 Exploratory Data Analysis (EDA) |
| `notebooks/02-preprocessing.ipynb` | 3.3 Preprocessing |
| `notebooks/03-train-baseline-xgboost.ipynb` | 3.4 Training — Classical ML model |
| `notebooks/04-train-nn-pytorch.ipynb` | 3.4 Training — Neural Network |
| `notebooks/05-evaluation-comparison.ipynb` | 3.5 Evaluation and comparison |
| `notebooks/06-conclusions.ipynb` | 3.6 Conclusions |

To execute all notebooks in order:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/*.ipynb
```

## Data layers

| Folder | Contents | Readers | Writers |
| --- | --- | --- | --- |
| `data/raw/` | Original, untouched data (read-only) | 00, 01, 02 | (manual download) |
| `data/interim/` | Intermediate results between notebooks | as needed | 02 |
| `data/processed/` | Splits and model-ready features | 03, 04, 05 | 02 |

The contents of `data/` are not versioned — only `.gitkeep` files are committed.

## Quality and linting

Configured tooling:

- `ruff` (lint + format) — pyflakes (`F`) rules, includes notebooks.
- `mypy` (on `src/`) and `nbqa` + `mypy` (on `notebooks/`).
- `nbmake` — executes each notebook end-to-end as a test.
- `pre-commit` — runs all quality hooks on commit:
  - `ruff-format` and `ruff-check --exit-non-zero-on-fix`
  - `mypy` (src) and `nbqa-mypy` (notebooks)
  - Basic hygiene: trailing-whitespace, end-of-file-fixer, check-yaml, debug-statements
  - `uv-lock` to keep `uv.lock` in sync with `pyproject.toml`
  - `gitleaks` to prevent committing secrets
- `poethepoet` — task runner: orchestrates all checks behind a single command.

### Setup (once per clone)

```bash
uv sync                       # install dev deps
uv run pre-commit install     # register git hooks
```

### Single command

```bash
uv run poe check              # lint + typecheck + nbtest in sequence
```

### Granular tasks

```bash
uv run poe lint               # ruff check + ruff format --check
uv run poe fmt                # ruff format (auto-fix)
uv run poe typecheck          # mypy src + nbqa mypy notebooks
uv run poe nbtest             # pytest --nbmake notebooks (slow, requires data/)
uv run poe precommit          # run all hooks against all files
uv run poe precommit-update   # manually bump hook revisions
```

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — package & project manager

## Setup

The project uses [`uv`](https://docs.astral.sh/uv/) to manage dependencies and the virtual environment.

```bash
uv sync                       # create .venv and install from lockfile
uv add <package>              # add a new dependency
uv run jupyter lab            # run commands inside the environment
uv run python src/script.py
```

## Stack

- Python 3.10+
- pandas, numpy, scikit-learn, matplotlib, seaborn, fg-data-profiling
- XGBoost (classical ML model)
- PyTorch (neural network, with Apple Silicon / MPS GPU support)
- Jupyter / JupyterLab
