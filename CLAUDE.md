# CLAUDE.md — diplo-mod-1

Final project for the Machine Learning & Deep Learning module of a diplomatura program.
The goal is a full ML pipeline: EDA → preprocessing → classical ML + neural net → evaluation/comparison.

## Setup

```bash
uv sync                       # install all deps (creates .venv)
uv run pre-commit install     # register git hooks (run once per clone)
```

## Running things

```bash
uv run jupyter lab            # start JupyterLab
uv run python src/script.py   # run a script inside the venv
```

## Quality checks

```bash
uv run poe check              # lint + typecheck + nbtest (full suite)
uv run poe lint               # ruff check + format --check
uv run poe fmt                # ruff format (auto-fix)
uv run poe typecheck          # mypy src/ + nbqa mypy notebooks/
uv run poe nbtest             # pytest --nbmake notebooks/ (slow, needs data/)
```

Always run `uv run poe lint`, `uv run poe typecheck`, and `uv run poe test` (unit tests) after making any change and fix any errors. Do **not** run `uv run poe nbtest`, `uv run poe check`, or `jupyter nbconvert --execute` on notebooks — full notebook execution (including any live W&B runs it triggers) is the user's to run themselves. Do not commit, push, or open a PR — even when working on a feature/experiment branch — until the user explicitly says to. Prepare the changes and wait for that instruction.

## Project structure

```
data/raw/        # Original files — READ ONLY, never modify
data/interim/    # Intermediate outputs written by notebook 02
data/processed/  # Train/test splits, model-ready features (written by 02, read by 03-05)
configs/         # User-editable JSON configs (e.g. Optuna search spaces) — versioned in git
notebooks/       # Numbered notebooks — must be run in order (00 → 06)
src/diplo_mod_1/ # Reusable Python package (helpers, transforms, etc.)
models/          # Saved model checkpoints — versioned in git
reports/         # Final report and figures
```

## Notebooks

Run in order. Each notebook maps to one report section:

| Notebook | Section |
|---|---|
| `00-intro-dataset.ipynb` | Introduction & dataset description |
| `01-eda.ipynb` | Exploratory Data Analysis |
| `02-preprocessing.ipynb` | Preprocessing, encoding, splits |
| `03-train-baseline-xgboost.ipynb` | Classical ML model (XGBoost) |
| `04-train-nn-pytorch.ipynb` | Neural network (PyTorch) |
| `05-evaluation-comparison.ipynb` | Evaluation & model comparison |
| `06-conclusions.ipynb` | Conclusions |

To execute all notebooks end-to-end:
```bash
uv run jupyter nbconvert --to notebook --execute notebooks/*.ipynb
```

## Notebook conventions

- **First cell is always an import cell**: all `import` statements go in the very first code cell of each notebook. Never scatter imports across later cells — if a new import is needed, add it to the first cell.
- Imports are sorted: stdlib → third-party → local, alphabetically within each group (ruff-compatible order).
- **Constants live in `src/diplo_mod_1/constants.py`**: shared values like paths (`RAW`, `INTERIM`, `PROCESSED`, `MODELS`, `REPORTS`), `RANDOM_STATE`, and thresholds are defined once in `constants.py` and imported in notebooks and modules. Never redefine them inline.

## Gotchas

- `data/` is not versioned — only `.gitkeep` files are committed. You must provide your own dataset in `data/raw/` before running notebooks.
- `nbtest` (`pytest --nbmake`) will fail if `data/` is empty.
- PyTorch is configured to use Apple Silicon MPS when available. Don't add CUDA-specific code.
- `notebooks/*.ipynb` ignores `F401` (unused imports) — exploratory cells intentionally import without always using.
- `gitleaks` pre-commit hook will block commits containing secrets/API keys. Never hardcode credentials.
- API keys (e.g. `WANDB_API_KEY`) go in `.env` (gitignored, loaded via `python-dotenv`), never inline. Copy `.env.example` to `.env` and fill in your own values.
- When adding a new dependency: `uv add <package>` (updates both `pyproject.toml` and `uv.lock`).
