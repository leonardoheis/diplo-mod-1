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

Always run `uv run poe check` after making any change and fix any errors. Do not commit or push — the user handles all commits and pushes explicitly.

## Project structure

```
data/raw/        # Original files — READ ONLY, never modify
data/interim/    # Intermediate outputs written by notebook 02
data/processed/  # Train/test splits, model-ready features (written by 02, read by 03-05)
notebooks/       # Numbered notebooks — must be run in order (00 → 06)
src/diplo_mod_1/ # Reusable Python package (helpers, transforms, etc.)
models/          # Saved model checkpoints — not versioned
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

## Gotchas

- `data/` is not versioned — only `.gitkeep` files are committed. You must provide your own dataset in `data/raw/` before running notebooks.
- `nbtest` (`pytest --nbmake`) will fail if `data/` is empty.
- PyTorch is configured to use Apple Silicon MPS when available. Don't add CUDA-specific code.
- `notebooks/*.ipynb` ignores `F401` (unused imports) — exploratory cells intentionally import without always using.
- `gitleaks` pre-commit hook will block commits containing secrets/API keys. Never hardcode credentials.
- When adding a new dependency: `uv add <package>` (updates both `pyproject.toml` and `uv.lock`).
