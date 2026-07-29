"""ModelRegistry — persists versioned XGBoost checkpoints and tracks the best run."""

import shutil
from pathlib import Path

import joblib
from xgboost import XGBRegressor

from diplo_mod_1.schemas.evaluation import EvaluationResult
from diplo_mod_1.training.config import RunRecord, TuningHistory


class ModelRegistry:
    """Writes one checkpoint file per tuning run and keeps a JSON history of all of them.

    Every run's model is kept (``<run_id>.joblib`` — ``run_id`` is already
    prefixed with the tuning config name by convention, e.g.
    ``xgboost_tuning_wide-20260728T175613Z``) rather than overwritten, and
    ``xgboost_best.joblib`` always points at whichever run has the lowest
    test-split RMSE on record.

    Usage::

        run_record, history = ModelRegistry.save_run(
            MODELS, REPORTS / "xgboost_metrics.json",
            best_model, run_id, tuning_config_name, study.best_params, result,
        )
    """

    @staticmethod
    def save_run(
        models_dir: Path,
        metrics_path: Path,
        model: XGBRegressor,
        run_id: str,
        tuning_config_name: str,
        best_params: dict[str, float],
        result: EvaluationResult,
    ) -> tuple[RunRecord, TuningHistory]:
        """Save ``model``, append its run to the history, and update the best pointer."""
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        model_filename = f"{run_id}.joblib"
        joblib.dump(model, models_dir / model_filename)

        history = ModelRegistry._load_history(metrics_path)
        record = RunRecord(
            run_id=run_id,
            tuning_config=tuning_config_name,
            model_filename=model_filename,
            best_params=best_params,
            metrics=result.metrics,
        )
        history.runs.append(record)

        # Only consider runs whose checkpoint actually exists on disk — a run
        # recorded in an older/stale history file may reference a filename
        # that's since been deleted or renamed (e.g. models/ was cleared).
        runnable = TuningHistory(
            runs=[r for r in history.runs if (models_dir / r.model_filename).exists()]
        )
        best = runnable.best_run(split="test")
        if best is not None:
            history.best_run_id = best.run_id
            shutil.copyfile(models_dir / best.model_filename, models_dir / "xgboost_best.joblib")

        metrics_path.write_text(history.model_dump_json(indent=2), encoding="utf-8")
        return record, history

    @staticmethod
    def _load_history(metrics_path: Path) -> TuningHistory:
        if not metrics_path.exists():
            return TuningHistory()
        try:
            return TuningHistory.model_validate_json(metrics_path.read_text(encoding="utf-8"))
        except ValueError:
            # Pre-existing file in an older/incompatible shape — start fresh rather than crash.
            return TuningHistory()
