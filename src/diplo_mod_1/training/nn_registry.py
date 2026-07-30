"""NNModelRegistry — persists versioned PyTorch checkpoints and tracks the best run.

Deliberately parallel to (not sharing a base class with) ``ModelRegistry``
(``registry.py``) — XGBoost's ``joblib.dump`` of a full picklable object and
PyTorch's ``torch.save(state_dict)`` + architecture config are different
enough serialization mechanics that a shared abstraction for exactly these
two call sites would be premature generalization.
"""

import shutil
from pathlib import Path

import torch

from diplo_mod_1.schemas.evaluation import EvaluationResult
from diplo_mod_1.training.config import RunRecord, TuningHistory
from diplo_mod_1.training.nn_model import WineScorePredictorNet


class NNModelRegistry:
    """Writes one checkpoint file per tuning run and keeps a JSON history of all of them.

    Every run's ``state_dict`` (plus enough config to reconstruct the
    network) is kept as ``<run_id>.pt`` rather than overwritten, and
    ``nn_best.pt`` always points at whichever run has the lowest test-split
    RMSE on record — same contract as ``ModelRegistry``.

    Usage::

        run_record, history = NNModelRegistry.save_run(
            MODELS, REPORTS / "nn_metrics.json",
            best_model, run_id, tuning_config_name, study.best_params, result,
        )
    """

    @staticmethod
    def save_run(
        models_dir: Path,
        metrics_path: Path,
        model: WineScorePredictorNet,
        run_id: str,
        tuning_config_name: str,
        best_params: dict[str, float | int | str],
        result: EvaluationResult,
    ) -> tuple[RunRecord, TuningHistory]:
        """Save ``model``, append its run to the history, and update the best pointer."""
        if not hasattr(model, "model_"):
            raise RuntimeError("Call fit() before save_run().")
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        model_filename = f"{run_id}.pt"
        torch.save(
            {
                "state_dict": model.model_.state_dict(),
                "input_dim": model.input_dim,
                "hidden_sizes": model.hidden_sizes,
                "dropout": model.dropout,
            },
            models_dir / model_filename,
        )

        history = NNModelRegistry._load_history(metrics_path)
        record = RunRecord(
            run_id=run_id,
            tuning_config=tuning_config_name,
            model_filename=model_filename,
            best_params=best_params,
            metrics=result.metrics,
        )
        history.runs.append(record)

        runnable = TuningHistory(
            runs=[r for r in history.runs if (models_dir / r.model_filename).exists()]
        )
        best = runnable.best_run(split="test")
        if best is not None:
            history.best_run_id = best.run_id
            shutil.copyfile(models_dir / best.model_filename, models_dir / "nn_best.pt")

        metrics_path.write_text(history.model_dump_json(indent=2), encoding="utf-8")
        return record, history

    @staticmethod
    def _load_history(metrics_path: Path) -> TuningHistory:
        if not metrics_path.exists():
            return TuningHistory()
        try:
            return TuningHistory.model_validate_json(metrics_path.read_text(encoding="utf-8"))
        except ValueError:
            return TuningHistory()
