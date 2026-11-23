"""Local predictor artifacts for orchestration tests that retain real checkpoint preflight."""

from pathlib import Path

import pytest
import yaml

from robot_sf.models import registry as model_registry


def stage_predictive_checkpoint_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Bind predictor defaults to a valid synthetic model without changing other arm metadata.

    Returns:
        The isolated registry path; its checkpoint is test setup, not benchmark evidence.
    """
    # Model classes initialize torch; unrelated selectors must collect without it.
    from robot_sf.planner.predictive_model import (
        PredictiveModelConfig,
        PredictiveTrajectoryModel,
        save_predictive_checkpoint,
    )

    checkpoint = tmp_path / "orchestrator_predictor.pt"
    save_predictive_checkpoint(
        checkpoint,
        model=PredictiveTrajectoryModel(PredictiveModelConfig(hidden_dim=8, horizon_steps=8)),
        optimizer=None,
        epoch=0,
    )
    root = Path(__file__).resolve().parents[2]
    entries = model_registry.load_registry(root / "model/registry.yaml")
    for model_id in ("predictive_proxy_selected_v1", "predictive_proxy_selected_v2_full"):
        entries[model_id] = {
            "model_id": model_id,
            "local_path": str(checkpoint),
            "local_only": True,
        }
    registry_path = tmp_path / "orchestrator_registry.yaml"
    registry_path.write_text(
        yaml.safe_dump({"version": 1, "models": list(entries.values())}), encoding="utf-8"
    )
    monkeypatch.setattr(model_registry, "DEFAULT_REGISTRY_PATH", registry_path)
    return registry_path
