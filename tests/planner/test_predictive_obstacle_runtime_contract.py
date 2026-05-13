"""Tests for predictive planner obstacle-feature runtime contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.planner.obstacle_features import (
    ObstacleFeatureSchemaError,
    PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
    predictive_feature_schema_metadata,
)
from robot_sf.planner.predictive_model import (
    PredictiveModelConfig,
    PredictiveTrajectoryModel,
    save_predictive_checkpoint,
)
from robot_sf.planner.socnav import PredictionPlannerAdapter, SocNavPlannerConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_prediction_planner_adapter_fails_closed_on_schema_mismatch(tmp_path: Path) -> None:
    """Runtime loading should reject obstacle-feature checkpoints under legacy config."""
    checkpoint = tmp_path / "predictive_obstacle.pt"
    cfg = PredictiveModelConfig(
        max_agents=3,
        horizon_steps=2,
        input_dim=10,
        hidden_dim=16,
        message_passing_steps=1,
        feature_schema_name=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
    )
    save_predictive_checkpoint(
        checkpoint,
        model=PredictiveTrajectoryModel(cfg),
        optimizer=None,
        epoch=1,
        feature_schema_metadata=predictive_feature_schema_metadata(
            model_family=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
            ego_conditioning=False,
        ),
    )
    adapter = PredictionPlannerAdapter(
        SocNavPlannerConfig(
            predictive_checkpoint_path=str(checkpoint),
            predictive_feature_schema_name="predictive_legacy_v1",
        )
    )

    try:
        adapter._build_model()
    except ObstacleFeatureSchemaError as exc:
        assert "Predictive feature schema mismatch" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure
        raise AssertionError("expected ObstacleFeatureSchemaError")
