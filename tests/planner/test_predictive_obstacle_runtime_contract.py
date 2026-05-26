"""Tests for predictive planner obstacle-feature runtime contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.nav.obstacle import Obstacle
from robot_sf.planner.obstacle_features import (
    PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
    ObstacleFeatureSchemaError,
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


def test_prediction_planner_adapter_places_legacy_obstacle_rows_after_legacy_base() -> None:
    """Obstacle rows should begin after the 4D legacy base, not after ego-only columns."""
    adapter = PredictionPlannerAdapter(
        SocNavPlannerConfig(
            predictive_feature_schema_name=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
            predictive_ego_conditioning=False,
            predictive_max_agents=2,
        )
    )
    adapter._ensure_model = lambda: SimpleNamespace(config=SimpleNamespace(input_dim=10))  # type: ignore[method-assign]

    state, mask, _robot_pos, _robot_heading = adapter._build_model_input(
        {
            "robot": {
                "position": np.array([0.0, 0.0], dtype=np.float32),
                "heading": np.array([0.0], dtype=np.float32),
                "speed": np.array([0.0, 0.0], dtype=np.float32),
            },
            "goal": {"current": np.array([1.0, 0.0], dtype=np.float32)},
            "pedestrians": {
                "positions": np.array([[1.0, 0.0]], dtype=np.float32),
                "velocities": np.array([[0.1, 0.0]], dtype=np.float32),
                "count": np.array([1.0], dtype=np.float32),
            },
        }
    )

    assert mask[0] == 1.0
    assert np.allclose(state[0, 0:4], np.array([1.0, 0.0, 0.1, 0.0], dtype=np.float32))
    assert state[0, 4] == np.float32(50.0)
    assert state[0, 9] == np.float32(0.0)


def test_prediction_planner_adapter_places_obstacle_rows_after_ego_base() -> None:
    """The combined ego+obstacle runtime contract should produce the expected 15D layout."""
    adapter = PredictionPlannerAdapter(
        SocNavPlannerConfig(
            predictive_feature_schema_name=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
            predictive_ego_conditioning=True,
            predictive_max_agents=2,
        )
    )
    adapter._ensure_model = lambda: SimpleNamespace(config=SimpleNamespace(input_dim=15))  # type: ignore[method-assign]

    state, mask, _robot_pos, _robot_heading = adapter._build_model_input(
        {
            "robot": {
                "position": np.array([0.0, 0.0], dtype=np.float32),
                "heading": np.array([0.0], dtype=np.float32),
                "speed": np.array([0.5, -0.1], dtype=np.float32),
            },
            "goal": {"current": np.array([3.0, 4.0], dtype=np.float32)},
            "pedestrians": {
                "positions": np.array([[1.0, 1.0]], dtype=np.float32),
                "velocities": np.array([[0.1, 0.2]], dtype=np.float32),
                "count": np.array([1.0], dtype=np.float32),
            },
            "map": {
                "obstacle_lines": np.array([[[0.0, 0.0], [2.0, 0.0]]], dtype=np.float32),
            },
        }
    )

    assert mask[0] == 1.0
    np.testing.assert_allclose(
        state[0, 0:9],
        [1.0, 1.0, 0.1, 0.2, 0.5, -0.1, 0.6, 0.8, 5.0],
    )
    np.testing.assert_allclose(state[0, 9:15], [1.0, 0.0, 1.0, 1.0, 0.0, 1.0])


def test_prediction_planner_adapter_uses_bound_map_obstacle_rows() -> None:
    """Runtime obstacle-feature inputs should use bound map geometry when available."""
    adapter = PredictionPlannerAdapter(
        SocNavPlannerConfig(
            predictive_feature_schema_name=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
            predictive_ego_conditioning=False,
            predictive_max_agents=2,
        )
    )
    adapter._ensure_model = lambda: SimpleNamespace(config=SimpleNamespace(input_dim=10))  # type: ignore[method-assign]
    map_def = SimpleNamespace(
        obstacles=[Obstacle([(0.0, 0.0), (2.0, 0.0), (2.0, 0.5), (0.0, 0.5)])],
        bounds=[],
    )
    adapter.bind_env(SimpleNamespace(map_def=map_def, simulator=SimpleNamespace(map_def=map_def)))

    state, mask, _robot_pos, _robot_heading = adapter._build_model_input(
        {
            "robot": {
                "position": np.array([0.0, 0.0], dtype=np.float32),
                "heading": np.array([0.0], dtype=np.float32),
                "speed": np.array([0.0, 0.0], dtype=np.float32),
            },
            "goal": {"current": np.array([1.0, 0.0], dtype=np.float32)},
            "pedestrians": {
                "positions": np.array([[1.0, 1.0]], dtype=np.float32),
                "velocities": np.array([[0.1, 0.0]], dtype=np.float32),
                "count": np.array([1.0], dtype=np.float32),
            },
        }
    )

    assert mask[0] == 1.0
    np.testing.assert_allclose(state[0, 4:10], [0.5, 0.0, 1.0, -1.0, 0.0, 1.0])


def test_prediction_planner_adapter_prefers_observation_obstacle_lines() -> None:
    """Runtime observations can carry explicit obstacle lines without using sentinels."""
    adapter = PredictionPlannerAdapter(
        SocNavPlannerConfig(
            predictive_feature_schema_name=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
            predictive_ego_conditioning=False,
            predictive_max_agents=2,
        )
    )
    adapter._ensure_model = lambda: SimpleNamespace(config=SimpleNamespace(input_dim=10))  # type: ignore[method-assign]

    state, mask, _robot_pos, _robot_heading = adapter._build_model_input(
        {
            "robot": {
                "position": np.array([0.0, 0.0], dtype=np.float32),
                "heading": np.array([0.0], dtype=np.float32),
                "speed": np.array([0.0, 0.0], dtype=np.float32),
            },
            "goal": {"current": np.array([1.0, 0.0], dtype=np.float32)},
            "pedestrians": {
                "positions": np.array([[1.0, 1.0]], dtype=np.float32),
                "velocities": np.array([[0.1, 0.0]], dtype=np.float32),
                "count": np.array([1.0], dtype=np.float32),
            },
            "map": {
                "obstacle_lines": np.array([[[0.0, 0.0], [2.0, 0.0]]], dtype=np.float32),
            },
        }
    )

    assert mask[0] == 1.0
    np.testing.assert_allclose(state[0, 4:10], [1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
