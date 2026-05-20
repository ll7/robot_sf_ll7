"""Tests for predictive hardcase data collection schemas."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.obstacle_features import PREDICTIVE_OBSTACLE_FEATURE_SCHEMA
from scripts.training import collect_predictive_hardcase_data as collect


def _frame(
    *,
    ped_positions: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
) -> collect.Frame:
    """Build one hardcase frame fixture."""
    positions_source = [(1.0, 1.0)] if ped_positions is None else ped_positions
    velocities_source = (
        [(0.1, 0.0)] * len(positions_source) if ped_velocities is None else ped_velocities
    )
    positions = np.asarray(positions_source, dtype=np.float32).reshape(-1, 2)
    velocities = np.asarray(velocities_source, dtype=np.float32).reshape(-1, 2)
    return collect.Frame(
        robot_pos=np.asarray((0.0, 0.0), dtype=np.float32),
        robot_heading=0.0,
        robot_speed=np.asarray((0.0, 0.0), dtype=np.float32),
        goal_current=np.asarray((3.0, 4.0), dtype=np.float32),
        ped_positions_world=positions,
        ped_velocities_world=velocities,
        ped_count=int(positions.shape[0]),
    )


def test_hardcase_frames_to_samples_appends_real_map_obstacle_rows() -> None:
    """Hardcase collection should emit real obstacle rows for obstacle-feature models."""
    frames = [
        _frame(ped_positions=[(1.0, 1.0)]),
        _frame(ped_positions=[(1.2, 1.0)]),
    ]

    state, _target, mask, _target_mask = collect._frames_to_samples(
        frames,
        max_agents=2,
        horizon_steps=1,
        ego_conditioning=False,
        model_family=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
        obstacle_lines=[((0.0, 0.0), (2.0, 0.0))],
    )

    assert state.shape == (1, 2, 10)
    np.testing.assert_allclose(state[0, 0, 4:10], [1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    assert state[0, 1, 9] == 0.0
    assert mask[0, 0] == 1.0


def test_hardcase_effective_schema_tracks_ego_obstacle_width() -> None:
    """Obstacle-family hardcase metadata should account for ego-conditioned base rows."""
    schema = collect._effective_predictive_feature_schema(
        model_family=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
        ego_conditioning=True,
    )

    assert schema["name"] == PREDICTIVE_OBSTACLE_FEATURE_SCHEMA
    assert schema["base_feature_dim"] == 9
    assert schema["input_dim"] == 15
