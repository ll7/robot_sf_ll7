"""Tests for graded benchmark observation-level contracts."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.algorithm_metadata import (
    enrich_algorithm_metadata,
    resolve_observation_mode,
)
from robot_sf.benchmark.observation_levels import (
    OBSERVATION_LEVEL_KEYS,
    observation_level_spec,
    resolve_observation_level_contract,
)
from robot_sf.benchmark.planner_command_contract import (
    PlannerContractValidationError,
    validate_planner_contract,
)


def test_observation_level_vocabulary_declares_expected_levels() -> None:
    """The benchmark-facing observation-level vocabulary should be stable and inspectable."""
    assert OBSERVATION_LEVEL_KEYS == (
        "oracle_full_state",
        "tracked_agents_no_noise",
        "tracked_agents_with_noise",
        "lidar_2d",
        "occluded_partial_state",
    )
    lidar = observation_level_spec("lidar_2d")
    assert lidar.key == "lidar_2d"
    assert lidar.perception_assumption == "range_sensor_projection"
    assert "sensor_fusion_state" in lidar.compatible_observation_modes


def test_observation_levels_materialize_same_planner_with_distinct_modes() -> None:
    """A deterministic fixture should materialize one planner under two observation levels."""
    oracle = resolve_observation_level_contract("goal", observation_level="oracle_full_state")
    tracked = resolve_observation_level_contract(
        "goal",
        observation_level="tracked_agents_no_noise",
    )

    assert oracle["observation_level"]["key"] == "oracle_full_state"
    assert oracle["active_observation_mode"] == "goal_state"
    assert tracked["observation_level"]["key"] == "tracked_agents_no_noise"
    assert tracked["active_observation_mode"] == "socnav_state"
    assert oracle["active_observation_mode"] != tracked["active_observation_mode"]


def test_algorithm_metadata_embeds_active_observation_level() -> None:
    """Algorithm metadata should distinguish the level from the raw observation mode."""
    meta = enrich_algorithm_metadata(
        algo="goal",
        observation_level="tracked_agents_no_noise",
    )

    assert meta["observation_level"]["key"] == "tracked_agents_no_noise"
    assert meta["observation_spec"]["active_mode"] == "socnav_state"
    assert meta["planner_contract"]["observation_contract"]["observation_level"] == (
        "tracked_agents_no_noise"
    )


def test_observation_level_rejects_unsupported_planner_combination() -> None:
    """Unsupported planner/observation-level pairs should fail closed with an actionable error."""
    with pytest.raises(ValueError, match="Observation level 'lidar_2d' is not supported"):
        resolve_observation_mode("goal", observation_level="lidar_2d")

    with pytest.raises(PlannerContractValidationError, match="Observation level 'lidar_2d'"):
        validate_planner_contract(
            algo="goal",
            robot_kinematics="differential_drive",
            algo_config={},
            observation_level="lidar_2d",
        )
