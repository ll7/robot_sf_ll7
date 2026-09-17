"""Regression coverage for the opt-in social-force goal-approach correction."""

from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.map_runner.map_runner import _run_map_episode
from robot_sf.planner.socnav import (
    SOCIAL_FORCE_GOAL_APPROACH_LEGACY_V1,
    SOCIAL_FORCE_GOAL_APPROACH_TERMINAL_V1,
    SocialForcePlannerAdapter,
    SocNavPlannerConfig,
)
from robot_sf.training.scenario_loader import load_scenarios

SCENARIO_PATH = Path("configs/scenarios/archetypes/classic_head_on_corridor.yaml")
FIXTURE_MANIFEST_PATH = Path("configs/benchmarks/issue_9428_social_force_goal_approach.yaml")
SCENARIO_ID = "classic_head_on_corridor_medium"
SEED = 24


def _make_observation(*, goal: tuple[float, float], next_goal: tuple[float, float]) -> dict:
    """Build a compact final-waypoint observation for adapter-level checks."""
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([0.0], dtype=np.float32),
            "speed": np.array([0.0, 0.0], dtype=np.float32),
            "radius": np.array([0.5], dtype=np.float32),
        },
        "goal": {
            "current": np.asarray(goal, dtype=np.float32),
            "next": np.asarray(next_goal, dtype=np.float32),
        },
        "pedestrians": {
            "positions": np.zeros((0, 2), dtype=np.float32),
            "velocities": np.zeros((0, 2), dtype=np.float32),
            "count": np.array([0.0], dtype=np.float32),
        },
        "sim": {"timestep": np.array([0.1], dtype=np.float32)},
    }


def _with_occupancy(observation: dict, *, wall: bool) -> dict:
    """Attach a compact occupancy grid, optionally blocking the goal segment."""
    observation["occupancy_grid"] = np.zeros((4, 4, 4), dtype=np.float32)
    if wall:
        observation["occupancy_grid"][0, 1, 2] = 1.0
    observation["occupancy_grid_meta_origin"] = np.array([-2.0, -2.0], dtype=np.float32)
    observation["occupancy_grid_meta_resolution"] = np.array([1.0], dtype=np.float32)
    observation["occupancy_grid_meta_size"] = np.array([4.0, 4.0], dtype=np.float32)
    observation["occupancy_grid_meta_use_ego_frame"] = np.array([1.0], dtype=np.float32)
    observation["occupancy_grid_meta_channel_indices"] = np.array([0, 1, 2, 3], dtype=np.float32)
    return observation


def test_goal_approach_version_is_opt_in_and_runtime_visible() -> None:
    """Missing config keeps legacy behavior while explicit version is visible."""
    legacy = SocialForcePlannerAdapter(SocNavPlannerConfig())
    legacy.plan(_make_observation(goal=(3.0, 0.0), next_goal=(0.0, 0.0)))
    legacy_meta = legacy.diagnostics()["goal_approach"]
    assert legacy_meta["version"] == SOCIAL_FORCE_GOAL_APPROACH_LEGACY_V1
    assert legacy_meta["enabled"] is False
    assert legacy_meta["applied"] is False

    fixed = SocialForcePlannerAdapter(
        SocNavPlannerConfig(
            social_force_goal_approach_version=SOCIAL_FORCE_GOAL_APPROACH_TERMINAL_V1
        )
    )
    fixed.plan(
        _with_occupancy(
            _make_observation(goal=(3.0, 0.0), next_goal=(0.0, 0.0)),
            wall=False,
        )
    )
    fixed_meta = fixed.diagnostics()["goal_approach"]
    assert fixed_meta["version"] == SOCIAL_FORCE_GOAL_APPROACH_TERMINAL_V1
    assert fixed_meta["enabled"] is True
    assert fixed_meta["applied"] is True


@pytest.mark.parametrize(
    "version",
    [SOCIAL_FORCE_GOAL_APPROACH_LEGACY_V1, SOCIAL_FORCE_GOAL_APPROACH_TERMINAL_V1],
)
def test_goal_approach_preserves_wall_avoidance(version: str) -> None:
    """A blocked final-goal segment retains obstacle-force handling in both arms."""
    adapter = SocialForcePlannerAdapter(
        SocNavPlannerConfig(social_force_goal_approach_version=version)
    )
    adapter.plan(
        _with_occupancy(
            _make_observation(goal=(3.0, 0.0), next_goal=(0.0, 0.0)),
            wall=True,
        )
    )
    diagnostics = adapter.diagnostics()
    assert diagnostics["goal_approach"]["applied"] is False
    if version == SOCIAL_FORCE_GOAL_APPROACH_TERMINAL_V1:
        assert diagnostics["goal_approach"]["parameters"].get("segment_clear") is False
    assert diagnostics["obstacle_force_law"]["applied"] is True


@pytest.mark.slow
def test_seed_24_legacy_limit_cycle_and_opt_in_completion() -> None:
    """The frozen native seed reproduces legacy timeout and fixed completion."""
    scenario = next(
        dict(row) for row in load_scenarios(SCENARIO_PATH) if row.get("name") == SCENARIO_ID
    )
    legacy = _run_map_episode(
        scenario,
        SEED,
        horizon=500,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="social_force",
        scenario_path=SCENARIO_PATH,
        algo_config={},
        record_simulation_step_trace=False,
    )
    fixed = _run_map_episode(
        scenario,
        SEED,
        horizon=500,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="social_force",
        scenario_path=SCENARIO_PATH,
        algo_config={"social_force_goal_approach_version": SOCIAL_FORCE_GOAL_APPROACH_TERMINAL_V1},
        record_simulation_step_trace=False,
    )

    assert FIXTURE_MANIFEST_PATH.is_file()
    assert legacy["outcome"] == {
        "route_complete": False,
        "collision_event": False,
        "timeout_event": True,
    }
    assert fixed["outcome"] == {
        "route_complete": True,
        "collision_event": False,
        "timeout_event": False,
    }
    runtime = fixed["algorithm_metadata"]["planner_runtime"]["goal_approach"]
    assert runtime["version"] == SOCIAL_FORCE_GOAL_APPROACH_TERMINAL_V1
    assert runtime["applied"] is True
