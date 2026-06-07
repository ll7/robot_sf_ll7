"""Tests for the stream-gap planner."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.stream_gap import (
    StreamGapPlannerAdapter,
    StreamGapPlannerConfig,
    build_stream_gap_config,
)
from robot_sf.representation import scenario_belief_from_simulator_oracle


def _obs(
    *, robot=(0.0, 0.0), heading=0.0, goal=(2.0, 0.0), ped_positions=None, ped_velocities=None
):
    """Build the compact observation payload used by stream-gap tests."""
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
        },
    }


def _single_blocker_uncertainty_agent() -> dict[str, object]:
    """Return one ScenarioBelief-derived uncertainty row for the blocking fixture."""
    simulator = SimpleNamespace(
        ped_pos=np.array([[0.8, 0.0]], dtype=np.float32),
        ped_vel=np.array([[0.0, 0.0]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=0.4),
            )
        ],
        goal_pos=[np.array([4.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=8.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    belief = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=RobotSimulationConfig(),
        max_pedestrians=4,
    )
    agent = dict(belief.to_uncertainty_report()["agents"][0])
    agent["class_probabilities"] = {"pedestrian": 0.2}
    agent["position_confidence"] = 0.2
    agent["existence_probability"] = 0.2
    agent["position_covariance_xy"] = [[4.0, 0.0], [0.0, 4.0]]
    return agent


def test_stream_gap_commits_in_open_space() -> None:
    """Planner should commit when no pedestrian blocks the corridor."""
    planner = StreamGapPlannerAdapter(StreamGapPlannerConfig(commit_speed=0.9))
    v, w = planner.plan(_obs(goal=(4.0, 0.0)))
    assert v >= 0.89
    assert abs(w) <= planner.config.max_angular_speed


def test_stream_gap_waits_for_blocking_crossing_pedestrian() -> None:
    """Planner should wait when a pedestrian currently blocks the goal corridor."""
    planner = StreamGapPlannerAdapter(StreamGapPlannerConfig())
    v, _w = planner.plan(
        _obs(
            goal=(4.0, 0.0),
            ped_positions=[(0.8, 0.0)],
            ped_velocities=[(0.0, 0.0)],
        )
    )
    assert v == 0.0


def test_stream_gap_uncertainty_gating_can_drop_low_confidence_blocker() -> None:
    """Opt-in gating can turn ScenarioBelief uncertainty into one planner-input decision."""
    observation = _obs(
        goal=(4.0, 0.0),
        ped_positions=[(0.8, 0.0)],
        ped_velocities=[(0.0, 0.0)],
    )
    observation["pedestrians"]["uncertainty"] = [_single_blocker_uncertainty_agent()]

    deterministic = StreamGapPlannerAdapter(StreamGapPlannerConfig())
    deterministic_v, _ = deterministic.plan(observation)

    uncertainty_aware = StreamGapPlannerAdapter(
        StreamGapPlannerConfig(
            uncertainty_gating_enabled=True,
            uncertainty_min_existence_probability=0.5,
            uncertainty_min_position_confidence=0.5,
            uncertainty_min_class_probability=0.5,
            uncertainty_max_position_variance=1.0,
        )
    )
    gated_v, _ = uncertainty_aware.plan(observation)

    assert deterministic_v == 0.0
    assert gated_v > 0.0
    assert uncertainty_aware.last_uncertainty_gate["status"] == "applied"
    assert uncertainty_aware.last_uncertainty_gate["dropped_count"] == 1
    assert set(uncertainty_aware.last_uncertainty_gate["dropped_reasons"][0]["reasons"]) == {
        "class_probability_below_threshold",
        "existence_probability_below_threshold",
        "position_confidence_below_threshold",
        "position_variance_above_threshold",
    }


def test_stream_gap_uncertainty_gating_fails_closed_for_malformed_metadata() -> None:
    """Malformed uncertainty sidecars must keep deterministic blocker handling."""
    observation = _obs(
        goal=(4.0, 0.0),
        ped_positions=[(0.8, 0.0)],
        ped_velocities=[(0.0, 0.0)],
    )
    observation["pedestrians"]["uncertainty"] = [{"position_confidence": "not-a-number"}]

    planner = StreamGapPlannerAdapter(StreamGapPlannerConfig(uncertainty_gating_enabled=True))
    v, _ = planner.plan(observation)

    assert v == 0.0
    assert planner.last_uncertainty_gate["status"] == "fail_closed"
    assert planner.last_uncertainty_gate["reason"] == "malformed_uncertainty_metadata"


def test_stream_gap_uncertainty_gating_fails_closed_for_malformed_covariance() -> None:
    """Malformed covariance rows must not crash the planner."""
    observation = _obs(
        goal=(4.0, 0.0),
        ped_positions=[(0.8, 0.0)],
        ped_velocities=[(0.0, 0.0)],
    )
    agent = _single_blocker_uncertainty_agent()
    agent["position_covariance_xy"] = None
    observation["pedestrians"]["uncertainty"] = [agent]

    planner = StreamGapPlannerAdapter(StreamGapPlannerConfig(uncertainty_gating_enabled=True))
    v, _ = planner.plan(observation)

    assert v == 0.0
    assert planner.last_uncertainty_gate["status"] == "fail_closed"
    assert planner.last_uncertainty_gate["reason"] == "malformed_uncertainty_metadata"


def test_build_stream_gap_config_parses_uncertainty_gating_fields() -> None:
    """YAML config parsing should expose the diagnostic uncertainty-gating knobs."""
    cfg = build_stream_gap_config(
        {
            "uncertainty_gating_enabled": True,
            "uncertainty_min_existence_probability": 0.4,
            "uncertainty_min_position_confidence": 0.45,
            "uncertainty_min_class_probability": 0.5,
            "uncertainty_max_position_variance": 0.75,
        }
    )

    assert cfg.uncertainty_gating_enabled is True
    assert cfg.uncertainty_min_existence_probability == pytest.approx(0.4)
    assert cfg.uncertainty_min_position_confidence == pytest.approx(0.45)
    assert cfg.uncertainty_min_class_probability == pytest.approx(0.5)
    assert cfg.uncertainty_max_position_variance == pytest.approx(0.75)


def test_stream_gap_approaches_when_gap_is_soon() -> None:
    """Planner should creep/approach when a free window starts shortly ahead."""
    planner = StreamGapPlannerAdapter(
        StreamGapPlannerConfig(
            safe_gap_time=0.8,
            approach_gap_time=0.8,
            corridor_half_width=0.5,
            sample_horizon=2.0,
            sample_dt=0.2,
            approach_speed=0.33,
        )
    )
    v, _w = planner.plan(
        _obs(
            goal=(4.0, 0.0),
            ped_positions=[(0.7, 0.45)],
            ped_velocities=[(0.0, 1.0)],
        )
    )
    assert 0.3 <= v <= 0.34


def test_stream_gap_commit_hold_persists_after_gap_opening() -> None:
    """Commit mode should persist briefly once the planner decides to go."""
    planner = StreamGapPlannerAdapter(StreamGapPlannerConfig(commit_hold_steps=3, commit_speed=0.8))
    open_obs = _obs(goal=(4.0, 0.0))
    v1, _ = planner.plan(open_obs)
    v2, _ = planner.plan(open_obs)
    assert v1 >= 0.79
    assert v2 >= 0.79
