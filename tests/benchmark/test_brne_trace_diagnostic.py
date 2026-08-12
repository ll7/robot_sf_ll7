"""Focused unit tests for the BRNE trace-diagnostic trace-table feature (issue #6923)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from robot_sf.baselines.brne import BRNE_PINNED_SHA
from robot_sf.benchmark.brne_trace_diagnostic import (
    TRACE_TABLE_SCHEMA_VERSION,
    _wrap_angle,
    build_trace_table,
)
from scripts.benchmark.run_brne_corridor_diagnostic_issue_6464 import _build_trace_tables


def _default_steps() -> list[dict[str, Any]]:
    """Two-step default fixture with time_s consistent with dt=0.1."""
    return [
        {
            "step": 0,
            "time_s": 0.1,
            "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
            "pedestrians": [],
            "planner": {
                "event": "step",
                "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
            },
            "rl": {"reward": 0.0, "terminated": False, "truncated": False},
        },
        {
            "step": 1,
            "time_s": 0.2,
            "robot": {"position": [1.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
            "pedestrians": [],
            "planner": {
                "event": "step",
                "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
            },
            "rl": {"reward": 1.0, "terminated": True, "truncated": False},
        },
    ]


def _minimal_trace(
    *,
    steps: list[dict[str, Any]] | None = None,
    goal_position: list[float] | None = None,
    initial_robot_position: list[float] | None = None,
    robot_radius: float = 1.0,
    ped_radius: float = 0.4,
    reached_goal_step: int | None = None,
    collision_step: int | None = None,
    dt: float = 0.1,
) -> dict[str, Any]:
    """Build a minimal simulation-step-trace.v1 envelope."""
    resolved_goal = goal_position or [10.0, 4.0]
    resolved_initial_robot = initial_robot_position or [0.0, 4.0]
    resolved_steps = []
    for raw_step in steps or _default_steps():
        normalized_step = dict(raw_step)
        normalized_step.setdefault("goal_position", list(resolved_goal))
        resolved_steps.append(normalized_step)
    trace: dict[str, Any] = {
        "schema_version": "simulation-step-trace.v1",
        "dt": dt,
        "initial_goal_distance_m": math.dist(resolved_initial_robot, resolved_goal),
        "robot_radius_m": robot_radius,
        "ped_radius_m": ped_radius,
        "goal_position": resolved_goal,
        "initial_goal_position": resolved_goal,
        "initial_robot_position": resolved_initial_robot,
        "termination_reason": "timeout",
        "steps": resolved_steps,
    }
    if goal_position is not None:
        trace["goal_position"] = goal_position
    if initial_robot_position is not None:
        trace["initial_robot_position"] = initial_robot_position
    if reached_goal_step is not None:
        trace["reached_goal_step"] = reached_goal_step
    if collision_step is not None:
        trace["collision_step"] = collision_step
    return trace


def _record(
    trace: dict[str, Any] | None = None,
    *,
    provenance: dict[str, Any] | None = None,
    algo: str = "social_force",
) -> dict[str, Any]:
    """Build a minimal episode record with algorithm_metadata."""
    metadata: dict[str, Any] = {
        "status": "ok",
        "simulation_step_trace": trace or _minimal_trace(),
    }
    if provenance is not None:
        metadata["planner_runtime"] = {"planner_metadata": provenance}
    return {
        "episode_id": "test_episode",
        "scenario_id": "classic_head_on_corridor_low",
        "seed": 111,
        "algo": algo,
        "algorithm_metadata": metadata,
    }


def _brne_metadata(
    *,
    effective_num_samples: int = 42,
    aggregation_layout: dict[str, Any] | None = None,
    status: str = "ok",
    runtime_status: str = "ok",
    failure_count: int = 0,
) -> dict[str, Any]:
    """Build minimal BRNE-native algorithm_metadata for provenance tests."""
    agg = aggregation_layout or {
        "method": "weighted_first_command",
        "ensemble_layout": "plan_step_first",
    }
    return {
        "status": status,
        "brne_diagnostic": {
            "status": "native_core_via_adapter",
            "execution_semantics": "native_upstream_core_through_robot_sf_adapter",
        },
        "planner_metadata": {
            "status": "ok",
            "aggregation_layout": agg,
        },
        "planner_runtime": {
            "planner_metadata": {
                "status": status,
                "runtime_status": runtime_status,
                "failure_count": failure_count,
                "source_commit": BRNE_PINNED_SHA,
                "source_pin": BRNE_PINNED_SHA,
                "source_integrity": "clean_pinned_worktree",
                "effective_num_samples": effective_num_samples,
                "step_count": 2,
                "aggregation_layout": agg,
            }
        },
    }


class TestTraceTableSchema:
    """Test trace-table schema version and structure."""

    def test_schema_version(self) -> None:
        table = build_trace_table(_record())
        assert table["schema_version"] == TRACE_TABLE_SCHEMA_VERSION

    def test_has_episode_and_steps(self) -> None:
        table = build_trace_table(_record())
        assert "episode" in table
        assert "steps" in table
        assert isinstance(table["steps"], list)

    def test_episode_summary_fields(self) -> None:
        table = build_trace_table(_record())
        ep = table["episode"]
        assert "episode_id" in ep
        assert "scenario_id" in ep
        assert "seed" in ep
        assert "num_steps" in ep
        assert "dt" in ep
        assert "initial_goal_distance_m" in ep
        assert "final_goal_distance_m" in ep
        assert "displacement_m" in ep
        assert "path_length_m" in ep
        assert "interaction_exposure_steps" in ep
        assert "interaction_exposure_fraction" in ep
        assert "progress_by_phase" in ep
        assert "goal_reached" in ep
        assert "collision_detected" in ep
        assert "termination_reason" in ep


class TestHeadingVelocityGoalAngles:
    """Test heading, velocity, and goal angle computations."""

    def test_velocity_derived_heading_positive(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps, dt=0.1)
        table = build_trace_table(_record(trace))
        heading = table["steps"][0]["velocity_derived_heading_rad"]
        assert heading is not None
        assert math.isclose(heading, 0.0, abs_tol=1e-10)

    def test_velocity_derived_heading_up(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [0.0, 1.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps, dt=0.1)
        table = build_trace_table(_record(trace))
        heading = table["steps"][0]["velocity_derived_heading_rad"]
        assert heading is not None
        assert math.isclose(heading, math.pi / 2, abs_tol=1e-10)

    def test_velocity_heading_zero_speed_returns_none(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 0.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps, dt=0.1)
        table = build_trace_table(_record(trace))
        assert table["steps"][0]["velocity_derived_heading_rad"] is None

    def test_velocity_heading_near_zero_returns_none(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1e-12, 1e-12]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 0.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps, dt=0.1)
        table = build_trace_table(_record(trace))
        assert table["steps"][0]["velocity_derived_heading_rad"] is None

    def test_goal_bearing_computed(self) -> None:
        trace = _minimal_trace(goal_position=[10.0, 4.0], initial_robot_position=[0.0, 4.0])
        table = build_trace_table(_record(trace))
        first_step = table["steps"][0]
        assert first_step["goal_bearing_rad"] is not None
        assert math.isclose(first_step["goal_bearing_rad"], 0.0, abs_tol=1e-10)

    def test_heading_to_goal_diff_wrapped(self) -> None:
        trace = _minimal_trace(goal_position=[0.0, 14.0], initial_robot_position=[0.0, 4.0])
        table = build_trace_table(_record(trace))
        first_step = table["steps"][0]
        assert first_step["heading_goal_angular_difference_rad"] is not None
        expected_diff = _wrap_angle(math.pi / 2 - 0.0)
        assert math.isclose(
            first_step["heading_goal_angular_difference_rad"], expected_diff, abs_tol=1e-10
        )

    def test_velocity_heading_in_step(self) -> None:
        table = build_trace_table(_record())
        first_step = table["steps"][0]
        assert first_step["velocity_derived_heading_rad"] is not None
        assert math.isclose(first_step["velocity_derived_heading_rad"], 0.0, abs_tol=1e-10)


class TestSignedPhaseProgress:
    """Test signed phase progress and route progress."""

    def test_distance_to_goal_decreases(self) -> None:
        trace = _minimal_trace(goal_position=[10.0, 4.0], initial_robot_position=[0.0, 4.0])
        table = build_trace_table(_record(trace))
        d0 = table["steps"][0]["distance_to_goal_m"]
        d1 = table["steps"][1]["distance_to_goal_m"]
        assert d0 is not None and d1 is not None
        assert d1 < d0

    def test_route_progress_positive(self) -> None:
        trace = _minimal_trace(goal_position=[10.0, 4.0], initial_robot_position=[0.0, 4.0])
        table = build_trace_table(_record(trace))
        rp = table["steps"][1]["progress_from_start_m"]
        assert rp is not None
        assert rp > 0.0

    def test_displacement_m(self) -> None:
        table = build_trace_table(_record())
        ep = table["episode"]
        assert "displacement_m" in ep
        assert math.isclose(ep["displacement_m"], 1.0, abs_tol=1e-10)

    def test_path_length_m(self) -> None:
        table = build_trace_table(_record())
        ep = table["episode"]
        assert "path_length_m" in ep
        assert math.isclose(ep["path_length_m"], 1.0, abs_tol=1e-10)

    def test_goal_switch_does_not_create_artificial_progress_jump(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "goal_position": [10.0, 4.0],
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
            {
                "step": 1,
                "time_s": 0.2,
                "robot": {"position": [1.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "goal_position": [20.0, 4.0],
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 1.0, "terminated": True, "truncated": False},
            },
        ]
        trace = _minimal_trace(goal_position=[10.0, 4.0], steps=steps)
        trace["goal_position"] = [20.0, 4.0]
        table = build_trace_table(_record(trace))
        switched = table["steps"][1]
        assert switched["goal_switched"] is True
        assert switched["signed_progress_delta_m"] == 0.0
        assert table["episode"]["progress_by_phase"]["all"]["goal_switch_steps"] == 1

    def test_first_step_goal_switch_does_not_create_artificial_progress_jump(self) -> None:
        """A route switch before step zero must not look like instantaneous progress."""
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "goal_position": [20.0, 4.0],
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
            {
                "step": 1,
                "time_s": 0.2,
                "robot": {"position": [1.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "goal_position": [20.0, 4.0],
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 1.0, "terminated": True, "truncated": False},
            },
        ]
        trace = _minimal_trace(goal_position=[20.0, 4.0], steps=steps)
        trace["initial_goal_position"] = [10.0, 4.0]
        trace["initial_goal_distance_m"] = 10.0
        table = build_trace_table(_record(trace))

        first_step = table["steps"][0]
        assert first_step["goal_switched"] is True
        assert first_step["signed_progress_delta_m"] == 0.0
        assert table["episode"]["progress_by_phase"]["all"]["goal_switch_steps"] == 1


class TestCommandDeltas:
    """Test command delta fields."""

    def test_no_amv_no_amv_deltas(self) -> None:
        table = build_trace_table(_record())
        for step in table["steps"]:
            assert step["amv_delta_linear_m_s"] is None
            assert step["amv_delta_angular_rad_s"] is None

    def test_selected_action_deltas_computed(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 0.5, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
            {
                "step": 1,
                "time_s": 0.2,
                "robot": {"position": [1.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.1},
                },
                "rl": {"reward": 1.0, "terminated": True, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps)
        table = build_trace_table(_record(trace))
        assert table["steps"][0]["selected_action_delta_linear_m_s"] is None
        assert table["steps"][0]["selected_action_delta_angular_rad_s"] is None
        assert table["steps"][1]["selected_action_delta_linear_m_s"] is not None
        assert math.isclose(
            table["steps"][1]["selected_action_delta_linear_m_s"], 0.5, abs_tol=1e-10
        )
        assert math.isclose(
            table["steps"][1]["selected_action_delta_angular_rad_s"], 0.1, abs_tol=1e-10
        )

    def test_amv_deltas_computed(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                    "amv": {
                        "requested_linear_m_s": 1.0,
                        "requested_angular_rad_s": 0.0,
                        "applied_linear_m_s": 0.8,
                        "applied_angular_rad_s": 0.0,
                        "command_clipped": False,
                        "yaw_rate_saturated": False,
                    },
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
            {
                "step": 1,
                "time_s": 0.2,
                "robot": {"position": [1.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.1},
                    "amv": {
                        "requested_linear_m_s": 1.0,
                        "requested_angular_rad_s": 0.1,
                        "applied_linear_m_s": 1.0,
                        "applied_angular_rad_s": 0.1,
                        "command_clipped": False,
                        "yaw_rate_saturated": False,
                    },
                },
                "rl": {"reward": 1.0, "terminated": True, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps)
        table = build_trace_table(_record(trace))
        assert table["steps"][0]["amv_delta_linear_m_s"] is None
        assert table["steps"][1]["amv_delta_linear_m_s"] is not None
        assert math.isclose(table["steps"][1]["amv_delta_linear_m_s"], 0.2, abs_tol=1e-10)
        assert math.isclose(table["steps"][1]["amv_delta_angular_rad_s"], 0.1, abs_tol=1e-10)


class TestClearance:
    """Test minimum surface clearance computation."""

    def test_no_peds_no_clearance(self) -> None:
        table = build_trace_table(_record())
        for step in table["steps"]:
            assert step["min_clearance_m"] is None

    def test_clearance_with_peds(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [
                    {"position": [1.0, 4.0], "velocity": [0.0, 0.0]},
                    {"position": [0.0, 5.0], "velocity": [0.0, 0.0]},
                ],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
            {
                "step": 1,
                "time_s": 0.2,
                "robot": {"position": [0.5, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 1.0, "terminated": True, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps, robot_radius=1.0, ped_radius=0.4)
        table = build_trace_table(_record(trace))
        # First step: ped at [1.0, 4.0] -> center distance 1.0, clearance 1.0 - 1.0 - 0.4 = -0.4
        assert table["steps"][0]["min_clearance_m"] is not None
        assert math.isclose(table["steps"][0]["min_clearance_m"], -0.4, abs_tol=1e-10)
        # Second step: no peds
        assert table["steps"][1]["min_clearance_m"] is None

    def test_clearance_episode_min(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [{"position": [1.0, 4.0], "velocity": [0.0, 0.0]}],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps, robot_radius=1.0, ped_radius=0.4)
        table = build_trace_table(_record(trace))
        ep = table["episode"]
        assert "min_clearance_m" in ep
        assert math.isclose(ep["min_clearance_m"], -0.4, abs_tol=1e-10)


class TestProvenanceAggregationFields:
    """Test provenance and aggregation fields."""

    def test_provenance_present_for_brne(self) -> None:
        metadata = _brne_metadata()
        trace = _minimal_trace(goal_position=[10.0, 4.0], initial_robot_position=[0.0, 4.0])
        record = _record(trace, algo="brne")
        record["algorithm_metadata"].update(metadata)
        table = build_trace_table(record, planner_key="brne")
        assert "provenance" in table
        assert table["provenance"]["status"] == "ok"
        assert table["provenance"]["runtime_status"] == "ok"
        assert table["provenance"]["effective_num_samples"] == 42

    def test_provenance_absent_for_comparator(self) -> None:
        table = build_trace_table(_record(), planner_key="social_force")
        assert "provenance" not in table

    def test_aggregation_layout_in_steps_for_brne(self) -> None:
        metadata = _brne_metadata()
        trace = _minimal_trace(goal_position=[10.0, 4.0], initial_robot_position=[0.0, 4.0])
        record = _record(trace, algo="brne")
        record["algorithm_metadata"].update(metadata)
        table = build_trace_table(record, planner_key="brne")
        assert "brne_aggregation_layout" in table["steps"][0]
        agg = table["steps"][0]["brne_aggregation_layout"]
        assert agg["method"] == "weighted_first_command"
        assert agg["ensemble_layout"] == "plan_step_first"

    def test_no_aggregation_layout_for_comparator(self) -> None:
        table = build_trace_table(_record(), planner_key="social_force")
        assert "brne_aggregation_layout" not in table["steps"][0]

    def test_provenance_expected_sample_count_mismatch_raises(self) -> None:
        metadata = _brne_metadata(effective_num_samples=42)
        trace = _minimal_trace(goal_position=[10.0, 4.0], initial_robot_position=[0.0, 4.0])
        record = _record(trace, algo="brne")
        record["algorithm_metadata"].update(metadata)
        with pytest.raises(ValueError, match="does not match the frozen contract"):
            build_trace_table(record, planner_key="brne", expected_effective_num_samples=99)

    def test_provenance_wrong_source_pin_raises(self) -> None:
        metadata = _brne_metadata()
        metadata["planner_runtime"]["planner_metadata"]["source_commit"] = "abc123"
        trace = _minimal_trace(goal_position=[10.0, 4.0], initial_robot_position=[0.0, 4.0])
        record = _record(trace, algo="brne")
        record["algorithm_metadata"].update(metadata)
        with pytest.raises(ValueError, match="source_commit does not match the frozen pin"):
            build_trace_table(record, planner_key="brne")


class TestMalformedMissingFallbackRuntimeFailed:
    """Test fail-closed behavior on malformed, missing, fallback, and runtime-failed data."""

    def test_missing_algorithm_metadata_raises(self) -> None:
        record = {"episode_id": "x"}
        with pytest.raises(ValueError, match="algorithm_metadata must be a mapping"):
            build_trace_table(record)

    def test_missing_trace_raises(self) -> None:
        record = {"algorithm_metadata": {"status": "ok"}}
        with pytest.raises(ValueError, match="simulation_step_trace.v1 is missing"):
            build_trace_table(record)

    def test_wrong_schema_version_raises(self) -> None:
        trace = _minimal_trace()
        trace["schema_version"] = "simulation-step-trace.v2"
        record = _record(trace)
        with pytest.raises(ValueError, match="simulation-step-trace.v1 is required"):
            build_trace_table(record)

    def test_empty_steps_raises(self) -> None:
        trace = _minimal_trace()
        trace["steps"] = []
        record = _record(trace)
        with pytest.raises(ValueError, match="non-empty list"):
            build_trace_table(record)

    def test_runtime_status_failed_raises(self) -> None:
        """Runtime-failed BRNE rows cannot enter the native diagnostic table."""
        metadata = _brne_metadata(runtime_status="failed")
        record = _record(_minimal_trace(), algo="brne")
        record["algorithm_metadata"].update(metadata)
        with pytest.raises(ValueError, match="BRNE runtime is not successful"):
            build_trace_table(record, planner_key="brne")

    def test_nonzero_failure_count_raises(self) -> None:
        """Any recorded BRNE failure count invalidates native evidence admission."""
        metadata = _brne_metadata(failure_count=1)
        record = _record(_minimal_trace(), algo="brne")
        record["algorithm_metadata"].update(metadata)
        with pytest.raises(ValueError, match="failure_count must be zero"):
            build_trace_table(record, planner_key="brne")

    def test_missing_robot_position_raises(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps)
        record = _record(trace)
        with pytest.raises(ValueError, match="robot.position must contain two coordinates"):
            build_trace_table(record)

    def test_nonfinite_goal_position_raises(self) -> None:
        trace = _minimal_trace(goal_position=[float("nan"), 4.0])
        record = _record(trace)
        with pytest.raises(ValueError, match="goal_position|initial_goal_distance"):
            build_trace_table(record)

    def test_nonfinite_step_position_raises(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [float("inf"), 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps)
        record = _record(trace)
        with pytest.raises(ValueError, match="robot.position"):
            build_trace_table(record)

    def test_zero_motion_detected(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 0.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
            {
                "step": 1,
                "time_s": 0.2,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 0.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps)
        table = build_trace_table(_record(trace))
        ep = table["episode"]
        # Zero-motion: displacement is 0, but episode doesn't have zero_motion_steps
        assert math.isclose(ep["displacement_m"], 0.0, abs_tol=1e-10)

    def test_fallback_metadata_rejected(self) -> None:
        metadata = {"status": "ok", "fallback_triggered": True}
        trace = _minimal_trace(goal_position=[10.0, 4.0], initial_robot_position=[0.0, 4.0])
        record = _record(trace, algo="social_force")
        record["algorithm_metadata"].update(metadata)
        with pytest.raises(ValueError, match="fallback"):
            build_trace_table(record, planner_key="social_force")

    def test_corridor_violation_is_retained_for_trace_diagnosis(self) -> None:
        trace = _minimal_trace(
            steps=[
                {
                    "step": 0,
                    "time_s": 0.1,
                    "robot": {
                        "position": [0.0, 2.0],
                        "heading": 0.0,
                        "velocity": [1.0, 0.0],
                    },
                    "pedestrians": [],
                    "planner": {
                        "event": "step",
                        "selected_action": {
                            "linear_velocity": 1.0,
                            "angular_velocity": 0.0,
                        },
                    },
                    "rl": {"reward": 0.0, "terminated": False, "truncated": False},
                },
                {
                    "step": 1,
                    "time_s": 0.2,
                    "robot": {
                        "position": [1.0, 2.0],
                        "heading": 0.0,
                        "velocity": [1.0, 0.0],
                    },
                    "pedestrians": [],
                    "planner": {
                        "event": "step",
                        "selected_action": {
                            "linear_velocity": 1.0,
                            "angular_velocity": 0.0,
                        },
                    },
                    "rl": {"reward": 0.0, "terminated": True, "truncated": False},
                },
            ],
            goal_position=[10.0, 2.0],
            initial_robot_position=[0.0, 2.0],
        )
        record = _record(trace, algo="social_force")
        record["status"] = "success"
        config = {
            "scenario_ids": ["classic_head_on_corridor_low"],
            "seeds": [111],
            "expected_effective_num_samples": 42,
            "max_pedestrians": 7,
            "claim_boundary": "diagnostic-only",
            "corridor": {
                "y_min": 2.5,
                "y_max": 37.5,
                "robot_radius_m": 1.0,
                "min_displacement_m": 0.5,
                "max_zero_motion_fraction": 0.95,
            },
        }
        tables = _build_trace_tables([record], planner_key="social_force", config=config)
        assert len(tables) == 1
        assert tables[0]["episode"]["num_steps"] == 2

    def test_missing_termination_reason_raises(self) -> None:
        trace = _minimal_trace()
        del trace["termination_reason"]
        record = _record(trace)
        with pytest.raises(ValueError, match="termination_reason must be a non-empty string"):
            build_trace_table(record)

    def test_empty_termination_reason_raises(self) -> None:
        trace = _minimal_trace()
        trace["termination_reason"] = "   "
        record = _record(trace)
        with pytest.raises(ValueError, match="termination_reason must be a non-empty string"):
            build_trace_table(record)

    def test_missing_initial_goal_distance_raises(self) -> None:
        trace = _minimal_trace()
        del trace["initial_goal_distance_m"]
        record = _record(trace)
        with pytest.raises(ValueError, match="initial_goal_distance_m"):
            build_trace_table(record)

    def test_missing_robot_radius_raises(self) -> None:
        trace = _minimal_trace()
        del trace["robot_radius_m"]
        record = _record(trace)
        with pytest.raises(ValueError, match="robot_radius_m"):
            build_trace_table(record)

    def test_missing_ped_radius_raises(self) -> None:
        trace = _minimal_trace()
        del trace["ped_radius_m"]
        record = _record(trace)
        with pytest.raises(ValueError, match="ped_radius_m"):
            build_trace_table(record)

    def test_step_index_mismatch_raises(self) -> None:
        steps = [
            {
                "step": 5,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps)
        record = _record(trace)
        with pytest.raises(ValueError, match="must equal 0"):
            build_trace_table(record)

    def test_time_s_inconsistent_with_dt_raises(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 99.9,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps, dt=0.1)
        record = _record(trace)
        with pytest.raises(ValueError, match="time_s is inconsistent"):
            build_trace_table(record)

    def test_negative_radius_raises(self) -> None:
        trace = _minimal_trace(robot_radius=-1.0)
        record = _record(trace)
        with pytest.raises(ValueError, match="radii must be non-negative"):
            build_trace_table(record)

    def test_missing_selected_action_raises(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {"event": "step"},
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps)
        record = _record(trace)
        with pytest.raises(ValueError, match="selected_action must be a mapping"):
            build_trace_table(record)

    def test_missing_rl_mapping_raises(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
            },
        ]
        trace = _minimal_trace(steps=steps)
        record = _record(trace)
        with pytest.raises(ValueError, match="rl must be a mapping"):
            build_trace_table(record)

    def test_missing_planner_mapping_raises(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps)
        record = _record(trace)
        with pytest.raises(ValueError, match="planner must be a mapping"):
            build_trace_table(record)


class TestEnvelopeFields:
    """Test envelope field propagation."""

    def test_goal_position_in_envelope(self) -> None:
        trace = _minimal_trace(goal_position=[10.0, 20.0])
        record = _record(trace)
        table = build_trace_table(record)
        assert table["episode"]["goal_position"] == [10.0, 20.0]

    def test_initial_robot_position_in_envelope(self) -> None:
        trace = _minimal_trace(initial_robot_position=[1.0, 2.0])
        record = _record(trace)
        table = build_trace_table(record)
        assert table["episode"]["initial_robot_position"] == [1.0, 2.0]

    def test_reached_goal_step_in_envelope(self) -> None:
        trace = _minimal_trace(reached_goal_step=5)
        record = _record(trace)
        table = build_trace_table(record)
        assert table["episode"]["reached_goal_step"] == 5
        assert table["episode"]["goal_reached"] is True

    def test_termination_reason_in_envelope(self) -> None:
        trace = _minimal_trace()
        trace["termination_reason"] = "goal_reached"
        record = _record(trace)
        table = build_trace_table(record)
        assert table["episode"]["termination_reason"] == "goal_reached"

    def test_collision_step_in_envelope(self) -> None:
        trace = _minimal_trace(collision_step=3)
        record = _record(trace)
        table = build_trace_table(record)
        assert table["episode"]["collision_step"] == 3
        assert table["episode"]["collision_detected"] is True

    def test_dt_propagated(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.05,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
            {
                "step": 1,
                "time_s": 0.10,
                "robot": {"position": [1.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 1.0, "terminated": True, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps, dt=0.05)
        record = _record(trace)
        table = build_trace_table(record)
        assert table["episode"]["dt"] == pytest.approx(0.05)

    def test_interaction_exposure_computed(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [{"position": [0.5, 4.0], "velocity": [0.0, 0.0]}],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
            {
                "step": 1,
                "time_s": 0.2,
                "robot": {"position": [1.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [{"position": [10.0, 4.0], "velocity": [0.0, 0.0]}],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 1.0, "terminated": True, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps)
        table = build_trace_table(_record(trace))
        ep = table["episode"]
        assert ep["interaction_exposure_steps"] == 1
        assert ep["interaction_exposure_fraction"] == pytest.approx(0.5)
        assert ep["interaction_exposure_radius_m"] == 2.0

    def test_progress_by_phase_structure(self) -> None:
        table = build_trace_table(_record())
        phases = table["episode"]["progress_by_phase"]
        assert "all" in phases
        assert "interaction" in phases
        assert "non_interaction" in phases
        for phase in phases.values():
            assert "steps" in phase
            assert "signed_progress_m" in phase
            assert "mean_signed_progress_delta_m" in phase

    def test_world_frame_pedestrians_propagated(self) -> None:
        steps = [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 4.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [{"position": [3.0, 4.0], "velocity": [0.0, -1.0]}],
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
                },
                "rl": {"reward": 0.0, "terminated": False, "truncated": False},
            },
        ]
        trace = _minimal_trace(steps=steps)
        table = build_trace_table(_record(trace))
        peds = table["steps"][0]["pedestrians_world"]
        assert len(peds) == 1
        assert peds[0]["position"] == [3.0, 4.0]
        assert peds[0]["velocity"] == [0.0, -1.0]


class TestWrapAngle:
    """Test angle wrapping utility."""

    def test_wrap_zero(self) -> None:
        assert math.isclose(_wrap_angle(0.0), 0.0, abs_tol=1e-10)

    def test_wrap_pi(self) -> None:
        assert math.isclose(_wrap_angle(math.pi), -math.pi, abs_tol=1e-10) or math.isclose(
            _wrap_angle(math.pi), math.pi, abs_tol=1e-10
        )

    def test_wrap_2pi(self) -> None:
        assert math.isclose(_wrap_angle(2 * math.pi), 0.0, abs_tol=1e-10)

    def test_wrap_negative_pi(self) -> None:
        result = _wrap_angle(-math.pi)
        assert math.isclose(result, -math.pi, abs_tol=1e-10) or math.isclose(
            result, math.pi, abs_tol=1e-10
        )
