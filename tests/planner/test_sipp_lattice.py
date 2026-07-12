"""Tests for kinodynamic state-time lattice primitives, collision model, and planner."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.planner.sipp_lattice import (
    MotionPrimitive,
    PrimitiveKind,
    SippKinodynamicCollisionModel,
    SippLatticeConfig,
    SippLatticePlannerAdapter,
    SippLatticePrimitiveSet,
    build_sipp_lattice_config,
)


def _obs(
    *,
    robot=(0.0, 0.0),
    heading=0.0,
    speed=0.0,
    goal=(2.0, 0.0),
    obstacle_cells=None,
    pedestrian_positions=None,
    count=0,
):
    """Build compact observation payload for SippLattice tests."""
    obstacle_cells = obstacle_cells or []
    pedestrian_positions = pedestrian_positions or []
    grid = np.zeros((4, 4, 4), dtype=np.float32)
    for row, col in obstacle_cells:
        grid[0, row, col] = 1.0
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(pedestrian_positions, dtype=float),
            "velocities": np.zeros((len(pedestrian_positions), 2), dtype=float)
            if pedestrian_positions
            else np.zeros((0, 2), dtype=float),
            "count": np.asarray([float(count)], dtype=float),
            "radius": 0.30,
        },
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": np.asarray([-2.0, -2.0], dtype=float),
        "occupancy_grid_meta_resolution": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_size": np.asarray([4.0, 4.0], dtype=float),
        "occupancy_grid_meta_use_ego_frame": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_channel_indices": np.asarray([0, 1, 2, 3], dtype=float),
    }


# -- MotionPrimitive tests --


class TestMotionPrimitive:
    """Unit tests for the MotionPrimitive dataclass."""

    def test_forward_primitive_properties(self) -> None:
        p = MotionPrimitive(
            linear_velocity=0.5,
            angular_velocity=0.3,
            duration=0.2,
            kind=PrimitiveKind.FORWARD,
        )
        assert p.distance_traveled == pytest.approx(0.1)
        assert p.delta_yaw == pytest.approx(0.06)
        assert p.as_command() == (0.5, 0.3)

    def test_zero_velocity_primitive(self) -> None:
        p = MotionPrimitive(
            linear_velocity=0.0,
            angular_velocity=0.0,
            duration=0.2,
            kind=PrimitiveKind.WAIT,
        )
        assert p.distance_traveled == pytest.approx(0.0)
        assert p.delta_yaw == pytest.approx(0.0)

    def test_negative_linear_velocity(self) -> None:
        p = MotionPrimitive(
            linear_velocity=-0.3,
            angular_velocity=0.0,
            duration=0.2,
            kind=PrimitiveKind.REVERSE,
        )
        assert p.distance_traveled == pytest.approx(0.06)

    def test_invalid_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            MotionPrimitive(
                linear_velocity=0.5,
                angular_velocity=0.0,
                duration=-0.1,
                kind=PrimitiveKind.FORWARD,
            )

    def test_invalid_duration_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            MotionPrimitive(
                linear_velocity=0.5,
                angular_velocity=0.0,
                duration=0.0,
                kind=PrimitiveKind.FORWARD,
            )

    def test_inf_velocity_raises(self) -> None:
        with pytest.raises(ValueError, match="linear_velocity"):
            MotionPrimitive(
                linear_velocity=float("inf"),
                angular_velocity=0.0,
                duration=0.2,
                kind=PrimitiveKind.FORWARD,
            )


# -- SippLatticePrimitiveSet tests --


class TestSippLatticePrimitiveSet:
    """Unit tests for the SippLatticePrimitiveSet builder."""

    def test_default_build_contains_forwards(self) -> None:
        ps = SippLatticePrimitiveSet()
        primes = ps.build()
        kinds = {p.kind for p in primes}
        assert PrimitiveKind.FORWARD in kinds

    def test_default_build_contains_wait(self) -> None:
        ps = SippLatticePrimitiveSet()
        primes = ps.build()
        kinds = {p.kind for p in primes}
        assert PrimitiveKind.WAIT in kinds

    def test_default_build_contains_decelerate(self) -> None:
        ps = SippLatticePrimitiveSet()
        primes = ps.build()
        kinds = {p.kind for p in primes}
        assert PrimitiveKind.DECELERATE in kinds

    def test_default_build_contains_recenter(self) -> None:
        ps = SippLatticePrimitiveSet()
        primes = ps.build()
        kinds = {p.kind for p in primes}
        assert PrimitiveKind.RECENTER in kinds

    def test_default_build_contains_reverse(self) -> None:
        ps = SippLatticePrimitiveSet()
        primes = ps.build()
        kinds = {p.kind for p in primes}
        assert PrimitiveKind.REVERSE in kinds

    def test_no_reverse_excludes_reverse(self) -> None:
        ps = SippLatticePrimitiveSet(allow_reverse=False)
        primes = ps.build()
        kinds = {p.kind for p in primes}
        assert PrimitiveKind.REVERSE not in kinds

    def test_count_matches_build(self) -> None:
        ps = SippLatticePrimitiveSet()
        assert ps.count() == len(ps.build())

    def test_count_nonzero(self) -> None:
        ps = SippLatticePrimitiveSet()
        assert ps.count() > 0

    def test_all_primitives_finite(self) -> None:
        ps = SippLatticePrimitiveSet()
        for p in ps.build():
            assert math.isfinite(p.linear_velocity)
            assert math.isfinite(p.angular_velocity)
            assert math.isfinite(p.duration)
            assert p.duration > 0.0

    def test_all_primitive_durations_match_config(self) -> None:
        ps = SippLatticePrimitiveSet(primitive_duration=0.25)
        for p in ps.build():
            assert p.duration == pytest.approx(0.25)

    def test_linear_resolution_filters(self) -> None:
        fine = SippLatticePrimitiveSet(linear_resolution=0.1, angular_resolution=0.5)
        coarse = SippLatticePrimitiveSet(linear_resolution=0.3, angular_resolution=1.0)
        assert fine.count() > coarse.count()

    def test_negative_max_linear_speed_raises(self) -> None:
        with pytest.raises(ValueError, match="max_linear_speed"):
            SippLatticePrimitiveSet(max_linear_speed=-1.0)

    def test_zero_deceleration_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="deceleration_steps"):
            SippLatticePrimitiveSet(deceleration_steps=0)

    def test_forward_primitives_positive_linear(self) -> None:
        ps = SippLatticePrimitiveSet()
        for p in ps.build():
            if p.kind == PrimitiveKind.FORWARD:
                assert p.linear_velocity > 0.0

    def test_reverse_primitives_negative_linear(self) -> None:
        ps = SippLatticePrimitiveSet(allow_reverse=True)
        for p in ps.build():
            if p.kind == PrimitiveKind.REVERSE:
                assert p.linear_velocity < 0.0

    def test_decelerate_monotonically_decreasing(self) -> None:
        ps = SippLatticePrimitiveSet(deceleration_steps=4)
        prizes = ps.build()
        decelerate = [p.linear_velocity for p in prizes if p.kind == PrimitiveKind.DECELERATE]
        assert decelerate == sorted(decelerate, reverse=True)

    def test_steering_rate_limits_angular_velocity_per_primitive(self) -> None:
        ps = SippLatticePrimitiveSet(
            max_angular_speed=3.0,
            max_steering_rate=2.0,
            primitive_duration=0.2,
        )
        forwards = [p for p in ps.build() if p.kind == PrimitiveKind.FORWARD]
        assert forwards
        assert max(abs(p.angular_velocity) for p in forwards) <= 0.4 + 1e-6

    def test_linear_acceleration_limits_speed_per_primitive(self) -> None:
        ps = SippLatticePrimitiveSet(
            max_linear_speed=3.0,
            max_linear_acceleration=0.5,
            primitive_duration=0.2,
        )
        assert max(abs(p.linear_velocity) for p in ps.build()) <= 0.1 + 1e-6

    def test_low_recenter_limit_produces_bounded_corrective_turns(self) -> None:
        ps = SippLatticePrimitiveSet(recenter_angular_max=0.1, angular_resolution=0.25)
        recenter = [p.angular_velocity for p in ps.build() if p.kind == PrimitiveKind.RECENTER]
        assert recenter == [-0.1, 0.1]

    def test_wait_primitive_is_zero_command(self) -> None:
        ps = SippLatticePrimitiveSet()
        for p in ps.build():
            if p.kind == PrimitiveKind.WAIT:
                assert p.linear_velocity == pytest.approx(0.0)
                assert p.angular_velocity == pytest.approx(0.0)


# -- SippKinodynamicCollisionModel tests --


class TestSippKinodynamicCollisionModel:
    """Unit tests for the SippKinodynamicCollisionModel."""

    def test_no_collision_when_far(self) -> None:
        cm = SippKinodynamicCollisionModel()
        pos = np.array([0.0, 0.0])
        obs = np.array([5.0, 0.0])
        assert not cm.check_circle_collision(pos, obs, 0.3)

    def test_collision_when_close(self) -> None:
        cm = SippKinodynamicCollisionModel()
        pos = np.array([0.0, 0.0])
        obs = np.array([0.3, 0.0])
        assert cm.check_circle_collision(pos, obs, 0.3)

    def test_continuous_arc_no_collision(self) -> None:
        cm = SippKinodynamicCollisionModel()
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 0.0])
        obstacles = np.array([[5.0, 0.0]])
        assert not cm.check_continuous_arc_collision(start, end, obstacles, 0.3)

    def test_continuous_arc_hits_midpoint(self) -> None:
        cm = SippKinodynamicCollisionModel()
        start = np.array([0.0, 0.0])
        end = np.array([2.0, 0.0])
        obstacles = np.array([[1.0, 0.0]])
        assert cm.check_continuous_arc_collision(start, end, obstacles, 0.3)

    def test_primitive_posture_no_collision(self) -> None:
        cm = SippKinodynamicCollisionModel()
        posture = cm.primitive_posture(
            command=(0.5, 0.0),
            heading=0.0,
            duration=0.2,
            start_pos=np.array([0.0, 0.0]),
            obstacle_positions=np.array([[5.0, 0.0]]),
            obstacle_radius=0.3,
        )
        assert not posture["endpoint_collides"]
        assert not posture["continuous_collides"]
        assert posture["endpoint_distance"] > 0.0

    def test_primitive_posture_collision(self) -> None:
        cm = SippKinodynamicCollisionModel()
        posture = cm.primitive_posture(
            command=(0.5, 0.0),
            heading=0.0,
            duration=0.2,
            start_pos=np.array([0.5, 0.0]),
            obstacle_positions=np.array([[0.8, 0.0]]),
            obstacle_radius=0.3,
        )
        assert posture["endpoint_collides"] or posture["continuous_collides"]

    def test_primitive_posture_straight_line(self) -> None:
        cm = SippKinodynamicCollisionModel()
        posture = cm.primitive_posture(
            command=(1.0, 0.0),
            heading=0.0,
            duration=0.2,
            start_pos=np.array([0.0, 0.0]),
            obstacle_positions=np.array([]).reshape(0, 2),
            obstacle_radius=0.3,
        )
        end = np.array(posture["end_position"])
        assert end[0] == pytest.approx(0.2, abs=0.01)
        assert end[1] == pytest.approx(0.0, abs=0.01)

    def test_primitive_posture_turning(self) -> None:
        cm = SippKinodynamicCollisionModel()
        posture = cm.primitive_posture(
            command=(1.0, 0.5),
            heading=0.0,
            duration=0.2,
            start_pos=np.array([0.0, 0.0]),
            obstacle_positions=np.array([]).reshape(0, 2),
            obstacle_radius=0.3,
        )
        end = np.array(posture["end_position"])
        assert end[0] == pytest.approx(math.sin(0.1) / 0.5)
        assert end[1] == pytest.approx((1.0 - math.cos(0.1)) / 0.5)

    def test_primitive_posture_checks_the_actual_turning_arc(self) -> None:
        cm = SippKinodynamicCollisionModel(continuous_check_steps=20)
        posture = cm.primitive_posture(
            command=(1.0, math.pi),
            heading=0.0,
            duration=1.0,
            start_pos=np.array([0.0, 0.0]),
            obstacle_positions=np.array([[0.32, 0.32]]),
            obstacle_radius=0.05,
        )
        assert posture["continuous_collides"]
        assert not posture["endpoint_collides"]

    def test_invalid_robot_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="robot_radius"):
            SippKinodynamicCollisionModel(robot_radius=-0.1)

    def test_zero_continuous_check_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="continuous_check_steps"):
            SippKinodynamicCollisionModel(continuous_check_steps=0)

    def test_empty_obstacle_array_no_collision(self) -> None:
        cm = SippKinodynamicCollisionModel()
        empty = np.array([]).reshape(0, 2)
        posture = cm.primitive_posture(
            command=(0.5, 0.0),
            heading=0.0,
            duration=0.2,
            start_pos=np.array([0.0, 0.0]),
            obstacle_positions=empty,
            obstacle_radius=0.3,
        )
        assert not posture["endpoint_collides"]
        assert not posture["continuous_collides"]


# -- SippLatticeConfig tests --


class TestSippLatticeConfig:
    """Unit tests for the SippLatticeConfig dataclass."""

    def test_default_config_valid(self) -> None:
        cfg = SippLatticeConfig()
        assert cfg.max_linear_speed > 0.0

    def test_config_to_primitive_set(self) -> None:
        cfg = SippLatticeConfig()
        ps = cfg.to_primitive_set()
        primes = ps.build()
        assert len(primes) > 0

    def test_config_to_collision_model(self) -> None:
        cfg = SippLatticeConfig()
        cm = cfg.to_collision_model()
        assert cm.robot_radius == cfg.robot_radius

    def test_invalid_max_linear_speed_raises(self) -> None:
        with pytest.raises(ValueError, match="max_linear_speed"):
            SippLatticeConfig(max_linear_speed=-1.0)

    def test_invalid_grid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="grid_obstacle_threshold"):
            SippLatticeConfig(grid_obstacle_threshold=2.0)

    def test_invalid_deceleration_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="deceleration_steps"):
            SippLatticeConfig(deceleration_steps=0)

    def test_inf_value_raises(self) -> None:
        with pytest.raises(ValueError, match="max_linear_speed"):
            SippLatticeConfig(max_linear_speed=float("inf"))


# -- build_sipp_lattice_config factory tests --


class TestBuildSippLatticeConfig:
    """Unit tests for the build_sipp_lattice_config factory."""

    def test_none_returns_defaults(self) -> None:
        cfg = build_sipp_lattice_config(None)
        assert isinstance(cfg, SippLatticeConfig)

    def test_empty_dict_returns_defaults(self) -> None:
        cfg = build_sipp_lattice_config({})
        assert cfg.max_linear_speed == 1.0

    def test_overrides_max_linear_speed(self) -> None:
        cfg = build_sipp_lattice_config({"max_linear_speed": 1.5})
        assert cfg.max_linear_speed == 1.5

    def test_overrides_allow_reverse(self) -> None:
        cfg = build_sipp_lattice_config({"allow_reverse": False})
        assert not cfg.allow_reverse

    def test_overrides_int(self) -> None:
        cfg = build_sipp_lattice_config({"deceleration_steps": 6})
        assert cfg.deceleration_steps == 6

    def test_overrides_primitive_duration(self) -> None:
        cfg = build_sipp_lattice_config({"primitive_duration": 0.15})
        assert cfg.primitive_duration == pytest.approx(0.15)

    def test_overrides_continuous_check_steps(self) -> None:
        cfg = build_sipp_lattice_config({"continuous_check_steps": 8})
        assert cfg.continuous_check_steps == 8

    def test_none_values_use_defaults(self) -> None:
        cfg = build_sipp_lattice_config(
            {"max_linear_speed": None, "deceleration_steps": None, "allow_reverse": None}
        )
        assert cfg.max_linear_speed == 1.0
        assert cfg.deceleration_steps == 4
        assert cfg.allow_reverse


# -- SippLatticePlannerAdapter tests --


class TestSippLatticePlannerAdapter:
    """Unit tests for the SippLatticePlannerAdapter."""

    def test_plan_returns_tuple(self) -> None:
        planner = SippLatticePlannerAdapter()
        result = planner.plan(_obs(goal=(3.0, 0.0)))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_plan_moves_toward_goal_in_open_space(self) -> None:
        planner = SippLatticePlannerAdapter()
        v, w = planner.plan(_obs(goal=(3.0, 0.0)))
        assert v > 0.0
        assert abs(w) <= planner.config.max_angular_speed

    def test_plan_zero_at_goal(self) -> None:
        planner = SippLatticePlannerAdapter()
        v, w = planner.plan(_obs(robot=(1.99, 0.0), goal=(2.0, 0.0)))
        assert v == pytest.approx(0.0)
        assert w == pytest.approx(0.0)

    def test_plan_bounds_obeyed(self) -> None:
        planner = SippLatticePlannerAdapter()
        v, w = planner.plan(_obs(goal=(3.0, 3.0)))
        assert abs(v) <= planner.config.max_linear_speed + 1e-6
        assert abs(w) <= planner.config.max_angular_speed + 1e-6

    def test_plan_with_pedestrians(self) -> None:
        planner = SippLatticePlannerAdapter()
        peds = [[0.0, 0.5], [0.0, -0.5]]
        v, w = planner.plan(
            _obs(
                goal=(3.0, 0.0),
                pedestrian_positions=peds,
                count=2,
            )
        )
        assert math.isfinite(v)
        assert math.isfinite(w)

    def test_diagnostics_returns_payload(self) -> None:
        planner = SippLatticePlannerAdapter()
        planner.plan(_obs(goal=(3.0, 0.0)))
        diag = planner.diagnostics()
        assert "last_decision" in diag
        assert diag["last_decision"] is not None

    def test_diagnostics_records_feasible_count(self) -> None:
        planner = SippLatticePlannerAdapter()
        planner.plan(_obs(goal=(3.0, 0.0)))
        diag = planner.diagnostics()
        last = diag["last_decision"]
        assert "feasible_count" in last
        assert last["feasible_count"] >= 0

    def test_diagnostics_goal_reached(self) -> None:
        planner = SippLatticePlannerAdapter()
        planner.plan(_obs(robot=(1.0, 0.0), goal=(1.05, 0.0)))
        diag = planner.diagnostics()
        last = diag["last_decision"]
        assert last["constraint_reason"] == "goal_reached"
        assert set(last) == {
            "primitive_count",
            "feasible_count",
            "infeasible_count",
            "best_score",
            "best_kind",
            "best_command",
            "constraint_reason",
            "distance_to_goal_m",
        }

    def test_reset_clears_last_decision(self) -> None:
        planner = SippLatticePlannerAdapter()
        planner.plan(_obs(goal=(3.0, 0.0)))
        assert planner._last_decision is not None
        planner.reset()
        assert planner._last_decision is None

    def test_plan_blocked_falls_back_to_zero(self) -> None:
        config = SippLatticeConfig(
            robot_radius=10.0,
            safety_margin=10.0,
            pedestrian_radius=0.3,
            min_clearance=100.0,
        )
        planner = SippLatticePlannerAdapter(config=config)
        peds = [[0.0, 0.01]]
        _v, _w = planner.plan(
            _obs(
                goal=(3.0, 0.0),
                pedestrian_positions=peds,
                count=1,
            )
        )
        diag = planner.diagnostics()
        last = diag["last_decision"]
        assert last["constraint_reason"] in {
            "all_primitives_infeasible_wait",
        }

    def test_primitive_count_reflected_in_diagnostics(self) -> None:
        planner = SippLatticePlannerAdapter()
        planner.plan(_obs(goal=(3.0, 0.0)))
        diag = planner.diagnostics()
        last = diag["last_decision"]
        assert last["primitive_count"] > 0

    def test_custom_config_applied(self) -> None:
        cfg = build_sipp_lattice_config({"max_linear_speed": 1.5, "primitive_duration": 0.3})
        planner = SippLatticePlannerAdapter(config=cfg)
        assert planner.config.max_linear_speed == 1.5
        assert planner.config.primitive_duration == pytest.approx(0.3)

    def test_no_reverse_config(self) -> None:
        cfg = SippLatticeConfig(allow_reverse=False)
        planner = SippLatticePlannerAdapter(config=cfg)
        v, _w = planner.plan(_obs(goal=(3.0, 0.0)))
        assert v >= 0.0

    def test_obstacle_grid_channel(self) -> None:
        grid = np.zeros((4, 4, 4), dtype=np.float32)
        grid[0, 1, 3] = 1.0
        obs = {
            "robot": {
                "position": np.asarray([0.0, 0.0], dtype=float),
                "heading": np.asarray([0.0], dtype=float),
                "speed": np.asarray([0.0], dtype=float),
                "radius": np.asarray([0.25], dtype=float),
            },
            "goal": {
                "current": np.asarray([3.0, 0.0], dtype=float),
                "next": np.asarray([3.0, 0.0], dtype=float),
            },
            "pedestrians": {
                "positions": np.zeros((0, 2), dtype=float),
                "velocities": np.zeros((0, 2), dtype=float),
                "count": np.asarray([0.0], dtype=float),
                "radius": 0.3,
            },
            "occupancy_grid": grid,
            "occupancy_grid_meta_origin": np.asarray([-2.0, -2.0], dtype=float),
            "occupancy_grid_meta_resolution": np.asarray([1.0], dtype=float),
            "occupancy_grid_meta_size": np.asarray([4.0, 4.0], dtype=float),
            "occupancy_grid_meta_use_ego_frame": np.asarray([1.0], dtype=float),
            "occupancy_grid_meta_channel_indices": np.asarray([0, 1, 2, 3], dtype=float),
        }
        planner = SippLatticePlannerAdapter()
        v, w = planner.plan(obs)
        assert math.isfinite(v)
        assert math.isfinite(w)
        assert abs(v) <= planner.config.max_linear_speed + 1e-6
        assert abs(w) <= planner.config.max_angular_speed + 1e-6

    def test_heading_direction_affects_output(self) -> None:
        planner = SippLatticePlannerAdapter()
        _v1, w1 = planner.plan(_obs(goal=(0.0, 3.0), heading=0.0))
        _v2, w2 = planner.plan(_obs(goal=(3.0, 0.0), heading=0.0))
        assert abs(w1) > 0.05
        assert abs(w2) < 0.5

    def test_multiple_plan_calls_are_independent(self) -> None:
        planner = SippLatticePlannerAdapter()
        v1, _ = planner.plan(_obs(goal=(3.0, 0.0)))
        v2, _ = planner.plan(_obs(goal=(0.0, 3.0)))
        assert math.isfinite(v1)
        assert math.isfinite(v2)


# -- Slice 2: time-indexed occupancy, bounded search, and commitment --

from robot_sf.planner.sipp_lattice import (  # noqa: E402
    SippLatticeSearch,
    SippLatticeSearchPlannerAdapter,
    _SearchNode,
    build_pedestrian_occupancy_forecast,
    build_sipp_lattice_search_adapter,
)


def _search_obs(
    *,
    robot=(0.0, 0.0),
    heading=0.0,
    speed=0.0,
    goal=(1.0, 0.0),
    pedestrian_positions=None,
    pedestrian_velocities=None,
    count=0,
):
    """Build an observation with time-indexed pedestrian dynamic state."""
    pedestrian_positions = pedestrian_positions or []
    if pedestrian_velocities is None:
        pedestrian_velocities = [[0.0, 0.0] for _ in pedestrian_positions]
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(pedestrian_positions, dtype=float).reshape(-1, 2),
            "velocities": np.asarray(pedestrian_velocities, dtype=float).reshape(-1, 2),
            "count": np.asarray([float(count)], dtype=float),
            "radius": 0.30,
        },
    }


def _fast_config(**overrides):
    """Config with per-primitive speed unclamped by acceleration for tractable search."""
    params = {
        "max_linear_acceleration": 5.0,
        "max_expansions": 6000,
        "planning_horizon_slots": 40,
    }
    params.update(overrides)
    return build_sipp_lattice_config(params)


class TestPedestrianOccupancyForecast:
    """Unit tests for time-indexed pedestrian safe intervals."""

    def test_no_pedestrians_is_static_and_unoccupied(self) -> None:
        fc = build_pedestrian_occupancy_forecast(
            positions=np.zeros((0, 2)),
            velocities=np.zeros((0, 2)),
            heading=0.0,
            config=SippLatticeConfig(),
            pedestrian_radius=0.3,
        )
        assert fc.status == "static"
        assert fc.usable
        arc = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        assert not fc.arc_occupied(arc, 5)

    def test_missing_velocities_is_static_stationary(self) -> None:
        fc = build_pedestrian_occupancy_forecast(
            positions=np.array([[0.3, 0.0]]),
            velocities=None,
            heading=0.0,
            config=SippLatticeConfig(),
            pedestrian_radius=0.3,
        )
        assert fc.status == "static"
        # Stationary pedestrian occupies the same cell at every slot.
        arc = np.array([[0.3, 0.0]])
        assert fc.arc_occupied(arc, 0)
        assert fc.arc_occupied(arc, 10)

    def test_geometrically_clear_arc_rejected_when_temporally_occupied(self) -> None:
        # Pedestrian is far now (arc clear at slot 0) but crosses the arc later.
        cfg = SippLatticeConfig()
        fc = build_pedestrian_occupancy_forecast(
            positions=np.array([[1.5, 0.0]]),
            velocities=np.array([[-1.0, 0.0]]),
            heading=0.0,
            config=cfg,
            pedestrian_radius=0.3,
        )
        arc = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        assert not fc.arc_occupied(arc, 0)  # geometrically clear right now
        assert fc.arc_occupied(arc, 6)  # temporally occupied once the pedestrian arrives

    def test_mid_primitive_crossing_is_checked_at_matching_sample_time(self) -> None:
        """A crossing pedestrian cannot evade an arc by clearing it at arrival."""
        cfg = SippLatticeConfig(
            time_slot_duration=0.2,
            robot_radius=0.02,
            pedestrian_radius=0.02,
            safety_margin=0.0,
            continuous_check_steps=10,
        )
        fc = build_pedestrian_occupancy_forecast(
            positions=np.array([[0.1, -0.25]]),
            velocities=np.array([[0.0, 2.5]]),
            heading=0.0,
            config=cfg,
            pedestrian_radius=0.02,
        )
        arc = SippKinodynamicCollisionModel(continuous_check_steps=10)._unicycle_arc_positions(
            (1.0, 0.0), 0.0, 0.2, np.zeros(2)
        )

        assert not fc.arc_occupied(arc, 1)
        assert fc.arc_occupied(arc, 0, 0.2)
        cfg.pedestrian_forecast_horizon_s = 0.2
        short_fc = build_pedestrian_occupancy_forecast(
            positions=np.array([[2.0, 0.0]]),
            velocities=np.array([[0.0, 0.0]]),
            heading=0.0,
            config=cfg,
            pedestrian_radius=0.02,
        )
        assert short_fc.arc_occupied(np.array([[0.0, 0.0], [0.2, 0.0]]), 0, 0.4)

    def test_malformed_velocities_fail_closed(self) -> None:
        fc = build_pedestrian_occupancy_forecast(
            positions=np.array([[0.5, 0.0]]),
            velocities=np.array([[float("inf"), 0.0]]),
            heading=0.0,
            config=SippLatticeConfig(),
            pedestrian_radius=0.3,
        )
        assert fc.status == "failed"
        assert not fc.usable

    def test_nonfinite_positions_fail_closed(self) -> None:
        fc = build_pedestrian_occupancy_forecast(
            positions=np.array([[float("nan"), 0.0]]),
            velocities=np.array([[0.0, 0.0]]),
            heading=0.0,
            config=SippLatticeConfig(),
            pedestrian_radius=0.3,
        )
        assert fc.status == "failed"

    def test_ego_velocity_rotated_to_world(self) -> None:
        # Ego +x velocity with heading pi/2 becomes world +y.
        fc = build_pedestrian_occupancy_forecast(
            positions=np.array([[0.0, 0.0]]),
            velocities=np.array([[1.0, 0.0]]),
            heading=math.pi / 2,
            config=SippLatticeConfig(time_slot_duration=1.0),
            pedestrian_radius=0.3,
        )
        moved = fc.positions + fc.slot_duration * fc.velocities
        assert moved[0][0] == pytest.approx(0.0, abs=1e-6)
        assert moved[0][1] == pytest.approx(1.0, abs=1e-6)


class TestSippLatticeSearch:
    """Unit tests for the bounded state-time lattice search."""

    def test_reaches_open_space_goal(self) -> None:
        cfg = _fast_config()
        search = SippLatticeSearch(cfg, cfg.to_primitive_set().build(), cfg.to_collision_model())
        fc = build_pedestrian_occupancy_forecast(
            positions=np.zeros((0, 2)),
            velocities=np.zeros((0, 2)),
            heading=0.0,
            config=cfg,
            pedestrian_radius=0.3,
        )
        result = search.search(
            start_pos=np.array([0.0, 0.0]),
            start_heading=0.0,
            start_speed=0.0,
            goal=np.array([0.6, 0.0]),
            forecast=fc,
        )
        assert result.result_type == "native_plan"
        assert result.bound_termination == "goal"
        assert len(result.plan) >= 1

    def test_expansion_bound_terminates_deterministically(self) -> None:
        # A tiny expansion budget with a far goal must return a classified result.
        cfg = _fast_config(max_expansions=1, planning_horizon_slots=3)
        search = SippLatticeSearch(cfg, cfg.to_primitive_set().build(), cfg.to_collision_model())
        fc = build_pedestrian_occupancy_forecast(
            positions=np.zeros((0, 2)),
            velocities=np.zeros((0, 2)),
            heading=0.0,
            config=cfg,
            pedestrian_radius=0.3,
        )
        result = search.search(
            start_pos=np.array([0.0, 0.0]),
            start_heading=0.0,
            start_speed=0.0,
            goal=np.array([50.0, 0.0]),
            forecast=fc,
        )
        assert result.bound_termination in {"expansions", "time", "horizon_only", "open_exhausted"}
        assert result.result_type in {"native_plan", "bounded_safe_wait"}
        assert result.expansions <= 1

    def test_search_is_deterministic(self) -> None:
        cfg = _fast_config()
        primitives = cfg.to_primitive_set().build()
        collision = cfg.to_collision_model()
        fc = build_pedestrian_occupancy_forecast(
            positions=np.array([[0.5, -1.1]]),
            velocities=np.array([[0.0, 2.0]]),
            heading=0.0,
            config=cfg,
            pedestrian_radius=0.3,
        )
        results = [
            SippLatticeSearch(cfg, primitives, collision).search(
                start_pos=np.array([0.0, 0.0]),
                start_heading=0.0,
                start_speed=0.0,
                goal=np.array([1.0, 0.0]),
                forecast=fc,
            )
            for _ in range(3)
        ]
        commands = [[p.as_command() for p in r.plan] for r in results]
        assert commands[0] == commands[1] == commands[2]

    def test_unreachable_kinodynamic_successor_is_rejected(self) -> None:
        """Search rejects unreachable commands and normalizes equivalent headings."""
        cfg = _fast_config(
            max_linear_acceleration=0.5,
            max_steering_rate=0.5,
            planning_horizon_slots=1,
            max_planning_time_s=0.2,
        )
        primitive = MotionPrimitive(
            linear_velocity=1.0,
            angular_velocity=1.0,
            duration=cfg.primitive_duration,
            kind=PrimitiveKind.FORWARD,
        )
        search = SippLatticeSearch(cfg, [primitive], cfg.to_collision_model())
        base = _SearchNode(np.zeros(2), 0.1, 0.0, 0.0, 0, 0.0, None)
        wrapped = _SearchNode(np.zeros(2), 0.1 + 2.0 * math.pi, 0.0, 0.0, 0, 0.0, None)
        assert search._state_key(base) == search._state_key(wrapped)
        fc = build_pedestrian_occupancy_forecast(
            positions=np.empty((0, 2)),
            velocities=None,
            heading=0.0,
            config=cfg,
            pedestrian_radius=0.3,
        )

        assert not search._transition_reachable(0.0, 0.0, primitive)
        result = search.search(
            start_pos=np.array([0.0, 0.0]),
            start_heading=0.0,
            start_speed=0.0,
            start_angular_velocity=0.2,
            goal=np.array([0.5, 0.0]),
            forecast=fc,
        )
        assert result.result_type == "bounded_safe_wait"


class TestSippLatticeSearchPlannerAdapter:
    """Unit tests for the bounded SIPP search planner with commitment."""

    def test_multi_primitive_path_around_time_indexed_conflict(self) -> None:
        # A pedestrian crosses the corridor when the robot would arrive going
        # straight at full speed. Greedy one-step scoring (Slice 1) ignores time
        # and drives straight into the future conflict; the state-time search
        # commits a multi-primitive plan that modulates speed to yield.
        cfg = _fast_config(
            robot_radius=0.15,
            pedestrian_radius=0.15,
            safety_margin=0.05,
            min_clearance=0.3,
            max_planning_time_s=0.2,
        )
        obs = _search_obs(
            goal=(1.0, 0.0),
            pedestrian_positions=[[0.5, -1.1]],
            pedestrian_velocities=[[0.0, 2.0]],
            count=1,
        )
        search = SippLatticeSearchPlannerAdapter(config=cfg)
        v, _w = search.plan(obs)
        decision = search.diagnostics()["last_decision"]

        assert decision["result_type"] == "native_plan"
        assert decision["bound_termination"] == "goal"
        assert decision["committed_length"] >= 2
        assert decision["safe_interval_rejections"] > 0
        # The committed plan modulates speed (a slowdown/wait) rather than driving
        # straight through: at least one committed primitive is slower than the
        # first executed command.
        committed_speeds = [p.linear_velocity for p in search._committed]
        assert min(committed_speeds) < v

        greedy = SippLatticePlannerAdapter(config=cfg)
        gv, gw = greedy.plan(obs)
        # Greedy sees the pedestrian's *current* (far) position as clear and drives
        # straight ahead at speed -- i.e. into the future conflict.
        assert gv > v
        assert abs(gw) < 0.2

    def test_safe_interval_rejects_geometrically_clear_but_occupied(self) -> None:
        # With a pedestrian that will occupy every forward arc in time, the
        # bounded search returns a classified safe wait rather than a collision.
        cfg = _fast_config(planning_horizon_slots=10, max_expansions=800)
        obs = _search_obs(
            goal=(1.0, 0.0),
            pedestrian_positions=[[0.35, 0.0]],
            pedestrian_velocities=[[0.0, 0.0]],
            count=1,
        )
        planner = SippLatticeSearchPlannerAdapter(config=cfg)
        v, w = planner.plan(obs)
        decision = planner.diagnostics()["last_decision"]
        assert decision["result_type"] in {"bounded_safe_wait", "native_plan"}
        if decision["result_type"] == "bounded_safe_wait":
            assert (v, w) == (0.0, 0.0)
            assert decision["safe_interval_rejections"] > 0

    def test_static_occupancy_checks_the_footprint_along_the_actual_arc(self) -> None:
        """A curved primitive is rejected when its midpoint crosses one occupied cell."""
        resolution = 0.01
        grid = np.zeros((4, 400, 400), dtype=np.float32)
        obs = _search_obs(goal=(1.0, 0.0))
        obs.update(
            {
                "occupancy_grid": grid,
                "occupancy_grid_meta_origin": np.asarray([-2.0, -2.0], dtype=float),
                "occupancy_grid_meta_resolution": np.asarray([resolution], dtype=float),
                "occupancy_grid_meta_size": np.asarray([4.0, 4.0], dtype=float),
                "occupancy_grid_meta_use_ego_frame": np.asarray([0.0], dtype=float),
                "occupancy_grid_meta_channel_indices": np.asarray([0, 1, 2, 3], dtype=float),
            }
        )
        cfg = _fast_config(robot_radius=0.005, safety_margin=0.0, continuous_check_steps=10)
        planner = SippLatticeSearchPlannerAdapter(config=cfg)
        arc = planner._collision_model._unicycle_arc_positions(
            (1.0, 4.0), 0.0, cfg.primitive_duration, np.zeros(2)
        )
        midpoint = arc[len(arc) // 2]
        row = int((midpoint[1] + 2.0) / resolution)
        col = int((midpoint[0] + 2.0) / resolution)
        grid[0, row, col] = 1.0
        static_blocked = planner._static_blocked_fn(obs)

        assert static_blocked(arc)
        assert not static_blocked(np.linspace(arc[0], arc[-1], len(arc)))

    def test_commitment_survives_one_unchanged_update(self) -> None:
        planner = SippLatticeSearchPlannerAdapter(config=_fast_config())
        obs = _search_obs(goal=(1.0, 0.0))
        planner.plan(obs)
        first = planner.diagnostics()["last_decision"]
        assert first["result_type"] == "native_plan"
        planner.plan(obs)
        second = planner.diagnostics()["last_decision"]
        assert second["result_type"] == "committed_plan"
        assert second["commit_index"] > first["commit_index"]

    def test_commitment_invalidates_on_new_occupancy(self) -> None:
        planner = SippLatticeSearchPlannerAdapter(config=_fast_config())
        clear = _search_obs(goal=(1.0, 0.0))
        planner.plan(clear)
        assert planner.diagnostics()["last_decision"]["result_type"] == "native_plan"
        blocked = _search_obs(
            goal=(1.0, 0.0),
            pedestrian_positions=[[0.4, 0.0]],
            pedestrian_velocities=[[0.0, 0.0]],
            count=1,
        )
        planner.plan(blocked)
        decision = planner.diagnostics()["last_decision"]
        assert decision["replanned"] is True
        assert decision["result_type"] != "committed_plan"

    def test_goal_change_clears_commitment(self) -> None:
        planner = SippLatticeSearchPlannerAdapter(config=_fast_config())
        planner.plan(_search_obs(goal=(1.0, 0.0)))
        planner.plan(_search_obs(goal=(0.0, 1.0)))
        decision = planner.diagnostics()["last_decision"]
        assert decision["replanned"] is True

    def test_reset_clears_commitment_and_diagnostics(self) -> None:
        planner = SippLatticeSearchPlannerAdapter(config=_fast_config())
        planner.plan(_search_obs(goal=(1.0, 0.0)))
        assert planner._committed
        planner.reset()
        assert planner._committed == []
        assert planner._commit_index == 0
        assert planner._last_goal is None
        assert planner._last_decision is None
        assert planner.diagnostics() == {"last_decision": {}}

    def test_failed_dynamic_input_is_classified_safe_wait(self) -> None:
        planner = SippLatticeSearchPlannerAdapter(config=_fast_config())
        obs = _search_obs(
            goal=(1.0, 0.0),
            pedestrian_positions=[[0.5, 0.0]],
            pedestrian_velocities=[[float("inf"), 0.0]],
            count=1,
        )
        v, w = planner.plan(obs)
        decision = planner.diagnostics()["last_decision"]
        assert decision["result_type"] == "failed_dynamic_input"
        assert decision["dynamic_state"] == "failed"
        assert (v, w) == (0.0, 0.0)

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda obs: obs["pedestrians"].update(
                positions=np.asarray([["not-a-number", 0.0]], dtype=object)
            ),
            lambda obs: obs["pedestrians"].update(positions=np.asarray([[0.5, 0.0, 0.1]])),
            lambda obs: obs["pedestrians"].update(count=np.asarray([1.5], dtype=float)),
            lambda obs: (
                obs["robot"].update(position=np.asarray([np.nan, 0.0]))
                or obs["goal"].update(current=np.asarray([np.inf, 0.0]))
            ),
        ],
    )
    def test_malformed_active_state_returns_failed_observation_stop(self, mutation) -> None:
        """Malformed simulator state must never become an empty-scene plan."""
        planner = SippLatticeSearchPlannerAdapter(config=_fast_config())
        obs = _search_obs(goal=(1.0, 0.0))
        mutation(obs)

        assert planner.plan(obs) == (0.0, 0.0)
        decision = planner.diagnostics()["last_decision"]
        assert decision["result_type"] == "failed_observation_input"
        assert decision["dynamic_state"] == "failed"

    def test_goal_reached_returns_zero_and_clears(self) -> None:
        planner = SippLatticeSearchPlannerAdapter(config=_fast_config())
        v, w = planner.plan(_search_obs(robot=(0.95, 0.0), goal=(1.0, 0.0)))
        decision = planner.diagnostics()["last_decision"]
        assert decision["result_type"] == "goal_reached"
        assert (v, w) == (0.0, 0.0)
        assert planner._committed == []

    def test_plan_bounds_obeyed(self) -> None:
        planner = SippLatticeSearchPlannerAdapter(config=_fast_config())
        v, w = planner.plan(_search_obs(goal=(2.0, 1.0)))
        assert abs(v) <= planner.config.max_linear_speed + 1e-6
        assert abs(w) <= planner.config.max_angular_speed + 1e-6

    def test_diagnostics_distinguish_result_types(self) -> None:
        planner = SippLatticeSearchPlannerAdapter(config=_fast_config())
        obs = _search_obs(goal=(1.0, 0.0))
        planner.plan(obs)
        assert planner.diagnostics()["last_decision"]["result_type"] == "native_plan"
        planner.plan(obs)
        assert planner.diagnostics()["last_decision"]["result_type"] == "committed_plan"

    def test_factory_builds_configured_adapter(self) -> None:
        planner = build_sipp_lattice_search_adapter(
            {"max_linear_speed": 1.5, "commitment_horizon": 2}
        )
        assert isinstance(planner, SippLatticeSearchPlannerAdapter)
        assert planner.config.max_linear_speed == 1.5
        assert planner.config.commitment_horizon == 2


class TestSippLatticeConfigSlice2:
    """Validation tests for the Slice-2 config fields."""

    def test_heuristic_weight_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="heuristic_weight"):
            SippLatticeConfig(heuristic_weight=0.5)

    def test_zero_max_expansions_raises(self) -> None:
        with pytest.raises(ValueError, match="max_expansions"):
            SippLatticeConfig(max_expansions=0)

    def test_negative_offtrack_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="offtrack_tolerance"):
            SippLatticeConfig(offtrack_tolerance=-0.1)

    def test_search_rejects_nonintegral_primitive_slot_ratio(self) -> None:
        """State-time search must reject cumulative primitive-slot drift."""
        cfg = SippLatticeConfig(primitive_duration=0.3, time_slot_duration=0.2)
        with pytest.raises(ValueError, match="integer multiple"):
            SippLatticeSearch(cfg, cfg.to_primitive_set().build(), cfg.to_collision_model())

    def test_search_fields_flow_through_factory(self) -> None:
        cfg = build_sipp_lattice_config(
            {
                "planning_horizon_slots": 12,
                "max_planning_time_s": 0.02,
                "commitment_horizon": 3,
                "pedestrian_forecast_horizon_s": 2.0,
            }
        )
        assert cfg.planning_horizon_slots == 12
        assert cfg.max_planning_time_s == pytest.approx(0.02)
        assert cfg.commitment_horizon == 3
        assert cfg.pedestrian_forecast_horizon_s == pytest.approx(2.0)
