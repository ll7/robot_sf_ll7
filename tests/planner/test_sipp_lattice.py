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
