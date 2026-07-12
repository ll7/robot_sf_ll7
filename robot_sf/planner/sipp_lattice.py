"""Kinodynamic state-time lattice primitives and collision model.

Foundational building blocks for a SIPP-class local planner in discretized SE(2)+time.
Provides AMV-feasible motion primitives with acceleration, steering-rate, footprint,
and continuous-collision constraints. This module covers Slice 1 of issue #5306:
primitive set, collision model, and unit-tested planner adapter.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

from robot_sf.common.math_utils import wrap_angle_pi
from robot_sf.nav.occupancy import is_circle_circle_intersection
from robot_sf.planner.socnav import OccupancyAwarePlannerMixin


class PrimitiveKind(Enum):
    """Categorization of kinodynamic lattice primitives."""

    FORWARD = "forward"
    DECELERATE = "decelerate"
    WAIT = "wait"
    RECENTER = "recenter"
    REVERSE = "reverse"


@dataclass(frozen=True)
class MotionPrimitive:
    """One discretized AMV-feasible unicycle command in SE(2).

    A primitive is a constant (v, omega) held for ``duration`` seconds, producing
    an arc through the state space.  AMV feasibility is enforced at construction time.

    Attributes:
        linear_velocity: Target linear velocity in m/s (can be negative for reverse).
        angular_velocity: Target angular velocity in rad/s.
        duration: How long to hold the command in seconds.
        kind: Primitive category for logging and diagnostics.
    """

    linear_velocity: float
    angular_velocity: float
    duration: float
    kind: PrimitiveKind

    def __post_init__(self) -> None:
        """Validate primitive parameters at construction."""
        if not isfinite(self.linear_velocity):
            raise ValueError("linear_velocity must be finite")
        if not isfinite(self.angular_velocity):
            raise ValueError("angular_velocity must be finite")
        if not (isfinite(self.duration) and self.duration > 0.0):
            raise ValueError("duration must be finite and positive")

    @property
    def distance_traveled(self) -> float:
        """Approximate arc length of a primitive held for its full duration."""
        return abs(self.linear_velocity) * self.duration

    @property
    def delta_yaw(self) -> float:
        """Total heading change over the primitive duration."""
        return self.angular_velocity * self.duration

    def as_command(self) -> tuple[float, float]:
        """Return ``(linear_velocity, angular_velocity)`` for adapter dispatch."""
        return (float(self.linear_velocity), float(self.angular_velocity))


@dataclass(frozen=True)
class SippLatticePrimitiveSet:
    """Discretized AMV-feasible motion primitive set for kinodynamic lattice search.

    Builds a lattice of unicycle commands covering forward arcs, controlled
    deceleration, wait/yield, recentering, and reverse maneuvers.  Every
    primitive is validated against the configured kinodynamic limits at
    construction time.

    Attributes:
        max_linear_speed: Maximum forward linear speed in m/s.
        max_angular_speed: Maximum angular speed in rad/s.
        max_linear_acceleration: Maximum linear acceleration in m/s^2.
        max_steering_rate: Maximum steering rate (alias of angular accel) in rad/s^2.
        primitive_duration: Duration to hold each primitive in seconds.
        linear_resolution: Spacing between sampled forward linear velocities.
        angular_resolution: Spacing between sampled angular velocities.
        allow_reverse: Whether reverse primitives are included.
        deceleration_steps: Number of deceleration primitives from max speed to stop.
        recenter_angular_max: Maximum angular rate for recentering primitives.
    """

    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.2
    max_linear_acceleration: float = 0.8
    max_steering_rate: float = 2.0
    primitive_duration: float = 0.2
    linear_resolution: float = 0.2
    angular_resolution: float = 0.25
    allow_reverse: bool = True
    deceleration_steps: int = 4
    recenter_angular_max: float = 0.4

    def __post_init__(self) -> None:
        """Validate kinodynamic limits at construction."""
        if not (isfinite(self.max_linear_speed) and self.max_linear_speed > 0.0):
            raise ValueError("max_linear_speed must be finite and positive")
        if not (isfinite(self.max_angular_speed) and self.max_angular_speed > 0.0):
            raise ValueError("max_angular_speed must be finite and positive")
        if not (isfinite(self.max_linear_acceleration) and self.max_linear_acceleration >= 0.0):
            raise ValueError("max_linear_acceleration must be finite and non-negative")
        if not (isfinite(self.max_steering_rate) and self.max_steering_rate >= 0.0):
            raise ValueError("max_steering_rate must be finite and non-negative")
        if not (isfinite(self.primitive_duration) and self.primitive_duration > 0.0):
            raise ValueError("primitive_duration must be finite and positive")
        if not (isfinite(self.linear_resolution) and self.linear_resolution > 0.0):
            raise ValueError("linear_resolution must be finite and positive")
        if not (isfinite(self.angular_resolution) and self.angular_resolution > 0.0):
            raise ValueError("angular_resolution must be finite and positive")
        if not (isfinite(self.recenter_angular_max) and self.recenter_angular_max >= 0.0):
            raise ValueError("recenter_angular_max must be finite and non-negative")
        if int(self.deceleration_steps) < 1:
            raise ValueError("deceleration_steps must be at least 1")

    def _generate_forwards(self) -> list[MotionPrimitive]:
        """Generate forward arc primitives.

        Returns:
            List of FORWARD and turn primitives.
        """
        dt = self.primitive_duration
        max_v = min(self.max_linear_speed, self.max_linear_acceleration * dt)
        max_w = min(
            self.max_angular_speed,
            self.max_steering_rate * dt,
        )

        linear_values = list(
            np.arange(self.linear_resolution, max_v + 1e-6, self.linear_resolution)
        )
        if max_v > 1e-6 and (not linear_values or linear_values[-1] < max_v - 1e-6):
            linear_values.append(max_v)

        if max_w <= 1e-6:
            angular_values = [0.0]
        else:
            positive_values = np.arange(0.0, max_w + 1e-6, self.angular_resolution)
            if positive_values[-1] < max_w - 1e-6:
                positive_values = np.append(positive_values, max_w)
            angular_values = list(np.concatenate((-positive_values[:0:-1], positive_values)))

        primitives: list[MotionPrimitive] = []
        for v in linear_values:
            for w in angular_values:
                abs_w = abs(w)
                if abs_w > max_w:
                    continue
                if abs_w > self.max_steering_rate * dt + 1e-6:
                    continue
                primitives.append(
                    MotionPrimitive(
                        linear_velocity=float(v),
                        angular_velocity=float(w),
                        duration=dt,
                        kind=PrimitiveKind.FORWARD,
                    )
                )
        return primitives

    def _generate_decelerate(self) -> list[MotionPrimitive]:
        """Generate controlled-deceleration primitives.

        Returns:
            List of primitives stepping velocity toward zero.
        """
        dt = self.primitive_duration
        steps = int(self.deceleration_steps)
        max_v = min(self.max_linear_speed, self.max_linear_acceleration * dt)
        primitives: list[MotionPrimitive] = []

        for i in range(1, steps + 1):
            frac = 1.0 - i / steps
            v = max_v * frac if frac > 1e-6 else 0.0
            primitives.append(
                MotionPrimitive(
                    linear_velocity=float(v),
                    angular_velocity=0.0,
                    duration=dt,
                    kind=PrimitiveKind.DECELERATE,
                )
            )
        return primitives

    def _generate_wait(self) -> list[MotionPrimitive]:
        """Generate wait/yield primitives.

        Returns:
            Single zero-velocity primitive.
        """
        return [
            MotionPrimitive(
                linear_velocity=0.0,
                angular_velocity=0.0,
                duration=self.primitive_duration,
                kind=PrimitiveKind.WAIT,
            )
        ]

    def _generate_recenter(self) -> list[MotionPrimitive]:
        """Generate small corrective recentering primitives.

        Returns:
            List of low-speed recentering arcs.
        """
        dt = self.primitive_duration
        max_w = min(
            self.recenter_angular_max,
            self.max_angular_speed,
            self.max_steering_rate * dt,
        )
        v = min(self.linear_resolution * 0.5, self.max_linear_acceleration * dt)

        if max_w > self.angular_resolution:
            steps = max(2, int(max_w / self.angular_resolution))
            angular_values = [
                w for w in np.linspace(-max_w, max_w, steps) if abs(w) <= max_w + 1e-6
            ]
        elif max_w > 1e-6:
            angular_values = [-max_w, max_w]
        else:
            angular_values = [0.0]

        primitives: list[MotionPrimitive] = []
        for w in angular_values:
            primitives.append(
                MotionPrimitive(
                    linear_velocity=float(v),
                    angular_velocity=float(w),
                    duration=dt,
                    kind=PrimitiveKind.RECENTER,
                )
            )
        return primitives

    def _generate_reverse(self) -> list[MotionPrimitive]:
        """Generate reverse primitives (only if kinematics allow).

        Returns:
            List of reverse primitives or empty list when disabled.
        """
        if not self.allow_reverse:
            return []

        dt = self.primitive_duration
        max_v = min(self.max_linear_speed * 0.4, self.max_linear_acceleration * dt)
        primitives: list[MotionPrimitive] = []

        magnitudes = list(np.arange(self.linear_resolution, max_v + 1e-6, self.linear_resolution))
        if max_v > 1e-6 and (not magnitudes or magnitudes[-1] < max_v - 1e-6):
            magnitudes.append(max_v)
        v_values = [-v for v in magnitudes]
        for v in v_values:
            primitives.append(
                MotionPrimitive(
                    linear_velocity=float(v),
                    angular_velocity=0.0,
                    duration=dt,
                    kind=PrimitiveKind.REVERSE,
                )
            )
        return primitives

    def build(self) -> list[MotionPrimitive]:
        """Build the full primitive set respecting kinodynamic limits.

        Returns:
            List of all validated motion primitives.
        """
        primitives: list[MotionPrimitive] = (
            self._generate_forwards()
            + self._generate_decelerate()
            + self._generate_wait()
            + self._generate_recenter()
            + self._generate_reverse()
        )
        return primitives

    def count(self) -> int:
        """Return the number of primitives in the default build."""
        return len(self.build())


@dataclass
class SippKinodynamicCollisionModel:
    """Collision-feasibility checks for kinodynamic lattice primitives.

    Enforces acceleration, steering-rate, footprint (circle-circle), and
    continuous-collision (circle-segment along arcs) constraints using
    existing Robot SF geometry helpers.

    Attributes:
        robot_radius: Robot safety radius in meters.
        safety_margin: Minimum clearance margin above contact distance in meters.
        continuous_check_steps: Number of interpolated points per arc for collision.
    """

    robot_radius: float = 0.25
    safety_margin: float = 0.10
    continuous_check_steps: int = 5

    def __post_init__(self) -> None:
        """Validate collision model parameters."""
        if not (isfinite(self.robot_radius) and self.robot_radius > 0.0):
            raise ValueError("robot_radius must be finite and positive")
        if not (isfinite(self.safety_margin) and self.safety_margin >= 0.0):
            raise ValueError("safety_margin must be finite and non-negative")
        if int(self.continuous_check_steps) < 1:
            raise ValueError("continuous_check_steps must be at least 1")

    def check_circle_collision(
        self, position: np.ndarray, obstacle_center: np.ndarray, obstacle_radius: float
    ) -> bool:
        """Check if the robot at ``position`` collides with a circular obstacle.

        Args:
            position: Robot center as ``(x, y)``.
            obstacle_center: Obstacle center as ``(x, y)``.
            obstacle_radius: Obstacle safety radius in meters.

        Returns:
            ``True`` if the robot collides (clearance < safety_margin).
        """
        robot_circle = (tuple(float(x) for x in position), self.robot_radius + self.safety_margin)
        obs_circle = (tuple(float(x) for x in obstacle_center), obstacle_radius)
        return is_circle_circle_intersection(robot_circle, obs_circle)

    def check_continuous_arc_collision(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        obstacle_centers: np.ndarray,
        obstacle_radius: float,
    ) -> bool:
        """Continuous collision check by linearly interpolating a straight segment.

        Args:
            start_pos: Robot start position as ``(x, y)``.
            end_pos: Robot end position after the primitive arc.
            obstacle_centers: Obstacle centers as ``(N, 2)``.
            obstacle_radius: Obstacle safety radius in meters.

        Returns:
            ``True`` if any interpolated point collides with any obstacle.
        """
        steps = max(int(self.continuous_check_steps), 1)
        fractions = np.linspace(0.0, 1.0, steps + 1)[:, None]
        positions = start_pos + fractions * (end_pos - start_pos)
        return self._positions_collide(positions, obstacle_centers, obstacle_radius)

    def _positions_collide(
        self,
        positions: np.ndarray,
        obstacle_centers: np.ndarray,
        obstacle_radius: float,
    ) -> bool:
        """Return whether sampled robot positions intersect any obstacle circle."""
        if len(obstacle_centers) == 0:
            return False
        combined_radius = self.robot_radius + self.safety_margin + float(obstacle_radius)
        diffs = positions[:, None, :] - obstacle_centers[None, :, :]
        distances_squared = np.sum(diffs * diffs, axis=2)
        return bool(np.any(distances_squared <= combined_radius * combined_radius))

    def _unicycle_arc_positions(
        self,
        command: tuple[float, float],
        heading: float,
        duration: float,
        start_pos: np.ndarray,
    ) -> np.ndarray:
        """Sample the exact constant-unicycle arc, including its endpoints.

        Returns:
            Sampled world-frame positions from the start through the arc endpoint.
        """
        velocity, angular_velocity = command
        steps = max(int(self.continuous_check_steps), 1)
        times = np.linspace(0.0, duration, steps + 1)
        if abs(angular_velocity) < 1e-6:
            direction = np.array([math.cos(heading), math.sin(heading)])
            return start_pos + times[:, None] * velocity * direction

        headings = heading + angular_velocity * times
        dx = velocity / angular_velocity * (np.sin(headings) - math.sin(heading))
        dy = -velocity / angular_velocity * (np.cos(headings) - math.cos(heading))
        return start_pos + np.column_stack((dx, dy))

    def primitive_posture(
        self,
        command: tuple[float, float],
        heading: float,
        duration: float,
        start_pos: np.ndarray,
        obstacle_positions: np.ndarray,
        obstacle_radius: float,
    ) -> dict[str, Any]:
        """Evaluate endpoint and continuous-collision posture for a primitive arc.

        Args:
            command: ``(v, omega)`` unicycle command.
            heading: Current robot heading in radians.
            duration: Duration of the primitive in seconds.
            start_pos: Robot start position as ``(x, y)``.
            obstacle_positions: Obstacle centers as ``(N, 2)``.
            obstacle_radius: Obstacle safety radius in meters.

        Returns:
            Dictionary with ``endpoint_collides``, ``continuous_collides``,
            ``endpoint_distance``, and ``continuous_clearance`` keys.
        """
        arc_positions = self._unicycle_arc_positions(command, heading, duration, start_pos)
        end_pos = arc_positions[-1]

        endpoint_dist = float("inf")
        if len(obstacle_positions) > 0:
            diffs = obstacle_positions - end_pos
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            endpoint_dist = float(
                np.min(dists) - self.robot_radius - self.safety_margin - float(obstacle_radius)
            )

        endpoint_collides = endpoint_dist <= 0.0 if len(obstacle_positions) > 0 else False
        continuous_collides = self._positions_collide(
            arc_positions, obstacle_positions, obstacle_radius
        )
        if len(obstacle_positions) > 0:
            diffs = arc_positions[:, None, :] - obstacle_positions[None, :, :]
            distances = np.sqrt(np.sum(diffs * diffs, axis=2))
            continuous_clearance = float(
                np.min(distances) - self.robot_radius - self.safety_margin - float(obstacle_radius)
            )
        else:
            continuous_clearance = float("inf")

        return {
            "endpoint_collides": bool(endpoint_collides),
            "continuous_collides": bool(continuous_collides),
            "endpoint_distance": endpoint_dist,
            "continuous_clearance": continuous_clearance,
            "end_position": end_pos.tolist(),
        }


def _validate_config_floats(
    *,
    positive_floats: dict[str, float],
    non_negative_floats: dict[str, float],
    positive_ints: dict[str, int],
) -> None:
    """Validate SippLatticeConfig float and int fields at construction."""
    all_floats = {**positive_floats, **non_negative_floats}
    for name, value in all_floats.items():
        if not isfinite(float(value)):
            raise ValueError(f"SippLatticeConfig.{name} must be finite")
    for name in positive_floats:
        if float(all_floats[name]) <= 0.0:
            raise ValueError(f"SippLatticeConfig.{name} must be positive")
    for name in non_negative_floats:
        if float(all_floats[name]) < 0.0:
            raise ValueError(f"SippLatticeConfig.{name} must be non-negative")
    for name, value in positive_ints.items():
        if int(value) < 1:
            raise ValueError(f"SippLatticeConfig.{name} must be at least 1")


@dataclass
class SippLatticeConfig:
    """Tunable parameters for the kinodynamic state-time lattice planner."""

    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.2
    max_linear_acceleration: float = 0.8
    max_steering_rate: float = 2.0
    primitive_duration: float = 0.2
    linear_resolution: float = 0.2
    angular_resolution: float = 0.25
    allow_reverse: bool = True
    deceleration_steps: int = 4
    recenter_angular_max: float = 0.4
    robot_radius: float = 0.25
    pedestrian_radius: float = 0.30
    safety_margin: float = 0.10
    min_clearance: float = 0.55
    grid_obstacle_threshold: float = 0.5
    continuous_check_steps: int = 5
    goal_tolerance: float = 0.25
    occupancy_candidates: int = 5
    occupancy_lookahead: float = 1.0
    occupancy_weight: float = 1.2
    occupancy_angle_weight: float = 0.3
    # -- Slice 2: bounded state-time search, commitment, and forecast tuning --
    time_slot_duration: float = 0.2
    planning_horizon_slots: int = 40
    max_expansions: int = 2000
    max_planning_time_s: float = 0.05
    heuristic_weight: float = 1.5
    commitment_horizon: int = 4
    offtrack_tolerance: float = 0.5
    xy_resolution: float = 0.1
    heading_resolution: float = 0.2618
    velocity_resolution: float = 0.1
    pedestrian_forecast_horizon_s: float = 3.0
    turn_cost_weight: float = 0.1
    reverse_cost_weight: float = 0.5
    wait_cost_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values at construction."""
        _validate_config_floats(
            positive_floats={
                "max_linear_speed": self.max_linear_speed,
                "max_angular_speed": self.max_angular_speed,
                "primitive_duration": self.primitive_duration,
                "linear_resolution": self.linear_resolution,
                "angular_resolution": self.angular_resolution,
                "robot_radius": self.robot_radius,
                "pedestrian_radius": self.pedestrian_radius,
                "goal_tolerance": self.goal_tolerance,
                "occupancy_lookahead": self.occupancy_lookahead,
                "time_slot_duration": self.time_slot_duration,
                "max_planning_time_s": self.max_planning_time_s,
                "heuristic_weight": self.heuristic_weight,
                "offtrack_tolerance": self.offtrack_tolerance,
                "xy_resolution": self.xy_resolution,
                "heading_resolution": self.heading_resolution,
                "velocity_resolution": self.velocity_resolution,
                "pedestrian_forecast_horizon_s": self.pedestrian_forecast_horizon_s,
            },
            non_negative_floats={
                "max_linear_acceleration": self.max_linear_acceleration,
                "max_steering_rate": self.max_steering_rate,
                "safety_margin": self.safety_margin,
                "min_clearance": self.min_clearance,
                "recenter_angular_max": self.recenter_angular_max,
                "occupancy_weight": self.occupancy_weight,
                "occupancy_angle_weight": self.occupancy_angle_weight,
                "turn_cost_weight": self.turn_cost_weight,
                "reverse_cost_weight": self.reverse_cost_weight,
                "wait_cost_weight": self.wait_cost_weight,
            },
            positive_ints={
                "deceleration_steps": self.deceleration_steps,
                "continuous_check_steps": self.continuous_check_steps,
                "occupancy_candidates": self.occupancy_candidates,
                "planning_horizon_slots": self.planning_horizon_slots,
                "max_expansions": self.max_expansions,
                "commitment_horizon": self.commitment_horizon,
            },
        )
        threshold = float(self.grid_obstacle_threshold)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("SippLatticeConfig.grid_obstacle_threshold must be in [0.0, 1.0]")
        if float(self.heuristic_weight) < 1.0:
            raise ValueError("SippLatticeConfig.heuristic_weight must be >= 1.0 (weighted A*)")

    def to_primitive_set(self) -> SippLatticePrimitiveSet:
        """Build a primitive set from this config.

        Returns:
            Configured SippLatticePrimitiveSet.
        """
        return SippLatticePrimitiveSet(
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
            max_linear_acceleration=self.max_linear_acceleration,
            max_steering_rate=self.max_steering_rate,
            primitive_duration=self.primitive_duration,
            linear_resolution=self.linear_resolution,
            angular_resolution=self.angular_resolution,
            allow_reverse=self.allow_reverse,
            deceleration_steps=self.deceleration_steps,
            recenter_angular_max=self.recenter_angular_max,
        )

    def to_collision_model(self) -> SippKinodynamicCollisionModel:
        """Build a collision model from this config.

        Returns:
            Configured SippKinodynamicCollisionModel.
        """
        return SippKinodynamicCollisionModel(
            robot_radius=self.robot_radius,
            safety_margin=self.safety_margin,
            continuous_check_steps=self.continuous_check_steps,
        )


class SippLatticePlannerAdapter(OccupancyAwarePlannerMixin):
    """Kinodynamic state-time lattice planner adapter (Slice 1 baseline).

    Uses a score-based primitive-selector over a kinodynamic primitive set
    with continuous-collision-aware scoring.  The full SIPP search with
    persistence and time-indexed occupancy (Slice 2) extends this foundation.

    Attributes:
        config: Planner configuration.
        _primitives: Pre-built primitive set from config.
        _collision_model: Collision-feasibility model from config.
        _last_decision: Diagnostic trace of the last planning step.
    """

    def __init__(self, config: SippLatticeConfig | None = None) -> None:
        """Initialize the lattice planner with optional config overrides."""
        self.config = config or SippLatticeConfig()
        self._primitives = self.config.to_primitive_set().build()
        self._collision_model = self.config.to_collision_model()
        self._last_decision: dict[str, Any] | None = None
        self._primitive_count = len(self._primitives)

    def _extract_state(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, float]:
        """Extract robot state, active goal, and pedestrian positions.

        Returns:
            Tuple of (robot_pos, heading, speed, active_goal, pedestrian_positions, ped_radius).
        """
        robot, goal, pedestrians = self._socnav_fields(observation)
        robot = robot or {}
        goal = goal or {}
        pedestrians = pedestrians or {}

        robot_pos = self._as_1d_float(robot.get("position", [0.0, 0.0]), pad=2)[:2]
        heading = float(self._as_1d_float(robot.get("heading", [0.0]), pad=1)[0])
        speed = float(self._as_1d_float(robot.get("speed", [0.0]), pad=1)[0])

        goal_current = self._as_1d_float(goal.get("current", [0.0, 0.0]), pad=2)[:2]
        goal_next = self._as_1d_float(goal.get("next", [0.0, 0.0]), pad=2)[:2]
        active_goal = (
            goal_next
            if np.linalg.norm(goal_next - robot_pos) > float(self.config.goal_tolerance)
            else goal_current
        )

        raw_positions = np.asarray(pedestrians.get("positions", []), dtype=float)
        if raw_positions.ndim == 1 and raw_positions.size % 2 == 0:
            raw_positions = raw_positions.reshape(-1, 2)
        if raw_positions.ndim != 2 or raw_positions.shape[-1] != 2:
            raw_positions = np.zeros((0, 2), dtype=float)
        count = max(
            int(self._as_1d_float(pedestrians.get("count", [raw_positions.shape[0]]), pad=1)[0]),
            0,
        )
        pedestrian_positions = raw_positions[:count]

        ped_rad = float(self.config.pedestrian_radius)
        return robot_pos, heading, speed, active_goal, pedestrian_positions, ped_rad

    def _score_primitive(
        self,
        primitive: MotionPrimitive,
        robot_pos: np.ndarray,
        heading: float,
        goal: np.ndarray,
        pedestrian_positions: np.ndarray,
        ped_rad: float,
        observation: dict[str, Any],
    ) -> float:
        """Score one primitive for goal alignment and collision safety.

        Returns:
            Higher-is-better score or negative infinity for blocked arcs.
        """
        command = primitive.as_command()
        posture = self._collision_model.primitive_posture(
            command=command,
            heading=heading,
            duration=primitive.duration,
            start_pos=robot_pos,
            obstacle_positions=pedestrian_positions,
            obstacle_radius=ped_rad,
        )

        if posture["endpoint_collides"] or posture["continuous_collides"]:
            return float("-inf")

        end_pos = np.array(posture["end_position"], dtype=float)
        desired_heading = float(np.arctan2(goal[1] - end_pos[1], goal[0] - end_pos[0]))
        end_heading = wrap_angle_pi(heading + primitive.angular_velocity * primitive.duration)
        heading_score = float(np.cos(wrap_angle_pi(desired_heading - end_heading)))

        start_dist = float(np.linalg.norm(goal - robot_pos))
        end_dist = float(np.linalg.norm(goal - end_pos))
        progress = start_dist - end_dist

        clearance_score = min(posture["endpoint_distance"], float(self.config.min_clearance)) / max(
            float(self.config.min_clearance), 1e-6
        )
        clearance_score = max(clearance_score, 0.0)

        velocity_score = abs(primitive.linear_velocity) / max(
            float(self.config.max_linear_speed), 1e-6
        )

        grid_penalty, ped_penalty = self._path_penalty(
            robot_pos,
            end_pos - robot_pos,
            observation,
            self.config.occupancy_lookahead,
            self.config.occupancy_candidates,
        )
        if grid_penalty >= float(self.config.grid_obstacle_threshold):
            return float("-inf")

        score = (
            1.5 * heading_score
            + 1.0 * clearance_score
            + 0.3 * velocity_score
            + 1.0 * progress
            - float(self.config.occupancy_weight) * grid_penalty
            - float(self.config.occupancy_angle_weight) * ped_penalty
        )
        return score

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Select the highest-scoring kinodynamic primitive.

        Returns:
            Bounded ``(v, omega)`` command from the chosen primitive,
            or ``(0.0, 0.0)`` when at goal or all primitives blocked.
        """
        (
            robot_pos,
            heading,
            _speed,
            goal,
            pedestrian_positions,
            ped_rad,
        ) = self._extract_state(observation)

        distance_to_goal = float(np.linalg.norm(goal - robot_pos))
        if distance_to_goal <= float(self.config.goal_tolerance):
            self._last_decision = {
                "primitive_count": self._primitive_count,
                "feasible_count": 0,
                "infeasible_count": 0,
                "best_score": 0.0,
                "best_kind": "goal_reached",
                "best_command": [0.0, 0.0],
                "constraint_reason": "goal_reached",
                "distance_to_goal_m": distance_to_goal,
            }
            return 0.0, 0.0

        scores: list[tuple[float, MotionPrimitive]] = []
        for primitive in self._primitives:
            score = self._score_primitive(
                primitive=primitive,
                robot_pos=robot_pos,
                heading=heading,
                goal=goal,
                pedestrian_positions=pedestrian_positions,
                ped_rad=ped_rad,
                observation=observation,
            )
            scores.append((score, primitive))

        feasible = [(s, p) for s, p in scores if math.isfinite(s) and s > float("-inf")]
        infeasible_count = len(scores) - len(feasible)

        if feasible:
            best_score, best_primitive = max(feasible, key=lambda x: x[0])
            cmd_v, cmd_w = best_primitive.as_command()
            constraint_reason = "best_feasible_primitive"
        else:
            best_score = float("-inf")
            cmd_v, cmd_w = 0.0, 0.0
            best_primitive = None
            constraint_reason = "all_primitives_infeasible_wait"

        self._last_decision = {
            "primitive_count": self._primitive_count,
            "feasible_count": len(feasible),
            "infeasible_count": infeasible_count,
            "best_score": float(best_score) if math.isfinite(best_score) else None,
            "best_kind": best_primitive.kind.value if best_primitive else None,
            "best_command": [float(cmd_v), float(cmd_w)],
            "constraint_reason": constraint_reason,
            "distance_to_goal_m": distance_to_goal,
        }

        return float(cmd_v), float(cmd_w)

    def diagnostics(self) -> dict[str, Any]:
        """Expose most recent lattice-planning decision detail.

        Returns:
            Dictionary with last planning step metadata.
        """
        return {"last_decision": dict(self._last_decision) if self._last_decision else {}}

    def reset(self, *, seed: int | None = None) -> None:
        """Reset per-episode state."""
        del seed
        self._last_decision = None


def build_sipp_lattice_config(cfg: dict[str, Any] | None) -> SippLatticeConfig:
    """Build a SippLatticeConfig from an algorithm-config mapping.

    Returns:
        Parsed configuration using defaults for omitted parameters.
    """
    if not isinstance(cfg, dict):
        return SippLatticeConfig()
    defaults = SippLatticeConfig()

    def _get_float(key: str) -> float:
        value = cfg.get(key)
        return float(getattr(defaults, key, 0.0) if value is None else value)

    def _get_int(key: str) -> int:
        value = cfg.get(key)
        return int(getattr(defaults, key, 1) if value is None else value)

    def _get_bool(key: str) -> bool:
        v = cfg.get(key)
        if v is None:
            v = getattr(defaults, key, False)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in {"true", "1", "yes"}
        return bool(v)

    return SippLatticeConfig(
        max_linear_speed=_get_float("max_linear_speed"),
        max_angular_speed=_get_float("max_angular_speed"),
        max_linear_acceleration=_get_float("max_linear_acceleration"),
        max_steering_rate=_get_float("max_steering_rate"),
        primitive_duration=_get_float("primitive_duration"),
        linear_resolution=_get_float("linear_resolution"),
        angular_resolution=_get_float("angular_resolution"),
        allow_reverse=_get_bool("allow_reverse"),
        deceleration_steps=_get_int("deceleration_steps"),
        recenter_angular_max=_get_float("recenter_angular_max"),
        robot_radius=_get_float("robot_radius"),
        pedestrian_radius=_get_float("pedestrian_radius"),
        safety_margin=_get_float("safety_margin"),
        min_clearance=_get_float("min_clearance"),
        grid_obstacle_threshold=_get_float("grid_obstacle_threshold"),
        continuous_check_steps=_get_int("continuous_check_steps"),
        goal_tolerance=_get_float("goal_tolerance"),
        occupancy_candidates=_get_int("occupancy_candidates"),
        occupancy_lookahead=_get_float("occupancy_lookahead"),
        occupancy_weight=_get_float("occupancy_weight"),
        occupancy_angle_weight=_get_float("occupancy_angle_weight"),
        time_slot_duration=_get_float("time_slot_duration"),
        planning_horizon_slots=_get_int("planning_horizon_slots"),
        max_expansions=_get_int("max_expansions"),
        max_planning_time_s=_get_float("max_planning_time_s"),
        heuristic_weight=_get_float("heuristic_weight"),
        commitment_horizon=_get_int("commitment_horizon"),
        offtrack_tolerance=_get_float("offtrack_tolerance"),
        xy_resolution=_get_float("xy_resolution"),
        heading_resolution=_get_float("heading_resolution"),
        velocity_resolution=_get_float("velocity_resolution"),
        pedestrian_forecast_horizon_s=_get_float("pedestrian_forecast_horizon_s"),
        turn_cost_weight=_get_float("turn_cost_weight"),
        reverse_cost_weight=_get_float("reverse_cost_weight"),
        wait_cost_weight=_get_float("wait_cost_weight"),
    )


# ---------------------------------------------------------------------------
# Slice 2 (#5306): time-indexed occupancy, bounded state-time search, commitment
# ---------------------------------------------------------------------------


def _rotate_ego_velocities_to_world(velocities: np.ndarray, heading: float) -> np.ndarray:
    """Rotate ego-frame pedestrian velocities into the world frame.

    The SocNav observation stores pedestrian velocities in the robot ego frame;
    the state-time forecast propagates positions in world coordinates.

    Returns:
        World-frame velocity array with the same shape as ``velocities``.
    """
    if velocities.size == 0:
        return velocities
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    vx = cos_h * velocities[:, 0] - sin_h * velocities[:, 1]
    vy = sin_h * velocities[:, 0] + cos_h * velocities[:, 1]
    return np.column_stack((vx, vy))


@dataclass(frozen=True)
class PedestrianOccupancyForecast:
    """Time-indexed pedestrian occupancy built from planner-facing dynamic state.

    Pedestrians use constant-velocity propagation; every arc sample is checked
    at its matching time so mid-primitive crossings cannot disappear at arrival.

    Attributes:
        positions: World-frame pedestrian positions as ``(N, 2)``.
        velocities: World-frame pedestrian velocities as ``(N, 2)``.
        slot_duration: Seconds represented by one discrete time slot.
        combined_radius: Robot radius + safety margin + pedestrian radius.
        horizon_slots: Slots beyond which the forecast is not trusted.
        status: ``"ok"`` (dynamic state usable), ``"static"`` (no velocities,
            stationary assumption), or ``"failed"`` (malformed dynamic input).
    """

    positions: np.ndarray
    velocities: np.ndarray
    slot_duration: float
    combined_radius: float
    horizon_slots: int
    status: str

    @property
    def usable(self) -> bool:
        """Return whether the forecast can back planner success evidence."""
        return self.status != "failed"

    @property
    def pedestrian_count(self) -> int:
        """Return the number of forecast pedestrians."""
        return int(self.positions.shape[0])

    def arc_occupied(
        self,
        arc_positions: np.ndarray,
        start_slot: int,
        duration: float | None = None,
    ) -> bool:
        """Return whether an arc collides with the forecast at matching times.

        Args:
            arc_positions: Sampled world-frame robot positions along the arc.
            start_slot: Start slot, or queried slot when ``duration`` is omitted.
            duration: Primitive duration; samples are checked at matching times.

        Returns:
            ``True`` for a collision or an arc beyond the trusted horizon.
        """
        if self.pedestrian_count == 0:
            return False
        try:
            arc = np.asarray(arc_positions, dtype=float)
        except (TypeError, ValueError, OverflowError):
            return True
        if arc.size == 0:
            return False
        if arc.ndim != 2 or arc.shape[-1] != 2 or not np.all(np.isfinite(arc)):
            return True

        if duration is None:
            if int(start_slot) < 0 or int(start_slot) > int(self.horizon_slots):
                return True
            sample_times = np.full(arc.shape[0], float(int(start_slot)) * float(self.slot_duration))
        else:
            if not (isfinite(float(duration)) and float(duration) > 0.0):
                return True
            start_time = float(int(start_slot)) * float(self.slot_duration)
            sample_times = start_time + np.linspace(0.0, float(duration), arc.shape[0])
            trusted_horizon = float(self.horizon_slots) * float(self.slot_duration)
            if (
                int(start_slot) < 0
                or np.any(sample_times < -1e-9)
                or float(sample_times[-1]) > trusted_horizon + 1e-9
            ):
                return True

        forecast = (
            self.positions[None, :, :] + sample_times[:, None, None] * self.velocities[None, :, :]
        )
        diffs = arc[:, None, :] - forecast
        distances_squared = np.sum(diffs * diffs, axis=2)
        return bool(np.any(distances_squared <= self.combined_radius * self.combined_radius))


def _failed_forecast(
    *, slot_duration: float, combined_radius: float, horizon_slots: int
) -> PedestrianOccupancyForecast:
    """Return the empty fail-closed forecast used for malformed active state."""
    return PedestrianOccupancyForecast(
        positions=np.zeros((0, 2), dtype=float),
        velocities=np.zeros((0, 2), dtype=float),
        slot_duration=slot_duration,
        combined_radius=combined_radius,
        horizon_slots=horizon_slots,
        status="failed",
    )


def build_pedestrian_occupancy_forecast(  # noqa: C901
    *,
    positions: np.ndarray,
    velocities: Any,
    heading: float,
    config: SippLatticeConfig,
    pedestrian_radius: float,
) -> PedestrianOccupancyForecast:
    """Construct a time-indexed pedestrian forecast, failing closed on bad input.

    Missing velocities mean stationary pedestrians; malformed dynamic state is
    classified ``"failed"`` and cannot back success evidence.

    Returns:
        A :class:`PedestrianOccupancyForecast` with an explicit ``status`` flag.
    """
    slot_duration = float(config.time_slot_duration)
    combined_radius = (
        float(config.robot_radius) + float(config.safety_margin) + float(pedestrian_radius)
    )
    horizon_slots = max(
        0,
        math.floor(float(config.pedestrian_forecast_horizon_s) / slot_duration + 1e-9),
    )
    if not isfinite(float(heading)):
        return _failed_forecast(
            slot_duration=slot_duration,
            combined_radius=combined_radius,
            horizon_slots=horizon_slots,
        )
    try:
        positions = np.asarray(positions, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return _failed_forecast(
            slot_duration=slot_duration,
            combined_radius=combined_radius,
            horizon_slots=horizon_slots,
        )
    if positions.ndim == 1 and positions.size == 0:
        positions = positions.reshape(0, 2)
    elif positions.ndim == 1 and positions.size % 2 == 0:
        positions = positions.reshape(-1, 2)
    if positions.ndim != 2 or positions.shape[-1] != 2:
        return _failed_forecast(
            slot_duration=slot_duration,
            combined_radius=combined_radius,
            horizon_slots=horizon_slots,
        )
    count = positions.shape[0]

    if count == 0:
        return PedestrianOccupancyForecast(
            positions=positions,
            velocities=np.zeros((0, 2), dtype=float),
            slot_duration=slot_duration,
            combined_radius=combined_radius,
            horizon_slots=horizon_slots,
            status="static",
        )

    if not np.all(np.isfinite(positions)):
        return _failed_forecast(
            slot_duration=slot_duration,
            combined_radius=combined_radius,
            horizon_slots=horizon_slots,
        )

    try:
        raw_velocities = np.asarray(velocities if velocities is not None else [], dtype=float)
    except (TypeError, ValueError, OverflowError):
        return _failed_forecast(
            slot_duration=slot_duration,
            combined_radius=combined_radius,
            horizon_slots=horizon_slots,
        )
    if raw_velocities.size == 0:
        return PedestrianOccupancyForecast(
            positions=positions,
            velocities=np.zeros_like(positions),
            slot_duration=slot_duration,
            combined_radius=combined_radius,
            horizon_slots=horizon_slots,
            status="static",
        )

    if raw_velocities.ndim == 1 and raw_velocities.size % 2 == 0:
        raw_velocities = raw_velocities.reshape(-1, 2)
    if (
        raw_velocities.ndim != 2
        or raw_velocities.shape[-1] != 2
        or raw_velocities.shape[0] < count
        or not np.all(np.isfinite(raw_velocities))
    ):
        return _failed_forecast(
            slot_duration=slot_duration,
            combined_radius=combined_radius,
            horizon_slots=horizon_slots,
        )

    world_velocities = _rotate_ego_velocities_to_world(raw_velocities[:count], float(heading))
    return PedestrianOccupancyForecast(
        positions=positions,
        velocities=world_velocities,
        slot_duration=slot_duration,
        combined_radius=combined_radius,
        horizon_slots=horizon_slots,
        status="ok",
    )


class _ObservationInputError(ValueError):
    """Raised when an active planner observation cannot satisfy its state contract."""


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    """Convert an observation value to a finite numeric array."""  # noqa: DOC201
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _ObservationInputError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise _ObservationInputError(f"{name} must contain only finite values")
    return array


def _finite_observation_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    """Convert one fixed-size observation vector, rejecting malformed state."""  # noqa: DOC201
    flattened = np.ravel(_finite_array(value, name=name))
    if flattened.size != size:
        raise _ObservationInputError(f"{name} must contain exactly {size} values")
    return np.asarray(flattened, dtype=float)


def _finite_pedestrian_positions(value: Any) -> np.ndarray:
    """Normalize active pedestrian positions to ``(N, 2)`` or fail closed."""  # noqa: DOC201
    if value is None:
        return np.zeros((0, 2), dtype=float)
    array = _finite_array(value, name="pedestrians.positions")
    if array.size == 0:
        return np.zeros((0, 2), dtype=float)
    if array.ndim == 1 and array.size % 2 == 0:
        array = array.reshape(-1, 2)
    if array.ndim != 2 or array.shape[-1] != 2:
        raise _ObservationInputError("pedestrians.positions must have shape (N, 2)")
    return np.asarray(array, dtype=float)


@dataclass
class _SearchNode:
    """One expanded state-time lattice node for the bounded SIPP search."""

    position: np.ndarray
    heading: float
    velocity: float
    angular_velocity: float
    slot: int
    g_cost: float
    primitive: MotionPrimitive | None
    parent: _SearchNode | None = field(default=None, repr=False)


@dataclass(frozen=True)
class SippSearchResult:
    """Outcome of one bounded state-time lattice search."""

    plan: list[MotionPrimitive]
    result_type: str
    bound_termination: str
    expansions: int
    horizon_reached: int
    safe_interval_rejections: int
    chosen_cost: float | None


class SippLatticeSearch:
    """Bounded weighted-A*/SIPP search over the kinodynamic state-time lattice.

    Expands AMV-feasible primitives while rejecting dynamic/static collisions;
    hard bounds guarantee safe-wait termination.
    """

    def __init__(
        self,
        config: SippLatticeConfig,
        primitives: list[MotionPrimitive],
        collision_model: SippKinodynamicCollisionModel,
    ) -> None:
        """Initialize the search with a config, primitive set, and collision model."""
        self.config = config
        self._primitives = primitives
        self._collision_model = collision_model
        forward_reach = [p.distance_traveled for p in primitives if p.linear_velocity > 0.0]
        self._max_step_distance = max(forward_reach) if forward_reach else 0.0
        slot_ratio = float(config.primitive_duration) / float(config.time_slot_duration)
        slots_per_primitive = round(slot_ratio)
        if slots_per_primitive < 1 or not math.isclose(
            slot_ratio, slots_per_primitive, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "SippLatticeConfig.primitive_duration must be an integer multiple of "
                "time_slot_duration for state-time search"
            )
        self._slots_per_primitive = int(slots_per_primitive)

    def _state_key(self, node: _SearchNode) -> tuple[int, int, int, int, int, int]:
        """Return the discretized closed-set key for a node."""
        return (
            round(float(node.position[0]) / float(self.config.xy_resolution)),
            round(float(node.position[1]) / float(self.config.xy_resolution)),
            round(float(wrap_angle_pi(node.heading)) / float(self.config.heading_resolution)),
            round(float(node.velocity) / float(self.config.velocity_resolution)),
            round(float(node.angular_velocity) / float(self.config.heading_resolution)),
            int(node.slot),
        )

    def _transition_reachable(
        self, current_velocity: float, current_angular_velocity: float, primitive: MotionPrimitive
    ) -> bool:
        """Return whether a primitive target is reachable from the current command."""
        duration = float(primitive.duration)
        linear_delta = float(self.config.max_linear_acceleration) * duration
        angular_delta = float(self.config.max_steering_rate) * duration
        linear_min = -float(self.config.max_linear_speed) if self.config.allow_reverse else 0.0
        linear_max = float(self.config.max_linear_speed)
        target_v = float(primitive.linear_velocity)
        target_w = float(primitive.angular_velocity)
        if not (isfinite(current_velocity) and isfinite(current_angular_velocity)):
            return False
        if not (linear_min - 1e-9 <= target_v <= linear_max + 1e-9):
            return False
        if not (
            -float(self.config.max_angular_speed) - 1e-9
            <= target_w
            <= float(self.config.max_angular_speed) + 1e-9
        ):
            return False
        return bool(
            abs(target_v - float(current_velocity)) <= linear_delta + 1e-9
            and abs(target_w - float(current_angular_velocity)) <= angular_delta + 1e-9
        )

    def _heuristic(self, position: np.ndarray, goal: np.ndarray) -> float:
        """Admissible-style time-to-go heuristic toward the goal.

        Returns:
            Estimated remaining cost (in time units) to reach the goal.
        """
        distance = float(np.linalg.norm(goal - position))
        if self._max_step_distance <= 1e-9:
            return distance
        steps = distance / self._max_step_distance
        return steps * float(self.config.primitive_duration)

    def _step_cost(self, primitive: MotionPrimitive) -> float:
        """Cost of committing to one primitive (time plus shaping penalties).

        Returns:
            Positive scalar cost.
        """
        cost = float(primitive.duration)
        cost += float(self.config.turn_cost_weight) * abs(primitive.delta_yaw)
        if primitive.linear_velocity < 0.0:
            cost += float(self.config.reverse_cost_weight) * primitive.duration
        if primitive.kind is PrimitiveKind.WAIT:
            cost += float(self.config.wait_cost_weight) * primitive.duration
        return cost

    @staticmethod
    def _reconstruct(node: _SearchNode) -> list[MotionPrimitive]:
        """Walk parent pointers to build an ordered primitive plan.

        Returns:
            Primitive list from the start state to ``node``.
        """
        plan: list[MotionPrimitive] = []
        cursor: _SearchNode | None = node
        while cursor is not None and cursor.primitive is not None:
            plan.append(cursor.primitive)
            cursor = cursor.parent
        plan.reverse()
        return plan

    def search(  # noqa: C901, PLR0915
        self,
        *,
        start_pos: np.ndarray,
        start_heading: float,
        start_speed: float,
        start_angular_velocity: float = 0.0,
        goal: np.ndarray,
        forecast: PedestrianOccupancyForecast,
        static_blocked: Callable[[np.ndarray], bool] | None = None,
    ) -> SippSearchResult:
        """Run the bounded state-time search from a start state toward a goal.

        Returns:
            A :class:`SippSearchResult` classifying the outcome and any plan.
        """
        goal_tolerance = float(self.config.goal_tolerance)
        horizon_slots = min(int(self.config.planning_horizon_slots), int(forecast.horizon_slots))
        max_expansions = int(self.config.max_expansions)
        deadline = time.perf_counter() + float(self.config.max_planning_time_s)

        start_position = np.asarray(start_pos, dtype=float)
        goal_position = np.asarray(goal, dtype=float)
        if not forecast.usable:
            return SippSearchResult(
                plan=[_wait_primitive(self.config)],
                result_type="bounded_safe_wait",
                bound_termination="invalid_forecast",
                expansions=0,
                horizon_reached=0,
                safe_interval_rejections=0,
                chosen_cost=None,
            )
        if (
            start_position.shape != (2,)
            or goal_position.shape != (2,)
            or not np.all(np.isfinite(start_position))
            or not np.all(np.isfinite(goal_position))
            or not isfinite(float(start_speed))
            or not isfinite(float(start_angular_velocity))
            or not isfinite(float(start_heading))
        ):
            raise ValueError("SippLatticeSearch state and goal must be finite (x, y) values")

        start = _SearchNode(
            position=start_position,
            heading=wrap_angle_pi(float(start_heading)),
            velocity=float(start_speed),
            angular_velocity=float(start_angular_velocity),
            slot=0,
            g_cost=0.0,
            primitive=None,
        )
        counter = 0
        start_h = self._heuristic(start.position, goal_position)
        open_heap: list[tuple[float, int, _SearchNode]] = [(start_h, counter, start)]
        best_cost: dict[tuple[int, int, int, int, int, int], float] = {self._state_key(start): 0.0}
        expansions = 0
        rejections = 0
        horizon_reached = 0
        bound_termination = "open_exhausted"

        while open_heap:
            if expansions >= max_expansions:
                bound_termination = "expansions"
                break
            if time.perf_counter() > deadline:
                bound_termination = "time"
                break

            _, _, node = heapq.heappop(open_heap)
            node_key = self._state_key(node)
            if best_cost.get(node_key, math.inf) < node.g_cost:
                continue

            goal_distance = float(np.linalg.norm(goal_position - node.position))
            if goal_distance <= goal_tolerance:
                plan = self._reconstruct(node)
                return SippSearchResult(
                    plan=plan if plan else [_wait_primitive(self.config)],
                    result_type="native_plan",
                    bound_termination="goal",
                    expansions=expansions,
                    horizon_reached=horizon_reached,
                    safe_interval_rejections=rejections,
                    chosen_cost=float(node.g_cost),
                )

            if node.slot >= horizon_slots:
                bound_termination = "horizon"
                continue

            expansions += 1
            for primitive in self._primitives:
                arrival_slot = node.slot + self._slots_per_primitive
                horizon_reached = max(horizon_reached, arrival_slot)
                if arrival_slot > horizon_slots:
                    bound_termination = "horizon"
                    continue
                if not self._transition_reachable(node.velocity, node.angular_velocity, primitive):
                    continue
                arc_positions = self._collision_model._unicycle_arc_positions(
                    primitive.as_command(), node.heading, primitive.duration, node.position
                )
                end_pos = arc_positions[-1]

                if static_blocked is not None and static_blocked(arc_positions):
                    continue
                if forecast.arc_occupied(arc_positions, node.slot, primitive.duration):
                    rejections += 1
                    continue

                child = _SearchNode(
                    position=end_pos,
                    heading=wrap_angle_pi(node.heading + primitive.delta_yaw),
                    velocity=float(primitive.linear_velocity),
                    angular_velocity=float(primitive.angular_velocity),
                    slot=arrival_slot,
                    g_cost=node.g_cost + self._step_cost(primitive),
                    primitive=primitive,
                    parent=node,
                )
                child_key = self._state_key(child)
                if best_cost.get(child_key, math.inf) <= child.g_cost:
                    continue
                best_cost[child_key] = child.g_cost
                counter += 1
                f_cost = child.g_cost + float(self.config.heuristic_weight) * self._heuristic(
                    child.position, goal_position
                )
                heapq.heappush(open_heap, (f_cost, counter, child))

        if bound_termination == "open_exhausted" and horizon_reached >= horizon_slots:
            bound_termination = "horizon"
        fallback = _controlled_deceleration_primitive(
            self.config, start.velocity, start.angular_velocity
        )
        fallback_arc = self._collision_model._unicycle_arc_positions(
            fallback.as_command(), start.heading, fallback.duration, start.position
        )
        fallback_safe = (
            self._transition_reachable(start.velocity, start.angular_velocity, fallback)
            and (static_blocked is None or not static_blocked(fallback_arc))
            and not forecast.arc_occupied(fallback_arc, 0, fallback.duration)
        )
        return SippSearchResult(
            plan=[fallback] if fallback_safe else [],
            result_type=(
                (
                    "bounded_safe_wait"
                    if fallback.kind is PrimitiveKind.WAIT
                    else "bounded_safe_deceleration"
                )
                if fallback_safe
                else "bounded_emergency_stop"
            ),
            bound_termination=bound_termination,
            expansions=expansions,
            horizon_reached=horizon_reached,
            safe_interval_rejections=rejections,
            chosen_cost=None,
        )


def _wait_primitive(config: SippLatticeConfig) -> MotionPrimitive:
    """Return the canonical zero-velocity wait primitive for a config.

    Returns:
        A WAIT :class:`MotionPrimitive` with the configured primitive duration.
    """
    return MotionPrimitive(
        linear_velocity=0.0,
        angular_velocity=0.0,
        duration=float(config.primitive_duration),
        kind=PrimitiveKind.WAIT,
    )


def _controlled_deceleration_primitive(
    config: SippLatticeConfig, speed: float, angular_velocity: float
) -> MotionPrimitive:
    """Return the closest-to-zero command reachable in one primitive duration."""
    duration = float(config.primitive_duration)
    linear = math.copysign(
        max(0.0, abs(float(speed)) - float(config.max_linear_acceleration) * duration), speed
    )
    if not config.allow_reverse:
        linear = max(0.0, linear)
    angular = math.copysign(
        max(0.0, abs(float(angular_velocity)) - float(config.max_steering_rate) * duration),
        angular_velocity,
    )
    kind = PrimitiveKind.WAIT if linear == 0.0 and angular == 0.0 else PrimitiveKind.DECELERATE
    return MotionPrimitive(linear, angular, duration, kind)


class SippLatticeSearchPlannerAdapter(OccupancyAwarePlannerMixin):
    """Bounded state-time SIPP planner with multi-step commitment (Slice 2).

    Extends Slice 1 with time-indexed occupancy, bounded weighted-A* search, and
    a committed primitive sequence that replans on exhaustion or invalidation.

    This adapter is testing-only/experimental: it produces exploratory
    implementation evidence, not safety, liveness, or benchmark superiority
    claims (Slice 3 of #5306 owns outcome evaluation).

    """

    def __init__(self, config: SippLatticeConfig | None = None) -> None:
        """Initialize the search planner with optional config overrides."""
        self.config = config or SippLatticeConfig()
        self._primitives = self.config.to_primitive_set().build()
        self._collision_model = self.config.to_collision_model()
        self._search = SippLatticeSearch(self.config, self._primitives, self._collision_model)
        self._committed: list[MotionPrimitive] = []
        self._commit_index = 0
        self._last_goal: np.ndarray | None = None
        self._expected_pos: np.ndarray | None = None
        self._last_decision: dict[str, Any] | None = None

    def _extract_state(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, float, float, np.ndarray, np.ndarray, Any]:
        """Extract finite robot/goal state and raw pedestrian dynamic state."""  # noqa: DOC201
        try:
            robot, goal, pedestrians = self._socnav_fields(observation)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            raise _ObservationInputError("observation must contain valid SocNav fields") from exc
        if not all(isinstance(field, dict) for field in (robot, goal, pedestrians)):
            raise _ObservationInputError("robot, goal, and pedestrians must be mappings")

        robot_pos = _finite_observation_vector(
            robot.get("position", [0.0, 0.0]), name="robot.position", size=2
        )
        heading = float(
            _finite_observation_vector(robot.get("heading", [0.0]), name="robot.heading", size=1)[0]
        )
        speed_values = np.ravel(_finite_array(robot.get("speed", [0.0]), name="robot.speed"))
        if speed_values.size not in {1, 2}:
            raise _ObservationInputError("robot.speed must contain one or two finite values")
        speed = float(speed_values[0])
        angular_source = robot.get("angular_velocity", robot.get("omega"))
        angular_velocity = (
            float(
                _finite_observation_vector(angular_source, name="robot.angular_velocity", size=1)[0]
            )
            if angular_source is not None
            else float(speed_values[1])
            if speed_values.size == 2
            else 0.0
        )

        goal_current = _finite_observation_vector(
            goal.get("current", [0.0, 0.0]), name="goal.current", size=2
        )
        goal_next = _finite_observation_vector(
            goal.get("next", [0.0, 0.0]), name="goal.next", size=2
        )
        active_goal = (
            goal_next
            if np.linalg.norm(goal_next - robot_pos) > float(self.config.goal_tolerance)
            else goal_current
        )

        raw_positions = _finite_pedestrian_positions(pedestrians.get("positions", []))
        count_value = (
            float(raw_positions.shape[0])
            if pedestrians.get("count") is None
            else _finite_observation_vector(
                pedestrians.get("count"), name="pedestrians.count", size=1
            )[0]
        )
        if count_value < 0.0 or not count_value.is_integer():
            raise _ObservationInputError("pedestrians.count must be a non-negative integer")
        count = int(count_value)
        if count > raw_positions.shape[0]:
            raise _ObservationInputError(
                "pedestrians.count cannot exceed pedestrians.positions rows"
            )
        ped_positions = raw_positions[:count]
        ped_velocities = pedestrians.get("velocities")
        return (
            robot_pos,
            heading,
            speed,
            angular_velocity,
            active_goal,
            ped_positions,
            ped_velocities,
        )

    def _static_blocked_fn(self, observation: dict[str, Any]) -> Callable[[np.ndarray], bool]:
        """Build a footprint-inflated static-occupancy checker over an arc."""  # noqa: DOC201

        payload = self._extract_grid_payload(observation)
        if payload is None:
            return lambda _arc_positions: False
        grid, meta = payload
        if grid.ndim < 3:
            return lambda _arc_positions: True
        try:
            channel = self._grid_channel_index(meta, "obstacles")
            if channel < 0:
                channel = self._grid_channel_index(meta, "combined")
            resolution = float(self._as_1d_float(meta.get("resolution", [0.0]), pad=1)[0])
        except (TypeError, ValueError, IndexError, KeyError):
            return lambda _arc_positions: True
        if channel < 0 or channel >= grid.shape[0] or not isfinite(resolution) or resolution <= 0.0:
            return lambda _arc_positions: True

        inflation = float(self.config.robot_radius + self.config.safety_margin)
        sample_radius = inflation + 0.5 * math.sqrt(2.0) * resolution
        cell_count = math.ceil(sample_radius / resolution)
        offsets = np.asarray(
            [
                [col * resolution, row * resolution]
                for row in range(-cell_count, cell_count + 1)
                for col in range(-cell_count, cell_count + 1)
                if math.hypot(col * resolution, row * resolution) <= sample_radius + 1e-9
            ],
            dtype=float,
        )

        def _blocked(arc_positions: np.ndarray) -> bool:
            """Reject malformed or footprint-overlapping arcs."""  # noqa: DOC201
            try:
                arc = np.asarray(arc_positions, dtype=float)
                if arc.ndim != 2 or arc.shape[-1] != 2 or not np.all(np.isfinite(arc)):
                    return True
                footprint_samples = (arc[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
                return any(
                    self._grid_value(point, grid, meta, channel)
                    >= float(self.config.grid_obstacle_threshold)
                    for point in footprint_samples
                )
            except (TypeError, ValueError, IndexError, KeyError, OverflowError):
                return True
            return False

        return _blocked

    def _commitment_valid(  # noqa: C901
        self,
        robot_pos: np.ndarray,
        active_goal: np.ndarray,
        forecast: PedestrianOccupancyForecast,
        heading: float,
        speed: float,
        angular_velocity: float,
        static_blocked: Callable[[np.ndarray], bool],
    ) -> bool:
        """Return whether the committed remainder remains safe and reachable."""
        if self._commit_index >= len(self._committed):
            return False
        if self._last_goal is None:
            return False
        if float(np.linalg.norm(active_goal - self._last_goal)) > float(self.config.goal_tolerance):
            return False
        if self._expected_pos is not None:
            drift = float(np.linalg.norm(robot_pos - self._expected_pos))
            if drift > float(self.config.offtrack_tolerance):
                return False

        # Re-validate the remaining committed arcs against the fresh forecast.
        cursor = np.asarray(robot_pos, dtype=float)
        cursor_heading = float(heading)
        cursor_speed = float(speed)
        cursor_angular_velocity = float(angular_velocity)
        slot = 0
        trusted_horizon = min(int(self.config.planning_horizon_slots), int(forecast.horizon_slots))
        for primitive in self._committed[self._commit_index :]:
            arrival_slot = slot + self._search._slots_per_primitive
            if arrival_slot > trusted_horizon:
                return False
            if not self._search._transition_reachable(
                cursor_speed, cursor_angular_velocity, primitive
            ):
                return False
            arc_positions = self._collision_model._unicycle_arc_positions(
                primitive.as_command(), cursor_heading, primitive.duration, cursor
            )
            if static_blocked(arc_positions):
                return False
            if forecast.arc_occupied(arc_positions, slot, primitive.duration):
                return False
            cursor = arc_positions[-1]
            cursor_heading = wrap_angle_pi(cursor_heading + primitive.delta_yaw)
            cursor_speed = float(primitive.linear_velocity)
            cursor_angular_velocity = float(primitive.angular_velocity)
            slot = arrival_slot
        return True

    def _record(
        self,
        *,
        result_type: str,
        primitive: MotionPrimitive | None,
        command: tuple[float, float],
        distance_to_goal: float | None,
        dynamic_state: str,
        replanned: bool,
        search_result: SippSearchResult | None,
    ) -> None:
        """Store the diagnostic trace for the most recent planning cycle."""
        self._last_decision = {
            "result_type": result_type,
            "primitive_count": len(self._primitives),
            "committed_length": len(self._committed),
            "commit_index": self._commit_index,
            "primitive_kind": primitive.kind.value if primitive else None,
            "primitive_command": [float(command[0]), float(command[1])],
            "distance_to_goal_m": (None if distance_to_goal is None else float(distance_to_goal)),
            "dynamic_state": dynamic_state,
            "replanned": bool(replanned),
            "expansions": search_result.expansions if search_result else 0,
            "bound_termination": search_result.bound_termination if search_result else None,
            "horizon_reached": search_result.horizon_reached if search_result else 0,
            "safe_interval_rejections": (
                search_result.safe_interval_rejections if search_result else 0
            ),
            "chosen_cost": search_result.chosen_cost if search_result else None,
        }

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return a bounded ``(v, omega)`` command from a committed state-time plan.

        Returns:
            The next committed primitive command, or ``(0.0, 0.0)`` for a
            classified safe wait, goal-reached, or failed dynamic-input state.
        """
        try:
            (
                robot_pos,
                heading,
                speed,
                angular_velocity,
                active_goal,
                ped_positions,
                ped_velocities,
            ) = self._extract_state(observation)
        except _ObservationInputError:
            self._clear_commitment()
            self._last_goal = None
            self._record(
                result_type="failed_observation_input",
                primitive=None,
                command=(0.0, 0.0),
                distance_to_goal=None,
                dynamic_state="failed",
                replanned=False,
                search_result=None,
            )
            return 0.0, 0.0

        distance_to_goal = float(np.linalg.norm(active_goal - robot_pos))

        forecast = build_pedestrian_occupancy_forecast(
            positions=ped_positions,
            velocities=ped_velocities,
            heading=heading,
            config=self.config,
            pedestrian_radius=float(self.config.pedestrian_radius),
        )
        if not forecast.usable:
            # Fail closed: malformed dynamic state never backs planner success.
            self._clear_commitment()
            self._last_goal = np.asarray(active_goal, dtype=float)
            self._record(
                result_type="failed_dynamic_input",
                primitive=None,
                command=(0.0, 0.0),
                distance_to_goal=distance_to_goal,
                dynamic_state="failed",
                replanned=False,
                search_result=None,
            )
            return 0.0, 0.0

        if distance_to_goal <= float(self.config.goal_tolerance):
            self._clear_commitment()
            self._last_goal = np.asarray(active_goal, dtype=float)
            self._record(
                result_type="goal_reached",
                primitive=None,
                command=(0.0, 0.0),
                distance_to_goal=distance_to_goal,
                dynamic_state=forecast.status,
                replanned=False,
                search_result=None,
            )
            return 0.0, 0.0

        static_blocked = self._static_blocked_fn(observation)

        if self._commitment_valid(
            robot_pos,
            active_goal,
            forecast,
            heading,
            speed,
            angular_velocity,
            static_blocked,
        ):
            primitive = self._committed[self._commit_index]
            self._commit_index += 1
            command = primitive.as_command()
            self._expected_pos = self._primitive_endpoint(primitive, robot_pos, heading)
            self._last_goal = np.asarray(active_goal, dtype=float)
            self._record(
                result_type="committed_plan",
                primitive=primitive,
                command=command,
                distance_to_goal=distance_to_goal,
                dynamic_state=forecast.status,
                replanned=False,
                search_result=None,
            )
            return float(command[0]), float(command[1])

        # Commitment exhausted or invalidated: run a fresh bounded search.
        result = self._search.search(
            start_pos=robot_pos,
            start_heading=heading,
            start_speed=speed,
            start_angular_velocity=angular_velocity,
            goal=active_goal,
            forecast=forecast,
            static_blocked=static_blocked,
        )
        commitment_horizon = int(self.config.commitment_horizon)
        self._committed = list(result.plan[:commitment_horizon])
        self._commit_index = 0
        self._last_goal = np.asarray(active_goal, dtype=float)

        if result.result_type == "bounded_emergency_stop" or not self._committed:
            self._clear_commitment()
            self._expected_pos = np.asarray(robot_pos, dtype=float)
            self._record(
                result_type=result.result_type,
                primitive=None,
                command=(0.0, 0.0),
                distance_to_goal=distance_to_goal,
                dynamic_state=forecast.status,
                replanned=True,
                search_result=result,
            )
            return 0.0, 0.0

        primitive = self._committed[self._commit_index]
        self._commit_index += 1
        command = primitive.as_command()
        self._expected_pos = self._primitive_endpoint(primitive, robot_pos, heading)
        self._record(
            result_type=result.result_type,
            primitive=primitive,
            command=command,
            distance_to_goal=distance_to_goal,
            dynamic_state=forecast.status,
            replanned=True,
            search_result=result,
        )
        return float(command[0]), float(command[1])

    def _primitive_endpoint(
        self, primitive: MotionPrimitive, start_pos: np.ndarray, heading: float
    ) -> np.ndarray:
        """Return the world-frame endpoint of a primitive from a start state.

        Returns:
            The arc endpoint as ``(x, y)``.
        """
        arc = self._collision_model._unicycle_arc_positions(
            primitive.as_command(), float(heading), primitive.duration, np.asarray(start_pos)
        )
        return np.asarray(arc[-1], dtype=float)

    def _clear_commitment(self) -> None:
        """Drop any committed sequence and tracking state."""
        self._committed = []
        self._commit_index = 0
        self._expected_pos = None

    def diagnostics(self) -> dict[str, Any]:
        """Expose the most recent state-time planning decision detail.

        Returns:
            Dictionary with the last planning-cycle metadata.
        """
        return {"last_decision": dict(self._last_decision) if self._last_decision else {}}

    def reset(self, *, seed: int | None = None) -> None:
        """Reset per-episode commitment and diagnostic state deterministically."""
        del seed
        self._clear_commitment()
        self._last_goal = None
        self._last_decision = None


def build_sipp_lattice_search_adapter(
    algo_config: dict[str, Any] | None,
) -> SippLatticeSearchPlannerAdapter:
    """Build the Slice-2 bounded SIPP search adapter from an algorithm config.

    Returns:
        A configured :class:`SippLatticeSearchPlannerAdapter`.
    """
    return SippLatticeSearchPlannerAdapter(config=build_sipp_lattice_config(algo_config))
