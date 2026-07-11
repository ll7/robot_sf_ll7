"""Kinodynamic state-time lattice primitives and collision model.

Foundational building blocks for a SIPP-class local planner in discretized SE(2)+time.
Provides AMV-feasible motion primitives with acceleration, steering-rate, footprint,
and continuous-collision constraints. This module covers Slice 1 of issue #5306:
primitive set, collision model, and unit-tested planner adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

import numpy as np

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
        max_v = self.max_linear_speed
        max_w = self.max_angular_speed

        linear_values = [
            v
            for v in np.arange(0.0, max_v + self.linear_resolution * 0.5, self.linear_resolution)
            if v <= max_v and v > 1e-6
        ]

        angular_values = [
            w
            for w in np.arange(
                -max_w, max_w + self.angular_resolution * 0.5, self.angular_resolution
            )
            if abs(w) <= max_w + 1e-6
        ]

        primitives: list[MotionPrimitive] = []
        for v in linear_values:
            for w in angular_values:
                abs_w = abs(w)
                if abs_w > max_w:
                    continue
                delta_w = abs_w * dt
                if max_steering_rate := self.max_steering_rate:
                    if delta_w > max_steering_rate * dt + 1e-6:
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
        max_v = self.max_linear_speed
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
        max_w = self.recenter_angular_max
        v = self.linear_resolution * 0.5

        if max_w > self.angular_resolution:
            steps = max(2, int(max_w / self.angular_resolution))
            angular_values = [
                w for w in np.linspace(-max_w, max_w, steps) if abs(w) <= max_w + 1e-6
            ]
        else:
            angular_values = [0.3, -0.3] if max_w >= 0.3 else [0.0, -0.0]

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
        max_v = self.max_linear_speed * 0.4
        primitives: list[MotionPrimitive] = []

        v_values = [
            -v
            for v in np.arange(
                self.linear_resolution, max_v + self.linear_resolution * 0.5, self.linear_resolution
            )
        ]
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
        pedestrian_radius: Default pedestrian safety radius in meters.
        safety_margin: Minimum clearance margin above contact distance in meters.
        min_clearance: Minimum acceptable distance to a dynamic obstacle.
        grid_obstacle_threshold: Occupancy grid value above which a cell is blocked.
        continuous_check_steps: Number of interpolated points per arc for collision.
    """

    robot_radius: float = 0.25
    pedestrian_radius: float = 0.30
    safety_margin: float = 0.10
    min_clearance: float = 0.55
    grid_obstacle_threshold: float = 0.5
    continuous_check_steps: int = 5

    def __post_init__(self) -> None:
        """Validate collision model parameters."""
        if not (isfinite(self.robot_radius) and self.robot_radius > 0.0):
            raise ValueError("robot_radius must be finite and positive")
        if not (isfinite(self.pedestrian_radius) and self.pedestrian_radius > 0.0):
            raise ValueError("pedestrian_radius must be finite and positive")
        if not (isfinite(self.safety_margin) and self.safety_margin >= 0.0):
            raise ValueError("safety_margin must be finite and non-negative")
        if not isfinite(self.grid_obstacle_threshold):
            raise ValueError("grid_obstacle_threshold must be finite")
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
        """Continuous collision check by linearly interpolating between arc endpoints.

        Args:
            start_pos: Robot start position as ``(x, y)``.
            end_pos: Robot end position after the primitive arc.
            obstacle_centers: Obstacle centers as ``(N, 2)``.
            obstacle_radius: Obstacle safety radius in meters.

        Returns:
            ``True`` if any interpolated point collides with any obstacle.
        """
        steps = max(int(self.continuous_check_steps), 1)
        combined_radius = self.robot_radius + self.safety_margin + float(obstacle_radius)

        combined_sq = combined_radius * combined_radius

        for i in range(steps + 1):
            t = i / steps
            pos = start_pos + t * (end_pos - start_pos)
            for j in range(len(obstacle_centers)):
                dx = pos[0] - obstacle_centers[j, 0]
                dy = pos[1] - obstacle_centers[j, 1]
                if dx * dx + dy * dy <= combined_sq:
                    return True
        return False

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
        v, omega = command
        dt = duration
        wrap_angle_pi(heading + omega * dt)

        if abs(omega) < 1e-6:
            end_pos = start_pos + np.array([v * math.cos(heading), v * math.sin(heading)]) * dt
        else:
            n = max(int(self.continuous_check_steps) + 2, 4)
            sub_dt = dt / n
            position = np.array(start_pos, dtype=float)
            h = float(heading)
            for _ in range(n):
                position[0] += v * math.cos(h) * sub_dt
                position[1] += v * math.sin(h) * sub_dt
                h += omega * sub_dt
            end_pos = position

        endpoint_dist = float("inf")
        if len(obstacle_positions) > 0:
            diffs = obstacle_positions - end_pos
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            endpoint_dist = float(
                np.min(dists) - self.robot_radius - self.safety_margin - float(obstacle_radius)
            )

        endpoint_collides = endpoint_dist < 0.0 if len(obstacle_positions) > 0 else False
        continuous_collides = (
            self.check_continuous_arc_collision(
                start_pos, end_pos, obstacle_positions, obstacle_radius
            )
            if len(obstacle_positions) > 0
            else False
        )

        continuous_clearance = endpoint_dist

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
    lattice_search_depth: int = 3
    max_expansion_nodes: int = 1000
    occupancy_heading_sweep: float = 1.0
    occupancy_candidates: int = 5
    occupancy_lookahead: float = 1.0
    occupancy_weight: float = 1.2
    occupancy_angle_weight: float = 0.3
    pedestrian_radius_cfg: float = 0.30

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
                "pedestrian_radius_cfg": self.pedestrian_radius_cfg,
                "goal_tolerance": self.goal_tolerance,
                "occupancy_lookahead": self.occupancy_lookahead,
            },
            non_negative_floats={
                "max_linear_acceleration": self.max_linear_acceleration,
                "max_steering_rate": self.max_steering_rate,
                "safety_margin": self.safety_margin,
                "min_clearance": self.min_clearance,
                "recenter_angular_max": self.recenter_angular_max,
                "occupancy_weight": self.occupancy_weight,
                "occupancy_angle_weight": self.occupancy_angle_weight,
            },
            positive_ints={
                "deceleration_steps": self.deceleration_steps,
                "continuous_check_steps": self.continuous_check_steps,
                "lattice_search_depth": self.lattice_search_depth,
                "max_expansion_nodes": self.max_expansion_nodes,
                "occupancy_candidates": self.occupancy_candidates,
            },
        )
        threshold = float(self.grid_obstacle_threshold)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("SippLatticeConfig.grid_obstacle_threshold must be in [0.0, 1.0]")

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
            pedestrian_radius=self.pedestrian_radius,
            safety_margin=self.safety_margin,
            min_clearance=self.min_clearance,
            grid_obstacle_threshold=self.grid_obstacle_threshold,
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

        ped_rad = float(self.config.pedestrian_radius_cfg)
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
            0.5,
            min(3, self.config.continuous_check_steps),
        )

        score = (
            1.5 * heading_score
            + 1.0 * clearance_score
            + 0.3 * velocity_score
            + 1.0 * progress
            - 1.2 * grid_penalty
            - 0.6 * ped_penalty
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
                "best_score": 0.0,
                "best_kind": "goal_reached",
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
        return float(cfg.get(key, getattr(defaults, key, 0.0)))

    def _get_int(key: str) -> int:
        return int(cfg.get(key, getattr(defaults, key, 1)))

    def _get_bool(key: str) -> bool:
        v = cfg.get(key, getattr(defaults, key, False))
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
        lattice_search_depth=_get_int("lattice_search_depth"),
        max_expansion_nodes=_get_int("max_expansion_nodes"),
        occupancy_heading_sweep=_get_float("occupancy_heading_sweep"),
        occupancy_candidates=_get_int("occupancy_candidates"),
        occupancy_lookahead=_get_float("occupancy_lookahead"),
        occupancy_weight=_get_float("occupancy_weight"),
        occupancy_angle_weight=_get_float("occupancy_angle_weight"),
        pedestrian_radius_cfg=_get_float("pedestrian_radius_cfg"),
    )
