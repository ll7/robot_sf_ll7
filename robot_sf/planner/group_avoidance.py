"""TAGA-like group-avoidance wrapper for scenario-declared social groups.

The adapter is diagnostic and opt-in. It uses group metadata declared on the
map to steer a goal-like command toward a tangent subgoal when the robot is near
or inside a group's o-space. It does not claim collision-safety improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, isfinite, sin
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon

from robot_sf.common.math_utils import wrap_angle_pi

CLAIM_BOUNDARY = "diagnostic group-space avoidance only; not a collision-safety guarantee"
SCHEMA_VERSION = "taga-like-group-avoidance.v1"


@dataclass(frozen=True)
class GroupAvoidanceConfig:
    """Configuration for the TAGA-like tangent-subgoal wrapper."""

    wrapped_algo: str = "goal"
    safety_margin_m: float = 0.4
    tangent_clearance_m: float = 0.2
    tangent_side: str = "auto"
    trigger_mode: str = "boundary_clearance"
    max_speed: float = 1.0


def build_group_avoidance_config(config: dict[str, Any] | None) -> GroupAvoidanceConfig:
    """Build and validate group-avoidance wrapper configuration.

    Returns:
        GroupAvoidanceConfig: Normalized wrapper configuration.
    """

    payload = dict(config or {})
    wrapped_algo = str(payload.get("wrapped_algo", "goal")).strip().lower()
    if wrapped_algo not in {"goal", "simple", "goal_policy", "simple_policy"}:
        raise ValueError("taga_group_avoidance supports only wrapped_algo='goal' in this slice")

    tangent_side = str(payload.get("tangent_side", "auto")).strip().lower()
    if tangent_side not in {"auto", "left", "right"}:
        raise ValueError("tangent_side must be one of: auto, left, right")

    trigger_mode = str(payload.get("trigger_mode", "boundary_clearance")).strip().lower()
    if trigger_mode != "boundary_clearance":
        raise ValueError("trigger_mode must be 'boundary_clearance'")

    cfg = GroupAvoidanceConfig(
        wrapped_algo="goal",
        safety_margin_m=float(payload.get("safety_margin_m", 0.4)),
        tangent_clearance_m=float(payload.get("tangent_clearance_m", 0.2)),
        tangent_side=tangent_side,
        trigger_mode=trigger_mode,
        max_speed=float(payload.get("max_speed", 1.0)),
    )
    if not isfinite(cfg.safety_margin_m) or cfg.safety_margin_m < 0.0:
        raise ValueError("safety_margin_m must be finite and non-negative")
    if not isfinite(cfg.tangent_clearance_m) or cfg.tangent_clearance_m < 0.0:
        raise ValueError("tangent_clearance_m must be finite and non-negative")
    if not isfinite(cfg.max_speed) or cfg.max_speed < 0.0:
        raise ValueError("max_speed must be finite and non-negative")
    return cfg


class TangentSubgoalGroupAvoidanceAdapter:
    """Thin TAGA-like tangent-subgoal wrapper around a goal command."""

    def __init__(self, config: GroupAvoidanceConfig | None = None) -> None:
        """Initialize adapter state and runtime diagnostics."""

        self.config = config or GroupAvoidanceConfig()
        self._group_specs: list[dict[str, Any]] = []
        self._trigger_count = 0
        self._last_selected_group_id: str | None = None
        self._last_subgoal: tuple[float, float] | None = None

    def bind_env(self, env: Any) -> None:
        """Cache JSON-safe social-group geometry from the benchmark environment."""

        simulator = getattr(env, "simulator", None)
        self._group_specs = _group_specs_from_source(simulator)

    def diagnostics(self) -> dict[str, Any]:
        """Return episode diagnostics for result provenance."""

        return {
            "group_avoidance": {
                "schema_version": SCHEMA_VERSION,
                "diagnostic_only": True,
                "wrapped_algo": self.config.wrapped_algo,
                "trigger_count": self._trigger_count,
                "last_selected_group_id": self._last_selected_group_id,
                "last_subgoal": list(self._last_subgoal)
                if self._last_subgoal is not None
                else None,
                "group_count": len(self._group_specs),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        }

    def plan(self, obs: dict[str, Any]) -> tuple[float, float]:
        """Return a goal-like command, detouring to a tangent subgoal near groups."""

        robot_pos, heading, goal_pos = _extract_robot_goal(obs)
        selected = self._nearest_triggering_group(robot_pos)
        if selected is None:
            self._last_selected_group_id = None
            self._last_subgoal = None
            return _goal_command(robot_pos, heading, goal_pos, self.config.max_speed)

        subgoal = self._tangent_subgoal(
            robot_pos=robot_pos,
            goal_pos=goal_pos,
            centroid=selected.centroid,
            effective_radius=selected.radius
            + self.config.safety_margin_m
            + self.config.tangent_clearance_m,
        )
        self._trigger_count += 1
        self._last_selected_group_id = selected.group_id
        self._last_subgoal = (float(subgoal[0]), float(subgoal[1]))
        return _goal_command(robot_pos, heading, subgoal, self.config.max_speed)

    def _nearest_triggering_group(self, robot_pos: np.ndarray) -> _GroupGeometry | None:
        nearest: _GroupGeometry | None = None
        nearest_clearance = float("inf")
        for spec in self._group_specs:
            group = _GroupGeometry.from_spec(spec)
            if group is None:
                continue
            clearance = group.boundary_clearance(robot_pos)
            if clearance < nearest_clearance:
                nearest = group
                nearest_clearance = clearance
        if nearest is None or nearest_clearance > self.config.safety_margin_m:
            return None
        return nearest

    def _tangent_subgoal(
        self,
        *,
        robot_pos: np.ndarray,
        goal_pos: np.ndarray,
        centroid: np.ndarray,
        effective_radius: float,
    ) -> np.ndarray:
        vec = robot_pos - centroid
        distance = float(np.linalg.norm(vec))
        radius = max(float(effective_radius), 1e-6)
        if distance <= radius + 1e-9:
            direction = goal_pos - centroid
            if float(np.linalg.norm(direction)) < 1e-9:
                direction = np.array([1.0, 0.0], dtype=float)
            unit = _unit(direction)
            left = np.array([-unit[1], unit[0]], dtype=float)
            right = -left
            side_vec = left if self.config.tangent_side != "right" else right
            return centroid + side_vec * radius

        unit_to_robot = vec / distance
        perp = np.array([-unit_to_robot[1], unit_to_robot[0]], dtype=float)
        alpha = acos(float(np.clip(radius / distance, -1.0, 1.0)))
        left = centroid + radius * (cos(alpha) * unit_to_robot + sin(alpha) * perp)
        right = centroid + radius * (cos(alpha) * unit_to_robot - sin(alpha) * perp)
        if self.config.tangent_side == "left":
            return left
        if self.config.tangent_side == "right":
            return right
        left_score = float(np.linalg.norm(left - goal_pos))
        right_score = float(np.linalg.norm(right - goal_pos))
        return left if left_score <= right_score else right


@dataclass(frozen=True)
class _GroupGeometry:
    group_id: str | None
    centroid: np.ndarray
    radius: float
    polygon: Polygon | None

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> _GroupGeometry | None:
        try:
            centroid = np.asarray(spec.get("centroid"), dtype=float).reshape(-1)[:2]
            radius = float(spec.get("radius", 0.0))
        except (TypeError, ValueError):
            return None
        if centroid.size < 2 or not np.all(np.isfinite(centroid)) or not isfinite(radius):
            return None
        poly_points = spec.get("o_space_polygon")
        polygon = None
        if isinstance(poly_points, list) and len(poly_points) >= 3:
            try:
                polygon = Polygon([(float(x), float(y)) for x, y in poly_points])
            except (TypeError, ValueError):
                polygon = None
            if polygon is not None and (polygon.is_empty or not polygon.is_valid):
                polygon = None
        return cls(
            group_id=str(spec.get("group_id")) if spec.get("group_id") is not None else None,
            centroid=centroid.astype(float),
            radius=max(radius, 0.0),
            polygon=polygon,
        )

    def boundary_clearance(self, robot_pos: np.ndarray) -> float:
        """Return signed clearance from group o-space boundary."""

        if self.polygon is not None:
            point = Point(float(robot_pos[0]), float(robot_pos[1]))
            distance = float(self.polygon.exterior.distance(point))
            return -distance if self.polygon.contains(point) else distance
        return float(np.linalg.norm(robot_pos - self.centroid) - self.radius)


def _extract_robot_goal(obs: dict[str, Any]) -> tuple[np.ndarray, float, np.ndarray]:
    robot = obs.get("robot") if isinstance(obs.get("robot"), dict) else {}
    goal = obs.get("goal") if isinstance(obs.get("goal"), dict) else {}
    robot_pos = np.asarray(
        robot.get("position", obs.get("robot_position", [0.0, 0.0])), dtype=float
    )
    heading_source = robot.get("heading", obs.get("robot_heading", [0.0]))
    heading = float(np.asarray(heading_source, dtype=float).reshape(-1)[0])
    goal_pos = np.asarray(goal.get("current", obs.get("goal_current", [0.0, 0.0])), dtype=float)
    return robot_pos.reshape(-1)[:2], heading, goal_pos.reshape(-1)[:2]


def _group_specs_from_source(source: Any) -> list[dict[str, Any]]:
    groups = getattr(source, "social_groups", None) or []
    specs: list[dict[str, Any]] = []
    for group in groups:
        as_spec = getattr(group, "as_spec", None)
        if callable(as_spec):
            specs.append(as_spec())
        elif isinstance(group, dict):
            specs.append(dict(group))
    return specs


def _goal_command(
    robot_pos: np.ndarray,
    heading: float,
    goal_pos: np.ndarray,
    max_speed: float,
) -> tuple[float, float]:
    vec = goal_pos - robot_pos
    distance = float(np.linalg.norm(vec))
    if distance < 1e-6:
        return 0.0, 0.0
    desired_heading = atan2(float(vec[1]), float(vec[0]))
    heading_error = wrap_angle_pi(desired_heading - heading)
    angular = float(np.clip(heading_error, -1.0, 1.0))
    linear = float(np.clip(distance, 0.0, max_speed * max(0.0, 1.0 - abs(heading_error) / np.pi)))
    return linear, angular


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return vec / norm


__all__ = [
    "CLAIM_BOUNDARY",
    "SCHEMA_VERSION",
    "GroupAvoidanceConfig",
    "TangentSubgoalGroupAvoidanceAdapter",
    "build_group_avoidance_config",
]
