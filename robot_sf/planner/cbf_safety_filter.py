"""Opt-in Control-Barrier-Function command filter for local planners.

The filter is intentionally current-state only: it consumes robot and pedestrian
positions/velocities from the planner observation, projects the nominal planar
velocity through linearized CBF half-space constraints, then converts the
projected velocity back to a unicycle ``(linear, angular)`` command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.planner.risk_dwa import _wrap_angle
from robot_sf.planner.safety_shield import ShieldDecision

ActionCommand = tuple[float, float]


@dataclass(frozen=True)
class CbfSafetyFilterConfig:
    """Configuration for the collision-cone CBF command filter."""

    enabled: bool = False
    variant: str = "collision_cone"
    alpha: float = 1.0
    safety_margin: float = 0.15
    robot_radius: float = 0.3
    pedestrian_radius: float = 0.3
    max_linear_speed: float | None = None
    max_angular_speed: float | None = None
    turn_gain: float = 2.0
    max_projection_passes: int = 3
    min_clearance_h: float = 1e-6


def build_cbf_safety_filter_config(config: dict[str, Any] | None) -> CbfSafetyFilterConfig:
    """Build a :class:`CbfSafetyFilterConfig` from an optional mapping.

    Returns:
        CBF safety-filter configuration.
    """

    raw = dict(config or {})
    if "cbf_safety_filter" in raw and isinstance(raw["cbf_safety_filter"], dict):
        raw = dict(raw["cbf_safety_filter"])
    allowed = set(CbfSafetyFilterConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown CBF safety filter config keys: {unknown}")
    cfg = CbfSafetyFilterConfig(**raw)
    variant = str(cfg.variant).strip().lower()
    if variant != "collision_cone":
        raise ValueError(
            "Only the collision_cone CBF safety-filter variant is implemented in this slice"
        )
    if cfg.alpha < 0.0:
        raise ValueError("CBF safety filter alpha must be non-negative")
    if cfg.safety_margin < 0.0:
        raise ValueError("CBF safety filter safety_margin must be non-negative")
    if cfg.max_projection_passes < 1:
        raise ValueError("CBF safety filter max_projection_passes must be >= 1")
    return cfg


@dataclass(frozen=True)
class _ObstacleState:
    position: np.ndarray
    velocity: np.ndarray
    radius: float


def _as_xy(value: Any, default: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Return a two-element float vector from common observation encodings."""

    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray(default, dtype=float)
    if arr.size < 2:
        return np.asarray(default, dtype=float)
    return arr[:2].astype(float)


def _as_float(value: Any, default: float) -> float:
    """Return a finite float from scalar/list/array payloads."""

    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return float(default)
    if arr.size == 0 or not np.isfinite(arr[0]):
        return float(default)
    return float(arr[0])


def _robot_state(
    observation: dict[str, Any], config: CbfSafetyFilterConfig
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Extract robot position, velocity, heading, and radius from observation.

    Returns:
        Position, velocity, heading, and robot radius.
    """

    robot = observation.get("robot")
    if isinstance(robot, dict):
        position = _as_xy(robot.get("position"))
        velocity = _as_xy(robot.get("velocity"))
        heading = _as_float(robot.get("heading"), math.atan2(velocity[1], velocity[0]))
        radius = _as_float(robot.get("radius"), config.robot_radius)
        return position, velocity, heading, radius

    drive_state = observation.get("drive_state")
    if drive_state is not None:
        arr = np.asarray(drive_state, dtype=float).reshape(-1)
        if arr.size >= 5:
            position = arr[:2].astype(float)
            heading = float(arr[2])
            speed = float(arr[3])
            velocity = np.asarray([math.cos(heading) * speed, math.sin(heading) * speed])
            return position, velocity, heading, float(config.robot_radius)

    return np.zeros(2, dtype=float), np.zeros(2, dtype=float), 0.0, float(config.robot_radius)


def _obstacles(observation: dict[str, Any], config: CbfSafetyFilterConfig) -> list[_ObstacleState]:
    """Extract dynamic obstacle states from supported observation payloads.

    Returns:
        Dynamic obstacle states visible in the observation.
    """

    obstacles: list[_ObstacleState] = []
    agents = observation.get("agents")
    if isinstance(agents, list):
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            obstacles.append(
                _ObstacleState(
                    position=_as_xy(agent.get("position")),
                    velocity=_as_xy(agent.get("velocity")),
                    radius=_as_float(agent.get("radius"), config.pedestrian_radius),
                )
            )

    pedestrians = observation.get("pedestrians")
    if isinstance(pedestrians, dict):
        positions = np.asarray(pedestrians.get("positions", []), dtype=float).reshape(-1, 2)
        velocities_raw = pedestrians.get("velocities", np.zeros_like(positions))
        velocities = np.asarray(velocities_raw, dtype=float)
        if velocities.size == 0:
            velocities = np.zeros_like(positions)
        velocities = velocities.reshape(-1, 2)
        radius = _as_float(pedestrians.get("radius"), config.pedestrian_radius)
        for idx, position in enumerate(positions):
            velocity = velocities[idx] if idx < len(velocities) else np.zeros(2, dtype=float)
            obstacles.append(
                _ObstacleState(
                    position=np.asarray(position, dtype=float),
                    velocity=np.asarray(velocity, dtype=float),
                    radius=radius,
                )
            )
    return obstacles


class CollisionConeCbfSafetyFilter:
    """Project nominal unicycle commands through current-state CBF constraints."""

    def __init__(self, config: CbfSafetyFilterConfig | None = None) -> None:
        """Initialize the filter and empty decision counters."""

        self.config = config or CbfSafetyFilterConfig()
        self._stats = {
            "decision_count": 0,
            "feasible_count": 0,
            "projected_count": 0,
            "fallback_count": 0,
            "last_decision": None,
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-serializable filter diagnostics."""

        return {
            "schema_version": "cbf-safety-filter-stats.v1",
            "variant": self.config.variant,
            **self._stats,
        }

    def filter_command(
        self, observation: dict[str, Any], proposed_command: ActionCommand
    ) -> ShieldDecision:
        """Return a shield decision for ``proposed_command`` in ``observation``."""

        if not self.config.enabled:
            return ShieldDecision(
                proposed_action=proposed_command,
                filtered_action=proposed_command,
                decision_label="cbf_disabled",
                intervention_reason="cbf_safety_filter_disabled",
                prediction_source="current_state",
            )

        robot_pos, _robot_vel, heading, robot_radius = _robot_state(observation, self.config)
        obstacles = _obstacles(observation, self.config)
        nominal = self._command_to_velocity(proposed_command, heading)
        selected = nominal.copy()
        constraints = self._constraints(robot_pos, robot_radius, obstacles)
        min_margin_before = self._min_margin(nominal, constraints)
        violated_before = min_margin_before < 0.0

        for _ in range(int(self.config.max_projection_passes)):
            changed = False
            for normal, lower_bound, _label in constraints:
                margin = float(np.dot(normal, selected) - lower_bound)
                if margin >= 0.0:
                    continue
                denom = float(np.dot(normal, normal))
                if denom <= 1e-12:
                    continue
                selected = (
                    selected + ((lower_bound - float(np.dot(normal, selected))) / denom) * normal
                )
                changed = True
            if not changed:
                break

        selected = self._cap_velocity(selected)
        min_margin_after = self._min_margin(selected, constraints)
        filtered = self._velocity_to_command(selected, heading, proposed_command)
        fallback = min_margin_after < -1e-9
        label = "cbf_projected" if violated_before else "cbf_feasible"
        if fallback:
            label = "cbf_best_effort"
        violated = tuple(
            label
            for normal, lower_bound, label in constraints
            if np.dot(normal, nominal) < lower_bound
        )
        decision = ShieldDecision(
            proposed_action=(float(proposed_command[0]), float(proposed_command[1])),
            filtered_action=filtered,
            decision_label=label,
            intervention_reason=(
                "collision_cone_cbf_projection"
                if violated_before
                else "proposed_command_satisfies_collision_cone_cbf"
            ),
            violated_constraints=violated,
            prediction_source="current_state",
            fallback_controller_state={
                "filter": "CollisionConeCbfSafetyFilter",
                "variant": self.config.variant,
                "fallback": fallback,
            },
            proposed_evaluation={"min_cbf_margin": float(min_margin_before)},
            selected_evaluation={"min_cbf_margin": float(min_margin_after)},
        )
        self._record_decision(decision, fallback=fallback)
        return decision

    def _command_to_velocity(self, command: ActionCommand, heading: float) -> np.ndarray:
        linear = float(command[0])
        if self.config.max_linear_speed is not None:
            linear = float(
                np.clip(linear, -self.config.max_linear_speed, self.config.max_linear_speed)
            )
        return np.asarray([math.cos(heading) * linear, math.sin(heading) * linear], dtype=float)

    def _velocity_to_command(
        self,
        velocity: np.ndarray,
        heading: float,
        proposed_command: ActionCommand,
    ) -> ActionCommand:
        speed = float(np.linalg.norm(velocity))
        if speed <= 1e-12:
            linear = 0.0
            angular = 0.0
        else:
            heading_unit = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
            linear = max(0.0, float(np.dot(velocity, heading_unit)))
            target_heading = math.atan2(float(velocity[1]), float(velocity[0]))
            angular = float(proposed_command[1]) + float(self.config.turn_gain) * _wrap_angle(
                target_heading - heading
            )
        if self.config.max_linear_speed is not None:
            linear = float(np.clip(linear, 0.0, self.config.max_linear_speed))
        if self.config.max_angular_speed is not None:
            angular = float(
                np.clip(angular, -self.config.max_angular_speed, self.config.max_angular_speed)
            )
        return float(linear), float(angular)

    def _constraints(
        self,
        robot_pos: np.ndarray,
        robot_radius: float,
        obstacles: list[_ObstacleState],
    ) -> list[tuple[np.ndarray, float, str]]:
        constraints: list[tuple[np.ndarray, float, str]] = []
        for idx, obstacle in enumerate(obstacles):
            relative_pos = np.asarray(obstacle.position - robot_pos, dtype=float)
            distance_sq = float(np.dot(relative_pos, relative_pos))
            combined_radius = float(robot_radius + obstacle.radius + self.config.safety_margin)
            h_value = distance_sq - combined_radius * combined_radius
            h_value = max(h_value, -abs(self.config.min_clearance_h))
            normal = -2.0 * relative_pos
            lower_bound = float(
                -self.config.alpha * h_value - 2.0 * np.dot(relative_pos, obstacle.velocity)
            )
            constraints.append((normal, lower_bound, f"collision_cone_cbf_agent_{idx}"))
        return constraints

    def _cap_velocity(self, velocity: np.ndarray) -> np.ndarray:
        if self.config.max_linear_speed is None:
            return velocity
        speed = float(np.linalg.norm(velocity))
        if speed <= float(self.config.max_linear_speed) or speed <= 1e-12:
            return velocity
        return velocity * (float(self.config.max_linear_speed) / speed)

    @staticmethod
    def _min_margin(
        velocity: np.ndarray, constraints: list[tuple[np.ndarray, float, str]]
    ) -> float:
        if not constraints:
            return float("inf")
        return min(
            float(np.dot(normal, velocity) - lower_bound) for normal, lower_bound, _ in constraints
        )

    def _record_decision(self, decision: ShieldDecision, *, fallback: bool) -> None:
        self._stats["decision_count"] = int(self._stats["decision_count"]) + 1
        if decision.decision_label == "cbf_feasible":
            self._stats["feasible_count"] = int(self._stats["feasible_count"]) + 1
        elif decision.decision_label == "cbf_projected":
            self._stats["projected_count"] = int(self._stats["projected_count"]) + 1
        if fallback:
            self._stats["fallback_count"] = int(self._stats["fallback_count"]) + 1
        self._stats["last_decision"] = decision.to_metadata()


class CbfSafetyFilterPlannerWrapper:
    """Planner wrapper that applies :class:`CollisionConeCbfSafetyFilter` after ``plan``."""

    def __init__(self, planner: Any, config: CbfSafetyFilterConfig | None = None) -> None:
        """Wrap a planner exposing ``plan(observation) -> (linear, angular)``."""

        self.planner = planner
        self.filter = CollisionConeCbfSafetyFilter(config)
        self.last_decision: ShieldDecision | None = None

    def plan(self, observation: dict[str, Any]) -> ActionCommand:
        """Return the planner command, filtered when enabled."""

        command = self.planner.plan(observation)
        if not self.filter.config.enabled:
            return command
        decision = self.filter.filter_command(
            observation,
            (float(command[0]), float(command[1])),
        )
        self.last_decision = decision
        return decision.filtered_action

    def diagnostics(self) -> dict[str, Any]:
        """Return wrapped planner and filter diagnostics when available."""

        payload = {"cbf_safety_filter": self.filter.diagnostics()}
        diagnostics = getattr(self.planner, "diagnostics", None)
        if callable(diagnostics):
            payload["wrapped_planner"] = diagnostics()
        return payload

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Forward reset to wrapped planner when supported.

        Returns:
            Wrapped planner reset result when available, otherwise ``None``.
        """

        reset = getattr(self.planner, "reset", None)
        if callable(reset):
            return reset(*args, **kwargs)
        return None

    def close(self) -> None:
        """Forward close to wrapped planner when supported."""

        close = getattr(self.planner, "close", None)
        if callable(close):
            close()


__all__ = [
    "CbfSafetyFilterConfig",
    "CbfSafetyFilterPlannerWrapper",
    "CollisionConeCbfSafetyFilter",
    "build_cbf_safety_filter_config",
]
