"""Opt-in Control-Barrier-Function command filter for local planners.

The filter is intentionally current-state only: it consumes robot and pedestrian
positions/velocities from the planner observation, projects the nominal planar
velocity through linearized CBF half-space constraints, then converts the
projected velocity back to a unicycle ``(linear, angular)`` command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any

import numpy as np

from robot_sf.planner.risk_dwa import _wrap_angle
from robot_sf.planner.safety_shield import ShieldDecision

ActionCommand = tuple[float, float]


@dataclass(frozen=True)
class CbfSafetyFilterConfig:
    """Configuration for collision-cone or DPCBF command filters."""

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
    dpcbf_lambda_gain: float = 1.0
    dpcbf_mu_gain: float = 1.0
    relative_speed_epsilon: float = 1e-6
    dpcbf_grid_samples: int = 161


def _validate_dpcbf_config_bounds(cfg: CbfSafetyFilterConfig) -> None:
    """Fail closed for malformed Dynamic Parabolic CBF tuning values."""

    if cfg.dpcbf_lambda_gain < 0.0:
        raise ValueError("CBF safety filter dpcbf_lambda_gain must be non-negative")
    if cfg.dpcbf_mu_gain < 0.0:
        raise ValueError("CBF safety filter dpcbf_mu_gain must be non-negative")
    if cfg.relative_speed_epsilon <= 0.0:
        raise ValueError("CBF safety filter relative_speed_epsilon must be positive")
    if cfg.dpcbf_grid_samples < 3:
        raise ValueError("CBF safety filter dpcbf_grid_samples must be >= 3")


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
    if variant not in {
        "collision_cone",
        "collision_cone_cbf_v1",
        "dynamic_parabolic",
        "dynamic_parabolic_cbf_v1",
    }:
        raise ValueError("CBF safety-filter variant must be collision_cone or dynamic_parabolic")
    if cfg.alpha < 0.0:
        raise ValueError("CBF safety filter alpha must be non-negative")
    if cfg.safety_margin < 0.0:
        raise ValueError("CBF safety filter safety_margin must be non-negative")
    if cfg.max_projection_passes < 1:
        raise ValueError("CBF safety filter max_projection_passes must be >= 1")
    _validate_dpcbf_config_bounds(cfg)
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


class DynamicParabolicCbfSafetyFilter(CollisionConeCbfSafetyFilter):
    """Apply the versioned Dynamic Parabolic CBF scalar-speed projection."""

    def filter_command(
        self, observation: dict[str, Any], proposed_command: ActionCommand
    ) -> ShieldDecision:
        """Return DPCBF shield decision for ``proposed_command`` in ``observation``."""

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
        max_linear = (
            float(self.config.max_linear_speed)
            if self.config.max_linear_speed is not None
            else max(2.0, abs(float(proposed_command[0])))
        )
        context = CBFFilterContext(
            robot_position_m=(float(robot_pos[0]), float(robot_pos[1])),
            robot_heading_rad=float(heading),
            robot_radius_m=float(robot_radius),
            obstacles=tuple(
                CBFObstacleState(
                    position_m=(float(obstacle.position[0]), float(obstacle.position[1])),
                    velocity_mps=(float(obstacle.velocity[0]), float(obstacle.velocity[1])),
                    radius_m=float(obstacle.radius),
                )
                for obstacle in obstacles
            ),
        )
        public_config = CBFSafetyFilterConfig(
            enabled=True,
            variant=CBF_VARIANT_DYNAMIC_PARABOLIC,
            alpha=float(self.config.alpha),
            safety_radius_margin_m=float(self.config.safety_margin),
            min_linear_velocity_m_s=0.0,
            max_linear_velocity_m_s=max_linear,
            fallback_mode="stop_keep_turn",
            dpcbf_lambda_gain=float(self.config.dpcbf_lambda_gain),
            dpcbf_mu_gain=float(self.config.dpcbf_mu_gain),
            relative_speed_epsilon=float(self.config.relative_speed_epsilon),
            dpcbf_grid_samples=int(self.config.dpcbf_grid_samples),
        )
        result = apply_cbf_safety_filter(
            float(proposed_command[0]),
            float(proposed_command[1]),
            context,
            public_config,
        )
        fallback = bool(result["fallback_applied"])
        label = (
            "cbf_best_effort"
            if fallback
            else "cbf_projected"
            if bool(result["intervened"])
            else "cbf_feasible"
        )
        filtered = (
            float(result["filtered_linear_velocity"]),
            float(result["filtered_angular_velocity"]),
        )
        violated = (
            ("dynamic_parabolic_cbf",)
            if result["min_barrier_before"] is not None and result["min_barrier_before"] < 0.0
            else ()
        )
        decision = ShieldDecision(
            proposed_action=(float(proposed_command[0]), float(proposed_command[1])),
            filtered_action=filtered,
            decision_label=label,
            intervention_reason=(
                "dynamic_parabolic_cbf_projection"
                if bool(result["intervened"])
                else "proposed_command_satisfies_dynamic_parabolic_cbf"
            ),
            violated_constraints=violated,
            prediction_source="current_state",
            fallback_controller_state={
                "filter": "DynamicParabolicCbfSafetyFilter",
                "variant": CBF_VARIANT_DYNAMIC_PARABOLIC,
                "fallback": fallback,
                "projection_method": result.get("projection_method"),
            },
            proposed_evaluation={"min_cbf_margin": result["min_barrier_before"]},
            selected_evaluation={"min_cbf_margin": result["min_barrier_after"]},
        )
        self._record_decision(decision, fallback=fallback)
        return decision


def build_cbf_safety_filter(config: CbfSafetyFilterConfig) -> CollisionConeCbfSafetyFilter:
    """Return the concrete CBF filter implementation for ``config.variant``."""

    variant = str(config.variant).strip().lower()
    if variant in {"collision_cone", CBF_VARIANT_COLLISION_CONE}:
        return CollisionConeCbfSafetyFilter(config)
    if variant in {"dynamic_parabolic", CBF_VARIANT_DYNAMIC_PARABOLIC}:
        return DynamicParabolicCbfSafetyFilter(
            dataclass_replace(config, variant=CBF_VARIANT_DYNAMIC_PARABOLIC)
        )
    raise ValueError("CBF safety-filter variant must be collision_cone or dynamic_parabolic")


class CbfSafetyFilterPlannerWrapper:
    """Planner wrapper applies a versioned CBF safety filter after ``plan``."""

    def __init__(self, planner: Any, config: CbfSafetyFilterConfig | None = None) -> None:
        """Wrap a planner exposing ``plan(observation) -> (linear, angular)``."""

        self.planner = planner
        self.filter = build_cbf_safety_filter(config or CbfSafetyFilterConfig())
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


CBF_SAFETY_FILTER_SCHEMA = "cbf_safety_filter.v1"
CBF_VARIANT_COLLISION_CONE = "collision_cone_cbf_v1"
CBF_VARIANT_DYNAMIC_PARABOLIC = "dynamic_parabolic_cbf_v1"


@dataclass(frozen=True, slots=True)
class CBFSafetyFilterConfig:
    """Pure-function CBF filter configuration for benchmark runtime binding."""

    enabled: bool = False
    variant: str = CBF_VARIANT_COLLISION_CONE
    alpha: float = 2.0
    safety_radius_margin_m: float = 0.05
    min_linear_velocity_m_s: float = 0.0
    max_linear_velocity_m_s: float = 2.0
    fallback_mode: str = "stop_keep_turn"
    angular_weight: float = 0.05
    dpcbf_lambda_gain: float = 1.0
    dpcbf_mu_gain: float = 1.0
    relative_speed_epsilon: float = 1e-6
    dpcbf_grid_samples: int = 161


@dataclass(frozen=True, slots=True)
class CBFObstacleState:
    """Current-state dynamic obstacle sample for the CBF filter."""

    position_m: tuple[float, float]
    velocity_mps: tuple[float, float]
    radius_m: float


@dataclass(frozen=True, slots=True)
class CBFFilterContext:
    """Robot and obstacle state needed for one CBF projection."""

    robot_position_m: tuple[float, float]
    robot_heading_rad: float
    robot_radius_m: float
    obstacles: tuple[CBFObstacleState, ...]


def _finite_xy_tuple(value: tuple[float, float], *, field_name: str) -> tuple[float, float]:
    """Validate finite xy tuple.

    Returns:
        Two finite float values.
    """

    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} must be two finite floats")
    return (float(arr[0]), float(arr[1]))


def _validate_public_context(context: CBFFilterContext) -> None:
    """Fail closed for malformed public CBF state."""

    _finite_xy_tuple(context.robot_position_m, field_name="robot_position_m")
    if not math.isfinite(float(context.robot_heading_rad)):
        raise ValueError("robot_heading_rad must be finite")
    if not math.isfinite(float(context.robot_radius_m)) or context.robot_radius_m < 0.0:
        raise ValueError("robot_radius_m must be finite and non-negative")
    for obstacle in context.obstacles:
        _finite_xy_tuple(obstacle.position_m, field_name="obstacle.position_m")
        _finite_xy_tuple(obstacle.velocity_mps, field_name="obstacle.velocity_mps")
        if not math.isfinite(float(obstacle.radius_m)) or obstacle.radius_m < 0.0:
            raise ValueError("obstacle.radius_m must be finite and non-negative")


def _dynamic_parabolic_barrier(
    *,
    linear_velocity: float,
    robot_position: np.ndarray,
    robot_heading: float,
    robot_radius: float,
    obstacle: CBFObstacleState,
    config: CBFSafetyFilterConfig,
) -> float:
    """Evaluate the DPCBF relative-velocity parabola for one scalar speed.

    Returns:
        Barrier value where non-negative means the scalar speed is feasible.
    """

    p_rel = np.asarray(obstacle.position_m, dtype=float) - robot_position
    v_robot = float(linear_velocity) * np.asarray(
        [math.cos(robot_heading), math.sin(robot_heading)], dtype=float
    )
    v_rel = np.asarray(obstacle.velocity_mps, dtype=float) - v_robot
    alpha = math.atan2(float(p_rel[1]), float(p_rel[0]))
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    v_tilde_x = float(cos_a * v_rel[0] + sin_a * v_rel[1])
    v_tilde_y = float(-sin_a * v_rel[0] + cos_a * v_rel[1])
    radius = float(robot_radius) + float(obstacle.radius_m) + float(config.safety_radius_margin_m)
    clearance_sq = max(float(np.dot(p_rel, p_rel)) - radius * radius, 0.0)
    clearance = math.sqrt(clearance_sq)
    relative_speed = max(float(np.linalg.norm(v_rel)), float(config.relative_speed_epsilon))
    lambda_term = float(config.dpcbf_lambda_gain) * clearance / relative_speed
    mu_term = float(config.dpcbf_mu_gain) * clearance
    return float(v_tilde_x + lambda_term * v_tilde_y * v_tilde_y + mu_term)


def _precompute_dynamic_parabolic_obstacles(
    *,
    robot_position: np.ndarray,
    robot_radius: float,
    obstacles: tuple[CBFObstacleState, ...],
    config: CBFSafetyFilterConfig,
) -> list[tuple[np.ndarray, float, float, float, float]]:
    """Precompute obstacle constants reused across DPCBF scalar-speed candidates.

    Returns:
        list[tuple[np.ndarray, float, float, float, float]]: Per-obstacle velocity,
        line-of-sight trigonometry, clearance, and static DPCBF offset.
    """

    constants: list[tuple[np.ndarray, float, float, float, float]] = []
    for obstacle in obstacles:
        p_rel = np.asarray(obstacle.position_m, dtype=float) - robot_position
        alpha = math.atan2(float(p_rel[1]), float(p_rel[0]))
        radius = (
            float(robot_radius) + float(obstacle.radius_m) + float(config.safety_radius_margin_m)
        )
        clearance_sq = max(float(np.dot(p_rel, p_rel)) - radius * radius, 0.0)
        clearance = math.sqrt(clearance_sq)
        constants.append(
            (
                np.asarray(obstacle.velocity_mps, dtype=float),
                math.cos(alpha),
                math.sin(alpha),
                clearance,
                float(config.dpcbf_mu_gain) * clearance,
            )
        )
    return constants


def _dynamic_parabolic_barrier_from_constants(
    *,
    linear_velocity: float,
    heading_unit: np.ndarray,
    obstacle_constants: tuple[np.ndarray, float, float, float, float],
    config: CBFSafetyFilterConfig,
) -> float:
    """Evaluate DPCBF barrier using obstacle constants shared across grid candidates.

    Returns:
        float: Barrier value where non-negative means scalar speed is feasible.
    """

    obstacle_velocity, cos_a, sin_a, clearance, mu_term = obstacle_constants
    v_robot = float(linear_velocity) * heading_unit
    v_rel = obstacle_velocity - v_robot
    v_tilde_x = float(cos_a * v_rel[0] + sin_a * v_rel[1])
    v_tilde_y = float(-sin_a * v_rel[0] + cos_a * v_rel[1])
    relative_speed = max(float(np.linalg.norm(v_rel)), float(config.relative_speed_epsilon))
    lambda_term = float(config.dpcbf_lambda_gain) * clearance / relative_speed
    return float(v_tilde_x + lambda_term * v_tilde_y * v_tilde_y + mu_term)


def _apply_dynamic_parabolic_cbf(
    linear_velocity: float,
    angular_velocity: float,
    context: CBFFilterContext,
    config: CBFSafetyFilterConfig,
) -> dict[str, Any]:
    """Project scalar speed by deterministic bounded DPCBF grid/refine search.

    Returns:
        Schema-tagged CBF filter result with projection diagnostics.
    """

    lower = float(config.min_linear_velocity_m_s)
    upper = float(config.max_linear_velocity_m_s)
    if lower > upper:
        raise ValueError("min_linear_velocity_m_s must be <= max_linear_velocity_m_s")
    if config.fallback_mode != "stop_keep_turn":
        raise ValueError("Only fallback_mode='stop_keep_turn' is implemented")
    if config.dpcbf_lambda_gain < 0.0:
        raise ValueError("dpcbf_lambda_gain must be non-negative")
    if config.dpcbf_mu_gain < 0.0:
        raise ValueError("dpcbf_mu_gain must be non-negative")
    if config.relative_speed_epsilon <= 0.0:
        raise ValueError("relative_speed_epsilon must be positive")
    if config.dpcbf_grid_samples < 3:
        raise ValueError("dpcbf_grid_samples must be >= 3")

    robot_position = np.asarray(context.robot_position_m, dtype=float)
    robot_heading = float(context.robot_heading_rad)
    heading_unit = np.asarray([math.cos(robot_heading), math.sin(robot_heading)], dtype=float)
    obstacle_constants = _precompute_dynamic_parabolic_obstacles(
        robot_position=robot_position,
        robot_radius=float(context.robot_radius_m),
        obstacles=context.obstacles,
        config=config,
    )
    nominal = float(np.clip(float(linear_velocity), lower, upper))
    barriers_before = [
        _dynamic_parabolic_barrier_from_constants(
            linear_velocity=nominal,
            heading_unit=heading_unit,
            obstacle_constants=constants,
            config=config,
        )
        for constants in obstacle_constants
    ]
    if not context.obstacles:
        selected = nominal
        feasible = True
    elif all(barrier >= 0.0 for barrier in barriers_before):
        selected = nominal
        feasible = True
    else:
        grid = np.linspace(lower, upper, int(config.dpcbf_grid_samples), dtype=float)
        feasible_values = [
            float(candidate)
            for candidate in grid
            if all(
                _dynamic_parabolic_barrier_from_constants(
                    linear_velocity=float(candidate),
                    heading_unit=heading_unit,
                    obstacle_constants=constants,
                    config=config,
                )
                >= 0.0
                for constants in obstacle_constants
            )
        ]
        if feasible_values:
            selected = min(feasible_values, key=lambda value: abs(value - nominal))
            feasible = True
        else:
            selected = float(np.clip(0.0, lower, upper))
            feasible = False

    barriers_after = [
        _dynamic_parabolic_barrier_from_constants(
            linear_velocity=selected,
            heading_unit=heading_unit,
            obstacle_constants=constants,
            config=config,
        )
        for constants in obstacle_constants
    ]
    min_before = min(barriers_before) if barriers_before else None
    min_after = min(barriers_after) if barriers_after else None
    intervened = not math.isclose(selected, float(linear_velocity), rel_tol=1.0e-9, abs_tol=1.0e-9)
    fallback_applied = not feasible
    qp_status = (
        "fallback_infeasible" if fallback_applied else "filtered" if intervened else "pass_through"
    )
    return {
        "schema_version": CBF_SAFETY_FILTER_SCHEMA,
        "variant": CBF_VARIANT_DYNAMIC_PARABOLIC,
        "enabled": True,
        "qp_status": qp_status,
        "qp_feasible": bool(feasible),
        "fallback_applied": bool(fallback_applied),
        "intervened": bool(intervened),
        "nominal_linear_velocity": float(linear_velocity),
        "nominal_angular_velocity": float(angular_velocity),
        "filtered_linear_velocity": float(selected),
        "filtered_angular_velocity": float(angular_velocity),
        "active_constraint_count": len(context.obstacles),
        "min_barrier_before": min_before,
        "min_barrier_after": min_after,
        "hard_constraint_violation": bool(min_after is not None and min_after < -1.0e-9),
        "projection_method": "bounded_scalar_dpcbf_grid_refine_v1",
    }


def apply_cbf_safety_filter(  # noqa: C901
    linear_velocity: float,
    angular_velocity: float,
    context: CBFFilterContext,
    config: CBFSafetyFilterConfig | None = None,
) -> dict[str, Any]:
    """Project a nominal unicycle command through collision-cone CBF constraints.

    Returns:
        JSON-safe decision payload containing the filtered command and CBF diagnostics.
    """

    config = config or CBFSafetyFilterConfig()
    _validate_public_context(context)
    if not math.isfinite(float(linear_velocity)) or not math.isfinite(float(angular_velocity)):
        raise ValueError("nominal command velocities must be finite")
    if config.variant == CBF_VARIANT_DYNAMIC_PARABOLIC:
        if not config.enabled:
            return {
                "schema_version": CBF_SAFETY_FILTER_SCHEMA,
                "variant": config.variant,
                "enabled": False,
                "qp_status": "disabled",
                "qp_feasible": True,
                "fallback_applied": False,
                "intervened": False,
                "nominal_linear_velocity": float(linear_velocity),
                "nominal_angular_velocity": float(angular_velocity),
                "filtered_linear_velocity": float(linear_velocity),
                "filtered_angular_velocity": float(angular_velocity),
                "active_constraint_count": 0,
                "min_barrier_before": None,
                "min_barrier_after": None,
                "hard_constraint_violation": False,
            }
        return _apply_dynamic_parabolic_cbf(linear_velocity, angular_velocity, context, config)
    if config.variant != CBF_VARIANT_COLLISION_CONE:
        raise ValueError("CBF variant must be collision_cone_cbf_v1")
    if not config.enabled:
        return {
            "schema_version": CBF_SAFETY_FILTER_SCHEMA,
            "variant": config.variant,
            "enabled": False,
            "qp_status": "disabled",
            "qp_feasible": True,
            "fallback_applied": False,
            "intervened": False,
            "nominal_linear_velocity": float(linear_velocity),
            "nominal_angular_velocity": float(angular_velocity),
            "filtered_linear_velocity": float(linear_velocity),
            "filtered_angular_velocity": float(angular_velocity),
            "active_constraint_count": 0,
            "min_barrier_before": None,
            "min_barrier_after": None,
            "hard_constraint_violation": False,
        }

    robot_pos = np.asarray(context.robot_position_m, dtype=float)
    heading = float(context.robot_heading_rad)
    e_theta = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
    lower = float(config.min_linear_velocity_m_s)
    upper = float(config.max_linear_velocity_m_s)
    if lower > upper:
        raise ValueError("min_linear_velocity_m_s must be <= max_linear_velocity_m_s")

    barriers_before: list[float] = []
    intervals: list[tuple[float, float]] = []
    for obstacle in context.obstacles:
        obstacle_pos = np.asarray(obstacle.position_m, dtype=float)
        obstacle_velocity = np.asarray(obstacle.velocity_mps, dtype=float)
        p_i = obstacle_pos - robot_pos
        radius = (
            float(context.robot_radius_m)
            + float(obstacle.radius_m)
            + float(config.safety_radius_margin_m)
        )
        h_i = float(np.dot(p_i, p_i) - radius * radius)
        coeff = float(-2.0 * np.dot(p_i, e_theta))
        rhs = float(-config.alpha * h_i - 2.0 * np.dot(p_i, obstacle_velocity))
        barrier_before = float(
            2.0 * np.dot(p_i, obstacle_velocity - float(linear_velocity) * e_theta)
            + float(config.alpha) * h_i
        )
        barriers_before.append(barrier_before)
        if abs(coeff) <= 1.0e-12:
            if 0.0 < rhs:
                intervals.append((math.inf, -math.inf))
            continue
        bound = rhs / coeff
        if coeff > 0.0:
            intervals.append((bound, math.inf))
        else:
            intervals.append((-math.inf, bound))

    for interval_lower, interval_upper in intervals:
        lower = max(lower, interval_lower)
        upper = min(upper, interval_upper)

    feasible = lower <= upper
    if feasible:
        filtered_linear = float(np.clip(float(linear_velocity), lower, upper))
    elif config.fallback_mode == "stop_keep_turn":
        filtered_linear = 0.0
    else:
        raise ValueError("unsupported CBF fallback_mode")
    filtered_angular = float(angular_velocity)
    barriers_after = [
        float(
            2.0
            * np.dot(
                np.asarray(obstacle.position_m, dtype=float) - robot_pos,
                np.asarray(obstacle.velocity_mps, dtype=float) - filtered_linear * e_theta,
            )
            + float(config.alpha)
            * (
                float(
                    np.dot(
                        np.asarray(obstacle.position_m, dtype=float) - robot_pos,
                        np.asarray(obstacle.position_m, dtype=float) - robot_pos,
                    )
                )
                - (
                    float(context.robot_radius_m)
                    + float(obstacle.radius_m)
                    + float(config.safety_radius_margin_m)
                )
                ** 2
            )
        )
        for obstacle in context.obstacles
    ]
    min_before = min(barriers_before) if barriers_before else None
    min_after = min(barriers_after) if barriers_after else None
    intervened = not math.isclose(
        filtered_linear, float(linear_velocity), rel_tol=1.0e-9, abs_tol=1.0e-9
    )
    fallback_applied = not feasible
    qp_status = (
        "fallback_infeasible" if fallback_applied else "filtered" if intervened else "pass_through"
    )
    return {
        "schema_version": CBF_SAFETY_FILTER_SCHEMA,
        "variant": config.variant,
        "enabled": True,
        "qp_status": qp_status,
        "qp_feasible": bool(feasible),
        "fallback_applied": bool(fallback_applied),
        "intervened": bool(intervened),
        "nominal_linear_velocity": float(linear_velocity),
        "nominal_angular_velocity": float(angular_velocity),
        "filtered_linear_velocity": float(filtered_linear),
        "filtered_angular_velocity": float(filtered_angular),
        "active_constraint_count": len(intervals),
        "min_barrier_before": min_before,
        "min_barrier_after": min_after,
        "hard_constraint_violation": bool(min_after is not None and min_after < -1.0e-9),
    }


__all__ = [
    "CBF_SAFETY_FILTER_SCHEMA",
    "CBF_VARIANT_COLLISION_CONE",
    "CBF_VARIANT_DYNAMIC_PARABOLIC",
    "CBFFilterContext",
    "CBFObstacleState",
    "CBFSafetyFilterConfig",
    "CbfSafetyFilterConfig",
    "CbfSafetyFilterPlannerWrapper",
    "CollisionConeCbfSafetyFilter",
    "DynamicParabolicCbfSafetyFilter",
    "apply_cbf_safety_filter",
    "build_cbf_safety_filter",
    "build_cbf_safety_filter_config",
]
