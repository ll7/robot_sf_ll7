"""Experimental prediction-aware MPC local planner."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, fields
from typing import Any, Protocol

import numpy as np
from scipy.optimize import NonlinearConstraint

from robot_sf.nav.uncertainty_envelope import envelope_diagnostics
from robot_sf.planner.nmpc_social import (
    NMPCSocialConfig,
    NMPCSocialPlannerAdapter,
    _parse_bool,
    _RolloutContext,
)


@dataclass
class PredictionMPCConfig:
    """Configuration for the experimental prediction-aware MPC planner."""

    max_linear_speed: float = 0.9
    max_angular_speed: float = 1.1
    horizon_steps: int = 6
    rollout_dt: float = 0.25
    goal_tolerance: float = 0.25
    waypoint_switch_distance: float = 0.75
    path_goal_weight: float = 1.5
    terminal_goal_weight: float = 5.0
    control_effort_weight: float = 0.05
    smoothness_weight: float = 0.2
    heading_weight: float = 0.5
    pedestrian_safety_margin: float = 0.35
    static_obstacle_soft_weight: float = 1.0
    solver_ftol: float = 1e-3
    solver_max_iterations: int = 40
    warm_start: bool = True
    fallback_to_stop: bool = True
    predictor_backend: str = "constant_velocity"
    allow_predictor_fallback: bool = False
    # Opt-in pedestrian uncertainty envelope (issue #4141). When enabled with a
    # positive alpha, per-horizon-step clearance uses an inflated effective
    # pedestrian radius r_eff(i) = ped_radius + alpha * i * rollout_dt. Disabled
    # or alpha == 0.0 reproduces the deterministic baseline exactly.
    pedestrian_uncertainty_envelope_enabled: bool = False
    pedestrian_uncertainty_alpha_mps: float = 0.0

    def __post_init__(self) -> None:
        """Validate the opt-in uncertainty-envelope conservatism rate."""
        if self.pedestrian_uncertainty_alpha_mps < 0.0:
            raise ValueError("pedestrian_uncertainty_alpha_mps must be >= 0")


@dataclass(frozen=True)
class PredictedPedestrianFutures:
    """Predicted pedestrian positions in world coordinates."""

    positions_world: np.ndarray
    mask: np.ndarray
    dt: float
    source: str


class PedestrianFuturePredictor(Protocol):
    """Protocol for prediction backends consumed by prediction-aware MPC."""

    def predict(
        self,
        observation: dict[str, Any],
        *,
        horizon_steps: int,
        dt: float,
    ) -> PredictedPedestrianFutures:
        """Return predicted pedestrian futures in world coordinates."""


class ConstantVelocityPedestrianPredictor(NMPCSocialPlannerAdapter):
    """Constant-velocity predictor that rotates SocNav ego velocities to world frame."""

    def __init__(self) -> None:
        """Initialize the predictor with the shared SocNav field helpers."""
        super().__init__(NMPCSocialConfig())

    def predict(
        self,
        observation: dict[str, Any],
        *,
        horizon_steps: int,
        dt: float,
    ) -> PredictedPedestrianFutures:
        """Predict pedestrian futures from current world positions and ego-frame velocities.

        Returns:
            PredictedPedestrianFutures: World-frame futures for active pedestrians.
        """
        robot_state, _goal_state, ped_state = self._socnav_fields(observation)
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        ped_positions = np.asarray(ped_state.get("positions", []), dtype=float)
        ped_velocities_ego = np.asarray(ped_state.get("velocities", []), dtype=float)
        if ped_positions.ndim == 1 and ped_positions.size % 2 == 0:
            ped_positions = ped_positions.reshape(-1, 2)
        if ped_velocities_ego.ndim == 1 and ped_velocities_ego.size % 2 == 0:
            ped_velocities_ego = ped_velocities_ego.reshape(-1, 2)
        if ped_positions.ndim != 2 or ped_positions.shape[-1] != 2:
            ped_positions = np.zeros((0, 2), dtype=float)
        if ped_velocities_ego.ndim != 2 or ped_velocities_ego.shape[-1] != 2:
            ped_velocities_ego = np.zeros_like(ped_positions)
        count = int(self._as_1d_float(ped_state.get("count", [ped_positions.shape[0]]), pad=1)[0])
        count = max(0, min(count, ped_positions.shape[0]))
        ped_positions = ped_positions[:count]
        if ped_velocities_ego.shape[0] < count:
            ped_velocities_ego = np.zeros_like(ped_positions)
        else:
            ped_velocities_ego = ped_velocities_ego[:count]

        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        ped_velocities_world = np.empty_like(ped_velocities_ego)
        ped_velocities_world[:, 0] = (
            cos_h * ped_velocities_ego[:, 0] - sin_h * ped_velocities_ego[:, 1]
        )
        ped_velocities_world[:, 1] = (
            sin_h * ped_velocities_ego[:, 0] + cos_h * ped_velocities_ego[:, 1]
        )
        future = np.zeros((count, max(int(horizon_steps), 1), 2), dtype=float)
        for step_idx in range(future.shape[1]):
            tau = float(step_idx + 1) * float(dt)
            future[:, step_idx, :] = ped_positions + ped_velocities_world * tau
        return PredictedPedestrianFutures(
            positions_world=future,
            mask=np.ones((count,), dtype=float),
            dt=float(dt),
            source="constant_velocity",
        )


class PredictionMPCPlannerAdapter(NMPCSocialPlannerAdapter):
    """Experimental MPC adapter with hard predicted pedestrian clearance constraints."""

    def __init__(
        self,
        config: PredictionMPCConfig | None = None,
        predictor: PedestrianFuturePredictor | None = None,
    ) -> None:
        """Initialize the adapter with strict constant-velocity prediction by default."""
        self.prediction_config = config or PredictionMPCConfig()
        if self.prediction_config.predictor_backend.strip().lower() not in {
            "constant_velocity",
            "cv",
        }:
            if not self.prediction_config.allow_predictor_fallback:
                raise ValueError(
                    "prediction_mpc currently supports predictor_backend='constant_velocity' "
                    "unless allow_predictor_fallback is true."
                )
            warnings.warn(
                "prediction_mpc falling back to constant_velocity predictor backend.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._future_predictor = predictor or ConstantVelocityPedestrianPredictor()
        super().__init__(_to_nmpc_config(self.prediction_config))

    def _optimizer_constraints(self, context: _RolloutContext) -> tuple[NonlinearConstraint, ...]:
        """Enforce hard time-varying pedestrian-future clearance constraints.

        Returns:
            tuple[NonlinearConstraint, ...]: Extra SLSQP constraints for the optimizer.
        """
        futures = self._future_predictor.predict(
            context.observation,
            horizon_steps=max(int(self.config.horizon_steps), 1),
            dt=float(self.config.rollout_dt),
        )
        if futures.positions_world.size == 0 or not np.any(futures.mask > 0.5):
            return ()
        return (
            NonlinearConstraint(
                lambda controls: self._pedestrian_clearance_constraints(
                    controls,
                    context=context,
                    predicted_futures=futures,
                ),
                0.0,
                np.inf,
            ),
        )

    def _pedestrian_clearance_constraints(
        self,
        controls_flat: np.ndarray,
        *,
        context: _RolloutContext,
        predicted_futures: PredictedPedestrianFutures,
    ) -> np.ndarray:
        """Return SLSQP constraint values for predicted pedestrian clearances.

        Returns:
            np.ndarray: Values constrained to be non-negative by SLSQP.
        """
        states = self._rollout_states(controls_flat, context=context)
        if states.size == 0 or predicted_futures.positions_world.size == 0:
            return np.ones((1,), dtype=float)
        active = predicted_futures.mask > 0.5
        if not np.any(active):
            return np.ones((1,), dtype=float)
        robot_xy = states[:, :2]
        enabled = bool(self.prediction_config.pedestrian_uncertainty_envelope_enabled)
        alpha = float(self.prediction_config.pedestrian_uncertainty_alpha_mps)
        dt = float(predicted_futures.dt)
        use_inflation = enabled and alpha > 0.0
        base_ped_radius = float(context.ped_radius)
        values: list[float] = []
        future_horizon = predicted_futures.positions_world.shape[1]
        for step_idx in range(robot_xy.shape[0]):
            ped_k = predicted_futures.positions_world[active, min(step_idx, future_horizon - 1), :]
            d2 = np.sum((ped_k - robot_xy[step_idx][None, :]) ** 2, axis=1)
            # Horizon-dependent effective pedestrian radius. inflation(0) == 0.0,
            # so the first step is unchanged and a disabled envelope (or
            # alpha == 0.0) is a bit-for-bit no-op versus the baseline.
            ped_eff_radius = base_ped_radius + (
                alpha * float(step_idx) * dt if use_inflation else 0.0
            )
            r_safe = (
                float(context.robot_radius)
                + ped_eff_radius
                + float(self.prediction_config.pedestrian_safety_margin)
            )
            values.extend((d2 - r_safe**2).tolist())
        return np.asarray(values or [1.0], dtype=float)

    def diagnostics(self) -> dict[str, Any]:
        """Return runtime diagnostics plus the uncertainty-envelope provenance.

        Returns:
            dict[str, Any]: Base solve/command statistics augmented with a
            ``pedestrian_uncertainty_envelope`` provenance payload recording the
            envelope settings and an explicit claim boundary.
        """
        payload: dict[str, Any] = copy.deepcopy(super().diagnostics())
        payload["pedestrian_uncertainty_envelope"] = envelope_diagnostics(
            enabled=bool(self.prediction_config.pedestrian_uncertainty_envelope_enabled),
            alpha=float(self.prediction_config.pedestrian_uncertainty_alpha_mps),
            dt=float(self.prediction_config.rollout_dt),
        )
        return payload


def _to_nmpc_config(config: PredictionMPCConfig) -> NMPCSocialConfig:
    """Map prediction-MPC config onto the reusable NMPC optimizer core.

    Returns:
        NMPCSocialConfig: Shared optimizer-core configuration.
    """
    return NMPCSocialConfig(
        max_linear_speed=config.max_linear_speed,
        max_angular_speed=config.max_angular_speed,
        horizon_steps=config.horizon_steps,
        rollout_dt=config.rollout_dt,
        goal_tolerance=config.goal_tolerance,
        waypoint_switch_distance=config.waypoint_switch_distance,
        path_goal_weight=config.path_goal_weight,
        terminal_goal_weight=config.terminal_goal_weight,
        progress_reward_weight=0.0,
        heading_weight=config.heading_weight,
        control_effort_weight=config.control_effort_weight,
        smoothness_weight=config.smoothness_weight,
        pedestrian_clearance_weight=0.0,
        obstacle_clearance_weight=config.static_obstacle_soft_weight,
        occupancy_cost_weight=config.static_obstacle_soft_weight,
        pedestrian_margin=config.pedestrian_safety_margin,
        solver_ftol=config.solver_ftol,
        solver_max_iterations=config.solver_max_iterations,
        warm_start=config.warm_start,
        fallback_to_stop=config.fallback_to_stop,
    )


def build_prediction_mpc_config(cfg: dict[str, Any] | None) -> PredictionMPCConfig:
    """Build prediction-MPC config from an algorithm YAML mapping.

    Returns:
        PredictionMPCConfig: Parsed configuration with invalid values defaulted.
    """
    raw = dict(cfg or {})
    defaults = PredictionMPCConfig()
    converters = {
        "max_linear_speed": float,
        "max_angular_speed": float,
        "horizon_steps": int,
        "rollout_dt": float,
        "goal_tolerance": float,
        "waypoint_switch_distance": float,
        "path_goal_weight": float,
        "terminal_goal_weight": float,
        "control_effort_weight": float,
        "smoothness_weight": float,
        "heading_weight": float,
        "pedestrian_safety_margin": float,
        "static_obstacle_soft_weight": float,
        "solver_ftol": float,
        "solver_max_iterations": int,
        "warm_start": _parse_bool,
        "fallback_to_stop": _parse_bool,
        "predictor_backend": str,
        "allow_predictor_fallback": _parse_bool,
        "pedestrian_uncertainty_envelope_enabled": _parse_bool,
        "pedestrian_uncertainty_alpha_mps": float,
    }
    kwargs: dict[str, Any] = {}
    for field in fields(PredictionMPCConfig):
        value = raw.get(field.name, getattr(defaults, field.name))
        try:
            kwargs[field.name] = converters[field.name](value)
        except (TypeError, ValueError):
            default_value = getattr(defaults, field.name)
            warnings.warn(
                (
                    f"Invalid prediction_mpc config value '{field.name}': {value!r}; "
                    f"falling back to default {default_value!r}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            kwargs[field.name] = default_value
    return PredictionMPCConfig(**kwargs)
