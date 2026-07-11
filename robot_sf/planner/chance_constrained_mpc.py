"""Experimental chance-constrained MPC over K-mode pedestrian forecasts.

The planner deliberately has no built-in predictor.  It accepts an injected
Gaussian-mixture forecast provider so that an unavailable learned predictor is
an explicit error, not a silent constant-velocity fallback.  The integration
with the learned predictor from issue #2844 belongs at that provider boundary.

The joint formulation applies Boole's inequality to the per-pedestrian,
per-timestep collision probabilities.  It is therefore conservative but does
not require an unimplemented cross-time covariance model.  Collision
probabilities are deterministic polar-quadrature approximations of each GMM
component's probability mass inside the collision disc.  This implementation
does not make a forecast-calibration or navigation-performance claim.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, fields
from typing import Any, Literal, Protocol

import numpy as np
from scipy.optimize import NonlinearConstraint

from robot_sf.planner.nmpc_social import NMPCSocialPlannerAdapter, _parse_bool, _RolloutContext
from robot_sf.planner.prediction_mpc import PredictionMPCConfig, _to_nmpc_config


@dataclass(frozen=True)
class GaussianMixturePedestrianForecast:
    """K-mode world-frame pedestrian trajectories with Gaussian uncertainty.

    ``means_world`` and ``covariances_world`` have shape ``(P, K, T, 2)`` and
    ``(P, K, T, 2, 2)`` respectively.  ``mode_weights`` has shape ``(P, K)``;
    each pedestrian's weights must sum to one.  Covariances are required to be
    positive definite because the collision-risk quadrature evaluates a 2-D
    Gaussian density rather than treating a singular component as a hidden
    deterministic fallback.
    """

    means_world: np.ndarray
    covariances_world: np.ndarray
    mode_weights: np.ndarray
    dt: float
    source: str

    def __post_init__(self) -> None:  # noqa: C901
        """Validate the GMM contract before an optimizer can consume it."""

        means = np.asarray(self.means_world, dtype=float)
        covariances = np.asarray(self.covariances_world, dtype=float)
        weights = np.asarray(self.mode_weights, dtype=float)
        if means.ndim != 4 or means.shape[-1] != 2:
            raise ValueError("means_world must have shape (P, K, T, 2)")
        if covariances.shape != means.shape[:-1] + (2, 2):
            raise ValueError("covariances_world must have shape (P, K, T, 2, 2)")
        if weights.shape != means.shape[:2]:
            raise ValueError("mode_weights must have shape (P, K)")
        if means.shape[1] == 0 or means.shape[2] == 0:
            raise ValueError("GMM forecasts require at least one mode and horizon step")
        if not all(np.all(np.isfinite(value)) for value in (means, covariances, weights)):
            raise ValueError("GMM forecast values must be finite")
        if np.any(weights < 0.0) or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("mode_weights must be non-negative and sum to one per pedestrian")
        if not np.allclose(covariances, np.swapaxes(covariances, -1, -2), atol=1e-8):
            raise ValueError("covariances_world must be symmetric")
        if np.any(np.linalg.eigvalsh(covariances) <= 1e-9):
            raise ValueError("covariances_world must be positive definite")
        if not np.isfinite(self.dt) or float(self.dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        object.__setattr__(self, "means_world", means)
        object.__setattr__(self, "covariances_world", covariances)
        object.__setattr__(self, "mode_weights", weights)
        object.__setattr__(self, "dt", float(self.dt))


class GaussianMixturePedestrianPredictor(Protocol):
    """Provider contract that the learned predictor must implement for #5307."""

    def predict(
        self,
        observation: dict[str, Any],
        *,
        horizon_steps: int,
        dt: float,
    ) -> GaussianMixturePedestrianForecast:
        """Return a K-mode/GMM forecast aligned with the MPC rollout horizon."""


@dataclass
class ChanceConstrainedMPCConfig(PredictionMPCConfig):
    """Configuration for the experimental GMM chance-constrained MPC arm."""

    predictor_backend: str = "multimodal_gmm"
    chance_constraint_formulation: Literal["marginal", "joint_horizon"] = "marginal"
    max_collision_risk: float = 0.05
    radial_quadrature_order: int = 6
    angular_quadrature_order: int = 16

    def __post_init__(self) -> None:
        """Reject ambiguous risk settings and incompatible envelope composition."""

        super().__post_init__()
        if self.chance_constraint_formulation not in {"marginal", "joint_horizon"}:
            raise ValueError("chance_constraint_formulation must be 'marginal' or 'joint_horizon'")
        if not 0.0 < float(self.max_collision_risk) < 1.0:
            raise ValueError("max_collision_risk must be in (0, 1)")
        if int(self.radial_quadrature_order) < 2 or int(self.angular_quadrature_order) < 4:
            raise ValueError("quadrature orders must be at least 2 radial and 4 angular")
        if self.allow_predictor_fallback:
            raise ValueError("chance_constrained_mpc never permits predictor fallback")
        if self.pedestrian_uncertainty_envelope_enabled or self.pedestrian_uncertainty_alpha_mps:
            raise ValueError(
                "chance_constrained_mpc does not compose the deterministic uncertainty envelope; "
                "use the separate prediction_mpc CV/envelope arms for matched-arm comparisons"
            )


class ChanceConstrainedMPCPlannerAdapter(NMPCSocialPlannerAdapter):
    """MPC adapter enforcing GMM collision-risk constraints without a fallback predictor."""

    def __init__(
        self,
        config: ChanceConstrainedMPCConfig | None = None,
        predictor: GaussianMixturePedestrianPredictor | None = None,
    ) -> None:
        """Initialize the adapter with the required K-mode forecast provider."""

        self.chance_config = config or ChanceConstrainedMPCConfig()
        if predictor is None:
            raise ValueError(
                "chance_constrained_mpc requires an injected GaussianMixturePedestrianPredictor; "
                "the #2844 learned K-mode/GMM provider is not wired yet"
            )
        self._multimodal_predictor = predictor
        self._last_forecast: GaussianMixturePedestrianForecast | None = None
        self._last_constraint_count = 0
        super().__init__(_to_nmpc_config(self.chance_config))

    def reset(self) -> None:
        """Clear solver and forecast diagnostics at an episode boundary."""

        super().reset()
        self._last_forecast = None
        self._last_constraint_count = 0
        reset = getattr(self._multimodal_predictor, "reset", None)
        if callable(reset):
            reset()

    def _optimizer_constraints(self, context: _RolloutContext) -> tuple[NonlinearConstraint, ...]:
        """Build the requested marginal or joint-horizon chance constraint.

        Returns:
            A single non-linear constraint unless no pedestrians are forecast.
        """

        forecast = self._multimodal_predictor.predict(
            context.observation,
            horizon_steps=max(int(self.config.horizon_steps), 1),
            dt=float(self.config.rollout_dt),
        )
        if not isinstance(forecast, GaussianMixturePedestrianForecast):
            raise TypeError("multimodal predictor must return GaussianMixturePedestrianForecast")
        if not np.isclose(forecast.dt, self.config.rollout_dt):
            raise ValueError("GMM forecast dt must match the MPC rollout_dt")
        if forecast.means_world.shape[2] < int(self.config.horizon_steps):
            raise ValueError("GMM forecast horizon is shorter than the MPC horizon")
        self._last_forecast = forecast
        if forecast.means_world.shape[0] == 0:
            self._last_constraint_count = 0
            return ()
        self._last_constraint_count = (
            forecast.means_world.shape[0] * int(self.config.horizon_steps)
            if self.chance_config.chance_constraint_formulation == "marginal"
            else 1
        )
        return (
            NonlinearConstraint(
                lambda controls: self._chance_constraint_values(
                    controls,
                    context=context,
                    forecast=forecast,
                ),
                0.0,
                np.inf,
            ),
        )

    def _chance_constraint_values(
        self,
        controls_flat: np.ndarray,
        *,
        context: _RolloutContext,
        forecast: GaussianMixturePedestrianForecast,
    ) -> np.ndarray:
        """Return non-negative chance-constraint slack for an MPC control sequence."""

        risks = self._marginal_collision_risks(controls_flat, context=context, forecast=forecast)
        if risks.size == 0:
            return np.ones((1,), dtype=float)
        alpha = float(self.chance_config.max_collision_risk)
        if self.chance_config.chance_constraint_formulation == "marginal":
            return alpha - risks.reshape(-1)
        # Boole's inequality bounds P(any collision over people and time) by the
        # sum of marginal probabilities without assuming cross-time independence.
        return np.asarray([alpha - float(np.sum(risks))], dtype=float)

    def _marginal_collision_risks(
        self,
        controls_flat: np.ndarray,
        *,
        context: _RolloutContext,
        forecast: GaussianMixturePedestrianForecast,
    ) -> np.ndarray:
        """Estimate per-pedestrian, per-timestep GMM collision probabilities.

        Returns:
            Array with shape ``(P, T)`` of collision-probability estimates.
        """

        states = self._rollout_states(controls_flat, context=context)
        steps = min(states.shape[0], forecast.means_world.shape[2])
        people = forecast.means_world.shape[0]
        risks = np.zeros((people, steps), dtype=float)
        collision_radius = (
            float(context.robot_radius)
            + float(context.ped_radius)
            + float(self.chance_config.pedestrian_safety_margin)
        )
        for pedestrian_index in range(people):
            for step_index in range(steps):
                risk = 0.0
                for mode_index, weight in enumerate(forecast.mode_weights[pedestrian_index]):
                    risk += float(weight) * self._gaussian_disc_probability(
                        mean=forecast.means_world[pedestrian_index, mode_index, step_index],
                        covariance=forecast.covariances_world[
                            pedestrian_index, mode_index, step_index
                        ],
                        center=states[step_index, :2],
                        radius=collision_radius,
                    )
                risks[pedestrian_index, step_index] = float(np.clip(risk, 0.0, 1.0))
        return risks

    def _gaussian_disc_probability(
        self,
        *,
        mean: np.ndarray,
        covariance: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> float:
        """Approximate probability mass of one Gaussian inside a collision disc.

        Returns:
            A deterministic quadrature estimate clipped to the unit interval.
        """

        radial_nodes, radial_weights = np.polynomial.legendre.leggauss(
            int(self.chance_config.radial_quadrature_order)
        )
        radii = 0.5 * (radial_nodes + 1.0) * radius
        radial_weights = 0.5 * radial_weights * radius
        angles = (
            2.0
            * np.pi
            * np.arange(int(self.chance_config.angular_quadrature_order), dtype=float)
            / float(self.chance_config.angular_quadrature_order)
        )
        points = (
            center[None, None, :]
            + radii[:, None, None] * np.stack((np.cos(angles), np.sin(angles)), axis=-1)[None, :, :]
        )
        delta = points - mean[None, None, :]
        inverse = np.linalg.inv(covariance)
        determinant = float(np.linalg.det(covariance))
        exponent = np.einsum("...i,ij,...j->...", delta, inverse, delta)
        density = np.exp(-0.5 * exponent) / (2.0 * np.pi * np.sqrt(determinant))
        angular_weight = 2.0 * np.pi / float(self.chance_config.angular_quadrature_order)
        probability = np.sum(density * radii[:, None] * radial_weights[:, None] * angular_weight)
        return float(np.clip(probability, 0.0, 1.0))

    def diagnostics(self) -> dict[str, Any]:
        """Return solver diagnostics with an explicit chance-constraint boundary."""

        payload = copy.deepcopy(super().diagnostics())
        forecast = self._last_forecast
        payload.update(
            {
                "chance_constraint": {
                    "formulation": self.chance_config.chance_constraint_formulation,
                    "max_collision_risk": self.chance_config.max_collision_risk,
                    "quadrature": "deterministic_polar_gaussian_disc",
                    "radial_order": self.chance_config.radial_quadrature_order,
                    "angular_order": self.chance_config.angular_quadrature_order,
                    "joint_bound": "Boole_union_bound"
                    if self.chance_config.chance_constraint_formulation == "joint_horizon"
                    else None,
                    "constraint_count": self._last_constraint_count,
                    "forecast_source": forecast.source if forecast else "not_run",
                    "forecast_modes": int(forecast.means_world.shape[1]) if forecast else 0,
                    "claim_boundary": (
                        "implementation-only; collision-risk calibration and planner benefit "
                        "require matched-arm benchmark evidence"
                    ),
                }
            }
        )
        return payload


def build_chance_constrained_mpc_config(
    cfg: dict[str, Any] | None,
) -> ChanceConstrainedMPCConfig:
    """Build chance-constrained MPC config from a YAML-compatible mapping.

    Returns:
        Parsed configuration with malformed scalar fields restored to defaults.
    """

    raw = dict(cfg or {})
    defaults = ChanceConstrainedMPCConfig()
    converters: dict[str, Any] = {
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
        "chance_constraint_formulation": str,
        "max_collision_risk": float,
        "radial_quadrature_order": int,
        "angular_quadrature_order": int,
    }
    kwargs: dict[str, Any] = {}
    for field in fields(ChanceConstrainedMPCConfig):
        value = raw.get(field.name, getattr(defaults, field.name))
        try:
            kwargs[field.name] = converters[field.name](value)
        except (TypeError, ValueError):
            default_value = getattr(defaults, field.name)
            warnings.warn(
                (
                    f"Invalid chance_constrained_mpc config value '{field.name}': {value!r}; "
                    f"falling back to default {default_value!r}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            kwargs[field.name] = default_value
    return ChanceConstrainedMPCConfig(**kwargs)


def build_chance_constrained_mpc_adapter(
    algo_config: dict[str, Any] | None,
) -> ChanceConstrainedMPCPlannerAdapter:
    """Fail closed until a #2844 provider is passed through an explicit integration path."""

    config = build_chance_constrained_mpc_config(algo_config)
    raise ValueError(
        "chance_constrained_mpc is unavailable: its #2844 K-mode/GMM predictor provider is not "
        f"configured (requested backend {config.predictor_backend!r})"
    )


__all__ = [
    "ChanceConstrainedMPCConfig",
    "ChanceConstrainedMPCPlannerAdapter",
    "GaussianMixturePedestrianForecast",
    "GaussianMixturePedestrianPredictor",
    "build_chance_constrained_mpc_adapter",
    "build_chance_constrained_mpc_config",
]
