"""Predictive, motion-aligned Gaussian human-cost primitive.

The primitive is a small Robot SF adaptation of the predictive Gaussian
interaction-field idea: pedestrian positions are advanced over the rollout
horizon and each future position contributes an anisotropic Gaussian penalty.
The longitudinal axis follows the pedestrian velocity and its spread grows
with speed and prediction time. The formula and parameters are explicit so a
future reproduction can replace them without silently changing an existing
benchmark row.

This module is a planner-cost primitive, not a reproduction of an external
paper and not a safety certificate. Its default configuration is disabled;
callers must opt in and record the resulting configuration digest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

PREDICTIVE_GAUSSIAN_HUMAN_COST_SCHEMA = "predictive_gaussian_human_cost.v1"


@dataclass(frozen=True, slots=True)
class PredictiveGaussianHumanCostConfig:
    """Parameters for the opt-in predictive Gaussian human cost.

    ``forward_speed_gain`` has units of seconds: the longitudinal standard
    deviation is ``longitudinal_sigma_m + forward_speed_gain * speed * time``.
    This intentionally names the adaptation rather than claiming that it is
    the external source's exact parameterization.
    """

    enabled: bool = False
    weight: float = 1.0
    longitudinal_sigma_m: float = 0.75
    lateral_sigma_m: float = 0.45
    forward_speed_gain: float = 0.8
    stationary_heading_rad: float = 0.0

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive Gaussian parameters."""

        if not isinstance(self.enabled, bool):
            raise ValueError("predictive human cost enabled must be boolean")
        for name in ("weight", "longitudinal_sigma_m", "lateral_sigma_m", "forward_speed_gain"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"predictive human cost {name} must be finite")
        if float(self.weight) < 0.0:
            raise ValueError("predictive human cost weight must be non-negative")
        if float(self.longitudinal_sigma_m) <= 0.0:
            raise ValueError("predictive human cost longitudinal_sigma_m must be positive")
        if float(self.lateral_sigma_m) <= 0.0:
            raise ValueError("predictive human cost lateral_sigma_m must be positive")
        if float(self.forward_speed_gain) < 0.0:
            raise ValueError("predictive human cost forward_speed_gain must be non-negative")
        if not np.isfinite(float(self.stationary_heading_rad)):
            raise ValueError("predictive human cost stationary_heading_rad must be finite")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration mapping."""

        return asdict(self)


def build_predictive_gaussian_human_cost_config(
    payload: dict[str, Any] | None,
) -> PredictiveGaussianHumanCostConfig:
    """Build the cost configuration from a nested planner mapping.

    Returns:
        Validated predictive Gaussian human-cost configuration.
    """

    if payload is None:
        return PredictiveGaussianHumanCostConfig()
    if not isinstance(payload, dict):
        raise ValueError("predictive_human_cost must be a mapping")
    allowed = set(PredictiveGaussianHumanCostConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown predictive human cost keys: {unknown}")
    values = dict(payload)
    if "enabled" in values and not isinstance(values["enabled"], bool):
        raise ValueError("predictive human cost enabled must be boolean")
    for name in (
        "weight",
        "longitudinal_sigma_m",
        "lateral_sigma_m",
        "forward_speed_gain",
        "stationary_heading_rad",
    ):
        if name in values:
            values[name] = float(values[name])
    return PredictiveGaussianHumanCostConfig(**values)


def _validate_pedestrian_arrays(
    pedestrian_positions: np.ndarray,
    pedestrian_velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize pedestrian position/velocity arrays.

    Returns:
        Position and velocity arrays with shape ``(P, 2)``.
    """

    positions = np.asarray(pedestrian_positions, dtype=float)
    velocities = np.asarray(pedestrian_velocities, dtype=float)
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError("pedestrian_positions must have shape (P, 2)")
    if velocities.shape != positions.shape:
        raise ValueError("pedestrian_velocities must match pedestrian_positions shape")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError("pedestrian positions and velocities must be finite")
    return positions, velocities


class PredictiveGaussianHumanCost:
    """Evaluate future motion-aligned Gaussian penalties for rollout points."""

    def __init__(self, config: PredictiveGaussianHumanCostConfig | None = None) -> None:
        """Store a validated opt-in configuration."""

        self.config = config or PredictiveGaussianHumanCostConfig()

    def evaluate(
        self,
        robot_positions: np.ndarray,
        pedestrian_positions: np.ndarray,
        pedestrian_velocities: np.ndarray,
        *,
        time_s: float,
    ) -> float | np.ndarray:
        """Return the unweighted Gaussian cost at one future time.

        ``robot_positions`` may be one point with shape ``(2,)`` or a batch of
        points with shape ``(N, 2)``. The returned value is a float for one
        point and an ``(N,)`` array for a batch. Empty pedestrian arrays return
        zero without inventing an unavailable metric.
        """

        points = np.asarray(robot_positions, dtype=float)
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, 2)
        if points.ndim != 2 or points.shape[-1] != 2:
            raise ValueError("robot_positions must have shape (2,) or (N, 2)")
        if not np.all(np.isfinite(points)):
            raise ValueError("robot_positions must be finite")
        time_value = float(time_s)
        if not np.isfinite(time_value) or time_value < 0.0:
            raise ValueError("time_s must be finite and non-negative")
        positions, velocities = _validate_pedestrian_arrays(
            pedestrian_positions,
            pedestrian_velocities,
        )
        if not self.config.enabled or positions.shape[0] == 0:
            result = np.zeros(points.shape[0], dtype=float)
            return float(result[0]) if single_point else result

        speed = np.linalg.norm(velocities, axis=1)
        headings = np.where(
            speed > 1e-9,
            np.arctan2(velocities[:, 1], velocities[:, 0]),
            float(self.config.stationary_heading_rad),
        )
        cos_heading = np.cos(headings)
        sin_heading = np.sin(headings)
        centers = positions + velocities * time_value
        delta = points[:, None, :] - centers[None, :, :]
        longitudinal = delta[..., 0] * cos_heading[None, :] + delta[..., 1] * sin_heading[None, :]
        lateral = -delta[..., 0] * sin_heading[None, :] + delta[..., 1] * cos_heading[None, :]
        longitudinal_sigma = (
            float(self.config.longitudinal_sigma_m)
            + float(self.config.forward_speed_gain) * speed * time_value
        )
        exponent = -0.5 * (
            np.square(longitudinal / longitudinal_sigma[None, :])
            + np.square(lateral / float(self.config.lateral_sigma_m))
        )
        result = np.sum(np.exp(exponent), axis=1)
        return float(result[0]) if single_point else result

    def evaluate_trajectory(
        self,
        robot_positions: np.ndarray,
        pedestrian_positions: np.ndarray,
        pedestrian_velocities: np.ndarray,
        *,
        dt: float,
    ) -> float | np.ndarray:
        """Sum the cost over a trajectory or a batch of trajectories.

        ``robot_positions`` has shape ``(T, 2)`` or ``(N, T, 2)``. The first
        trajectory point is evaluated at ``dt`` so it matches a one-step
        forward rollout rather than the current observation timestamp.

        Returns:
            Scalar trajectory cost or one cost per batch trajectory.
        """

        trajectory = np.asarray(robot_positions, dtype=float)
        if trajectory.ndim == 2:
            batch = False
            trajectory = trajectory[None, ...]
        elif trajectory.ndim == 3:
            batch = True
        else:
            raise ValueError("robot_positions must have shape (T, 2) or (N, T, 2)")
        if trajectory.shape[-1] != 2 or trajectory.shape[1] == 0:
            raise ValueError("robot_positions must contain one or more xy rollout points")
        if not np.all(np.isfinite(trajectory)):
            raise ValueError("robot_positions must be finite")
        dt_value = float(dt)
        if not np.isfinite(dt_value) or dt_value <= 0.0:
            raise ValueError("dt must be finite and positive")
        values = np.zeros(trajectory.shape[0], dtype=float)
        for step in range(trajectory.shape[1]):
            values += np.asarray(
                self.evaluate(
                    trajectory[:, step, :],
                    pedestrian_positions,
                    pedestrian_velocities,
                    time_s=(step + 1) * dt_value,
                ),
                dtype=float,
            )
        return values if batch else float(values[0])


__all__ = [
    "PREDICTIVE_GAUSSIAN_HUMAN_COST_SCHEMA",
    "PredictiveGaussianHumanCost",
    "PredictiveGaussianHumanCostConfig",
    "build_predictive_gaussian_human_cost_config",
]
