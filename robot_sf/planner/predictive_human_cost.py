"""Predictive, motion-aligned anisotropic Gaussian human-cost primitive.

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

Formula (per pedestrian *p* at future time *t*)::

    heading_p = atan2(vy_p, vx_p)  if  ||v_p|| > eps,  else stationary_heading_rad
    center_p  = pos_p + v_p * t
    delta     = robot_pos - center_p
    longitudinal = dot(delta, [cos(heading_p), sin(heading_p)])
    lateral      = dot(delta, [-sin(heading_p), cos(heading_p)])
    sigma_L(t)   = longitudinal_sigma_m + forward_speed_gain * ||v_p|| * t
    exponent     = -0.5 * ((longitudinal / sigma_L(t))^2 + (lateral / lateral_sigma_m)^2)
    cost_p       = exp(exponent)              if ||delta|| <= cutoff_distance_m  else 0

Aggregation across pedestrians: ``sum`` (default), ``max``, or ``mean``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

PREDICTIVE_GAUSSIAN_HUMAN_COST_SCHEMA = "predictive_gaussian_human_cost.v2"

_AGGREGATION_MODES = frozenset({"sum", "max", "mean"})


def _coerce_numeric(value: Any, name: str) -> float:
    """Convert one numeric parameter while rejecting boolean masquerades.

    Returns:
        float: The converted numeric value.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"predictive human cost {name} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"predictive human cost {name} must be numeric") from exc


def _validate_positive_finite(value: float, name: str) -> None:
    """Reject non-finite or non-positive values."""

    if not np.isfinite(value):
        raise ValueError(f"predictive human cost {name} must be finite")
    if value <= 0.0:
        raise ValueError(f"predictive human cost {name} must be positive")


def _validate_non_negative_finite(value: float, name: str) -> None:
    """Reject non-finite or negative values."""

    if not np.isfinite(value):
        raise ValueError(f"predictive human cost {name} must be finite")
    if value < 0.0:
        raise ValueError(f"predictive human cost {name} must be non-negative")


@dataclass(frozen=True, slots=True)
class PredictiveGaussianHumanCostConfig:
    """Immutable opt-in configuration for the anisotropic Gaussian human cost.

    ``forward_speed_gain`` has units of seconds: the longitudinal standard
    deviation is ``longitudinal_sigma_m + forward_speed_gain * speed * time``.
    This intentionally names the adaptation rather than claiming that it is
    the external source's exact parameterization.

    ``cutoff_distance_m`` clips per-pedestrian contributions: when the
    Euclidean distance from the robot position to the pedestrian's predicted
    center exceeds this threshold the contribution is zero rather than
    numerically negligible.  The default ``None`` preserves the original
    unbounded Gaussian tail and remains standard-JSON serializable.

    ``aggregation`` controls how per-pedestrian cost scalars are combined:
    ``sum`` (default, matching prior behavior), ``max``, or ``mean``.
    """

    enabled: bool = False
    weight: float = 1.0
    longitudinal_sigma_m: float = 0.75
    lateral_sigma_m: float = 0.45
    forward_speed_gain: float = 0.8
    stationary_heading_rad: float = 0.0
    cutoff_distance_m: float | None = None
    aggregation: Literal["sum", "max", "mean"] = "sum"

    def __post_init__(self) -> None:
        """Reject non-finite, non-positive, or unsupported parameters."""

        if not isinstance(self.enabled, bool):
            raise ValueError("predictive human cost enabled must be boolean")
        for name in ("weight", "longitudinal_sigma_m", "lateral_sigma_m", "forward_speed_gain"):
            _validate_non_negative_finite(_coerce_numeric(getattr(self, name), name), name)
        _validate_positive_finite(
            _coerce_numeric(self.longitudinal_sigma_m, "longitudinal_sigma_m"),
            "longitudinal_sigma_m",
        )
        _validate_positive_finite(
            _coerce_numeric(self.lateral_sigma_m, "lateral_sigma_m"),
            "lateral_sigma_m",
        )
        if self.cutoff_distance_m is not None:
            _validate_positive_finite(
                _coerce_numeric(self.cutoff_distance_m, "cutoff_distance_m"),
                "cutoff_distance_m",
            )
        _validate_non_negative_finite(_coerce_numeric(self.weight, "weight"), "weight")
        if not np.isfinite(_coerce_numeric(self.stationary_heading_rad, "stationary_heading_rad")):
            raise ValueError("predictive human cost stationary_heading_rad must be finite")
        if not isinstance(self.aggregation, str) or self.aggregation not in _AGGREGATION_MODES:
            raise ValueError(
                f"predictive human cost aggregation must be one of {sorted(_AGGREGATION_MODES)}; "
                f"got {self.aggregation!r}"
            )

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
        "cutoff_distance_m",
    ):
        if name in values and values[name] is not None:
            values[name] = _coerce_numeric(values[name], name)
    if "aggregation" in values and not isinstance(values["aggregation"], str):
        raise ValueError("predictive human cost aggregation must be a string")
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

        The aggregation mode (``sum``, ``max``, or ``mean``) controls how per-
        pedestrian cost scalars are combined across the pedestrian axis.
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
        per_ped = np.exp(exponent)

        cutoff = self.config.cutoff_distance_m
        if cutoff is not None:
            dists = np.linalg.norm(delta, axis=2)
            beyond = dists > float(cutoff)
            per_ped = np.where(beyond, 0.0, per_ped)

        agg = str(self.config.aggregation)
        if agg == "max":
            result = np.max(per_ped, axis=1)
        elif agg == "mean":
            result = np.mean(per_ped, axis=1)
        else:
            result = np.sum(per_ped, axis=1)
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
