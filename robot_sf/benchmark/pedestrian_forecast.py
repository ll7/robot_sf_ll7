"""Deterministic pedestrian forecast baselines for benchmark step traces."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_FORECAST_HORIZONS_S = (0.5, 1.0, 2.0)
PEDESTRIAN_ACTOR_TYPES = frozenset({"pedestrian", "person"})


@dataclass(frozen=True)
class PedestrianState:
    """One trace-compatible pedestrian state at a single timestep."""

    id: int
    position: np.ndarray
    velocity: np.ndarray
    intent: str | None = None
    signal: str | None = None
    signal_available: bool = False
    actor_type: str = "pedestrian"

    @classmethod
    def from_trace(cls, payload: dict[str, Any]) -> PedestrianState:
        """Build a state from ``simulation_step_trace.steps[].pedestrians[]``.

        Returns:
            Trace-compatible pedestrian state.
        """

        signal_state = payload.get("signal_state")
        signal_available = False
        signal: str | None = None
        if isinstance(signal_state, dict):
            signal_available = bool(
                signal_state.get("available")
                if "available" in signal_state
                else signal_state.get("label") is not None
            )
            if signal_available and signal_state.get("label") is not None:
                signal = str(signal_state["label"])
        elif payload.get("signal_label") is not None:
            signal_available = True
            signal = str(payload["signal_label"])

        return cls(
            id=int(payload["id"]),
            position=np.asarray(payload["position"], dtype=float),
            velocity=np.asarray(payload["velocity"], dtype=float),
            intent=str(payload["intent_label"])
            if payload.get("intent_label") is not None
            else None,
            signal=signal,
            signal_available=signal_available,
            actor_type=str(payload.get("actor_type") or "pedestrian"),
        )


def is_pedestrian_actor(actor_type: str | None) -> bool:
    """True when the actor type represents a pedestrian (the default).

    Returns:
        True for pedestrians, persons, or missing types.
    """
    if actor_type is None:
        return True
    return _canonical_actor_type(actor_type) in PEDESTRIAN_ACTOR_TYPES


def _canonical_actor_type(actor_type: str | None) -> str:
    """Return the stable actor-type label used by forecast denominator metadata."""
    if actor_type is None:
        return "pedestrian"
    label = str(actor_type).strip().lower()
    return label or "pedestrian"


def actor_type_metric_key(actor_type: str | None) -> str:
    """Normalize actor-type labels for flat metric keys.

    Returns:
        Lowercase alphanumeric/underscore key segment.
    """
    label = _canonical_actor_type(actor_type)
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "unknown"


@dataclass(frozen=True)
class ForecastDistribution:
    """Gaussian occupancy ellipse for one pedestrian and horizon."""

    horizon_s: float
    mean: np.ndarray
    covariance: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PedestrianForecast:
    """Predicted distributions for one pedestrian."""

    id: int
    predictions: list[ForecastDistribution]


def constant_velocity_gaussian_baseline(
    state: PedestrianState,
    horizons_s: list[float] | tuple[float, ...] = DEFAULT_FORECAST_HORIZONS_S,
    *,
    base_std_m: float = 0.3,
    velocity_std_rate: float = 0.4,
    unavailable_context_multiplier: float = 1.5,
) -> PedestrianForecast:
    """Forecast future pedestrian occupancy with a deterministic Gaussian baseline.

    The mean follows constant velocity. Uncertainty grows with horizon and is
    widened when intent or signal context is unavailable, so unknown signal
    state is represented as uncertainty rather than an assumed phase.

    Returns:
        Forecast distributions for the requested horizons.
    """

    context_available = state.intent is not None and state.signal_available
    multiplier = 1.0 if context_available else unavailable_context_multiplier
    predictions: list[ForecastDistribution] = []

    for horizon_s in horizons_s:
        horizon = float(horizon_s)
        std_m = (base_std_m + velocity_std_rate * horizon) * multiplier
        if std_m <= 0.0:
            raise ValueError("forecast standard deviation must be positive")
        predictions.append(
            ForecastDistribution(
                horizon_s=horizon,
                mean=state.position + state.velocity * horizon,
                covariance=np.eye(2, dtype=float) * (std_m**2),
                metadata={
                    "model": "constant_velocity_gaussian",
                    "std_m": float(std_m),
                    "is_intent_aware": state.intent is not None,
                    "is_signal_aware": state.signal_available,
                    "signal_state": state.signal if state.signal_available else "unknown",
                    "context_status": "available" if context_available else "uncertain",
                },
            )
        )

    return PedestrianForecast(id=state.id, predictions=predictions)


def evaluate_forecast(
    forecast: PedestrianForecast,
    ground_truth: dict[float, np.ndarray],
    *,
    confidence_level: float = 0.95,
    robot_positions: dict[float, np.ndarray] | None = None,
    collision_distance_m: float = 0.8,
) -> dict[str, float]:
    """Evaluate likelihood, calibration, miss rate, and collision relevance.

    Returns:
        Flat metric dictionary keyed by metric name and horizon suffix.
    """

    metrics: dict[str, float] = {}
    confidence_threshold = chi_square_2d_threshold(confidence_level)
    for prediction in forecast.predictions:
        horizon = prediction.horizon_s
        if horizon not in ground_truth:
            continue

        actual_position = np.asarray(ground_truth[horizon], dtype=float)
        offset = actual_position - prediction.mean
        inverse_covariance = np.linalg.inv(prediction.covariance)
        determinant = float(np.linalg.det(prediction.covariance))
        mahalanobis_sq = float(offset.T @ inverse_covariance @ offset)
        log_likelihood = -0.5 * (
            mahalanobis_sq + math.log(determinant) + 2.0 * math.log(2.0 * math.pi)
        )
        within_confidence = mahalanobis_sq <= confidence_threshold
        suffix = f"{horizon:g}s"

        metrics[f"ade_{suffix}"] = float(np.linalg.norm(offset))
        metrics[f"log_likelihood_{suffix}"] = float(log_likelihood)
        metrics[f"negative_log_likelihood_{suffix}"] = float(-log_likelihood)
        metrics[f"mahalanobis_dist_{suffix}"] = float(math.sqrt(mahalanobis_sq))
        metrics[f"within_{int(confidence_level * 100)}ci_{suffix}"] = (
            1.0 if within_confidence else 0.0
        )
        metrics[f"miss_rate_{suffix}"] = 0.0 if within_confidence else 1.0
        metrics[f"calibration_error_{suffix}"] = abs(
            (1.0 if within_confidence else 0.0) - confidence_level
        )

        if robot_positions is not None and horizon in robot_positions:
            robot_position = np.asarray(robot_positions[horizon], dtype=float)
            actual_relevant = float(
                np.linalg.norm(actual_position - robot_position) <= collision_distance_m
            )
            predicted_relevant = float(
                ellipse_overlaps_point(
                    mean=prediction.mean,
                    covariance=prediction.covariance,
                    point=robot_position,
                    confidence_threshold=confidence_threshold,
                    radius_m=collision_distance_m,
                )
            )
            metrics[f"collision_relevance_error_{suffix}"] = abs(
                predicted_relevant - actual_relevant
            )

    return metrics


def chi_square_2d_threshold(confidence_level: float) -> float:
    """Return the chi-square threshold for a 2D Gaussian confidence ellipse."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    return float(-2.0 * math.log(1.0 - confidence_level))


def ellipse_overlaps_point(
    *,
    mean: np.ndarray,
    covariance: np.ndarray,
    point: np.ndarray,
    confidence_threshold: float,
    radius_m: float,
) -> bool:
    """Approximate whether an ellipse overlaps a circular relevance region.

    The first baseline emits isotropic covariance, but this helper uses the
    maximum covariance eigenvalue as a conservative bounding radius so later
    non-isotropic forecasts can still be scored deterministically.

    Returns:
        True when the confidence ellipse overlaps the circular relevance region.
    """

    max_variance = float(np.max(np.linalg.eigvalsh(covariance)))
    ellipse_radius = math.sqrt(max(0.0, confidence_threshold * max_variance))
    return bool(np.linalg.norm(np.asarray(mean) - np.asarray(point)) <= ellipse_radius + radius_m)


def compute_batch_forecast_metrics(
    trace_steps: list[dict[str, Any]],
    horizons_s: list[float] | tuple[float, ...] = DEFAULT_FORECAST_HORIZONS_S,
    *,
    dt_s: float = 0.1,
    confidence_level: float = 0.95,
    collision_distance_m: float = 0.8,
) -> dict[str, float]:
    """Compute aggregate forecast metrics over benchmark trace steps.

    Returns:
        Mean metrics plus per-metric denominator counts and actor exclusion metadata.
    """

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    (
        candidate_count,
        excluded_count,
        excluded_by_type,
        sample_metrics,
    ) = _collect_sample_metrics(
        trace_steps,
        horizons_s,
        dt_s,
        confidence_level,
        collision_distance_m,
    )

    summary = {
        "pedestrian_forecast_candidate_count": float(candidate_count),
        "pedestrian_forecast_included_actor_count": float(candidate_count - excluded_count),
        "pedestrian_forecast_excluded_actor_count": float(excluded_count),
    }
    for actor_type, count in excluded_by_type.items():
        summary[f"pedestrian_forecast_excluded_{actor_type}_count"] = float(count)

    if not sample_metrics:
        summary["forecast_evaluable_samples"] = 0.0
        return summary

    values_by_key: dict[str, list[float]] = defaultdict(list)
    for sample in sample_metrics:
        for key, value in sample.items():
            values_by_key[key].append(float(value))

    summary["forecast_evaluable_samples"] = float(len(sample_metrics))
    for key, values in sorted(values_by_key.items()):
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"count_{key}"] = float(len(values))
    return summary


def _collect_sample_metrics(
    trace_steps: list[dict[str, Any]],
    horizons_s: list[float] | tuple[float, ...],
    dt_s: float,
    confidence_level: float,
    collision_distance_m: float,
) -> tuple[int, int, dict[str, int], list[dict[str, float]]]:
    candidate_count = 0
    excluded_count = 0
    excluded_by_type: dict[str, int] = defaultdict(int)
    sample_metrics: list[dict[str, float]] = []

    for step_index, step in enumerate(trace_steps):
        for pedestrian_payload in step.get("pedestrians", []):
            candidate_count += 1
            actor_type = pedestrian_payload.get("actor_type")
            if not is_pedestrian_actor(actor_type):
                excluded_count += 1
                excluded_by_type[actor_type_metric_key(actor_type)] += 1
                continue

            metrics = _evaluate_single_pedestrian(
                pedestrian_payload,
                step_index,
                trace_steps,
                horizons_s,
                dt_s,
                confidence_level,
                collision_distance_m,
            )
            if metrics:
                sample_metrics.append(metrics)

    return candidate_count, excluded_count, excluded_by_type, sample_metrics


def _evaluate_single_pedestrian(
    pedestrian_payload: dict[str, Any],
    step_index: int,
    trace_steps: list[dict[str, Any]],
    horizons_s: list[float] | tuple[float, ...],
    dt_s: float,
    confidence_level: float,
    collision_distance_m: float,
) -> dict[str, float] | None:
    state = PedestrianState.from_trace(pedestrian_payload)
    ground_truth = _future_pedestrian_positions(
        state.id,
        step_index,
        trace_steps,
        horizons_s,
        dt_s,
    )
    if not ground_truth:
        return None
    return evaluate_forecast(
        constant_velocity_gaussian_baseline(state, horizons_s),
        ground_truth,
        confidence_level=confidence_level,
        robot_positions=_future_robot_positions(
            step_index,
            trace_steps,
            horizons_s,
            dt_s,
        ),
        collision_distance_m=collision_distance_m,
    )


def _future_pedestrian_positions(
    pedestrian_id: int,
    start_index: int,
    trace_steps: list[dict[str, Any]],
    horizons_s: list[float] | tuple[float, ...],
    dt_s: float,
) -> dict[float, np.ndarray]:
    positions: dict[float, np.ndarray] = {}
    for horizon_s in horizons_s:
        future_step = start_index + round(float(horizon_s) / dt_s)
        if future_step >= len(trace_steps):
            continue
        for pedestrian in trace_steps[future_step].get("pedestrians", []):
            if int(pedestrian["id"]) == pedestrian_id:
                positions[float(horizon_s)] = np.asarray(pedestrian["position"], dtype=float)
                break
    return positions


def _future_robot_positions(
    start_index: int,
    trace_steps: list[dict[str, Any]],
    horizons_s: list[float] | tuple[float, ...],
    dt_s: float,
) -> dict[float, np.ndarray]:
    positions: dict[float, np.ndarray] = {}
    for horizon_s in horizons_s:
        future_step = start_index + round(float(horizon_s) / dt_s)
        if future_step >= len(trace_steps):
            continue
        robot = trace_steps[future_step].get("robot")
        if isinstance(robot, dict) and robot.get("position") is not None:
            positions[float(horizon_s)] = np.asarray(robot["position"], dtype=float)
    return positions
