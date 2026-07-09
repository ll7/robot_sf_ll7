"""Deterministic pedestrian forecast baselines for benchmark step traces."""

from __future__ import annotations

import inspect
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np

from robot_sf.nav.predictive_types import NeighborContext, PedestrianState

DEFAULT_FORECAST_HORIZONS_S = (0.5, 1.0, 2.0)
PEDESTRIAN_ACTOR_TYPES = frozenset({"pedestrian", "person"})


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


ForecastBaselineFunction = Callable[
    [PedestrianState, list[float] | tuple[float, ...]],
    PedestrianForecast,
]


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


def signal_aware_cv_baseline(
    state: PedestrianState,
    horizons_s: list[float] | tuple[float, ...] = DEFAULT_FORECAST_HORIZONS_S,
    *,
    base_std_m: float = 0.3,
    velocity_std_rate: float = 0.4,
    red_slowdown_factor: float = 0.4,
    signal_unavailable_multiplier: float = 1.5,
) -> PedestrianForecast:
    """Signal-aware constant-velocity baseline.

    When signal is available: red reduces mean displacement (slowdown_factor),
    green preserves mean displacement, and unrecognized present values preserve
    plain-CV without widening uncertainty.  When signal is unavailable the mean
    stays at plain-CV and uncertainty is widened rather than assuming a phase.

    This is deterministic and labeled; it does not claim human realism.

    Returns:
        Forecast distributions for the requested horizons.
    """

    predictions: list[ForecastDistribution] = []

    for horizon_s in horizons_s:
        horizon = float(horizon_s)
        std_m = base_std_m + velocity_std_rate * horizon

        if not state.signal_available or state.signal is None:
            mean = state.position + state.velocity * horizon
            signal_status = "unknown_widened"
            std_m *= signal_unavailable_multiplier
        elif state.signal == "red":
            mean = state.position + state.velocity * horizon * red_slowdown_factor
            signal_status = "red_slowed"
        elif state.signal == "green":
            mean = state.position + state.velocity * horizon
            signal_status = "green_preserved"
        else:
            mean = state.position + state.velocity * horizon
            signal_status = "unrecognized_preserved"

        if std_m <= 0.0:
            raise ValueError("forecast standard deviation must be positive")

        predictions.append(
            ForecastDistribution(
                horizon_s=horizon,
                mean=mean,
                covariance=np.eye(2, dtype=float) * (std_m**2),
                metadata={
                    "model": "signal_aware_cv",
                    "std_m": float(std_m),
                    "is_intent_aware": state.intent is not None,
                    "is_signal_aware": state.signal_available,
                    "signal_state": state.signal if state.signal_available else "unknown",
                    "signal_status": signal_status,
                },
            )
        )

    return PedestrianForecast(id=state.id, predictions=predictions)


def goal_aware_cv_baseline(
    state: PedestrianState,
    horizons_s: list[float] | tuple[float, ...] = DEFAULT_FORECAST_HORIZONS_S,
    *,
    base_std_m: float = 0.3,
    velocity_std_rate: float = 0.4,
    crossing_speed_factor: float = 1.2,
    walking_along_speed_factor: float = 0.8,
    intent_unavailable_multiplier: float = 1.3,
) -> PedestrianForecast:
    """Goal-aware constant-velocity baseline.

    When intent metadata is present: crossing increases mean displacement,
    walking_along decreases mean displacement, and unrecognized present values
    preserve plain-CV without widening uncertainty.  Absent intent keeps
    plain-CV mean and widens uncertainty rather than assuming a goal.

    This is deterministic and labeled; it does not claim human realism.

    Returns:
        Forecast distributions for the requested horizons.
    """

    predictions: list[ForecastDistribution] = []

    for horizon_s in horizons_s:
        horizon = float(horizon_s)
        std_m = base_std_m + velocity_std_rate * horizon

        if state.intent is None:
            mean = state.position + state.velocity * horizon
            intent_status = "unknown_widened"
            std_m *= intent_unavailable_multiplier
        elif state.intent == "crossing":
            mean = state.position + state.velocity * horizon * crossing_speed_factor
            intent_status = "crossing_adjusted"
        elif state.intent == "walking_along":
            mean = state.position + state.velocity * horizon * walking_along_speed_factor
            intent_status = "walking_along_adjusted"
        else:
            mean = state.position + state.velocity * horizon
            intent_status = "unrecognized_preserved"

        if std_m <= 0.0:
            raise ValueError("forecast standard deviation must be positive")

        predictions.append(
            ForecastDistribution(
                horizon_s=horizon,
                mean=mean,
                covariance=np.eye(2, dtype=float) * (std_m**2),
                metadata={
                    "model": "goal_aware_cv",
                    "std_m": float(std_m),
                    "is_intent_aware": state.intent is not None,
                    "is_signal_aware": state.signal_available,
                    "intent_state": state.intent or "unknown",
                    "intent_status": intent_status,
                },
            )
        )

    return PedestrianForecast(id=state.id, predictions=predictions)


def semantic_cv_baseline(  # noqa: PLR0913
    state: PedestrianState,
    horizons_s: list[float] | tuple[float, ...] = DEFAULT_FORECAST_HORIZONS_S,
    *,
    base_std_m: float = 0.3,
    velocity_std_rate: float = 0.4,
    red_slowdown_factor: float = 0.4,
    signal_unavailable_multiplier: float = 1.5,
    crossing_speed_factor: float = 1.2,
    walking_along_speed_factor: float = 0.8,
    intent_unavailable_multiplier: float = 1.3,
) -> PedestrianForecast:
    """Combined semantic CV baseline composing signal and goal factors.

    Applies signal_aware adjustments to the mean, then goal_aware adjustments.
    Unavailable context widens uncertainty; present but unrecognized context is
    labeled and preserved as neutral.  This is deterministic and labeled; it
    does not claim human realism.

    Returns:
        Forecast distributions for the requested horizons.
    """

    predictions: list[ForecastDistribution] = []

    for horizon_s in horizons_s:
        horizon = float(horizon_s)
        std_m = base_std_m + velocity_std_rate * horizon

        speed_factor = 1.0
        signal_status = "unknown_widened"
        intent_status = "unknown_widened"

        if not state.signal_available or state.signal is None:
            std_m *= signal_unavailable_multiplier
        elif state.signal == "red":
            speed_factor *= red_slowdown_factor
            signal_status = "red_slowed"
        elif state.signal == "green":
            signal_status = "green_preserved"
        else:
            signal_status = "unrecognized_preserved"

        if state.intent is None:
            std_m *= intent_unavailable_multiplier
        elif state.intent == "crossing":
            speed_factor *= crossing_speed_factor
            intent_status = "crossing_adjusted"
        elif state.intent == "walking_along":
            speed_factor *= walking_along_speed_factor
            intent_status = "walking_along_adjusted"
        else:
            intent_status = "unrecognized_preserved"

        mean = state.position + state.velocity * horizon * speed_factor

        if std_m <= 0.0:
            raise ValueError("forecast standard deviation must be positive")

        predictions.append(
            ForecastDistribution(
                horizon_s=horizon,
                mean=mean,
                covariance=np.eye(2, dtype=float) * (std_m**2),
                metadata={
                    "model": "semantic_cv",
                    "std_m": float(std_m),
                    "is_intent_aware": state.intent is not None,
                    "is_signal_aware": state.signal_available,
                    "signal_state": state.signal if state.signal_available else "unknown",
                    "signal_status": signal_status,
                    "intent_state": state.intent or "unknown",
                    "intent_status": intent_status,
                },
            )
        )

    return PedestrianForecast(id=state.id, predictions=predictions)


def interaction_aware_cv_baseline(
    state: PedestrianState,
    horizons_s: list[float] | tuple[float, ...] = DEFAULT_FORECAST_HORIZONS_S,
    *,
    base_std_m: float = 0.3,
    velocity_std_rate: float = 0.4,
    interaction_radius_m: float = 2.0,
    avoidance_strength: float = 0.15,
    crowding_std_factor: float = 0.25,
    neighbors: list[NeighborContext] | None = None,
) -> PedestrianForecast:
    """Interaction-aware constant-velocity baseline.

    Modulates the mean displacement and uncertainty based on proximity to
    neighboring pedestrians: nearby agents push the forecast away (repulsive
    interaction) and increase uncertainty proportionally to crowding density.

    When no neighbor context is provided the forecast degrades to plain CV
    without widening uncertainty, preserving a deterministic diagnostic-only
    baseline.  This is labeled and deterministic; it does not claim human
    realism.

    Returns:
        Forecast distributions for the requested horizons.
    """
    predictions: list[ForecastDistribution] = []

    effective_neighbors = neighbors or []
    active_neighbors = (
        _active_neighbor_repulsion(
            state.position,
            effective_neighbors,
            interaction_radius_m,
        )
        if effective_neighbors
        else []
    )
    active_neighbor_count = len(active_neighbors)
    repulsion = np.zeros(2, dtype=float)
    crowding_std_increment = 0.0
    interaction_status = "no_neighbors"
    if effective_neighbors:
        if active_neighbors:
            repulsion = _compute_repulsion(
                state.position,
                active_neighbors,
                interaction_radius_m,
                avoidance_strength,
            )
            crowding_density = active_neighbor_count / (math.pi * interaction_radius_m**2)
            crowding_std_increment = crowding_std_factor * math.sqrt(crowding_density)
            interaction_status = "repulsion_active"
        else:
            interaction_status = "no_neighbors_in_radius"

    for horizon_s in horizons_s:
        horizon = float(horizon_s)
        std_m = base_std_m + velocity_std_rate * horizon + crowding_std_increment
        mean = state.position + state.velocity * horizon + repulsion * horizon

        if std_m <= 0.0:
            raise ValueError("forecast standard deviation must be positive")

        predictions.append(
            ForecastDistribution(
                horizon_s=horizon,
                mean=mean,
                covariance=np.eye(2, dtype=float) * (std_m**2),
                metadata={
                    "model": "interaction_aware_cv",
                    "std_m": float(std_m),
                    "is_intent_aware": state.intent is not None,
                    "is_signal_aware": state.signal_available,
                    "signal_state": state.signal if state.signal_available else "unknown",
                    "interaction_status": interaction_status,
                    "neighbor_count": float(len(effective_neighbors)),
                    "active_neighbor_count": float(active_neighbor_count),
                },
            )
        )

    return PedestrianForecast(id=state.id, predictions=predictions)


def _active_neighbor_repulsion(
    ego_position: np.ndarray,
    neighbors: list[NeighborContext],
    interaction_radius_m: float,
) -> list[NeighborContext]:
    """Return neighbors within the interaction radius."""
    return [
        n
        for n in neighbors
        if float(np.linalg.norm(n.position - ego_position)) <= interaction_radius_m
    ]


def _compute_repulsion(
    ego_position: np.ndarray,
    active_neighbors: list[NeighborContext],
    interaction_radius_m: float,
    avoidance_strength: float,
) -> np.ndarray:
    """Compute a mean repulsive displacement vector from active neighbors.

    Each neighbor contributes a repulsive force inversely proportional to
    distance, bounded by the interaction radius.  The result is a per-step
    displacement correction that accumulates linearly with horizon.

    Returns:
        2D repulsive displacement vector.
    """
    repulsion = np.zeros(2, dtype=float)
    for neighbor in active_neighbors:
        diff = ego_position - neighbor.position
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6:
            continue
        weight = 1.0 - dist / interaction_radius_m
        repulsion += (diff / dist) * weight * avoidance_strength
    return repulsion


def risk_filtered_cv_baseline(
    state: PedestrianState,
    horizons_s: list[float] | tuple[float, ...] = DEFAULT_FORECAST_HORIZONS_S,
    *,
    base_std_m: float = 0.3,
    velocity_std_rate: float = 0.4,
    robot_position: np.ndarray | None = None,
    risk_distance_m: float = 3.0,
    filter_std_multiplier: float = 2.0,
) -> PedestrianForecast:
    """Risk-filtered constant-velocity baseline.

    Generates a plain-CV forecast and then widens uncertainty for predictions
    whose mean is far from the robot.  When ``robot_position`` is omitted the
    baseline degrades to plain CV without widening, preserving a deterministic
    diagnostic-only mode.  The filter is intentionally conservative: it never
    removes a prediction, only marks low-relevance predictions as filtered and
    inflates their covariance.

    Returns:
        Forecast distributions for the requested horizons.
    """

    cv_forecast = constant_velocity_gaussian_baseline(
        state,
        horizons_s=horizons_s,
        base_std_m=base_std_m,
        velocity_std_rate=velocity_std_rate,
    )
    predictions: list[ForecastDistribution] = []

    for cv_prediction in cv_forecast.predictions:
        std_m = float(cv_prediction.metadata["std_m"])
        relevance_status = "robot_unavailable"
        if robot_position is not None:
            robot_pos = np.asarray(robot_position, dtype=float)
            distance = float(np.linalg.norm(cv_prediction.mean - robot_pos))
            if distance <= risk_distance_m:
                relevance_status = "collision_relevant"
            else:
                relevance_status = "filtered_low_relevance"
                std_m *= filter_std_multiplier

        if std_m <= 0.0:
            raise ValueError("forecast standard deviation must be positive")

        predictions.append(
            ForecastDistribution(
                horizon_s=cv_prediction.horizon_s,
                mean=cv_prediction.mean,
                covariance=np.eye(2, dtype=float) * (std_m**2),
                metadata={
                    "model": "risk_filtered_cv",
                    "std_m": float(std_m),
                    "is_intent_aware": state.intent is not None,
                    "is_signal_aware": state.signal_available,
                    "signal_state": state.signal if state.signal_available else "unknown",
                    "relevance_status": relevance_status,
                    "risk_distance_m": float(risk_distance_m),
                    "filter_std_multiplier": float(filter_std_multiplier),
                },
            )
        )

    return PedestrianForecast(id=state.id, predictions=predictions)


BASELINE_FUNCTIONS: dict[str, ForecastBaselineFunction] = {
    "cv": constant_velocity_gaussian_baseline,
    "signal_aware": signal_aware_cv_baseline,
    "goal_aware": goal_aware_cv_baseline,
    "semantic": semantic_cv_baseline,
    "interaction_aware": interaction_aware_cv_baseline,
}


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
    baseline_function: ForecastBaselineFunction | None = None,
) -> dict[str, float]:
    """Compute aggregate forecast metrics over benchmark trace steps.

    Args:
        trace_steps: Sequence of simulation step dicts with pedestrian lists.
        horizons_s: Forecast horizons to evaluate.
        dt_s: Simulation timestep for horizon alignment.
        confidence_level: Confidence level for evaluation ellipse.
        collision_distance_m: Collision proximity threshold.
        baseline_function: Forecast function taking (state, horizons_s) and
            returning a PedestrianForecast. Defaults to constant_velocity_gaussian_baseline.

    Returns:
        Mean metrics plus per-metric denominator counts and actor exclusion metadata.
    """

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    if baseline_function is None:
        baseline_function = constant_velocity_gaussian_baseline

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
        baseline_function,
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
    baseline_function: ForecastBaselineFunction,
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
                baseline_function,
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
    baseline_function: ForecastBaselineFunction,
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

    neighbors = _collect_neighbor_context(pedestrian_payload, step_index, trace_steps)
    forecast = _call_baseline_with_neighbors(baseline_function, state, horizons_s, neighbors)

    return evaluate_forecast(
        forecast,
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


def _collect_neighbor_context(
    ego_payload: dict[str, Any],
    step_index: int,
    trace_steps: list[dict[str, Any]],
) -> list[NeighborContext]:
    """Build neighbor context from the current step, excluding the ego pedestrian.

    Returns:
        List of NeighborContext for nearby pedestrians.
    """
    if step_index >= len(trace_steps):
        return []
    ego_id = int(ego_payload["id"])
    neighbors: list[NeighborContext] = []
    for ped in trace_steps[step_index].get("pedestrians", []):
        if int(ped["id"]) == ego_id:
            continue
        neighbors.append(
            NeighborContext(
                position=np.asarray(ped["position"], dtype=float),
                velocity=np.asarray(ped["velocity"], dtype=float),
                actor_type=str(ped.get("actor_type") or "pedestrian"),
            )
        )
    return neighbors


def _call_baseline_with_neighbors(
    baseline_function: ForecastBaselineFunction,
    state: PedestrianState,
    horizons_s: list[float] | tuple[float, ...],
    neighbors: list[NeighborContext],
) -> PedestrianForecast:
    """Call baseline, passing neighbors if the function accepts them.

    Returns:
        Forecast from the baseline function.
    """

    if _baseline_accepts_neighbors(baseline_function):
        return baseline_function(state, horizons_s, neighbors=neighbors)  # type: ignore[call-arg]
    return baseline_function(state, horizons_s)


@cache
def _baseline_accepts_neighbors(baseline_function: ForecastBaselineFunction) -> bool:
    """Return whether a baseline accepts neighbor context.

    Returns:
        True when ``baseline_function`` declares a ``neighbors`` parameter.
    """
    return "neighbors" in inspect.signature(baseline_function).parameters


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
