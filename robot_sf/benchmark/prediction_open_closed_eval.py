"""Paired open-loop and closed-loop pedestrian-prediction evaluation.

This module is a diagnostic reporting layer for issue #3973.  It projects
existing ForecastBatch.v1 forecast metrics and existing benchmark episode
metrics into one predictor-level report without changing collision, near-miss,
timeout, or forecast metric semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.benchmark.forecast_batch import ForecastBatch, validate_forecast_batch
from robot_sf.benchmark.forecast_metrics import GroundTruthPositions, evaluate_forecast_batch
from robot_sf.benchmark.pedestrian_forecast import chi_square_2d_threshold

PREDICTION_OPEN_CLOSED_SCHEMA_VERSION = "PredictionOpenClosedEval.v1"
PREDICTION_OPEN_CLOSED_CLAIM_BOUNDARY = (
    "Diagnostic CPU evaluation protocol. Open-loop forecast quality is reported side by side "
    "with closed-loop navigation outcomes; no causal safety or planner-promotion claim."
)
_TIMEOUT_REASONS = frozenset({"timeout", "truncated", "max_steps"})


@dataclass(frozen=True)
class _ScalarSummary:
    value: float | None
    status: str
    denominator: int
    note: str | None = None


def compute_open_loop_prediction_metrics(
    *,
    forecast_batches: list[ForecastBatch | dict[str, Any]],
    ground_truth_by_batch: Mapping[str, GroundTruthPositions] | GroundTruthPositions,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Summarize open-loop ForecastBatch.v1 metrics for issue #3973.

    Args:
        forecast_batches: ForecastBatch objects or JSON-compatible dictionaries.
        ground_truth_by_batch: Either one ground-truth mapping for a single batch,
            or a mapping keyed by batch id, scenario id, or list index.
        confidence_level: Gaussian confidence level used for calibration coverage.

    Returns:
        JSON-compatible summary with ADE, FDE, calibration error, prediction spread,
        and denominator/status metadata.
    """
    if not forecast_batches:
        return _open_loop_unavailable("no forecast batches supplied")

    ade_values: list[float] = []
    fde_values: list[float] = []
    calibration_hits = 0
    calibration_count = 0
    spread_values: list[float] = []
    deterministic_spread_only = True
    missing_truth_batches: list[str] = []

    for index, raw_batch in enumerate(forecast_batches):
        batch = validate_forecast_batch(raw_batch)
        batch_key = _batch_key(batch, index)
        ground_truth = _ground_truth_for_batch(
            batch=batch,
            index=index,
            batch_count=len(forecast_batches),
            ground_truth_by_batch=ground_truth_by_batch,
        )
        if ground_truth is None:
            missing_truth_batches.append(batch_key)
            continue

        metric_report = evaluate_forecast_batch(batch, ground_truth)
        ade_values.extend(_metric_values(metric_report, "ade"))
        fde_values.extend(_metric_values(metric_report, "fde"))

        hits, count = _gaussian_calibration_counts(
            batch=batch,
            ground_truth=ground_truth,
            confidence_level=confidence_level,
        )
        calibration_hits += hits
        calibration_count += count

        spread = _prediction_spread_values(batch)
        spread_values.extend(spread.values)
        deterministic_spread_only = deterministic_spread_only and spread.deterministic_only

    ade = _mean_summary(ade_values, unavailable_note="no deterministic ADE rows available")
    fde = _mean_summary(fde_values, unavailable_note="no deterministic FDE rows available")
    calibration = _calibration_summary(
        hits=calibration_hits,
        count=calibration_count,
        confidence_level=confidence_level,
    )
    prediction_spread = _spread_summary(spread_values, deterministic_spread_only)

    return {
        "ade": ade.value,
        "fde": fde.value,
        "calibration_error": calibration.value,
        "prediction_spread": prediction_spread.value,
        "sample_count": ade.denominator,
        "statuses": {
            "ade": _status_dict(ade),
            "fde": _status_dict(fde),
            "calibration_error": _status_dict(calibration),
            "prediction_spread": _status_dict(prediction_spread),
        },
        "denominators": {
            "ade": ade.denominator,
            "fde": fde.denominator,
            "calibration_error": calibration.denominator,
            "prediction_spread": prediction_spread.denominator,
            "batch_count": len(forecast_batches),
            "missing_ground_truth_batch_count": len(missing_truth_batches),
        },
        "missing_ground_truth_batches": missing_truth_batches,
    }


def summarize_closed_loop_prediction_metrics(
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize existing closed-loop episode metrics for issue #3973.

    The source rows are benchmark episode JSONL-style dictionaries.  This function
    only projects existing fields; it does not redefine collision, near-miss, or
    timeout semantics.

    Returns:
        JSON-compatible summary with closed-loop metrics and denominator metadata.
    """
    episode_count = len(episode_rows)
    if episode_count == 0:
        return {
            "collision_rate": None,
            "near_miss_rate": None,
            "min_distance": None,
            "timeout_rate": None,
            "time_to_goal": None,
            "jerk": None,
            "closed_loop_denominators": _closed_loop_denominators(0, 0, 0, 0, 0, 0),
            "statuses": {
                field: {"status": "unavailable", "note": "no episode rows supplied"}
                for field in (
                    "collision_rate",
                    "near_miss_rate",
                    "min_distance",
                    "timeout_rate",
                    "time_to_goal",
                    "jerk",
                )
            },
        }

    collision_events = [_collision_event(row) for row in episode_rows]
    near_miss_events = [_near_miss_event(row) for row in episode_rows]
    timeout_events = [_timeout_event(row) for row in episode_rows]
    min_distances = [_number_from_row(row, "min_distance") for row in episode_rows]
    time_to_goal_values = [_number_from_row(row, "time_to_goal") for row in episode_rows]
    jerk_values = [_number_from_row(row, "jerk_mean") for row in episode_rows]

    min_distance_values = [value for value in min_distances if value is not None]
    time_to_goal_present = [value for value in time_to_goal_values if value is not None]
    jerk_present = [value for value in jerk_values if value is not None]

    return {
        "collision_rate": float(sum(collision_events) / episode_count),
        "near_miss_rate": float(sum(near_miss_events) / episode_count),
        "min_distance": float(min(min_distance_values)) if min_distance_values else None,
        "timeout_rate": float(sum(timeout_events) / episode_count),
        "time_to_goal": _mean_or_none(time_to_goal_present),
        "jerk": _mean_or_none(jerk_present),
        "closed_loop_denominators": _closed_loop_denominators(
            episode_count,
            episode_count,
            episode_count,
            len(min_distance_values),
            len(time_to_goal_present),
            len(jerk_present),
        ),
        "statuses": {
            "collision_rate": {"status": "ok", "denominator": episode_count},
            "near_miss_rate": {"status": "ok", "denominator": episode_count},
            "min_distance": _availability_status(
                len(min_distance_values), "metrics.min_distance unavailable"
            ),
            "timeout_rate": {"status": "ok", "denominator": episode_count},
            "time_to_goal": _availability_status(
                len(time_to_goal_present), "metrics.time_to_goal unavailable"
            ),
            "jerk": _availability_status(len(jerk_present), "metrics.jerk_mean unavailable"),
        },
    }


def build_open_closed_prediction_report(
    *,
    predictor_id: str,
    open_loop_report: dict[str, Any],
    closed_loop_report: dict[str, Any],
    predictor_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one predictor row in the paired evaluation report.

    Returns:
        JSON-compatible predictor report containing open-loop, closed-loop, and
        divergence fields.
    """
    metadata = dict(predictor_metadata or {})
    return {
        "predictor_id": predictor_id,
        "predictor_family": metadata.get("predictor_family", "unspecified"),
        "open_loop": open_loop_report,
        "closed_loop": closed_loop_report,
        "divergence": {
            "open_loop_rank_key": "fde",
            "closed_loop_rank_key": "collision_rate_then_near_miss_then_min_distance",
            "rank_comparison_available": False,
            "reason": "single_predictor_smoke",
        },
        "predictor_metadata": metadata,
    }


def build_paired_prediction_eval_report(
    predictor_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a schema-versioned open-loop/closed-loop prediction report.

    Returns:
        JSON-compatible paired report for one or more predictors.
    """
    reports = [dict(report) for report in predictor_reports]
    _attach_rank_comparisons(reports)
    return {
        "schema_version": PREDICTION_OPEN_CLOSED_SCHEMA_VERSION,
        "issue": 3973,
        "claim_boundary": PREDICTION_OPEN_CLOSED_CLAIM_BOUNDARY,
        "predictors": reports,
    }


@dataclass(frozen=True)
class _SpreadValues:
    values: list[float]
    deterministic_only: bool


def _open_loop_unavailable(note: str) -> dict[str, Any]:
    status = _ScalarSummary(value=None, status="unavailable", denominator=0, note=note)
    return {
        "ade": None,
        "fde": None,
        "calibration_error": None,
        "prediction_spread": None,
        "sample_count": 0,
        "statuses": {
            "ade": _status_dict(status),
            "fde": _status_dict(status),
            "calibration_error": _status_dict(status),
            "prediction_spread": _status_dict(status),
        },
        "denominators": {
            "ade": 0,
            "fde": 0,
            "calibration_error": 0,
            "prediction_spread": 0,
            "batch_count": 0,
            "missing_ground_truth_batch_count": 0,
        },
        "missing_ground_truth_batches": [],
    }


def _batch_key(batch: ForecastBatch, index: int) -> str:
    metadata = batch.metadata if isinstance(batch.metadata, dict) else {}
    for key in ("batch_id", "trace_id", "episode_id"):
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return str(batch.provenance.scenario_id or index)


def _ground_truth_for_batch(
    *,
    batch: ForecastBatch,
    index: int,
    batch_count: int,
    ground_truth_by_batch: Mapping[str, GroundTruthPositions] | GroundTruthPositions,
) -> GroundTruthPositions | None:
    if batch_count == 1 and _looks_like_ground_truth(ground_truth_by_batch):
        return ground_truth_by_batch  # type: ignore[return-value]
    key_candidates = [
        _batch_key(batch, index),
        str(batch.provenance.scenario_id),
        str(index),
    ]
    for key in key_candidates:
        value = ground_truth_by_batch.get(key)  # type: ignore[union-attr]
        if value is not None:
            return value
    return None


def _looks_like_ground_truth(value: Mapping[str, Any]) -> bool:
    if not value:
        return False
    first_value = next(iter(value.values()))
    return not isinstance(first_value, Mapping)


def _metric_values(metric_report: dict[str, Any], metric: str) -> list[float]:
    values: list[float] = []
    for row in metric_report.get("metric_rows", []):
        if row.get("metric") != metric or row.get("status") != "ok":
            continue
        value = _finite_float(row.get("value"))
        if value is not None:
            values.append(value)
    return values


def _gaussian_calibration_counts(
    *,
    batch: ForecastBatch,
    ground_truth: GroundTruthPositions,
    confidence_level: float,
) -> tuple[int, int]:
    threshold = chi_square_2d_threshold(confidence_level)
    hits = 0
    count = 0
    horizons = list(batch.provenance.horizons_s)
    for forecast in batch.forecasts:
        truth = _truth_array(ground_truth, forecast.actor_id, len(horizons))
        if truth is None or not forecast.gaussian:
            continue
        for index, gaussian in enumerate(forecast.gaussian):
            mean = _position_array(gaussian.get("mean"))
            covariance = _covariance_array(gaussian.get("covariance"))
            if mean is None or covariance is None:
                continue
            offset = truth[index] - mean
            try:
                mahalanobis_sq = float(offset.T @ np.linalg.inv(covariance) @ offset)
            except np.linalg.LinAlgError:
                continue
            if np.isfinite(mahalanobis_sq):
                count += 1
                hits += int(mahalanobis_sq <= threshold)
    return hits, count


def _prediction_spread_values(batch: ForecastBatch) -> _SpreadValues:
    values: list[float] = []
    deterministic_only = True
    for forecast in batch.forecasts:
        if forecast.gaussian:
            deterministic_only = False
            for gaussian in forecast.gaussian:
                covariance = _covariance_array(gaussian.get("covariance"))
                if covariance is not None:
                    values.append(float(np.sqrt(np.trace(covariance) / 2.0)))
        elif forecast.samples is not None:
            deterministic_only = False
            samples = np.asarray(forecast.samples, dtype=float)
            per_horizon_std = np.std(samples, axis=0)
            values.extend(float(np.linalg.norm(std_xy)) for std_xy in per_horizon_std)
    return _SpreadValues(values=values, deterministic_only=deterministic_only)


def _truth_array(
    ground_truth: GroundTruthPositions,
    actor_id: str,
    expected_steps: int,
) -> np.ndarray | None:
    if actor_id not in ground_truth:
        return None
    truth = np.asarray(ground_truth[actor_id], dtype=float)
    if truth.shape != (expected_steps, 2) or not np.all(np.isfinite(truth)):
        return None
    return truth


def _position_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    position = np.asarray(value, dtype=float)
    if position.shape != (2,) or not np.all(np.isfinite(position)):
        return None
    return position


def _covariance_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return None
    return covariance


def _mean_summary(values: Sequence[float], *, unavailable_note: str) -> _ScalarSummary:
    if not values:
        return _ScalarSummary(
            value=None,
            status="unavailable",
            denominator=0,
            note=unavailable_note,
        )
    return _ScalarSummary(value=float(np.mean(values)), status="ok", denominator=len(values))


def _calibration_summary(
    *,
    hits: int,
    count: int,
    confidence_level: float,
) -> _ScalarSummary:
    if count == 0:
        return _ScalarSummary(
            value=None,
            status="unavailable",
            denominator=0,
            note="no Gaussian uncertainty representation available",
        )
    empirical_coverage = hits / count
    return _ScalarSummary(
        value=float(abs(empirical_coverage - confidence_level)),
        status="ok",
        denominator=count,
    )


def _spread_summary(values: Sequence[float], deterministic_spread_only: bool) -> _ScalarSummary:
    if values:
        return _ScalarSummary(value=float(np.mean(values)), status="ok", denominator=len(values))
    if deterministic_spread_only:
        return _ScalarSummary(
            value=0.0,
            status="deterministic_no_spread",
            denominator=0,
            note="deterministic forecast has no uncertainty representation",
        )
    return _ScalarSummary(
        value=None,
        status="unavailable",
        denominator=0,
        note="spread inputs unavailable or invalid",
    )


def _status_dict(summary: _ScalarSummary) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": summary.status,
        "denominator": summary.denominator,
    }
    if summary.note is not None:
        payload["note"] = summary.note
    return payload


def _closed_loop_denominators(
    episode_count: int,
    collision_count: int,
    near_miss_count: int,
    min_distance_count: int,
    time_to_goal_count: int,
    jerk_count: int,
) -> dict[str, int]:
    return {
        "episode_count": episode_count,
        "collision_rate_denominator": collision_count,
        "near_miss_rate_denominator": near_miss_count,
        "min_distance_denominator": min_distance_count,
        "timeout_rate_denominator": episode_count,
        "time_to_goal_denominator": time_to_goal_count,
        "jerk_denominator": jerk_count,
    }


def _collision_event(row: Mapping[str, Any]) -> bool:
    metrics = _metrics(row)
    outcome = _mapping(row.get("outcome"))
    return bool(
        outcome.get("collision_event")
        or _number(metrics.get("total_collision_count"), default=0.0) > 0.0
        or _number(metrics.get("collisions"), default=0.0) > 0.0
        or _number(row.get("collision"), default=0.0) > 0.0
    )


def _near_miss_event(row: Mapping[str, Any]) -> bool:
    metrics = _metrics(row)
    return bool(
        _number(metrics.get("near_misses"), default=0.0) > 0.0
        or _number(row.get("near_misses"), default=0.0) > 0.0
    )


def _timeout_event(row: Mapping[str, Any]) -> bool:
    metrics = _metrics(row)
    outcome = _mapping(row.get("outcome"))
    termination_reason = str(row.get("termination_reason") or "").lower()
    return bool(
        outcome.get("timeout_event")
        or termination_reason in _TIMEOUT_REASONS
        or _number(metrics.get("timeout"), default=0.0) > 0.0
    )


def _number_from_row(row: Mapping[str, Any], key: str) -> float | None:
    metrics = _metrics(row)
    value = _finite_float(metrics.get(key))
    if value is not None:
        return value
    return _finite_float(row.get(key))


def _metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row.get("metrics"))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object, *, default: float) -> float:
    number = _finite_float(value)
    return default if number is None else number


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _availability_status(denominator: int, note: str) -> dict[str, Any]:
    if denominator:
        return {"status": "ok", "denominator": denominator}
    return {"status": "unavailable", "denominator": 0, "note": note}


def _attach_rank_comparisons(reports: list[dict[str, Any]]) -> None:
    if len(reports) < 2:
        return
    open_ranks = _rank_by_metric(
        reports,
        section="open_loop",
        key="fde",
        reverse=False,
    )
    closed_ranks = _rank_closed_loop(reports)
    for report in reports:
        predictor_id = report.get("predictor_id")
        divergence = dict(report.get("divergence") or {})
        if predictor_id in open_ranks and predictor_id in closed_ranks:
            divergence.update(
                {
                    "rank_comparison_available": True,
                    "open_loop_rank": open_ranks[predictor_id],
                    "closed_loop_rank": closed_ranks[predictor_id],
                    "rank_delta": closed_ranks[predictor_id] - open_ranks[predictor_id],
                    "reason": "multi_predictor_rank_projection",
                }
            )
        report["divergence"] = divergence


def _rank_by_metric(
    reports: Iterable[dict[str, Any]],
    *,
    section: str,
    key: str,
    reverse: bool,
) -> dict[Any, int]:
    sortable: list[tuple[float, Any]] = []
    for report in reports:
        value = _finite_float(_mapping(report.get(section)).get(key))
        if value is not None:
            sortable.append((value, report.get("predictor_id")))
    sortable.sort(reverse=reverse)
    return {predictor_id: rank for rank, (_, predictor_id) in enumerate(sortable, start=1)}


def _rank_closed_loop(reports: Iterable[dict[str, Any]]) -> dict[Any, int]:
    sortable: list[tuple[float, float, float, Any]] = []
    for report in reports:
        closed_loop = _mapping(report.get("closed_loop"))
        collision_rate = _finite_float(closed_loop.get("collision_rate"))
        near_miss_rate = _finite_float(closed_loop.get("near_miss_rate"))
        min_distance = _finite_float(closed_loop.get("min_distance"))
        if collision_rate is None or near_miss_rate is None or min_distance is None:
            continue
        sortable.append((collision_rate, near_miss_rate, -min_distance, report.get("predictor_id")))
    sortable.sort()
    return {predictor_id: rank for rank, (*_, predictor_id) in enumerate(sortable, start=1)}
