"""Predictor-agnostic metrics for ForecastBatch.v1 artifacts.

These helpers evaluate forecast artifacts without making navigation-benefit
claims. The output keeps metric availability, denominators, horizon metadata,
actor class, scenario id, and observation tier explicit so deterministic and
probabilistic predictors cannot be compared through silently merged rows.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.benchmark.forecast_batch import (
    FORECAST_BATCH_SCHEMA_VERSION,
    ActorForecast,
    ForecastBatch,
    validate_forecast_batch,
)

FORECAST_METRICS_SCHEMA_VERSION = "ForecastMetrics.v1"


@dataclass(frozen=True)
class ForecastMetricRow:
    """One per-actor or aggregate forecast metric row."""

    metric: str
    horizon_s: float
    value: float | None
    status: str
    denominator: int
    actor_class: str
    scenario_id: str
    observation_tier: str
    dt_s: float
    actor_id: str | None = None
    scenario_family: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible metric row.

        Returns:
            Metric row dictionary.
        """
        data: dict[str, Any] = {
            "metric": self.metric,
            "horizon_s": float(self.horizon_s),
            "value": self.value,
            "status": self.status,
            "denominator": int(self.denominator),
            "actor_class": self.actor_class,
            "scenario_id": self.scenario_id,
            "observation_tier": self.observation_tier,
            "dt_s": float(self.dt_s),
        }
        if self.actor_id is not None:
            data["actor_id"] = self.actor_id
        if self.scenario_family is not None:
            data["scenario_family"] = self.scenario_family
        if self.note is not None:
            data["note"] = self.note
        return data


GroundTruthPositions = dict[str, list[list[float]] | tuple[tuple[float, float], ...] | np.ndarray]


def evaluate_forecast_batch(
    batch: ForecastBatch | dict[str, Any],
    ground_truth: GroundTruthPositions,
    *,
    miss_threshold_m: float = 2.0,
) -> dict[str, Any]:
    """Evaluate ForecastBatch.v1 predictions against future positions.

    Args:
        batch: ForecastBatch object or JSON-compatible batch dictionary.
        ground_truth: Mapping from actor id to future positions with shape
            ``(T, 2)`` aligned to ``batch.provenance.horizons_s``.
        miss_threshold_m: Displacement threshold used for final-horizon miss-rate rows.

    Returns:
        JSON-compatible metric report with per-actor rows and aggregate rows.
    """
    if miss_threshold_m <= 0:
        raise ValueError("miss_threshold_m must be positive")

    forecast_batch = validate_forecast_batch(batch)
    rows: list[ForecastMetricRow] = []

    active_actor_ids = {
        actor_id
        for actor_id, included in zip(
            forecast_batch.provenance.actor_ids,
            forecast_batch.provenance.actor_mask,
            strict=True,
        )
        if included
    }
    forecasts_by_actor = {forecast.actor_id: forecast for forecast in forecast_batch.forecasts}
    scenario_family = _scenario_family(forecast_batch)

    for actor_id in sorted(active_actor_ids):
        forecast = forecasts_by_actor[actor_id]
        actor_class = _actor_class(forecast, forecast_batch)
        truth = _ground_truth_array(
            actor_id, ground_truth, len(forecast_batch.provenance.horizons_s)
        )
        rows.extend(
            _actor_metric_rows(
                forecast=forecast,
                truth=truth,
                actor_class=actor_class,
                batch=forecast_batch,
                scenario_family=scenario_family,
                miss_threshold_m=miss_threshold_m,
            )
        )

    aggregate_rows = _aggregate_rows(rows)
    excluded_actor_count = int(len(forecast_batch.provenance.actor_ids) - len(active_actor_ids))
    unavailable_count = int(sum(row.status == "unavailable" for row in rows + aggregate_rows))

    return {
        "schema_version": FORECAST_METRICS_SCHEMA_VERSION,
        "forecast_schema_version": FORECAST_BATCH_SCHEMA_VERSION,
        "evaluator": "ForecastBatchMetrics.v1",
        "provenance": {
            "predictor_id": forecast_batch.provenance.predictor_id,
            "predictor_family": forecast_batch.provenance.predictor_family,
            "scenario_id": forecast_batch.provenance.scenario_id,
            "scenario_family": scenario_family,
            "observation_tier": forecast_batch.provenance.observation_tier,
            "dt_s": forecast_batch.provenance.dt_s,
            "horizons_s": list(forecast_batch.provenance.horizons_s),
        },
        "metric_parameters": {
            "miss_threshold_m": float(miss_threshold_m),
        },
        "denominator_health": {
            "active_actor_count": len(active_actor_ids),
            "excluded_actor_count": excluded_actor_count,
            "metric_row_count": len(rows),
            "aggregate_row_count": len(aggregate_rows),
            "unavailable_row_count": unavailable_count,
            "has_empty_denominator": any(row.denominator == 0 for row in rows + aggregate_rows),
        },
        "metric_rows": [row.to_dict() for row in rows],
        "aggregate_rows": [row.to_dict() for row in aggregate_rows],
        "claim_boundary": (
            "Forecast metrics are open-loop evidence only and do not prove navigation "
            "success, calibration, or planner benefit."
        ),
    }


def _actor_metric_rows(
    *,
    forecast: ActorForecast,
    truth: np.ndarray | None,
    actor_class: str,
    batch: ForecastBatch,
    scenario_family: str | None,
    miss_threshold_m: float,
) -> list[ForecastMetricRow]:
    rows: list[ForecastMetricRow] = []
    horizons_s = batch.provenance.horizons_s

    for index, horizon_s in enumerate(horizons_s):
        if truth is None:
            rows.extend(
                _unavailable_rows(
                    forecast=forecast,
                    horizon_s=horizon_s,
                    actor_class=actor_class,
                    batch=batch,
                    scenario_family=scenario_family,
                    note="missing ground truth",
                )
            )
            continue
        target = truth[index]
        deterministic_error = _point_error(forecast.deterministic, index, target)
        sample_error = _min_sample_error(forecast.samples, index, target)
        expected_error = _expected_sample_error(forecast, index, target)
        final_horizon = index == len(horizons_s) - 1
        rows.append(
            _row(
                forecast=forecast,
                metric="ade",
                horizon_s=horizon_s,
                value=deterministic_error,
                actor_class=actor_class,
                batch=batch,
                scenario_family=scenario_family,
                note=None
                if deterministic_error is not None
                else "deterministic trajectory unavailable",
            )
        )
        rows.append(
            _row(
                forecast=forecast,
                metric="fde",
                horizon_s=horizon_s,
                value=deterministic_error if final_horizon else None,
                actor_class=actor_class,
                batch=batch,
                scenario_family=scenario_family,
                note=_final_horizon_note(
                    final_horizon=final_horizon,
                    value=deterministic_error,
                    missing_note="deterministic trajectory unavailable",
                ),
            )
        )
        rows.append(
            _row(
                forecast=forecast,
                metric="minade@k",
                horizon_s=horizon_s,
                value=sample_error,
                actor_class=actor_class,
                batch=batch,
                scenario_family=scenario_family,
                note=None if forecast.samples is not None else "sampled trajectories unavailable",
            )
        )
        rows.append(
            _row(
                forecast=forecast,
                metric="minfde@k",
                horizon_s=horizon_s,
                value=sample_error if final_horizon else None,
                actor_class=actor_class,
                batch=batch,
                scenario_family=scenario_family,
                note=_final_horizon_note(
                    final_horizon=final_horizon,
                    value=sample_error,
                    missing_note="sampled trajectories unavailable",
                ),
            )
        )
        rows.append(
            _row(
                forecast=forecast,
                metric="expected_ade",
                horizon_s=horizon_s,
                value=expected_error,
                actor_class=actor_class,
                batch=batch,
                scenario_family=scenario_family,
                note=None
                if forecast.samples is not None and forecast.mode_probabilities is not None
                else "mode probabilities unavailable",
            )
        )
        rows.append(
            _row(
                forecast=forecast,
                metric="expected_fde",
                horizon_s=horizon_s,
                value=expected_error if final_horizon else None,
                actor_class=actor_class,
                batch=batch,
                scenario_family=scenario_family,
                note=_final_horizon_note(
                    final_horizon=final_horizon,
                    value=expected_error,
                    missing_note="mode probabilities unavailable",
                ),
            )
        )
        miss_value = _miss_rate_value(
            deterministic_error=deterministic_error,
            min_sample_error=sample_error,
            final_horizon=final_horizon,
            threshold_m=miss_threshold_m,
        )
        rows.append(
            _row(
                forecast=forecast,
                metric="miss_rate",
                horizon_s=horizon_s,
                value=miss_value,
                actor_class=actor_class,
                batch=batch,
                scenario_family=scenario_family,
                note=_final_horizon_note(
                    final_horizon=final_horizon,
                    value=miss_value,
                    missing_note="requires deterministic or sampled final-horizon displacement",
                ),
            )
        )
        rows.extend(
            _unavailable_hook_rows(
                forecast=forecast,
                horizon_s=horizon_s,
                actor_class=actor_class,
                batch=batch,
                scenario_family=scenario_family,
            )
        )

    return rows


def _unavailable_rows(
    *,
    forecast: ActorForecast,
    horizon_s: float,
    actor_class: str,
    batch: ForecastBatch,
    scenario_family: str | None,
    note: str,
) -> list[ForecastMetricRow]:
    return [
        _row(
            forecast=forecast,
            metric=metric,
            horizon_s=horizon_s,
            value=None,
            actor_class=actor_class,
            batch=batch,
            scenario_family=scenario_family,
            note=note,
        )
        for metric in (
            "ade",
            "fde",
            "minade@k",
            "minfde@k",
            "expected_ade",
            "expected_fde",
            "miss_rate",
            "likelihood",
            "coverage",
            "collision_relevance",
        )
    ]


def _unavailable_hook_rows(
    *,
    forecast: ActorForecast,
    horizon_s: float,
    actor_class: str,
    batch: ForecastBatch,
    scenario_family: str | None,
) -> list[ForecastMetricRow]:
    return [
        _row(
            forecast=forecast,
            metric=metric,
            horizon_s=horizon_s,
            value=None,
            actor_class=actor_class,
            batch=batch,
            scenario_family=scenario_family,
            note=note,
        )
        for metric, note in (
            ("likelihood", "requires forecast distribution density metadata"),
            ("coverage", "requires calibrated set or occupancy coverage metadata"),
            ("collision_relevance", "requires forecast-scene collision relevance metadata"),
        )
    ]


def _row(
    *,
    forecast: ActorForecast,
    metric: str,
    horizon_s: float,
    value: float | None,
    actor_class: str,
    batch: ForecastBatch,
    scenario_family: str | None,
    note: str | None,
) -> ForecastMetricRow:
    return ForecastMetricRow(
        metric=metric,
        horizon_s=float(horizon_s),
        value=None if value is None else float(value),
        status="unavailable" if value is None else "ok",
        denominator=0 if value is None else 1,
        actor_id=forecast.actor_id,
        actor_class=actor_class,
        scenario_id=batch.provenance.scenario_id,
        scenario_family=scenario_family,
        observation_tier=batch.provenance.observation_tier,
        dt_s=batch.provenance.dt_s,
        note=note,
    )


def _final_horizon_note(
    *,
    final_horizon: bool,
    value: float | None,
    missing_note: str,
) -> str | None:
    if not final_horizon:
        return "final horizon only"
    if value is None:
        return missing_note
    return None


def _aggregate_rows(rows: list[ForecastMetricRow]) -> list[ForecastMetricRow]:
    grouped: dict[tuple[str, float, str, str, str, float, str | None], list[float]] = defaultdict(
        list
    )
    for row in rows:
        if row.status != "ok" or row.value is None:
            continue
        key = (
            row.metric,
            row.horizon_s,
            row.actor_class,
            row.scenario_id,
            row.observation_tier,
            row.dt_s,
            row.scenario_family,
        )
        grouped[key].append(row.value)

    aggregate_rows: list[ForecastMetricRow] = []
    seen_keys = {
        (
            row.metric,
            row.horizon_s,
            row.actor_class,
            row.scenario_id,
            row.observation_tier,
            row.dt_s,
            row.scenario_family,
        )
        for row in rows
    }
    for key in sorted(seen_keys, key=_aggregate_sort_key):
        metric, horizon_s, actor_class, scenario_id, observation_tier, dt_s, scenario_family = key
        values = grouped.get(key, [])
        aggregate_rows.append(
            ForecastMetricRow(
                metric=f"mean_{metric}",
                horizon_s=horizon_s,
                value=float(np.mean(values)) if values else None,
                status="ok" if values else "unavailable",
                denominator=len(values),
                actor_class=actor_class,
                scenario_id=scenario_id,
                scenario_family=scenario_family,
                observation_tier=observation_tier,
                dt_s=dt_s,
                note=None if values else "empty denominator",
            )
        )
    return aggregate_rows


def _aggregate_sort_key(key: tuple[str, float, str, str, str, float, str | None]) -> tuple:
    metric, horizon_s, actor_class, scenario_id, observation_tier, dt_s, scenario_family = key
    return (
        metric,
        horizon_s,
        actor_class,
        scenario_id,
        observation_tier,
        dt_s,
        "" if scenario_family is None else scenario_family,
    )


def _ground_truth_array(
    actor_id: str,
    ground_truth: GroundTruthPositions,
    expected_steps: int,
) -> np.ndarray | None:
    if actor_id not in ground_truth:
        return None
    array = np.asarray(ground_truth[actor_id], dtype=float)
    if array.shape != (expected_steps, 2):
        raise ValueError(
            "ground_truth trajectories must align with horizons_s and have shape (T, 2)"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("ground_truth trajectories must contain only finite values")
    return array


def _point_error(
    trajectory: np.ndarray | None, horizon_index: int, target: np.ndarray
) -> float | None:
    if trajectory is None:
        return None
    return float(np.linalg.norm(trajectory[horizon_index] - target))


def _min_sample_error(
    samples: np.ndarray | None, horizon_index: int, target: np.ndarray
) -> float | None:
    if samples is None:
        return None
    errors = np.linalg.norm(samples[:, horizon_index, :] - target, axis=1)
    return float(np.min(errors))


def _miss_rate_value(
    *,
    deterministic_error: float | None,
    min_sample_error: float | None,
    final_horizon: bool,
    threshold_m: float,
) -> float | None:
    if not final_horizon:
        return None
    error = min_sample_error if min_sample_error is not None else deterministic_error
    if error is None:
        return None
    return float(error > threshold_m)


def _expected_sample_error(
    forecast: ActorForecast,
    horizon_index: int,
    target: np.ndarray,
) -> float | None:
    if forecast.samples is None or forecast.mode_probabilities is None:
        return None
    errors = np.linalg.norm(forecast.samples[:, horizon_index, :] - target, axis=1)
    probabilities = np.asarray(forecast.mode_probabilities, dtype=float)
    return float(np.sum(errors * probabilities))


def _actor_class(forecast: ActorForecast, batch: ForecastBatch) -> str:
    metadata_sources = (
        forecast.uncertainty_metadata,
        forecast.occupancy_summary,
        batch.metadata.get("actor_classes"),
    )
    for source in metadata_sources:
        if isinstance(source, dict):
            actor_class = source.get(forecast.actor_id)
            if actor_class is not None:
                return str(actor_class)
            default_actor_class = source.get("actor_class")
            if default_actor_class is not None:
                return str(default_actor_class)
    return "pedestrian"


def _scenario_family(batch: ForecastBatch) -> str | None:
    value = batch.metadata.get("scenario_family")
    return str(value) if value is not None else None


def format_forecast_metrics_markdown(report: dict[str, Any]) -> str:
    """Format a compact Markdown summary for context evidence promotion.

    Args:
        report: JSON-compatible report returned by :func:`evaluate_forecast_batch`.

    Returns:
        Markdown summary with provenance, denominator health, and aggregate metrics.
    """
    provenance = report["provenance"]
    health = report["denominator_health"]
    lines = [
        "# Forecast Metrics Summary",
        "",
        f"- Predictor: {provenance['predictor_id']} ({provenance['predictor_family']})",
        f"- Scenario: {provenance['scenario_id']}",
        f"- Observation tier: {provenance['observation_tier']}",
        (
            f"- Denominators: active={health['active_actor_count']}, "
            f"excluded={health['excluded_actor_count']}, unavailable={health['unavailable_row_count']}"
        ),
        "",
        "| metric | horizon_s | actor_class | denominator | status | value |",
        "| --- | ---: | --- | ---: | --- | ---: |",
    ]
    for row in report["aggregate_rows"]:
        value = row["value"]
        rendered_value = "NA" if value is None else f"{float(value):.6g}"
        lines.append(
            "| {metric} | {horizon_s:g} | {actor_class} | {denominator} | {status} | "
            "{value} |".format(
                metric=row["metric"],
                horizon_s=float(row["horizon_s"]),
                actor_class=row["actor_class"],
                denominator=int(row["denominator"]),
                status=row["status"],
                value=rendered_value,
            )
        )
    lines.extend(["", report["claim_boundary"]])
    return "\n".join(lines) + "\n"


__all__ = [
    "FORECAST_METRICS_SCHEMA_VERSION",
    "ForecastMetricRow",
    "evaluate_forecast_batch",
    "format_forecast_metrics_markdown",
]
