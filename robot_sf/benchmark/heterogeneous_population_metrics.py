"""Per-archetype metrics + mean-matched heterogeneity-effect isolation (issue #3574).

The #3206 heterogeneous-pedestrian smoke showed a mean-clearance change under a mixed population but
could not (a) isolate heterogeneity from a population-mean shift, nor (b) compute per-archetype
metrics. The literature is explicit that heterogeneity effects are real but planner-specific and are
not a single difficulty scalar. This module provides the pure analysis primitives needed before any
heterogeneity effect can be claimed:

1. a **per-archetype metric harness** — per-archetype mean, worst-stratum, and CVaR over the
   dangerous tail (so the worst-served archetype is visible, not averaged away);
2. **mean-matched heterogeneity-effect isolation** — comparing a heterogeneous population against a
   homogeneous one whose parameter is the population mean (``theta_i = E[theta]``), so the effect is
   separated from a mean shift.

The per-pedestrian control-trace logging that feeds the per-archetype harness, and the mean-matched
paired ablation runs, are deferred; this module is pure and side-effect free, mirroring the accepted
analysis layers in this run (#3484, #3558, #3557, #3573).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.benchmark.finite_checks import require_finite_array, require_finite_scalar

HETEROGENEOUS_POPULATION_METRICS_SCHEMA = "heterogeneous_population_metrics.v1"

_CONTROL_TRACE_REDUCERS = frozenset({"mean", "min", "max", "final"})


def _require_finite(name: str, value: float) -> None:
    """Fail closed on a non-finite (NaN/Inf) metric value.

    A degraded trace can leak NaN/Inf, after which ``min``/``max`` for the worst
    stratum and ``np.mean`` for the CVaR evaluate inconsistently by element order
    (fail-open). Raising names the offending field so the caller drops the input.

    Raises:
        ValueError: If ``value`` is not finite.
    """
    require_finite_scalar(name, value)


@dataclass(frozen=True, slots=True)
class PedestrianMetric:
    """One per-pedestrian metric observation tagged with its archetype.

    Attributes:
        archetype: Pedestrian archetype label (e.g. ``cooperative``, ``static``, ``rushed``).
        value: The per-pedestrian metric value (e.g. clearance, near-field exposure).
    """

    archetype: str
    value: float


@dataclass(frozen=True, slots=True)
class ControlTraceReadiness:
    """Readiness diagnostic for per-pedestrian control-trace metric extraction."""

    status: str
    metric_key: str
    pedestrian_count: int
    step_count: int
    archetype_counts: dict[str, int]
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        """Return true when the trace can feed the per-archetype metric harness."""

        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable diagnostic payload."""

        return {
            "schema_version": HETEROGENEOUS_POPULATION_METRICS_SCHEMA,
            "source": "pedestrian_control_trace",
            "metric_key": self.metric_key,
            "status": self.status,
            "ready": self.ready,
            "pedestrian_count": self.pedestrian_count,
            "step_count": self.step_count,
            "archetype_counts": dict(sorted(self.archetype_counts.items())),
            "blockers": list(self.blockers),
        }


def cvar(values: Sequence[float], alpha: float, *, higher_is_safer: bool) -> float:
    """Return the CVaR (mean of the worst ``alpha`` tail) of a metric.

    For a metric where higher is safer (e.g. clearance), the worst tail is the lowest values; for a
    metric where lower is safer (e.g. exposure), it is the highest values.

    Returns:
        float: Mean of the worst ``ceil(alpha * n)`` observations.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    arr = require_finite_array("values", values)
    arr = np.sort(arr)  # ascending
    k = max(1, math.ceil(alpha * arr.size))
    tail = arr[:k] if higher_is_safer else arr[-k:]
    return float(np.mean(tail))


def assess_control_trace_readiness(
    control_trace: Mapping[str, Any],
    metric_key: str,
) -> ControlTraceReadiness:
    """Assess whether a control trace has the fields required for per-archetype metrics.

    The readiness check is intentionally non-raising so orchestration code can emit a
    blocked diagnostic packet. Metric extraction still raises through
    ``pedestrian_metric_observations_from_control_trace`` when a caller tries to use
    an incomplete trace.

    Returns:
        Readiness diagnostic with coverage counts and fail-closed blockers.
    """

    normalized_metric_key = str(metric_key).strip()
    blockers: list[str] = []
    pedestrian_count = 0
    step_count = 0
    archetype_counts: dict[str, int] = {}

    if not normalized_metric_key:
        blockers.append("metric_key must be non-empty")

    try:
        pedestrians = _control_trace_pedestrians(control_trace)
    except ValueError as exc:
        blockers.append(str(exc))
        return ControlTraceReadiness(
            status="blocked",
            metric_key=normalized_metric_key,
            pedestrian_count=0,
            step_count=0,
            archetype_counts={},
            blockers=tuple(blockers),
        )

    pedestrian_count = len(pedestrians)
    for pedestrian_index, pedestrian in enumerate(pedestrians):
        if not isinstance(pedestrian, Mapping):
            blockers.append(f"control_trace.pedestrians[{pedestrian_index}] must be mapping")
            continue

        try:
            archetype = _control_trace_archetype(pedestrian, pedestrian_index)
        except ValueError as exc:
            blockers.append(str(exc))
        else:
            archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1

        if not normalized_metric_key:
            continue

        try:
            values = _control_trace_metric_values(
                pedestrian,
                pedestrian_index,
                normalized_metric_key,
            )
        except (TypeError, ValueError) as exc:
            blockers.append(str(exc))
        else:
            step_count += int(values.size)

    return ControlTraceReadiness(
        status="ready" if not blockers else "blocked",
        metric_key=normalized_metric_key,
        pedestrian_count=pedestrian_count,
        step_count=step_count,
        archetype_counts=archetype_counts,
        blockers=tuple(blockers),
    )


def per_archetype_metrics(
    observations: Sequence[PedestrianMetric],
    *,
    higher_is_safer: bool = True,
    cvar_alpha: float = 0.1,
) -> dict[str, Any]:
    """Aggregate a per-pedestrian metric by archetype with mean, worst-stratum, and CVaR.

    Returns:
        dict[str, Any]: Versioned report with a per-archetype block and the worst archetype by mean.
    """
    if not observations:
        raise ValueError("at least one observation is required")
    grouped: dict[str, list[float]] = {}
    for observation in observations:
        value = float(observation.value)
        _require_finite(f"observation value for archetype {observation.archetype!r}", value)
        grouped.setdefault(observation.archetype, []).append(value)

    per_archetype: dict[str, Any] = {}
    for archetype in sorted(grouped):
        values = grouped[archetype]
        per_archetype[archetype] = {
            "n": len(values),
            "mean": float(np.mean(values)),
            "worst_stratum": float(min(values) if higher_is_safer else max(values)),
            "cvar": cvar(values, cvar_alpha, higher_is_safer=higher_is_safer),
        }

    worst_archetype = (
        min(per_archetype, key=lambda a: per_archetype[a]["mean"])
        if higher_is_safer
        else max(per_archetype, key=lambda a: per_archetype[a]["mean"])
    )
    return {
        "schema_version": HETEROGENEOUS_POPULATION_METRICS_SCHEMA,
        "evidence_kind": "diagnostic_proxy",
        "higher_is_safer": higher_is_safer,
        "cvar_alpha": cvar_alpha,
        "per_archetype": per_archetype,
        "worst_archetype_by_mean": worst_archetype,
    }


def pedestrian_metric_observations_from_control_trace(
    control_trace: Mapping[str, Any],
    metric_key: str,
    *,
    reducer: str = "mean",
) -> list[PedestrianMetric]:
    """Extract per-pedestrian metric observations from a control trace.

    The input is the ``pedestrian_control_trace`` payload attached to episode
    metadata. This is intentionally a pure trace-to-observation bridge so later
    ablation code can reuse the per-archetype harness without running benchmark
    campaigns inside the metric layer.

    Returns:
        Per-pedestrian observations tagged by archetype.
    """

    metric_key = str(metric_key).strip()
    if not metric_key:
        raise ValueError("metric_key must be non-empty")
    if reducer not in _CONTROL_TRACE_REDUCERS:
        raise ValueError(f"reducer must be one of {sorted(_CONTROL_TRACE_REDUCERS)}")

    observations: list[PedestrianMetric] = []
    for pedestrian_index, pedestrian in enumerate(_control_trace_pedestrians(control_trace)):
        archetype = _control_trace_archetype(pedestrian, pedestrian_index)
        arr = _control_trace_metric_values(pedestrian, pedestrian_index, metric_key)
        value = _reduce_control_trace_metric(arr, reducer)
        observations.append(PedestrianMetric(archetype=archetype, value=value))

    return observations


def _control_trace_pedestrians(control_trace: Mapping[str, Any]) -> Sequence[Any]:
    """Return the non-empty ``pedestrians`` sequence, failing closed on bad shapes."""
    if not isinstance(control_trace, Mapping):
        raise ValueError("control_trace must be a mapping")
    pedestrians = control_trace.get("pedestrians")
    if not isinstance(pedestrians, Sequence) or isinstance(pedestrians, str):
        raise ValueError("control_trace.pedestrians must be a sequence")
    if not pedestrians:
        raise ValueError("control_trace.pedestrians must be non-empty")
    return pedestrians


def _control_trace_archetype(pedestrian: Any, pedestrian_index: int) -> str:
    """Return a non-empty archetype label, rejecting null/blank values fail-closed."""
    if not isinstance(pedestrian, Mapping):
        raise ValueError(f"control_trace.pedestrians[{pedestrian_index}] must be a mapping")
    archetype_value = pedestrian.get("archetype")
    # Guard against an explicit ``archetype: null`` payload: ``str(None)`` would
    # coerce to the truthy string ``"None"`` and silently group pedestrians under
    # a fake archetype, defeating the fail-closed contract of this helper.
    archetype = str(archetype_value).strip() if archetype_value is not None else ""
    if not archetype:
        raise ValueError(
            f"control_trace.pedestrians[{pedestrian_index}].archetype must be non-empty"
        )
    return archetype


def _control_trace_metric_values(
    pedestrian: Mapping[str, Any],
    pedestrian_index: int,
    metric_key: str,
) -> np.ndarray:
    """Collect the per-step ``metric_key`` values as a finite array, failing closed.

    Missing keys, null values, non-mapping steps, and non-finite values each raise a
    descriptive ``ValueError`` instead of silently degrading the extracted support.

    Returns:
        Finite per-step values for ``metric_key`` as a 1-D array.
    """
    steps = pedestrian.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, str) or not steps:
        raise ValueError(f"control_trace.pedestrians[{pedestrian_index}].steps must be non-empty")

    values: list[float] = []
    for step_index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(
                "control_trace.pedestrians"
                f"[{pedestrian_index}].steps[{step_index}] must be a mapping"
            )
        if metric_key not in step:
            raise ValueError(
                "control_trace.pedestrians"
                f"[{pedestrian_index}].steps[{step_index}] missing {metric_key!r}"
            )
        raw_value = step[metric_key]
        # A null trace value would raise a bare ``TypeError`` from ``float(None)``;
        # raise a descriptive ``ValueError`` instead so the offending field is named.
        if raw_value is None:
            raise ValueError(
                "control_trace.pedestrians"
                f"[{pedestrian_index}].steps[{step_index}].{metric_key} must not be null"
            )
        values.append(float(raw_value))

    return require_finite_array(
        f"control_trace.pedestrians[{pedestrian_index}].steps.{metric_key}",
        values,
    )


def _reduce_control_trace_metric(values: np.ndarray, reducer: str) -> float:
    """Reduce a per-step value array to one scalar per the validated reducer name.

    Returns:
        The reduced scalar value.
    """
    if reducer == "mean":
        return float(np.mean(values))
    if reducer == "min":
        return float(np.min(values))
    if reducer == "max":
        return float(np.max(values))
    return float(values[-1])


def per_archetype_metrics_from_control_trace(
    control_trace: Mapping[str, Any],
    metric_key: str,
    *,
    higher_is_safer: bool = True,
    cvar_alpha: float = 0.1,
    reducer: str = "mean",
) -> dict[str, Any]:
    """Build a per-archetype metric report from a pedestrian control trace.

    Returns:
        Versioned per-archetype metric report with trace provenance fields.
    """

    # Record the same normalized key used for lookup so the provenance field never
    # disagrees with the metric actually extracted (e.g. a padded " speed_m_s ").
    normalized_metric_key = str(metric_key).strip()
    observations = pedestrian_metric_observations_from_control_trace(
        control_trace,
        normalized_metric_key,
        reducer=reducer,
    )
    report = per_archetype_metrics(
        observations,
        higher_is_safer=higher_is_safer,
        cvar_alpha=cvar_alpha,
    )
    report["source"] = "pedestrian_control_trace"
    report["metric_key"] = normalized_metric_key
    report["pedestrian_metric_reducer"] = reducer
    return report


def mean_matched_heterogeneity_effect(
    homogeneous_mean: float,
    heterogeneous_mean: float,
    *,
    homogeneous_is_mean_matched: bool,
) -> dict[str, Any]:
    """Isolate the heterogeneity effect against a mean-matched homogeneous baseline.

    The homogeneous arm must use the population mean of the heterogeneous parameter distribution
    (``theta_i = E[theta]``); otherwise the difference confounds heterogeneity with a mean shift and
    is flagged invalid.

    Returns:
        dict[str, Any]: Versioned report with the isolated effect and a validity flag.
    """
    _require_finite("homogeneous_mean", float(homogeneous_mean))
    _require_finite("heterogeneous_mean", float(heterogeneous_mean))
    effect = float(heterogeneous_mean) - float(homogeneous_mean)
    return {
        "schema_version": HETEROGENEOUS_POPULATION_METRICS_SCHEMA,
        "evidence_kind": "diagnostic_proxy",
        "homogeneous_mean": float(homogeneous_mean),
        "heterogeneous_mean": float(heterogeneous_mean),
        "heterogeneity_effect": effect,
        "isolated_from_mean_shift": bool(homogeneous_is_mean_matched),
        "validity": "isolated" if homogeneous_is_mean_matched else "confounded_by_mean_shift",
    }


__all__ = [
    "HETEROGENEOUS_POPULATION_METRICS_SCHEMA",
    "ControlTraceReadiness",
    "PedestrianMetric",
    "assess_control_trace_readiness",
    "cvar",
    "mean_matched_heterogeneity_effect",
    "pedestrian_metric_observations_from_control_trace",
    "per_archetype_metrics",
    "per_archetype_metrics_from_control_trace",
]
