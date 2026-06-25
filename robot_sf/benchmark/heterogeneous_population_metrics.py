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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

HETEROGENEOUS_POPULATION_METRICS_SCHEMA = "heterogeneous_population_metrics.v1"


def _require_finite(name: str, value: float) -> None:
    """Fail closed on a non-finite (NaN/Inf) metric value.

    A degraded trace can leak NaN/Inf, after which ``min``/``max`` for the worst
    stratum and ``np.mean`` for the CVaR evaluate inconsistently by element order
    (fail-open). Raising names the offending field so the caller drops the input.

    Raises:
        ValueError: If ``value`` is not finite.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


@dataclass(frozen=True, slots=True)
class PedestrianMetric:
    """One per-pedestrian metric observation tagged with its archetype.

    Attributes:
        archetype: Pedestrian archetype label (e.g. ``cooperative``, ``static``, ``rushed``).
        value: The per-pedestrian metric value (e.g. clearance, near-field exposure).
    """

    archetype: str
    value: float


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
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must contain only finite values")
    arr = np.sort(arr)  # ascending
    k = max(1, math.ceil(alpha * arr.size))
    tail = arr[:k] if higher_is_safer else arr[-k:]
    return float(np.mean(tail))


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
    "PedestrianMetric",
    "cvar",
    "mean_matched_heterogeneity_effect",
    "per_archetype_metrics",
]
