"""Helpers to keep training-side SNQI computation aligned with benchmark logic."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from robot_sf.benchmark.snqi import WEIGHT_NAMES, compute_snqi
from robot_sf.benchmark.snqi.weights_validation import validate_weights_mapping

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_SNQI_WEIGHTS: dict[str, float] = {
    "w_success": 1.0,
    "w_time": 0.8,
    "w_collisions": 2.0,
    "w_near": 1.0,
    "w_comfort": 0.5,
    "w_force_exceed": 1.5,
    "w_jerk": 0.3,
}
_DEFAULT_SNQI_BASELINE: dict[str, dict[str, float]] = {
    "collisions": {"med": 0.0, "p95": 1.0},
    "near_misses": {"med": 0.0, "p95": 1.0},
    "force_exceed_events": {"med": 0.0, "p95": 1.0},
    "jerk_mean": {"med": 0.0, "p95": 1.0},
}


@dataclass(frozen=True, slots=True)
class TrainingSNQIContext:
    """Resolved SNQI settings used by training scripts."""

    weights: dict[str, float]
    baseline_stats: dict[str, dict[str, float]]
    weights_source: str
    baseline_source: str
    baseline_fallback_keys: tuple[str, ...]


def _safe_float(value: object, *, default: float) -> float:
    """Return a finite float or a default fallback."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _extract_weights_mapping(raw: object) -> dict[str, object]:
    """Extract a weights mapping from supported JSON shapes.

    Returns:
        Mapping with direct SNQI weight keys.
    """
    if not isinstance(raw, dict):
        raise ValueError("SNQI weights payload must be a JSON object.")
    if "weights" in raw and isinstance(raw["weights"], dict):
        return dict(cast("dict[str, object]", raw["weights"]))
    recommended = raw.get("recommended")
    if isinstance(recommended, dict) and isinstance(recommended.get("weights"), dict):
        return dict(cast("dict[str, object]", recommended["weights"]))
    if all(key in raw for key in WEIGHT_NAMES):
        return dict(cast("dict[str, object]", raw))
    raise ValueError(
        "Unable to locate SNQI weights mapping. Expected keys under top-level mapping, "
        "'weights', or 'recommended.weights'.",
    )


def _extract_baseline_mapping(raw: object) -> dict[str, object]:
    """Extract a baseline stats mapping from supported JSON shapes.

    Returns:
        Mapping keyed by metric name containing ``med``/``p95`` entries.
    """
    if not isinstance(raw, dict):
        raise ValueError("SNQI baseline payload must be a JSON object.")
    if "baseline_stats" in raw and isinstance(raw["baseline_stats"], dict):
        return dict(cast("dict[str, object]", raw["baseline_stats"]))
    return dict(cast("dict[str, object]", raw))


def _normalize_baseline_mapping(
    raw: dict[str, object],
) -> tuple[dict[str, dict[str, float]], tuple[str, ...]]:
    """Coerce baseline entries to {med, p95} floats and fill missing defaults.

    Returns:
        Tuple of normalized baseline mapping and required keys that were backfilled.
    """
    normalized: dict[str, dict[str, float]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        med = _safe_float(value.get("med"), default=0.0)
        p95 = _safe_float(value.get("p95"), default=med + 1.0)
        if p95 <= med:
            p95 = med + 1.0
        normalized[str(key)] = {"med": med, "p95": p95}

    fallback_keys: list[str] = []
    for key, fallback in _DEFAULT_SNQI_BASELINE.items():
        if key in normalized:
            continue
        normalized[key] = dict(fallback)
        fallback_keys.append(key)

    return normalized, tuple(fallback_keys)


def default_training_snqi_context() -> TrainingSNQIContext:
    """Return the default SNQI context used when no files are configured."""
    return TrainingSNQIContext(
        weights=dict(_DEFAULT_SNQI_WEIGHTS),
        baseline_stats={key: dict(value) for key, value in _DEFAULT_SNQI_BASELINE.items()},
        weights_source="default",
        baseline_source="default",
        baseline_fallback_keys=(),
    )


def resolve_training_snqi_context(
    *,
    weights_path: Path | None = None,
    baseline_path: Path | None = None,
) -> TrainingSNQIContext:
    """Resolve SNQI weights/baseline settings for training scripts.

    Returns:
        Fully resolved context with sources and any required baseline key fallbacks.
    """
    if weights_path is None:
        weights = dict(_DEFAULT_SNQI_WEIGHTS)
        weights_source = "default"
    else:
        raw_weights = json.loads(weights_path.read_text(encoding="utf-8"))
        weights_mapping = _extract_weights_mapping(raw_weights)
        weights = validate_weights_mapping(weights_mapping)
        weights_source = str(weights_path)

    if baseline_path is None:
        baseline_stats = {key: dict(value) for key, value in _DEFAULT_SNQI_BASELINE.items()}
        baseline_source = "default"
        baseline_fallback_keys: tuple[str, ...] = ()
    else:
        raw_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_mapping = _extract_baseline_mapping(raw_baseline)
        baseline_stats, baseline_fallback_keys = _normalize_baseline_mapping(baseline_mapping)
        baseline_source = str(baseline_path)

    return TrainingSNQIContext(
        weights=weights,
        baseline_stats=baseline_stats,
        weights_source=weights_source,
        baseline_source=baseline_source,
        baseline_fallback_keys=baseline_fallback_keys,
    )


def compute_training_snqi(
    metric_values: dict[str, float | int | bool],
    *,
    context: TrainingSNQIContext,
) -> float:
    """Compute SNQI using the canonical benchmark implementation.

    Returns:
        Canonical SNQI score for the provided episode metrics.
    """
    return float(compute_snqi(metric_values, context.weights, context.baseline_stats))
