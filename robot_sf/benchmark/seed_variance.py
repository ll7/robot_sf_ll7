"""Seed variance analysis for Social Navigation Benchmark.

This module computes paper-facing per-scenario/per-planner seed-variability
artifacts from benchmark episode records.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.aggregate import flatten_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence


_BOOTSTRAP_METHOD = "bootstrap_mean_over_seed_means"
_PAPER_METRIC_ALIASES = {
    "success": "success",
    "collisions": "collision",
    "near_misses": "near_miss",
    "time_to_goal_norm": "time_to_goal",
    "snqi": "snqi",
}


def _get_nested(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted-path value from a nested dict, falling back to default.

    Returns:
        The nested value when present, otherwise ``default``.
    """
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _numeric_items(d: dict[str, Any]) -> dict[str, float]:
    """Extract numeric scalar items from a flattened metrics row.

    Returns:
        Mapping of metric names to finite numeric values.
    """
    out: dict[str, float] = {}
    for k, v in d.items():
        if k in ("episode_id", "scenario_id", "seed"):
            continue
        if v is None:
            continue
        if isinstance(v, int | float) and not (isinstance(v, float) and math.isnan(v)):
            out[k] = float(v)
    return out


def _group_rows(
    records: Sequence[dict[str, Any]],
    group_by: str,
    fallback_group_by: str,
) -> dict[str, list[dict[str, Any]]]:
    """Group flattened records by a dotted-path key with fallback.

    Returns:
        Mapping of group key to flattened rows.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        g = _get_nested(rec, group_by)
        if g is None:
            g = _get_nested(rec, fallback_group_by)
        if g is None:
            g = "unknown"
        groups[str(g)].append(flatten_metrics(rec))
    return groups


def _collect_metric_names(
    groups: dict[str, list[dict[str, Any]]],
    metrics: Sequence[str] | None,
) -> set[str]:
    """Collect metric names from groups unless explicitly provided.

    Returns:
        Set of metric names to evaluate.
    """
    if metrics is not None:
        return set(metrics)
    names: set[str] = set()
    for rows in groups.values():
        for row in rows:
            names.update(_numeric_items(row).keys())
    return names


def _coerce_float(value: Any) -> float | None:
    """Convert a value to a finite float when possible.

    Returns:
        Finite float value, or ``None`` when conversion fails.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _confidence_alpha(confidence: float) -> tuple[float, float]:
    """Return lower/upper percentile alphas for a confidence level.

    Returns:
        Lower and upper quantile values in ``[0, 1]``.
    """
    bounded = min(0.999999, max(0.0, float(confidence)))
    alpha = 1.0 - bounded
    return alpha / 2.0, 1.0 - alpha / 2.0


def _bootstrap_mean_ci(
    vals: list[float],
    *,
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> tuple[float, float]:
    """Bootstrap CI over the mean of per-seed values.

    Returns:
        Low/high confidence interval bounds.
    """
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        value = float(vals[0])
        return value, value
    arr = np.asarray(vals, dtype=float)
    if bootstrap_samples <= 0:
        mean = float(np.mean(arr))
        return mean, mean
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(arr), size=(int(bootstrap_samples), len(arr)))
    sampled_means = arr[indices].mean(axis=1)
    lo_q, hi_q = _confidence_alpha(confidence)
    low = float(np.quantile(sampled_means, lo_q, method="linear"))
    high = float(np.quantile(sampled_means, hi_q, method="linear"))
    return low, high


def _stats_for_vals(
    vals: list[float],
    *,
    bootstrap_samples: int = 0,
    confidence: float = 0.95,
    bootstrap_seed: int = 123,
) -> dict[str, float]:
    """Compute mean/std/cv and CI for a list of values, ignoring non-finite.

    Returns:
        Mapping with central tendency, spread, and confidence interval fields.
    """
    if not vals:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "cv": float("nan"),
            "count": 0.0,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_half_width": float("nan"),
        }
    arr = np.asarray(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    n = finite.size
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "cv": float("nan"),
            "count": 0.0,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_half_width": float("nan"),
        }
    values = finite.tolist()
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=0)) if n >= 2 else 0.0
    cv = float(std / mean) if (n >= 2 and mean != 0.0 and math.isfinite(std)) else float("nan")
    ci_low, ci_high = _bootstrap_mean_ci(
        values,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )
    half_width = (
        float(max(abs(mean - ci_low), abs(ci_high - mean)))
        if math.isfinite(ci_low) and math.isfinite(ci_high)
        else float("nan")
    )
    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "count": float(n),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half_width": half_width,
    }


def compute_seed_variance(
    records: list[dict[str, Any]] | Sequence[dict[str, Any]],
    *,
    group_by: str = "scenario_id",
    fallback_group_by: str = "scenario_id",
    metrics: Sequence[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-metric seed variability for groups.

    Returns:
        Nested dictionary of group -> metric -> summary statistics.
    """
    groups = _group_rows(records, group_by, fallback_group_by)
    metric_names = _collect_metric_names(groups, metrics)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for g, rows in groups.items():
        cols: dict[str, list[float]] = {m: [] for m in metric_names}
        for row in rows:
            num = _numeric_items(row)
            for m in metric_names:
                v = num.get(m)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                cols[m].append(float(v))
        out[g] = {m: _stats_for_vals(vals) for m, vals in cols.items()}

    return out


def _metric_alias(metric: str) -> str:
    """Return the paper-facing alias for a source metric name."""
    return _PAPER_METRIC_ALIASES.get(metric, metric)


def build_seed_variability_rows(
    records: list[dict[str, Any]] | Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str],
    campaign_id: str,
    config_hash: str,
    git_hash: str,
    seed_policy: dict[str, Any] | None = None,
    confidence_settings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build paper-facing seed-variability rows from benchmark episode records.

    Returns:
        Aggregate rows grouped by scenario and planner across seeds.
    """
    confidence_settings = dict(confidence_settings or {})
    seed_policy = dict(seed_policy or {})
    confidence_settings.setdefault("method", _BOOTSTRAP_METHOD)
    confidence_settings.setdefault("confidence", 0.95)
    confidence_settings.setdefault("bootstrap_samples", 0)
    confidence_settings.setdefault("bootstrap_seed", 123)
    bootstrap_samples = int(confidence_settings.get("bootstrap_samples", 0) or 0)
    bootstrap_confidence = float(confidence_settings.get("confidence", 0.95) or 0.95)
    bootstrap_seed = int(confidence_settings.get("bootstrap_seed", 123) or 123)
    grouped: dict[
        tuple[str, str, str, str, str, str],
        dict[int, list[dict[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))

    for record in records:
        scenario_id = str(record.get("scenario_id") or "unknown")
        planner_key = str(
            record.get("planner_key")
            or _get_nested(record, "scenario_params.planner_key", default=None)
            or record.get("algo")
            or _get_nested(record, "scenario_params.algo", default="unknown")
        )
        algo = str(
            record.get("algo") or _get_nested(record, "scenario_params.algo", default="unknown")
        )
        planner_group = str(record.get("planner_group") or "unknown")
        kinematics = str(record.get("kinematics") or "unknown")
        benchmark_profile = str(record.get("benchmark_profile") or "unknown")
        seed = int(record.get("seed", -1))
        grouped[(scenario_id, planner_key, algo, planner_group, kinematics, benchmark_profile)][
            seed
        ].append(flatten_metrics(record))

    rows: list[dict[str, Any]] = []
    for (
        scenario_id,
        planner_key,
        algo,
        planner_group,
        kinematics,
        benchmark_profile,
    ), seed_groups in sorted(grouped.items()):
        per_seed_rows: list[dict[str, Any]] = []
        across_seed_values: dict[str, list[float]] = {metric: [] for metric in metrics}
        total_episodes = 0
        for seed, seed_records in sorted(seed_groups.items()):
            total_episodes += len(seed_records)
            metric_rows: dict[str, float] = {}
            for metric in metrics:
                metric_values: list[float] = []
                for seed_record in seed_records:
                    numeric = _coerce_float(seed_record.get(metric))
                    if numeric is not None:
                        metric_values.append(numeric)
                if metric_values:
                    mean_value = float(np.mean(metric_values))
                    metric_rows[metric] = mean_value
                    across_seed_values[metric].append(mean_value)
            per_seed_rows.append(
                {
                    "seed": seed,
                    "episode_count": len(seed_records),
                    "metrics": metric_rows,
                }
            )
        summary = {
            metric: _stats_for_vals(
                values,
                bootstrap_samples=bootstrap_samples,
                confidence=bootstrap_confidence,
                bootstrap_seed=bootstrap_seed,
            )
            for metric, values in across_seed_values.items()
            if values
        }
        rows.append(
            {
                "scenario_id": scenario_id,
                "planner_key": planner_key,
                "algo": algo,
                "planner_group": planner_group,
                "kinematics": kinematics,
                "benchmark_profile": benchmark_profile,
                "n": len(seed_groups),
                "seed_count": len(seed_groups),
                "episode_count": total_episodes,
                "seed_list": [entry["seed"] for entry in per_seed_rows],
                "per_seed": per_seed_rows,
                "summary": summary,
                "provenance": {
                    "campaign_id": campaign_id,
                    "config_hash": config_hash,
                    "git_hash": git_hash,
                    "seed_policy": seed_policy,
                    "confidence": dict(confidence_settings),
                },
            }
        )
    return rows


def build_seed_variability_csv_rows(
    rows: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    """Flatten seed-variability rows into CSV-friendly per-seed rows.

    Returns:
        One CSV row per scenario/planner/seed combination.
    """
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        provenance = dict(row.get("provenance") or {})
        summary = dict(row.get("summary") or {})
        confidence = dict(provenance.get("confidence") or {})
        for seed_row in row.get("per_seed") or []:
            csv_row = {
                "scenario_id": row.get("scenario_id"),
                "planner_key": row.get("planner_key"),
                "algo": row.get("algo"),
                "planner_group": row.get("planner_group"),
                "kinematics": row.get("kinematics"),
                "benchmark_profile": row.get("benchmark_profile"),
                "seed": seed_row.get("seed"),
                "seed_episode_count": seed_row.get("episode_count"),
                "seed_list": ",".join(str(seed) for seed in row.get("seed_list") or []),
                "campaign_id": provenance.get("campaign_id"),
                "config_hash": provenance.get("config_hash"),
                "git_hash": provenance.get("git_hash"),
                "seed_policy_mode": (provenance.get("seed_policy") or {}).get("mode"),
                "seed_policy_seed_set": (provenance.get("seed_policy") or {}).get("seed_set"),
                "confidence_method": confidence.get("method"),
                "confidence_level": confidence.get("confidence"),
                "bootstrap_samples": confidence.get("bootstrap_samples"),
                "bootstrap_seed": confidence.get("bootstrap_seed"),
            }
            per_seed_metrics = dict(seed_row.get("metrics") or {})
            for metric in metrics:
                csv_row[f"{metric}_per_seed_mean"] = per_seed_metrics.get(metric)
                metric_summary = dict(summary.get(metric) or {})
                csv_row[f"{metric}_across_seed_mean"] = metric_summary.get("mean")
                csv_row[f"{metric}_across_seed_std"] = metric_summary.get("std")
                csv_row[f"{metric}_across_seed_cv"] = metric_summary.get("cv")
                csv_row[f"{metric}_across_seed_count"] = metric_summary.get("count")
                csv_row[f"{metric}_ci_low"] = metric_summary.get("ci_low")
                csv_row[f"{metric}_ci_high"] = metric_summary.get("ci_high")
                csv_row[f"{metric}_ci_half_width"] = metric_summary.get("ci_half_width")
            csv_rows.append(csv_row)
    return csv_rows


def build_seed_episode_rows(
    records: list[dict[str, Any]] | Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build planner-aware per-episode seed traceability rows for paper tooling.

    Returns:
        One flat row per executed episode with deterministic repeat indices.
    """
    grouped: dict[
        tuple[str, str, str, int],
        list[tuple[int, str, dict[str, Any]]],
    ] = defaultdict(list)
    for index, record in enumerate(records):
        flattened = flatten_metrics(record)
        scenario_id = str(record.get("scenario_id") or "unknown")
        planner_key = str(
            record.get("planner_key")
            or _get_nested(record, "scenario_params.planner_key", default=None)
            or record.get("algo")
            or _get_nested(record, "scenario_params.algo", default="unknown")
        )
        kinematics = str(record.get("kinematics") or flattened.get("kinematics") or "unknown")
        seed = int(record.get("seed", -1))
        grouped[(scenario_id, planner_key, kinematics, seed)].append((index, kinematics, flattened))

    rows: list[dict[str, Any]] = []
    for (_, _, _, _), group_rows in sorted(grouped.items()):
        ordered = sorted(
            group_rows,
            key=lambda item: (
                str(item[2].get("episode_id", "")),
                int(item[0]),
            ),
        )
        for repeat_index, (_, kinematics, flat) in enumerate(ordered):
            rows.append(
                {
                    "episode_id": flat.get("episode_id"),
                    "scenario_id": flat.get("scenario_id"),
                    "planner_key": flat.get("planner_key")
                    or _get_nested(flat, "scenario_params.planner_key", default=None)
                    or flat.get("algo"),
                    "kinematics": kinematics,
                    "algo": flat.get("algo")
                    or _get_nested(flat, "scenario_params.algo", default="unknown"),
                    "seed": flat.get("seed"),
                    "repeat_index": repeat_index,
                    "success": _coerce_float(flat.get("success")),
                    "collision": _coerce_float(flat.get("collisions")),
                    "near_miss": _coerce_float(flat.get("near_misses")),
                    "time_to_goal": _coerce_float(flat.get("time_to_goal_norm")),
                    "snqi": _coerce_float(flat.get("snqi")),
                }
            )
    rows.sort(
        key=lambda row: (
            str(row.get("scenario_id", "")),
            str(row.get("planner_key", "")),
            str(row.get("kinematics", "")),
            int(row.get("seed", -1) or -1),
            int(row.get("repeat_index", -1) or -1),
        )
    )
    return rows


def build_statistical_sufficiency_rows(
    rows: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    """Build a thin statistical-sufficiency summary over seed-variability rows.

    Returns:
        One row per scenario/planner summarizing seed counts and CI half-widths.
    """
    sufficiency_rows: list[dict[str, Any]] = []
    for row in rows:
        summary = dict(row.get("summary") or {})
        metric_half_widths: dict[str, float | None] = {}
        metric_entries: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            metric_summary = dict(summary.get(metric) or {})
            half_width = metric_summary.get("ci_half_width")
            metric_half_widths[_metric_alias(metric)] = half_width
            metric_entries[_metric_alias(metric)] = {
                "n": metric_summary.get("count"),
                "ci_half_width": half_width,
            }
        sufficiency_rows.append(
            {
                "scenario_id": row.get("scenario_id"),
                "planner_key": row.get("planner_key"),
                "kinematics": row.get("kinematics"),
                "algo": row.get("algo"),
                "seed_count": row.get("seed_count"),
                "episode_count": row.get("episode_count"),
                "sufficiency_status": "reported",
                "metric_half_widths": metric_half_widths,
                "metrics": metric_entries,
            }
        )
    return sufficiency_rows


__all__ = [
    "build_seed_episode_rows",
    "build_seed_variability_csv_rows",
    "build_seed_variability_rows",
    "build_statistical_sufficiency_rows",
    "compute_seed_variance",
]
