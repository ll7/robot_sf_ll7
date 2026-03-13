"""Seed variance analysis for Social Navigation Benchmark.

This module computes per-metric variability across seeds for grouped
episode records. The primary output is the coefficient of variation (CV),
defined as std/mean for each metric within a group.

Design notes:
- Inputs are the same JSONL episode records used by aggregation.
- Grouping supports dotted paths (e.g., "scenario_params.algo"). When the
  primary path is missing, a fallback (default: "scenario_id") is used.
- NaN metric values are ignored for statistics. If fewer than 2 finite values
  remain in a group for a metric, std and cv are reported as NaN.

Public API
----------
- compute_seed_variance(records, ...): return mapping group -> metric -> stats
- build_seed_variability_rows(records, ...): paper-facing per-scenario/per-planner export rows

CLI wiring is implemented in `robot_sf.benchmark.cli` as subcommand
`seed-variance`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.aggregate import flatten_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence


def _get_nested(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted-path value from a nested dict, falling back to default.

    Returns:
        Value at the path or the default.
    """
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _numeric_items(d: dict[str, Any]) -> dict[str, float]:
    """Extract numeric scalar items from a metrics row.

    Returns:
        Mapping of metric names to numeric values.
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


def _stats_for_vals(vals: list[float]) -> dict[str, float]:
    """Compute mean/std/cv for a list of values, ignoring non-finite.

    Returns:
        Dictionary with mean, std, cv, and count.
    """
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "cv": float("nan"), "count": 0.0}
    arr = np.asarray(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    n = finite.size
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "cv": float("nan"), "count": 0.0}
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=0)) if n >= 2 else float("nan")
    cv = float(std / mean) if (n >= 2 and mean != 0.0 and math.isfinite(std)) else float("nan")
    return {"mean": mean, "std": std, "cv": cv, "count": float(n)}


def compute_seed_variance(
    records: list[dict[str, Any]] | Sequence[dict[str, Any]],
    *,
    group_by: str = "scenario_id",
    fallback_group_by: str = "scenario_id",
    metrics: Sequence[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-metric seed variability for groups.

    Returns mapping group -> metric -> stats(mean, std, cv, count).

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        Nested dictionary: group -> metric -> statistics (mean, std, cv, count).
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


def build_seed_variability_rows(
    records: list[dict[str, Any]] | Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str],
    campaign_id: str,
    config_hash: str,
    git_hash: str,
    seed_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build paper-facing seed-variability rows from benchmark episode records.

    The rows are grouped by scenario and planner identity, then summarized
    across seeds while retaining per-seed metric means.

    Returns:
        List of paper-facing variability rows with per-seed metrics and
        across-seed summary statistics.
    """
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
        rows.append(
            {
                "scenario_id": scenario_id,
                "planner_key": planner_key,
                "algo": algo,
                "planner_group": planner_group,
                "kinematics": kinematics,
                "benchmark_profile": benchmark_profile,
                "seed_count": len(seed_groups),
                "episode_count": total_episodes,
                "seed_list": [entry["seed"] for entry in per_seed_rows],
                "per_seed": per_seed_rows,
                "summary": {
                    metric: _stats_for_vals(values)
                    for metric, values in across_seed_values.items()
                    if values
                },
                "provenance": {
                    "campaign_id": campaign_id,
                    "config_hash": config_hash,
                    "git_hash": git_hash,
                    "seed_policy": seed_policy,
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
        Flat rows suitable for CSV export, one row per scenario/planner/seed.
    """
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        provenance = dict(row.get("provenance") or {})
        summary = dict(row.get("summary") or {})
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
            }
            per_seed_metrics = dict(seed_row.get("metrics") or {})
            for metric in metrics:
                csv_row[f"{metric}_per_seed_mean"] = per_seed_metrics.get(metric)
                metric_summary = dict(summary.get(metric) or {})
                csv_row[f"{metric}_across_seed_mean"] = metric_summary.get("mean")
                csv_row[f"{metric}_across_seed_std"] = metric_summary.get("std")
                csv_row[f"{metric}_across_seed_cv"] = metric_summary.get("cv")
                csv_row[f"{metric}_across_seed_count"] = metric_summary.get("count")
            csv_rows.append(csv_row)
    return csv_rows


__all__ = [
    "build_seed_variability_csv_rows",
    "build_seed_variability_rows",
    "compute_seed_variance",
]
