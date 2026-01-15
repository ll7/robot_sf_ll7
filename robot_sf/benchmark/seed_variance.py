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


__all__ = ["compute_seed_variance"]
