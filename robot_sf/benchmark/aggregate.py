"""Aggregation utilities for Social Navigation Benchmark JSONL outputs.

Functions:
- read_jsonl(paths): load episode records from one or more JSONL files
- flatten_metrics(record): flatten nested metrics dict for CSV writing
- write_episode_csv(records, out_csv, ...): write per-episode metrics CSV
- compute_aggregates(records, group_by=..., ...): grouped mean/median/p95 (+SNQI)

Notes:
- group_by supports dotted paths (e.g., "scenario_params.algo"). If the path
  is missing for a record, it falls back to the record's "scenario_id".
- If SNQI is not present and weights/baseline are provided, it's recomputed.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, cast

import numpy as np

from robot_sf.benchmark.metrics import snqi as snqi_fn


def read_jsonl(paths: Sequence[str | Path] | str | Path) -> List[Dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records: List[Dict[str, Any]] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(rec)
    return records


def _get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def flatten_metrics(rec: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "episode_id": rec.get("episode_id"),
        "scenario_id": rec.get("scenario_id"),
        "seed": rec.get("seed"),
    }
    metrics = dict(rec.get("metrics") or {})
    fq = metrics.pop("force_quantiles", {}) or {}
    # Flatten known force quantiles
    for qk in ("q50", "q90", "q95"):
        key = f"force_{qk}"
        base[key] = fq.get(qk)
    # Remainder metrics (flat numbers)
    for k, v in metrics.items():
        base[k] = v
    return base


def _ensure_snqi(
    rec: Dict[str, Any],
    weights: Dict[str, float] | None,
    baseline: Dict[str, Dict[str, float]] | None,
) -> None:
    if rec.get("metrics") is None:
        return
    if "snqi" in rec["metrics"]:
        return
    if weights is None:
        return
    try:
        rec["metrics"]["snqi"] = float(snqi_fn(rec["metrics"], weights, baseline_stats=baseline))
    except Exception:
        # Leave untouched if computation fails
        pass


def write_episode_csv(
    records: List[Dict[str, Any]],
    out_csv: str | Path,
    *,
    snqi_weights: Dict[str, float] | None = None,
    snqi_baseline: Dict[str, Dict[str, float]] | None = None,
) -> str:
    # Optionally compute SNQI per record if missing
    for rec in records:
        _ensure_snqi(rec, snqi_weights, snqi_baseline)

    # Determine all metric keys across records for CSV header
    flat_rows = [flatten_metrics(r) for r in records]
    keys = set()
    for row in flat_rows:
        keys.update(row.keys())
    # Ensure stable ordering: id fields first
    id_keys = ["episode_id", "scenario_id", "seed"]
    metric_keys = sorted(k for k in keys if k not in id_keys)
    header = id_keys + metric_keys

    out_csv = str(out_csv)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow({k: row.get(k) for k in header})
    return out_csv


def _numeric_items(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if k in ("episode_id", "scenario_id", "seed"):
            continue
        if v is None:
            continue
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            out[k] = float(v)
    return out


def compute_aggregates(
    records: List[Dict[str, Any]],
    *,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
    snqi_weights: Dict[str, float] | None = None,
    snqi_baseline: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    # Optionally compute SNQI per record if missing
    for rec in records:
        _ensure_snqi(rec, snqi_weights, snqi_baseline)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        g = _get_nested(rec, group_by)
        if g is None:
            g = _get_nested(rec, fallback_group_by)
        if g is None:
            g = "unknown"
        groups[str(g)].append(flatten_metrics(rec))

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for g, rows in groups.items():
        # collect numeric columns
        cols: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            num = _numeric_items(row)
            for k, v in num.items():
                cols[k].append(v)
        agg: Dict[str, Dict[str, float]] = {}
        for k, vals in cols.items():
            arr = np.asarray(vals, dtype=float)
            agg[k] = {
                "mean": float(np.nanmean(arr)),
                "median": float(np.nanmedian(arr)),
                "p95": float(np.nanpercentile(arr, 95)),
            }
        summary[g] = agg
    return summary


# --- Optional bootstrap confidence intervals ---


def _bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a statistic.

    Parameters
    - data: 1D numpy array (NaNs will be ignored).
    - stat_fn: callable computing a scalar statistic on a 1D array.
    - samples: number of bootstrap resamples (B).
    - confidence: confidence level (e.g., 0.95).
    - seed: optional RNG seed for reproducibility.

    Returns (low, high) CI bounds; (nan, nan) if insufficient data or samples=0.
    """
    if samples <= 0:
        return (float("nan"), float("nan"))
    x = np.asarray(data, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        try:
            stats[i] = float(stat_fn(xb))
        except Exception:
            stats[i] = float("nan")
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return (float("nan"), float("nan"))
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(stats, 100.0 * alpha))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha)))
    return (lo, hi)


def _group_flattened(
    records: List[Dict[str, Any]],
    *,
    group_by: str,
    fallback_group_by: str,
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        g = _get_nested(rec, group_by)
        if g is None:
            g = _get_nested(rec, fallback_group_by)
        if g is None:
            g = "unknown"
        groups[str(g)].append(flatten_metrics(rec))
    return groups


def _attach_ci_for_group(
    dst_group: Dict[str, Dict[str, Any]],
    values_by_metric: Dict[str, List[float]],
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int | None,
) -> None:
    for metric_name, values in values_by_metric.items():
        arr = np.asarray(values, dtype=float)

        def mean_fn(a: np.ndarray) -> float:
            return float(np.mean(a))

        def median_fn(a: np.ndarray) -> float:
            return float(np.median(a))

        def p95_fn(a: np.ndarray) -> float:
            return float(np.percentile(a, 95))

        lo_hi_mean = _bootstrap_ci(
            arr,
            mean_fn,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
        lo_hi_median = _bootstrap_ci(
            arr,
            median_fn,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
        lo_hi_p95 = _bootstrap_ci(
            arr,
            p95_fn,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
        dst_group.setdefault(metric_name, {})
        dst_group[metric_name]["mean_ci"] = [float(lo_hi_mean[0]), float(lo_hi_mean[1])]
        dst_group[metric_name]["median_ci"] = [float(lo_hi_median[0]), float(lo_hi_median[1])]
        dst_group[metric_name]["p95_ci"] = [float(lo_hi_p95[0]), float(lo_hi_p95[1])]


def compute_aggregates_with_ci(
    records: List[Dict[str, Any]],
    *,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
    snqi_weights: Dict[str, float] | None = None,
    snqi_baseline: Dict[str, Dict[str, float]] | None = None,
    return_ci: bool = True,
    bootstrap_samples: int = 1000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int | None = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute grouped aggregates and optional bootstrap CIs.

    This preserves the original aggregate keys (mean, median, p95) and, when
    return_ci is True and bootstrap_samples>0, adds parallel keys 'mean_ci',
    'median_ci', 'p95_ci' with [low, high] bounds using percentile bootstrap.
    """
    # Start from base aggregates (no CI) for consistency
    base = compute_aggregates(
        records,
        group_by=group_by,
        fallback_group_by=fallback_group_by,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
    )
    if not return_ci or bootstrap_samples <= 0:
        # Upcast type to Any container for compatibility, but keep content unchanged
        return cast(Dict[str, Dict[str, Dict[str, Any]]], base)

    # Rebuild groups with flattened numeric values to avoid rework
    for rec in records:
        _ensure_snqi(rec, snqi_weights, snqi_baseline)
    groups = _group_flattened(records, group_by=group_by, fallback_group_by=fallback_group_by)

    out: Dict[str, Dict[str, Dict[str, Any]]] = {k: dict(v) for k, v in base.items()}
    for g, rows in groups.items():
        # collect numeric columns per group
        cols: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            num = _numeric_items(row)
            for k, v in num.items():
                cols[k].append(v)
        _attach_ci_for_group(
            out.setdefault(g, {}),
            cols,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
            bootstrap_seed=bootstrap_seed,
        )


__all__ = [
    "read_jsonl",
    "flatten_metrics",
    "write_episode_csv",
    "compute_aggregates",
    "compute_aggregates_with_ci",
]
