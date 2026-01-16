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
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from loguru import logger

from robot_sf.benchmark.errors import AggregationMetadataError
from robot_sf.benchmark.metrics import snqi as snqi_fn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def read_jsonl(paths: Sequence[str | Path] | str | Path) -> list[dict[str, Any]]:
    """Read one or more JSONL files into a list of records.

    Returns:
        List of parsed episode records.
    """
    if isinstance(paths, str | Path):
        path_list = [paths]
    else:
        path_list = list(paths)  # type: ignore[arg-type]
    records: list[dict[str, Any]] = []
    for p in path_list:
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


def _get_nested(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Resolve a dotted-path value from a dict.

    Returns:
        Value at the path or default when missing.
    """
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


_EFFECTIVE_GROUP_KEY = "scenario_params.algo | algo | scenario_id"


def _normalize_algo(value: Any) -> str | None:
    """Normalize algorithm identifiers to a non-empty string.

    Returns:
        Normalized string or None if empty/invalid.
    """
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return None


def _ensure_mapping(record: dict[str, Any], key: str, episode_id: str | None) -> None:
    """Ensure a record field is a mapping when nested access is required."""
    value = record.get(key)
    if value is not None and not isinstance(value, dict):
        raise AggregationMetadataError(
            f"{key} must be a mapping to access nested algorithm metadata.",
            episode_id=episode_id,
            missing_fields=(f"{key}", f"{key}.algo"),
            advice="Regenerate the benchmark data to include structured scenario parameters.",
        )


def _resolve_group_key(
    record: dict[str, Any],
    *,
    group_by: str,
    fallback_group_by: str,
) -> str:
    """Resolve the aggregation group key with metadata fallbacks.

    Returns:
        Group key string.
    """
    episode_id = record.get("episode_id")
    episode_ref = str(episode_id) if episode_id is not None else None

    if group_by.startswith("scenario_params"):
        _ensure_mapping(record, "scenario_params", episode_ref)

    nested_algo = _normalize_algo(_get_nested(record, group_by))
    if nested_algo is not None:
        return nested_algo

    top_level_algo = _normalize_algo(record.get("algo"))
    if top_level_algo is not None:
        return top_level_algo

    # Check algorithm_metadata.algorithm as additional fallback
    metadata_algo = _normalize_algo(_get_nested(record, "algorithm_metadata.algorithm"))
    if metadata_algo is not None:
        return metadata_algo

    fallback_value = _get_nested(record, fallback_group_by)
    if fallback_value is not None:
        if group_by == "scenario_params.algo":
            raise AggregationMetadataError(
                "Episode lacks algorithm metadata required for aggregation.",
                episode_id=episode_ref,
                missing_fields=("scenario_params.algo", "algo", "algorithm_metadata.algorithm"),
                advice="Ensure the orchestrator mirrors algorithm identifiers before aggregation.",
            )
        return str(fallback_value)

    raise AggregationMetadataError(
        "Unable to determine aggregation group key for episode.",
        episode_id=episode_ref,
        missing_fields=(group_by, "algo"),
        advice="Verify that episode records include algorithm metadata.",
    )


def flatten_metrics(rec: dict[str, Any]) -> dict[str, Any]:
    """Flatten metrics dict into a flat per-episode row.

    Returns:
        Flattened metrics row for CSV or aggregation.
    """
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
    base.update(metrics)
    return base


def _ensure_snqi(
    rec: dict[str, Any],
    weights: dict[str, float] | None,
    baseline: dict[str, dict[str, float]] | None,
) -> None:
    """Compute and attach SNQI when missing and inputs are provided."""
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
    records: list[dict[str, Any]],
    out_csv: str | Path,
    *,
    snqi_weights: dict[str, float] | None = None,
    snqi_baseline: dict[str, dict[str, float]] | None = None,
) -> str:
    # Optionally compute SNQI per record if missing
    """Write per-episode metrics to CSV.

    Returns:
        Path string to the written CSV file.
    """
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


def _numeric_items(d: dict[str, Any]) -> dict[str, float]:
    """Extract numeric values from a flattened metrics row.

    Returns:
        Mapping of numeric metric values.
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


def compute_aggregates(
    records: list[dict[str, Any]],
    *,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
    snqi_weights: dict[str, float] | None = None,
    snqi_baseline: dict[str, dict[str, float]] | None = None,
    expected_algorithms: set[str] | None = None,
    logger_ctx=None,
) -> dict[str, dict[str, dict[str, float]]]:
    # Optionally compute SNQI per record if missing
    """Aggregate metrics by group and compute summary statistics.

    Returns:
        Nested dict of group -> metric -> summary statistics.
    """
    for rec in records:
        _ensure_snqi(rec, snqi_weights, snqi_baseline)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    present_algorithms: set[str] = set()
    for rec in records:
        key = _resolve_group_key(rec, group_by=group_by, fallback_group_by=fallback_group_by)
        key_str = str(key)
        present_algorithms.add(key_str)
        groups[key_str].append(flatten_metrics(rec))

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for g, rows in groups.items():
        # collect numeric columns
        cols: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            num = _numeric_items(row)
            for k, v in num.items():
                cols[k].append(v)
        agg: dict[str, dict[str, float]] = {}
        for k, vals in cols.items():
            arr = np.asarray(vals, dtype=float)
            agg[k] = {
                "mean": float(np.nanmean(arr)),
                "median": float(np.nanmedian(arr)),
                "p95": float(np.nanpercentile(arr, 95)),
            }
        summary[g] = agg

    meta: dict[str, Any] = {
        "group_by": group_by,
        "effective_group_key": _EFFECTIVE_GROUP_KEY,
        "missing_algorithms": [],
        "warnings": [],
    }

    if expected_algorithms:
        expected_set = {str(v) for v in expected_algorithms}
        missing = sorted(expected_set - present_algorithms)
        meta["missing_algorithms"] = missing
        if missing:
            warning_text = f"Missing algorithms detected: {', '.join(missing)}"
            meta["warnings"] = [warning_text]
            (logger_ctx or logger).bind(
                event="aggregation_missing_algorithms",
                expected=sorted(expected_set),
                present=sorted(present_algorithms),
                missing=missing,
            ).warning(warning_text)

    summary["_meta"] = meta
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

    Args:
        data: One-dimensional array of samples; NaNs are ignored.
        stat_fn: Callable that computes a scalar statistic over a 1D array.
        samples: Number of bootstrap resamples to draw.
        confidence: Confidence level for the returned interval (e.g., 0.95).
        seed: Optional RNG seed for reproducibility.

    Returns:
        tuple[float, float]: ``(low, high)`` bounds, ``(nan, nan)`` when insufficient data.
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
    records: list[dict[str, Any]],
    *,
    group_by: str,
    fallback_group_by: str,
) -> dict[str, list[dict[str, Any]]]:
    """Group flattened episode rows by aggregation key.

    Returns:
        Mapping of group key to flattened rows.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = _resolve_group_key(rec, group_by=group_by, fallback_group_by=fallback_group_by)
        groups[str(key)].append(flatten_metrics(rec))
    return groups


def _attach_ci_for_group(
    dst_group: dict[str, dict[str, Any]],
    values_by_metric: dict[str, list[float]],
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int | None,
) -> None:
    """Attach bootstrap confidence intervals for each metric in a group."""
    for metric_name, values in values_by_metric.items():
        arr = np.asarray(values, dtype=float)

        def mean_fn(a: np.ndarray) -> float:
            """Return mean for bootstrap sample."""
            return float(np.mean(a))

        def median_fn(a: np.ndarray) -> float:
            """Return median for bootstrap sample."""
            return float(np.median(a))

        def p95_fn(a: np.ndarray) -> float:
            """Return p95 for bootstrap sample."""
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
    records: list[dict[str, Any]],
    *,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
    snqi_weights: dict[str, float] | None = None,
    snqi_baseline: dict[str, dict[str, float]] | None = None,
    return_ci: bool = True,
    bootstrap_samples: int = 1000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int | None = None,
    expected_algorithms: set[str] | None = None,
    logger_ctx=None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute grouped aggregates and optional bootstrap CIs.

    This preserves the original aggregate keys (mean, median, p95) and, when
    return_ci is True and bootstrap_samples>0, adds parallel keys 'mean_ci',
    'median_ci', 'p95_ci' with [low, high] bounds using percentile bootstrap.

    Returns:
        Nested dictionary mapping group names to metric names to aggregate statistics.
    """
    # Start from base aggregates (no CI) for consistency
    base = compute_aggregates(
        records,
        group_by=group_by,
        fallback_group_by=fallback_group_by,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
        expected_algorithms=expected_algorithms,
        logger_ctx=logger_ctx,
    )
    if not return_ci or bootstrap_samples <= 0:
        # Upcast type to Any container for compatibility, but keep content unchanged
        return cast("dict[str, dict[str, dict[str, Any]]]", base)

    # Rebuild groups with flattened numeric values to avoid rework
    for rec in records:
        _ensure_snqi(rec, snqi_weights, snqi_baseline)
    groups = _group_flattened(records, group_by=group_by, fallback_group_by=fallback_group_by)

    out: dict[str, dict[str, dict[str, Any]]] = {
        k: dict(v) for k, v in base.items() if k != "_meta"
    }
    if "_meta" in base:
        out["_meta"] = dict(base["_meta"])  # type: ignore[assignment]
    for g, rows in groups.items():
        # collect numeric columns per group
        cols: dict[str, list[float]] = defaultdict(list)
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
    return out


__all__ = [
    "compute_aggregates",
    "compute_aggregates_with_ci",
    "flatten_metrics",
    "read_jsonl",
    "write_episode_csv",
]
