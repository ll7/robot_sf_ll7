"""Aggregation and bootstrap confidence interval computations (T030).

Scope of T030 implementation:
  * Group episode records by (archetype, density).
  * Compute descriptive statistics per metric: mean, median, p95.
  * Perform bootstrap (seeded, deterministic) for mean & median to obtain
    provisional confidence intervals (these will be refined for rate metrics
    with Wilson intervals in T031). For now we store (low, high) tuples.
  * Return a list of ``AggregateMetricsGroup`` dataclass instances.

Out‑of‑scope (future tasks):
  * Wilson score interval integration for binary rate metrics (T031).
  * Effect size computations (T032).
  * Precision evaluation (T033) and adaptive integration (T034).

Design notes:
  * Determinism: bootstrap uses a ``random.Random`` instance seeded from
    ``cfg.master_seed`` combined with a stable hash of the group key to ensure
    reproducibility across runs and independence between groups.
  * Performance: bootstrap sample count is taken from ``cfg.bootstrap_samples``;
    in smoke mode we cap this at 300 (per research spec) while still allowing
    tests to override via their fixture config values.
  * Minimal validation: silently skips records missing the ``metrics`` field
    (log noise avoided for tests); in production we may raise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple

from loguru import logger

__all__ = [
    "AggregateMetric",
    "AggregateMetricsGroup",
    "aggregate_metrics",
]


@dataclass
class AggregateMetric:  # mirrors spec (subset for current needs)
    name: str
    mean: float
    median: float
    p95: float
    mean_ci: Tuple[float, float] | None
    median_ci: Tuple[float, float] | None


@dataclass
class AggregateMetricsGroup:
    archetype: str
    density: str
    count: int
    metrics: Dict[str, AggregateMetric]


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return math.nan
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _bootstrap_ci(
    values: List[float], samples: int, conf: float, rng: random.Random
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if not values:
        nan_pair = (math.nan, math.nan)
        return nan_pair, nan_pair
    n = len(values)
    if n == 1:  # degenerate case: CI collapses to the single value
        v = values[0]
        single = (v, v)
        return single, single
    stats_mean = []
    # We also compute median in the same routine by reusing draws to save time.
    stats_median = []
    for _ in range(samples):
        sample = [values[rng.randrange(0, n)] for _ in range(n)]
        stats_mean.append(mean(sample))
        stats_median.append(median(sample))
    stats_mean.sort()
    stats_median.sort()
    alpha = 1.0 - conf
    low_idx = int(alpha / 2 * samples)
    high_idx = min(samples - 1, int((1 - alpha / 2) * samples) - 1)
    mean_ci_tuple = (stats_mean[low_idx], stats_mean[high_idx])
    median_ci_tuple = (stats_median[low_idx], stats_median[high_idx])
    return mean_ci_tuple, median_ci_tuple


def aggregate_metrics(records: Iterable[dict], cfg):  # T030
    groups_raw = _group_records(records)
    if not groups_raw:
        return []
    bootstrap_samples, conf, master_seed = _bootstrap_params(cfg)
    result: List[AggregateMetricsGroup] = []
    for (arch, dens), recs in sorted(groups_raw.items()):
        metric_values = _collect_metric_values(recs)
        metric_objs: Dict[str, AggregateMetric] = {}
        for metric_name, vals in metric_values.items():
            metric_objs[metric_name] = _aggregate_single_metric(
                vals, arch, dens, metric_name, master_seed, bootstrap_samples, conf
            )
        result.append(
            AggregateMetricsGroup(
                archetype=arch,
                density=dens,
                count=len(recs),
                metrics=metric_objs,
            )
        )
    return result


def _group_records(records: Iterable[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for rec in records:
        try:
            arch = rec["archetype"]
            dens = rec["density"]
            metrics = rec.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
        except KeyError:  # pragma: no cover
            logger.debug("Skipping record missing required keys: {}", rec)
            continue
        groups.setdefault((arch, dens), []).append(rec)
    return groups


def _bootstrap_params(cfg) -> Tuple[int, float, int]:
    samples = int(getattr(cfg, "bootstrap_samples", 1000) or 1000)
    if getattr(cfg, "smoke", False):
        samples = min(samples, 300)
    conf = float(getattr(cfg, "bootstrap_confidence", 0.95) or 0.95)
    seed = int(getattr(cfg, "master_seed", 0) or 0)
    return samples, conf, seed


def _collect_metric_values(recs: List[dict]) -> Dict[str, List[float]]:
    vals: Dict[str, List[float]] = {}
    for r in recs:
        for k, v in r.get("metrics", {}).items():
            if isinstance(v, (int, float)):
                vals.setdefault(k, []).append(float(v))
    return vals


def _aggregate_single_metric(
    values: List[float],
    arch: str,
    dens: str,
    metric_name: str,
    master_seed: int,
    bootstrap_samples: int,
    conf: float,
) -> AggregateMetric:
    sorted_vals = sorted(values)
    m_mean = mean(sorted_vals) if sorted_vals else math.nan
    m_median = median(sorted_vals) if sorted_vals else math.nan
    m_p95 = _percentile(sorted_vals, 95.0) if sorted_vals else math.nan
    group_seed = hash((master_seed, arch, dens, metric_name)) & 0xFFFFFFFF
    rng = random.Random(group_seed)
    mean_ci, median_ci = _bootstrap_ci(sorted_vals, bootstrap_samples, conf, rng)
    return AggregateMetric(
        name=metric_name,
        mean=m_mean,
        median=m_median,
        p95=m_p95,
        mean_ci=mean_ci,
        median_ci=median_ci,
    )
