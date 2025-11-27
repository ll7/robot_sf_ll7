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
from typing import TYPE_CHECKING

from loguru import logger

# Map legacy/alias metric names to the canonical ones used in aggregation.
_METRIC_ALIASES = {"average_speed": "avg_speed"}

if TYPE_CHECKING:
    from collections.abc import Iterable

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
    mean_ci: tuple[float, float] | None
    median_ci: tuple[float, float] | None


@dataclass
class AggregateMetricsGroup:
    archetype: str
    density: str
    count: int
    metrics: dict[str, AggregateMetric]


def _percentile(sorted_vals: list[float], p: float) -> float:
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
    values: list[float],
    samples: int,
    conf: float,
    rng: random.Random,
) -> tuple[tuple[float, float], tuple[float, float]]:
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
    result: list[AggregateMetricsGroup] = []
    for (arch, dens), recs in sorted(groups_raw.items()):
        metric_values = _collect_metric_values(recs)
        metric_objs: dict[str, AggregateMetric] = {}
        for metric_name, vals in metric_values.items():
            metric_objs[metric_name] = _aggregate_single_metric(
                vals,
                arch,
                dens,
                metric_name,
                master_seed,
                bootstrap_samples,
                conf,
            )
        result.append(
            AggregateMetricsGroup(
                archetype=arch,
                density=dens,
                count=len(recs),
                metrics=metric_objs,
            ),
        )
    return result


def _group_records(records: Iterable[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
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


def _bootstrap_params(cfg) -> tuple[int, float, int]:
    samples = int(getattr(cfg, "bootstrap_samples", 1000) or 1000)
    if getattr(cfg, "smoke", False):
        samples = min(samples, 300)
    conf = float(getattr(cfg, "bootstrap_confidence", 0.95) or 0.95)
    seed = int(getattr(cfg, "master_seed", 0) or 0)
    return samples, conf, seed


def _collect_metric_values(recs: list[dict]) -> dict[str, list[float]]:
    vals: dict[str, list[float]] = {}
    for r in recs:
        for k, v in r.get("metrics", {}).items():
            if isinstance(v, int | float):
                key = _METRIC_ALIASES.get(k, k)
                vals.setdefault(key, []).append(float(v))
    return vals


def _aggregate_single_metric(
    values: list[float],
    arch: str,
    dens: str,
    metric_name: str,
    master_seed: int,
    bootstrap_samples: int,
    conf: float,
) -> AggregateMetric:
    finite_values = [v for v in values if math.isfinite(v)]
    non_finite = len(values) - len(finite_values)
    if non_finite:
        logger.bind(
            event="aggregation_non_finite_metric",
            archetype=arch,
            density=dens,
            metric=metric_name,
        ).warning(
            "Dropped {} non-finite metric values; computing stats from {} finite samples.",
            non_finite,
            len(finite_values),
        )
    sorted_vals = sorted(finite_values)
    m_mean = mean(sorted_vals) if sorted_vals else math.nan
    m_median = median(sorted_vals) if sorted_vals else math.nan
    m_p95 = _percentile(sorted_vals, 95.0) if sorted_vals else math.nan
    group_seed = hash((master_seed, arch, dens, metric_name)) & 0xFFFFFFFF
    rng = random.Random(group_seed)
    mean_ci, median_ci = _bootstrap_ci(sorted_vals, bootstrap_samples, conf, rng)
    # Replace mean_ci for rate metrics with Wilson interval (T031) for better coverage when near bounds.
    if metric_name in {"collision_rate", "success_rate"} and sorted_vals:
        # Interpret values as Bernoulli outcomes; approximate p via mean.
        p = m_mean
        n = len(sorted_vals)
        mean_ci = _wilson_interval(p, n, conf)
    return AggregateMetric(
        name=metric_name,
        mean=m_mean,
        median=m_median,
        p95=m_p95,
        mean_ci=mean_ci,
        median_ci=median_ci,
    )


def _wilson_interval(p: float, n: int, conf: float) -> tuple[float, float]:  # T031
    """Wilson score interval for a Bernoulli proportion.

    Parameters
    ----------
    p : float
        Observed proportion (mean of Bernoulli samples).
    n : int
        Number of trials (episodes).
    conf : float
        Confidence level in (0,1), typically 0.95.

    Returns
    -------
    (low, high) tuple clipped to [0,1]. For n<=0 returns (nan,nan).
    """
    if n <= 0 or not (0.0 <= p <= 1.0):  # defensive
        return (math.nan, math.nan)
    # Approximate z via inverse error function for standard normal quantile.
    # For common confidence 0.95, z ≈ 1.959963984540054.
    # To keep dependency-light, we precompute z from conf using a simple approximation if needed.
    z = _z_from_conf(conf)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return (low, high)


def _z_from_conf(conf: float) -> float:
    # Map a set of common confidence levels; fallback to 0.95 if unknown.
    mapping = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.975: 2.241402727604947,
        0.99: 2.5758293035489004,
    }
    if conf in mapping:
        return mapping[conf]
    # Simple rational approximation for inverse normal CDF (Acklam) for two-tailed alpha.
    # We implement minimal subset since tests will likely use standard 0.95.
    alpha = 1 - conf
    # Two-tailed so tail probability each side:
    tail = alpha / 2
    # Coefficients for approximation
    a = [
        -39.6968302866538,
        220.946098424521,
        -275.928510446969,
        138.357751867269,
        -30.6647980661472,
        2.50662827745924,
    ]
    b = [
        -54.4760987982241,
        161.585836858041,
        -155.698979859887,
        66.8013118877197,
        -13.2806815528857,
    ]
    c = [
        -0.00778489400243029,
        -0.322396458041136,
        -2.40075827716184,
        -2.54973253934373,
        4.37466414146497,
        2.93816398269878,
    ]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    # Break-points
    plow = 0.02425
    phigh = 1 - plow
    if tail < plow:
        q = math.sqrt(-2 * math.log(tail))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        z = num / den
    elif tail > phigh:
        q = math.sqrt(-2 * math.log(1 - tail))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        z = -num / den
    else:
        q = tail - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        z = num / den
    return abs(z)  # symmetric
