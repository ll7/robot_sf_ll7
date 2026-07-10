"""Aggregation and bootstrap confidence interval computations (T030).

Scope of T030 implementation:
  * Group episode records by (archetype, density).
  * Compute descriptive statistics per metric: mean, median, p95.
  * Perform bootstrap (seeded, deterministic) for mean & median to obtain
    provisional confidence intervals (these will be refined for rate metrics
    with Wilson intervals in T031). For now we store (low, high) tuples.
  * Return a list of ``AggregateMetricsGroup`` dataclass instances.

Out‑of‑scope (future tasks):
  * Effect size computations (T032).
  * Precision evaluation (T033) and adaptive integration (T034).

Resampling modes (issue #5139):
  * ``flat`` (default): resample episodes i.i.d. within a single
    (archetype, density) group. This is the original T030 behaviour and the
    backward-compatible default. For binary rate metrics it pairs with the
    Wilson score interval (T031), which assumes i.i.d. Bernoulli trials.
  * ``hierarchical``: two-stage cluster bootstrap. Episodes are nested in
    scenario cells (and seeds); flat i.i.d. resampling understates uncertainty
    because it ignores within-cell correlation. The hierarchical procedure
    resamples cluster cells with replacement and then episodes within each
    resampled cell with replacement before pooling, which is the documented
    "successor-campaign procedure". The cluster field is selected by
    ``cfg.bootstrap_cluster`` (``"scenario"`` -> ``scenario_id``, the default;
    ``"seed"`` -> ``seed`` for a seed-level cluster bootstrap).
    For binary rate metrics in hierarchical mode the flat Wilson interval is
    replaced by a cluster-robust interval (see ``_cluster_robust_interval``)
    that derives its standard error from the between-cell dispersion of
    cell-level proportions, so the interval widens with intra-cluster
    correlation rather than assuming independence.

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

import hashlib
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
    "cluster_robust_interval",
    "hierarchical_bootstrap_ci",
]


@dataclass
class AggregateMetric:  # mirrors spec (subset for current needs)
    """Aggregated statistics for a single metric."""

    name: str
    mean: float
    median: float
    p95: float
    mean_ci: tuple[float, float] | None
    median_ci: tuple[float, float] | None


@dataclass
class AggregateMetricsGroup:
    """Aggregated statistics for an archetype/density group."""

    archetype: str
    density: str
    count: int
    metrics: dict[str, AggregateMetric]


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Return percentile value from a sorted list."""
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
    """Bootstrap mean/median confidence intervals for values.

    Returns:
        Tuple of (mean_ci, median_ci).
    """
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


def _resolve_cluster_field(cfg) -> str:
    """Resolve the record field that defines a bootstrap cluster (cell).

    Returns:
        Record key used as the cluster identifier in hierarchical mode
        (``"scenario_id"`` for ``bootstrap_cluster == "scenario"``,
        ``"seed"`` for ``bootstrap_cluster == "seed"``). Defaults to
        ``scenario_id`` for any unrecognised value so callers always get a
        concrete field name.
    """
    cluster = str(getattr(cfg, "bootstrap_cluster", "scenario") or "scenario").lower()
    return "seed" if cluster == "seed" else "scenario_id"


def _percentile_of_sorted(sorted_vals: list[float], conf: float) -> tuple[float, float]:
    """Return the (low, high) percentile interval of a sorted sample list.

    Args:
        sorted_vals: Values sorted ascending. Must be non-empty.
        conf: Confidence level in (0, 1).

    Returns:
        ``(lower, upper)`` bounds of the two-sided central interval.
    """
    n = len(sorted_vals)
    alpha = 1.0 - conf
    low_idx = int(alpha / 2 * n)
    high_idx = min(n - 1, int((1 - alpha / 2) * n) - 1)
    high_idx = max(high_idx, low_idx)  # guard against empty upper slice
    return (sorted_vals[low_idx], sorted_vals[high_idx])


def hierarchical_bootstrap_ci(
    clustered_values: list[list[float]],
    samples: int,
    conf: float,
    rng: random.Random,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two-stage (cluster-then-episode) bootstrap mean/median CIs.

    This is the hierarchical (scenario-then-episode) resampling procedure.
    For each bootstrap iteration it resamples cluster cells with replacement
    and then episodes within each resampled cell with replacement, pools the
    drawn episodes, and records the mean and median of the pool. The
    percentile interval of those statistics is the reported CI. Unlike flat
    i.i.d. resampling, this preserves within-cell correlation and therefore
    widens intervals when episodes within a cell are similar (the
    anti-conservatism correction documented in issue #5139).

    Args:
        clustered_values: Per-cluster lists of metric values. Each inner list
            is the set of episode values observed in one cluster (cell). Must
            contain at least one non-empty cluster.
        samples: Number of bootstrap iterations.
        conf: Confidence level in (0, 1).
        rng: Seeded ``random.Random`` for reproducibility.

    Returns:
        Tuple of ``(mean_ci, median_ci)`` as ``(low, high)`` tuples. Returns
        ``(nan, nan)`` pairs when there is no data and collapses to the single
        value when there is exactly one episode overall.
    """
    nan_pair = (math.nan, math.nan)
    cells = [c for c in clustered_values if c]
    if not cells:
        return nan_pair, nan_pair
    total = sum(len(c) for c in cells)
    if total == 1:  # degenerate: CI collapses to the single observed value
        v = next(v for c in cells for v in c)
        single = (v, v)
        return single, single
    n_cells = len(cells)
    stats_mean: list[float] = []
    stats_median: list[float] = []
    for _ in range(samples):
        pooled: list[float] = []
        for _ in range(n_cells):
            cell = cells[rng.randrange(0, n_cells)]
            m = len(cell)
            pooled.extend(cell[rng.randrange(0, m)] for _ in range(m))
        stats_mean.append(mean(pooled))
        stats_median.append(median(pooled))
    stats_mean.sort()
    stats_median.sort()
    return _percentile_of_sorted(stats_mean, conf), _percentile_of_sorted(stats_median, conf)


def cluster_robust_interval(
    clustered_values: list[list[float]],
    conf: float,
) -> tuple[float, float]:
    """Cluster-robust normal interval for a binary (rate) endpoint.

    Counterpart to the flat Wilson interval (T031) for the hierarchical mode.
    The flat Wilson interval assumes i.i.d. Bernoulli trials and therefore
    understates uncertainty when episodes within a cluster are correlated.
    This interval estimates the standard error of the overall proportion from
    the between-cluster dispersion of cluster-level proportions
    (cluster-robust / sandwich-style variance) and forms a symmetric normal
    interval clipped to ``[0, 1]``.

    Let ``C`` be the number of clusters, ``p_c`` the proportion in cluster
    ``c``, and ``p_bar`` the mean of the ``p_c``. The cluster-robust variance
    of the overall mean is ``sum_c (p_c - p_bar)^2 / (C * (C - 1))`` and the
    interval is ``p_hat +/- z * sqrt(var)`` clipped to ``[0, 1]``, where
    ``p_hat`` is the pooled episode-level proportion (consistent with the
    bootstrap point estimate).

    Args:
        clustered_values: Per-cluster lists of 0/1 (or [0,1]) outcomes. Each
            inner list is the set of episode outcomes in one cluster.
        conf: Confidence level in (0, 1).

    Returns:
        ``(low, high)`` clipped to ``[0, 1]``. Returns ``(nan, nan)`` when
        there is no data, collapses to the single value when there is exactly
        one episode, and falls back to the pooled Wilson interval when there
        is only one cluster (cluster-robust SE is undefined with C < 2).
    """
    cells = [c for c in clustered_values if c]
    if not cells:
        return (math.nan, math.nan)
    total_vals = [v for c in cells for v in c if math.isfinite(v)]
    if not total_vals:
        return (math.nan, math.nan)
    n = len(total_vals)
    p_hat = sum(total_vals) / n
    if n == 1:
        return (p_hat, p_hat)
    c = len(cells)
    if c < 2:
        # Cluster-robust SE is undefined with a single cluster; fall back to
        # the flat Wilson interval so the rate metric still reports a finite,
        # defensible interval rather than a zero-width degenerate one.
        return _wilson_interval(p_hat, n, conf)
    cell_means = [sum(cl) / len(cl) for cl in cells]
    p_bar = sum(cell_means) / c
    var = sum((pc - p_bar) ** 2 for pc in cell_means) / (c * (c - 1))
    se = math.sqrt(var) if var > 0 else 0.0
    z = _z_from_conf(conf)
    half = z * se
    low = max(0.0, p_hat - half)
    high = min(1.0, p_hat + half)
    return (low, high)


def aggregate_metrics(records: Iterable[dict], cfg):  # T030
    """Aggregate metrics grouped by archetype and density.

    Resampling mode is selected by ``cfg.bootstrap_mode``: ``"flat"`` (default,
    backward-compatible i.i.d. episode bootstrap) or ``"hierarchical"`` (two-stage
    cluster bootstrap, issue #5139). In hierarchical mode the cluster field is
    ``cfg.bootstrap_cluster`` (``"scenario"`` -> ``scenario_id`` by default,
    ``"seed"`` -> ``seed``).

    Returns:
        List of AggregateMetricsGroup entries.
    """
    groups_raw = _group_records(records)
    if not groups_raw:
        return []
    bootstrap_samples, conf, master_seed, mode, cluster_field = _bootstrap_params(cfg)
    result: list[AggregateMetricsGroup] = []
    for (arch, dens), recs in sorted(groups_raw.items()):
        metric_values = _collect_metric_values(recs)
        clustered_values = (
            _collect_clustered_metric_values(recs, cluster_field) if mode == "hierarchical" else {}
        )
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
                clustered_values.get(metric_name),
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
    """Group records by (archetype, density).

    Returns:
        Mapping of (archetype, density) to record lists.
    """
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


def _bootstrap_params(cfg) -> tuple[int, float, int, str, str]:
    """Resolve bootstrap configuration from the config object.

    Returns:
        Tuple of (samples, confidence, master_seed, bootstrap_mode,
        cluster_field). ``bootstrap_mode`` is ``"flat"`` (default) or
        ``"hierarchical"``; ``cluster_field`` is the record key used as the
        cluster identifier in hierarchical mode.
    """
    samples = int(getattr(cfg, "bootstrap_samples", 1000) or 1000)
    if getattr(cfg, "smoke", False):
        samples = min(samples, 300)
    conf = float(getattr(cfg, "bootstrap_confidence", 0.95) or 0.95)
    seed = int(getattr(cfg, "master_seed", 0) or 0)
    mode = str(getattr(cfg, "bootstrap_mode", "flat") or "flat").lower()
    if mode not in {"flat", "hierarchical"}:
        logger.bind(
            event="aggregation_unknown_bootstrap_mode",
            bootstrap_mode=mode,
        ).warning(
            "Unknown bootstrap_mode '{}'; falling back to 'flat'.",
            mode,
        )
        mode = "flat"
    cluster_field = _resolve_cluster_field(cfg)
    return samples, conf, seed, mode, cluster_field


def _collect_clustered_metric_values(
    recs: list[dict],
    cluster_field: str,
) -> dict[str, list[list[float]]]:
    """Collect numeric metric values grouped by cluster (cell).

    Used by the hierarchical bootstrap. Records missing the cluster field are
    placed in a synthetic ``"__no_cluster__"`` cell so their values are still
    resampled rather than dropped.

    Returns:
        Mapping of metric name to a list of per-cluster value lists.
    """
    cells: dict[str, dict[str, list[float]]] = {}
    for r in recs:
        cluster_key = r.get(cluster_field)
        cell_id = "__no_cluster__" if cluster_key is None else str(cluster_key)
        bucket = cells.setdefault(cell_id, {})
        for k, v in r.get("metrics", {}).items():
            if isinstance(v, int | float):
                key = _METRIC_ALIASES.get(k, k)
                bucket.setdefault(key, []).append(float(v))
    result: dict[str, list[list[float]]] = {}
    for cell_metrics in cells.values():
        for metric_name, vals in cell_metrics.items():
            result.setdefault(metric_name, []).append(vals)
    return result


def _finite_clusters(
    clustered_values: list[list[float]] | None,
) -> list[list[float]]:
    """Drop non-finite values and empty cells from clustered metric values.

    Returns:
        List of non-empty per-cluster lists containing only finite values.
        Returns ``[]`` when ``clustered_values`` is ``None`` or has no usable
        data, signalling the caller to fall back to the flat bootstrap.
    """
    if not clustered_values:
        return []
    cleaned = [[v for v in cell if math.isfinite(v)] for cell in clustered_values]
    return [c for c in cleaned if c]


def _collect_metric_values(recs: list[dict]) -> dict[str, list[float]]:
    """Collect numeric metric values from records.

    Returns:
        Mapping of metric name to list of values.
    """
    vals: dict[str, list[float]] = {}
    for r in recs:
        for k, v in r.get("metrics", {}).items():
            if isinstance(v, int | float):
                key = _METRIC_ALIASES.get(k, k)
                vals.setdefault(key, []).append(float(v))
    return vals


def _stable_group_seed(master_seed: int, arch: str, dens: str, metric_name: str) -> int:
    """Return a process-independent bootstrap seed for one metric group."""
    material = "\x1f".join((str(master_seed), arch, dens, metric_name)).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _aggregate_single_metric(
    values: list[float],
    arch: str,
    dens: str,
    metric_name: str,
    master_seed: int,
    bootstrap_samples: int,
    conf: float,
    clustered_values: list[list[float]] | None = None,
) -> AggregateMetric:
    """Compute aggregate statistics and CIs for one metric.

    When ``clustered_values`` is provided (hierarchical mode) the bootstrap
    mean/median CIs use the two-stage cluster bootstrap
    (``hierarchical_bootstrap_ci``) and, for binary rate metrics, the mean CI
    uses the cluster-robust interval (``cluster_robust_interval``) instead of
    the flat Wilson interval. When cluster data is unavailable (``None``, the
    flat-mode default) the implementation uses the original i.i.d. bootstrap,
    so a missing cluster field never produces a crash.

    Returns:
        AggregateMetric instance.
    """
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
    group_seed = _stable_group_seed(master_seed, arch, dens, metric_name)
    rng = random.Random(group_seed)
    is_rate = metric_name in {"collision_rate", "success_rate"}
    finite_clusters = _finite_clusters(clustered_values)
    if finite_clusters:
        mean_ci, median_ci = hierarchical_bootstrap_ci(
            finite_clusters,
            bootstrap_samples,
            conf,
            rng,
        )
        if is_rate:
            mean_ci = cluster_robust_interval(finite_clusters, conf)
    else:
        mean_ci, median_ci = _bootstrap_ci(sorted_vals, bootstrap_samples, conf, rng)
        # Replace mean_ci for rate metrics with Wilson interval (T031) for better coverage when near bounds.
        if is_rate and sorted_vals:
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
    """Return a normal z value for a confidence level."""
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
