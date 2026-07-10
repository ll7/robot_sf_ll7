"""Canonical SNQI computation helpers.

The SNQI (Social Navigation Quality Index) combines task performance and
social safety/comfort metrics into a single scalar score:

    SNQI = w_success * success
           - w_time * time_norm
           - w_collisions * coll_norm
           - w_near * near_norm
           - w_comfort * comfort_exposure
           - w_force_exceed * force_exceed_norm
           - w_jerk * jerk_norm

Normalization for metrics using baseline stats (median / p95):

    norm = (value - med) / (p95 - med)   (clamped to [0,1])

This module is intentionally lightweight so it can be imported by scripts
and tests without pulling heavy dependencies.

Clamping Implications:
- Values below baseline median never produce a negative normalized penalty (floor at 0) – rewarding improvements implicitly happens only via lower adverse metrics and preserved success term.
- Values above p95 saturate at 1; extreme outliers do not further worsen the normalized term, trading tail sensitivity for robustness.
- Future modes may allow soft saturation (sigmoid) or asymmetric scaling; current design favors reproducibility & bounded optimization landscapes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from subprocess import run

from robot_sf.benchmark.snqi.types import SNQIWeights

WEIGHT_NAMES = [
    "w_success",
    "w_time",
    "w_collisions",
    "w_near",
    "w_comfort",
    "w_force_exceed",
    "w_jerk",
]

SNQI_SCORE_VERSION_V0 = "SNQI-v0"
SNQI_SCORE_VERSION_V1 = "SNQI-v1"
SNQI_V1_PENALTY_METRICS = (
    "time_to_goal_norm",
    "collisions",
    "near_misses",
    "comfort_exposure",
    "force_exceed_events",
    "jerk_mean",
)

MetricName = str
BaselineStats = Mapping[MetricName, Mapping[str, float]]
Weights = Mapping[str, float]
Metrics = Mapping[str, float | int | bool]


def _weighted_term(weight: float, value: float) -> float:
    """Weighted contribution of one SNQI term with an explicit zero-weight contract.

    A zero weight means the term is excluded from the score, so it must contribute
    exactly ``0.0`` -- even when ``value`` is ``NaN``. IEEE 754 makes ``0.0 * NaN``
    equal ``NaN``, which would otherwise silently collapse the whole score (issue
    #5132: a legitimately absent ``comfort_exposure`` with ``w_comfort = 0.0``).
    Short-circuiting on a zero weight keeps the score finite without masking real
    signals: a non-zero weight still multiplies through, so a genuine ``NaN`` under
    a non-zero weight still propagates as before.

    Args:
        weight: Weight coefficient for the term.
        value: Pre-normalized (or raw) metric value for the term.

    Returns:
        ``weight * value``, or ``0.0`` when ``weight == 0.0``.
    """
    if weight == 0.0:
        return 0.0
    return weight * value


def normalize_metric(name: str, value: float | int | bool, baseline_stats: BaselineStats) -> float:
    """Normalize a raw metric using median/p95 baseline statistics.

    Args:
        name: Metric name (e.g., "collisions", "near_misses", "jerk_mean").
        value: Raw metric value from an episode record. Booleans are coerced to float.
        baseline_stats: Mapping of metric -> {"med": float, "p95": float} used for scaling.

    Returns:
        A float in [0.0, 1.0] representing the clamped normalized value where 0.0 corresponds
        to the baseline median and 1.0 corresponds to (and above) the baseline p95.

    Notes:
        - If the metric is missing in the baseline statistics, returns 0.0 (neutral contribution).
        - Division-by-zero is guarded by using a denominator of 1.0 when p95 == med.
        - Result is clamped to [0,1] to prevent extreme outliers from dominating.
    """
    if name not in baseline_stats:
        return 0.0
    med = float(baseline_stats[name].get("med", 0.0))
    p95 = float(baseline_stats[name].get("p95", med))
    denom = p95 - med
    if abs(denom) < 1e-9:
        denom = 1.0
    norm = (float(value) - med) / denom
    if norm < 0.0:
        return 0.0
    if norm > 1.0:
        return 1.0
    return norm


def compute_snqi_v0(metrics: Metrics, weights: Weights, baseline_stats: BaselineStats) -> float:
    """Compute the SNQI score for a single episode.

    Args:
        metrics: Episode metrics mapping.
        weights: Weight coefficients for each component.
        baseline_stats: Baseline normalization statistics (median/p95) for
            collision-like metrics.

    Returns:
        SNQI score (higher is better).

    Notes:
        - A term whose weight is ``0.0`` is excluded from the score and never
          propagates ``NaN`` (e.g. a legitimately absent ``comfort_exposure``
          when ``w_comfort = 0.0``). A non-zero weight still multiplies through,
          so a genuine ``NaN`` under a non-zero weight propagates as before
          (issue #5132).
    """
    success_raw = metrics.get("success", 0.0)
    success = 1.0 if isinstance(success_raw, bool) and success_raw else float(success_raw)

    time_norm = float(metrics.get("time_to_goal_norm", 1.0))
    coll_norm = normalize_metric("collisions", metrics.get("collisions", 0.0), baseline_stats)
    near_norm = normalize_metric("near_misses", metrics.get("near_misses", 0.0), baseline_stats)
    comfort_exposure = float(metrics.get("comfort_exposure", 0.0))
    force_exceed_norm = normalize_metric(
        "force_exceed_events",
        metrics.get("force_exceed_events", 0.0),
        baseline_stats,
    )
    jerk_norm = normalize_metric("jerk_mean", metrics.get("jerk_mean", 0.0), baseline_stats)

    score = (
        _weighted_term(weights.get("w_success", 1.0), success)
        - _weighted_term(weights.get("w_time", 1.0), time_norm)
        - _weighted_term(weights.get("w_collisions", 1.0), coll_norm)
        - _weighted_term(weights.get("w_near", 1.0), near_norm)
        - _weighted_term(weights.get("w_comfort", 1.0), comfort_exposure)
        - _weighted_term(weights.get("w_force_exceed", 1.0), force_exceed_norm)
        - _weighted_term(weights.get("w_jerk", 1.0), jerk_norm)
    )
    return float(score)


def normalize_metric_required(
    name: str,
    value: float | int | bool,
    baseline_stats: BaselineStats,
) -> float:
    """Normalize an ``SNQI-v1`` metric and fail closed on missing baseline coverage.

    Returns:
        Baseline-normalized metric value clamped to ``[0, 1]``.
    """
    entry = baseline_stats.get(name)
    if not isinstance(entry, Mapping) or "med" not in entry or "p95" not in entry:
        raise ValueError(f"SNQI-v1 baseline_stats missing med/p95 for {name!r}")
    if float(entry["p95"]) <= float(entry["med"]):
        raise ValueError(f"SNQI-v1 baseline_stats has non-positive spread for {name!r}")
    return normalize_metric(name, value, baseline_stats)


def compute_snqi_v1(metrics: Metrics, weights: Weights, baseline_stats: BaselineStats) -> float:
    """Compute opt-in ``SNQI-v1`` with bounded baseline-relative penalty terms.

    Returns:
        Versioned SNQI score where every penalty term is baseline-normalized.

    Notes:
        - Shares the zero-weight contract of :func:`compute_snqi_v0`: a term with
          weight ``0.0`` never propagates ``NaN`` (issue #5132).
    """
    success_raw = metrics.get("success", 0.0)
    success = 1.0 if isinstance(success_raw, bool) and success_raw else float(success_raw)

    time_norm = normalize_metric_required(
        "time_to_goal_norm",
        metrics.get("time_to_goal_norm", 1.0),
        baseline_stats,
    )
    coll_norm = normalize_metric_required(
        "collisions",
        metrics.get("collisions", 0.0),
        baseline_stats,
    )
    near_norm = normalize_metric_required(
        "near_misses",
        metrics.get("near_misses", 0.0),
        baseline_stats,
    )
    comfort_norm = normalize_metric_required(
        "comfort_exposure",
        metrics.get("comfort_exposure", 0.0),
        baseline_stats,
    )
    force_exceed_norm = normalize_metric_required(
        "force_exceed_events",
        metrics.get("force_exceed_events", 0.0),
        baseline_stats,
    )
    jerk_norm = normalize_metric_required(
        "jerk_mean",
        metrics.get("jerk_mean", 0.0),
        baseline_stats,
    )

    score = (
        _weighted_term(weights.get("w_success", 1.0), success)
        - _weighted_term(weights.get("w_time", 1.0), time_norm)
        - _weighted_term(weights.get("w_collisions", 1.0), coll_norm)
        - _weighted_term(weights.get("w_near", 1.0), near_norm)
        - _weighted_term(weights.get("w_comfort", 1.0), comfort_norm)
        - _weighted_term(weights.get("w_force_exceed", 1.0), force_exceed_norm)
        - _weighted_term(weights.get("w_jerk", 1.0), jerk_norm)
    )

    return float(score)


def compute_snqi(
    metrics: Metrics,
    weights: Weights,
    baseline_stats: BaselineStats,
    *,
    score_version: str = SNQI_SCORE_VERSION_V0,
) -> float:
    """Compute a versioned SNQI score for a single episode.

    Returns:
        SNQI score for the requested score version.
    """
    if score_version == SNQI_SCORE_VERSION_V0:
        return compute_snqi_v0(metrics, weights, baseline_stats)
    if score_version == SNQI_SCORE_VERSION_V1:
        return compute_snqi_v1(metrics, weights, baseline_stats)
    raise ValueError(f"unknown SNQI score version: {score_version}")


def recompute_snqi_weights(
    baseline_stats: BaselineStats,
    method: str = "canonical",
    seed: int | None = None,
) -> SNQIWeights:
    """Recompute SNQI weights based on baseline statistics.

    Args:
        baseline_stats: Baseline statistics for metrics normalization.
        method: Weight computation method ("canonical", "balanced", "optimized").
        seed: Random seed for reproducible weight generation.

    Returns:
        SNQIWeights instance with computed weights and metadata.
    """
    # Default canonical weights
    canonical_weights = {
        "w_success": 1.0,
        "w_time": 0.8,
        "w_collisions": 2.0,
        "w_near": 1.0,
        "w_comfort": 0.5,
        "w_force_exceed": 1.5,
        "w_jerk": 0.3,
    }

    # Compute weights based on method
    if method == "canonical":
        weights = canonical_weights.copy()
    elif method == "balanced":
        # Balanced approach - all weights equal
        weights = dict.fromkeys(WEIGHT_NAMES, 1.0)
    elif method == "optimized":
        # Could implement optimization here - for now use canonical
        weights = canonical_weights.copy()
        # Add some variation based on baseline stats if needed
        if "collisions" in baseline_stats:
            collision_p95 = baseline_stats["collisions"].get("p95", 1.0)
            weights["w_collisions"] = max(1.0, collision_p95 * 0.5)
    else:
        msg = f"Unknown weight computation method: {method}"
        raise ValueError(msg)

    # Get git SHA for provenance. NOTE: ``capture_output=True`` is equivalent to
    # ``stdout=PIPE, stderr=PIPE``, so also passing ``stderr=DEVNULL`` is an
    # invalid combination that always raises ``ValueError`` (swallowed by the
    # handler below) -- which silently degraded ``git_sha`` to ``"unknown"`` for
    # every recorded weight config (issue #4895). Do not reintroduce it.
    try:
        result = run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        git_sha = result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        git_sha = "unknown"

    # Compute hash of baseline stats for reproducibility
    baseline_hash = hashlib.sha256(json.dumps(baseline_stats, sort_keys=True).encode()).hexdigest()[
        :16
    ]

    return SNQIWeights(
        weights_version="1.0",
        created_at=datetime.now().isoformat(),
        git_sha=git_sha,
        baseline_stats_path="computed",
        baseline_stats_hash=baseline_hash,
        normalization_strategy="median_p95_clamp",
        bootstrap_params={"method": method, "seed": seed} if seed else {"method": method},
        components=list(WEIGHT_NAMES),
        weights=weights,
    )


def compute_snqi_ablation(
    episodes_data: list[dict],
    weights: dict[str, float],
    baseline_stats: BaselineStats,
    components: list[str] | None = None,
) -> dict[str, float]:
    """Compute SNQI ablation analysis by removing each component.

    Args:
        episodes_data: List of episode records with metrics.
        weights: Weight coefficients for each component.
        baseline_stats: Baseline statistics for normalization.
        components: Components to ablate (default: all weight components).

    Returns:
        Dictionary mapping component names to SNQI scores without that component.
    """
    if components is None:
        components = list(WEIGHT_NAMES)

    results = {}

    # Compute baseline SNQI with all components
    baseline_scores = []
    for episode in episodes_data:
        metrics = episode.get("metrics", {})
        score = compute_snqi(metrics, weights, baseline_stats)
        baseline_scores.append(score)

    baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0

    # Compute SNQI for each ablation (removing one component at a time)
    for component in components:
        ablated_weights = weights.copy()
        ablated_weights[component] = 0.0  # Remove component by setting weight to 0

        ablated_scores = []
        for episode in episodes_data:
            metrics = episode.get("metrics", {})
            score = compute_snqi(metrics, ablated_weights, baseline_stats)
            ablated_scores.append(score)

        ablated_mean = sum(ablated_scores) / len(ablated_scores) if ablated_scores else 0.0

        # Store the impact (difference from baseline)
        results[component] = baseline_mean - ablated_mean

    return results


__all__ = [
    "SNQI_SCORE_VERSION_V0",
    "SNQI_SCORE_VERSION_V1",
    "SNQI_V1_PENALTY_METRICS",
    "WEIGHT_NAMES",
    "compute_snqi",
    "compute_snqi_ablation",
    "compute_snqi_v0",
    "compute_snqi_v1",
    "normalize_metric",
    "normalize_metric_required",
    "recompute_snqi_weights",
]
