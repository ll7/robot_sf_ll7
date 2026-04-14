"""SNQI campaign diagnostics, calibration, and contract evaluation helpers."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.snqi.compute import WEIGHT_NAMES, compute_snqi, normalize_metric

if TYPE_CHECKING:
    from pathlib import Path

_STARTUP_METRIC_NAMES = (
    "time_to_goal_norm",
    "collisions",
    "near_misses",
    "force_exceed_events",
    "jerk_mean",
)

_WEIGHT_COMPONENT_SPECS: tuple[dict[str, str], ...] = (
    {
        "weight_name": "w_success",
        "metric_name": "success",
        "contribution_name": "success_reward",
        "direction": "positive",
    },
    {
        "weight_name": "w_time",
        "metric_name": "time_to_goal_norm",
        "contribution_name": "time_penalty",
        "direction": "negative",
    },
    {
        "weight_name": "w_collisions",
        "metric_name": "collisions",
        "contribution_name": "collisions_penalty",
        "direction": "negative",
    },
    {
        "weight_name": "w_near",
        "metric_name": "near_misses",
        "contribution_name": "near_penalty",
        "direction": "negative",
    },
    {
        "weight_name": "w_comfort",
        "metric_name": "comfort_exposure",
        "contribution_name": "comfort_penalty",
        "direction": "negative",
    },
    {
        "weight_name": "w_force_exceed",
        "metric_name": "force_exceed_events",
        "contribution_name": "force_exceed_penalty",
        "direction": "negative",
    },
    {
        "weight_name": "w_jerk",
        "metric_name": "jerk_mean",
        "contribution_name": "jerk_penalty",
        "direction": "negative",
    },
)


@dataclass(frozen=True)
class SnqiContractThresholds:
    """Threshold configuration for SNQI campaign contract evaluation."""

    rank_alignment_warn: float = 0.5
    rank_alignment_fail: float = 0.3
    outcome_separation_warn: float = 0.05
    outcome_separation_fail: float = 0.0
    max_component_dominance_warn: float = 0.24
    max_component_dominance_fail: float = 0.27


@dataclass(frozen=True)
class SnqiContractEvaluation:
    """Computed contract metrics and resolved status."""

    status: str
    rank_alignment_spearman: float
    outcome_separation: float
    objective_score: float
    dominant_component: str
    dominant_component_mean_abs: float


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(v) for v in values)
    q_clamped = max(0.0, min(1.0, float(q)))
    pos = (len(sorted_values) - 1) * q_clamped
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def default_weight_mapping() -> dict[str, float]:
    """Return canonical default SNQI weights (all components weight=1)."""
    return dict.fromkeys(WEIGHT_NAMES, 1.0)


def resolve_weight_mapping(raw: Mapping[str, Any] | None) -> dict[str, float]:
    """Resolve SNQI weight mapping from direct mapping or nested ``weights`` payload.

    Returns:
        Complete weight mapping covering all SNQI components.
    """
    if raw is None:
        return default_weight_mapping()
    if not isinstance(raw, Mapping):
        raise ValueError("SNQI weights payload must be a mapping or contain a 'weights' mapping.")
    if "weights" in raw:
        weights_payload = raw.get("weights")
        if not isinstance(weights_payload, Mapping):
            raise ValueError("SNQI weights payload key 'weights' must be a mapping when provided.")
        source: Mapping[str, Any] = weights_payload
    else:
        source = raw
    resolved = default_weight_mapping()
    for name in WEIGHT_NAMES:
        value = source.get(name)
        if _is_finite(value):
            resolved[name] = float(value)
    return resolved


def sanitize_baseline_stats(
    raw: Mapping[str, Any] | None,
    *,
    metric_names: Sequence[str] = _STARTUP_METRIC_NAMES,
    epsilon: float = 1e-6,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Return validated baseline stats and warnings for degenerate entries.

    Degenerate entries (``p95 <= med``) are widened with a deterministic fallback.
    """
    warnings: list[str] = []
    source = raw or {}
    sanitized: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        entry = source.get(metric) if isinstance(source, Mapping) else None
        med = 0.0
        p95 = 1.0
        if isinstance(entry, Mapping):
            if _is_finite(entry.get("med")):
                med = float(entry.get("med"))
            if _is_finite(entry.get("p95")):
                p95 = float(entry.get("p95"))
            else:
                p95 = med
        if p95 <= med:
            # Keep normalization monotonic by forcing a minimum positive width.
            width = max(epsilon, abs(med) * 0.05, 1.0 if med == 0.0 else 0.0)
            p95 = med + width
            warnings.append(
                f"Adjusted degenerate baseline for '{metric}' (p95 <= med) using fallback width {width:.6g}"
            )
        sanitized[metric] = {"med": med, "p95": p95}
    return sanitized, warnings


def compute_baseline_stats_from_episodes(
    episodes: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str] = _STARTUP_METRIC_NAMES,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Compute robust median/p95 baseline stats from episode metrics.

    Returns:
        Tuple of sanitized baseline mapping and adjustment warnings.
    """
    values_by_metric: dict[str, list[float]] = {name: [] for name in metric_names}
    for episode in episodes:
        metrics = episode.get("metrics") if isinstance(episode, Mapping) else None
        if not isinstance(metrics, Mapping):
            continue
        for name in metric_names:
            value = metrics.get(name)
            if _is_finite(value):
                values_by_metric[name].append(float(value))
    baseline: dict[str, dict[str, float]] = {}
    for metric, values in values_by_metric.items():
        if values:
            med = float(statistics.median(values))
            p95 = float(_quantile(values, 0.95))
        else:
            med = 0.0
            p95 = 1.0
        baseline[metric] = {"med": med, "p95": p95}
    sanitized, warnings = sanitize_baseline_stats(baseline, metric_names=metric_names)
    return sanitized, warnings


def _rank(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=False))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return float(cov / math.sqrt(vx * vy))


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman rank correlation without SciPy dependency.

    Returns:
        Correlation coefficient in ``[-1, 1]``.
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    return _pearson(_rank(list(x)), _rank(list(y)))


def _target_quality(row: Mapping[str, Any]) -> float:
    success = float(row.get("success_mean", 0.0) or 0.0)
    collisions = float(row.get("collisions_mean", 0.0) or 0.0)
    near_misses = float(row.get("near_misses_mean", 0.0) or 0.0)
    comfort = float(row.get("comfort_exposure_mean", 0.0) or 0.0)
    # Paper-facing aggregate quality proxy emphasizing completion+safety.
    return success - 0.7 * collisions - 0.25 * near_misses - 0.2 * comfort


def _planner_key(row: Mapping[str, Any]) -> str:
    planner = str(row.get("planner_key", "")).strip()
    kinematics = str(row.get("kinematics", "")).strip()
    if planner and kinematics:
        return f"{planner}::{kinematics}"
    return planner or "unknown"


def _episode_planner_key(episode: Mapping[str, Any]) -> str:
    planner = str(episode.get("planner_key", "")).strip()
    kinematics = str(episode.get("kinematics", "")).strip()
    if planner and kinematics:
        return f"{planner}::{kinematics}"
    return planner or "unknown"


def _episode_score(
    metrics: Mapping[str, Any],
    *,
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
) -> float:
    return compute_snqi(metrics, weights, baseline)


def evaluate_snqi_contract(  # noqa: C901, PLR0912
    planner_rows: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
    thresholds: SnqiContractThresholds,
) -> SnqiContractEvaluation:
    """Evaluate rank-alignment and outcome-separation contract metrics.

    Returns:
        Contract evaluation with status and objective diagnostics.
    """
    planner_target: dict[str, float] = {}
    for row in planner_rows:
        if not _is_finite(row.get("success_mean")) or not _is_finite(row.get("collisions_mean")):
            continue
        target = _target_quality(row)
        if not _is_finite(target):
            continue
        planner_target[_planner_key(row)] = target

    score_sum: dict[str, float] = {}
    score_count: dict[str, int] = {}
    positive_scores: list[float] = []
    negative_scores: list[float] = []

    for episode in episodes:
        metrics = episode.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        score = _episode_score(metrics, weights=weights, baseline=baseline)
        if not _is_finite(score):
            continue
        success_raw = metrics.get("success")
        collisions_raw = metrics.get("collisions")
        if not _is_finite(success_raw) or not _is_finite(collisions_raw):
            continue
        key = _episode_planner_key(episode)
        score_sum[key] = score_sum.get(key, 0.0) + score
        score_count[key] = score_count.get(key, 0) + 1

        success = float(success_raw)
        collisions = float(collisions_raw)
        is_positive = success >= 1.0 and collisions <= 0.0
        if is_positive:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    aligned_targets: list[float] = []
    aligned_scores: list[float] = []
    for key, target in planner_target.items():
        if key not in score_count or score_count[key] <= 0:
            continue
        aligned_targets.append(target)
        aligned_scores.append(score_sum[key] / score_count[key])

    rank_alignment = spearman_correlation(aligned_targets, aligned_scores)
    if positive_scores and negative_scores:
        outcome_separation = (sum(positive_scores) / len(positive_scores)) - (
            sum(negative_scores) / len(negative_scores)
        )
    else:
        outcome_separation = 0.0

    component_dominance = compute_component_dominance(
        episodes,
        weights=weights,
        baseline=baseline,
    )
    dominant_component = "none"
    dominant_component_mean_abs = 0.0
    if component_dominance:
        candidate_component, candidate_mean_abs = max(
            component_dominance.items(),
            key=lambda item: float(item[1]),
        )
        if float(candidate_mean_abs) > 0.0:
            dominant_component = str(candidate_component)
            dominant_component_mean_abs = float(candidate_mean_abs)

    objective = rank_alignment + 0.25 * outcome_separation

    if (
        rank_alignment < thresholds.rank_alignment_fail
        or outcome_separation < thresholds.outcome_separation_fail
        or dominant_component_mean_abs > thresholds.max_component_dominance_fail
    ):
        status = "fail"
    elif (
        rank_alignment < thresholds.rank_alignment_warn
        or outcome_separation < thresholds.outcome_separation_warn
        or dominant_component_mean_abs > thresholds.max_component_dominance_warn
    ):
        status = "warn"
    else:
        status = "pass"

    return SnqiContractEvaluation(
        status=status,
        rank_alignment_spearman=float(rank_alignment),
        outcome_separation=float(outcome_separation),
        objective_score=float(objective),
        dominant_component=str(dominant_component),
        dominant_component_mean_abs=float(dominant_component_mean_abs),
    )


def compute_component_dominance(
    episodes: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Compute mean absolute component contribution magnitudes across episodes.

    Returns:
        Mapping of component contribution names to mean absolute magnitudes.
    """
    accum = {
        "success_reward": 0.0,
        "time_penalty": 0.0,
        "collisions_penalty": 0.0,
        "near_penalty": 0.0,
        "comfort_penalty": 0.0,
        "force_exceed_penalty": 0.0,
        "jerk_penalty": 0.0,
    }
    count = 0
    for episode in episodes:
        metrics = episode.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        count += 1
        success_raw = metrics.get("success")
        time_to_goal_raw = metrics.get("time_to_goal_norm")
        comfort_raw = metrics.get("comfort_exposure")
        collisions_raw = metrics.get("collisions")
        near_misses_raw = metrics.get("near_misses")
        force_exceed_raw = metrics.get("force_exceed_events")
        jerk_raw = metrics.get("jerk_mean")

        success = float(success_raw) if _is_finite(success_raw) else 0.0
        time_to_goal = float(time_to_goal_raw) if _is_finite(time_to_goal_raw) else 0.0
        comfort = float(comfort_raw) if _is_finite(comfort_raw) else 0.0
        collisions = collisions_raw if _is_finite(collisions_raw) else 0.0
        near_misses = near_misses_raw if _is_finite(near_misses_raw) else 0.0
        force_exceed = force_exceed_raw if _is_finite(force_exceed_raw) else 0.0
        jerk = jerk_raw if _is_finite(jerk_raw) else 0.0

        accum["success_reward"] += abs(weights.get("w_success", 1.0) * success)
        accum["time_penalty"] += abs(weights.get("w_time", 1.0) * time_to_goal)
        accum["collisions_penalty"] += abs(
            weights.get("w_collisions", 1.0) * normalize_metric("collisions", collisions, baseline)
        )
        accum["near_penalty"] += abs(
            weights.get("w_near", 1.0) * normalize_metric("near_misses", near_misses, baseline)
        )
        accum["comfort_penalty"] += abs(weights.get("w_comfort", 1.0) * comfort)
        accum["force_exceed_penalty"] += abs(
            weights.get("w_force_exceed", 1.0)
            * normalize_metric("force_exceed_events", force_exceed, baseline)
        )
        accum["jerk_penalty"] += abs(
            weights.get("w_jerk", 1.0) * normalize_metric("jerk_mean", jerk, baseline)
        )
    if count <= 0:
        return dict.fromkeys(accum, 0.0)
    return {key: value / count for key, value in accum.items()}


def compute_component_correlations(
    episodes: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, Any]]:
    """Compute per-metric Spearman correlations against episode-level SNQI scores.

    Returns:
        Mapping of metric name to correlation payload, including sign alignment and
        degeneracy flags for non-varying metrics.
    """
    scores: list[float] = []
    metric_values: dict[str, list[float]] = {
        str(spec["metric_name"]): [] for spec in _WEIGHT_COMPONENT_SPECS
    }
    for episode in episodes:
        metrics = episode.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        score = _episode_score(metrics, weights=weights, baseline=baseline)
        if not _is_finite(score):
            continue
        row: dict[str, float] = {}
        valid_row = True
        for spec in _WEIGHT_COMPONENT_SPECS:
            metric_name = str(spec["metric_name"])
            value = metrics.get(metric_name)
            if not _is_finite(value):
                valid_row = False
                break
            row[metric_name] = float(value)
        if not valid_row:
            continue
        scores.append(float(score))
        for metric_name, value in row.items():
            metric_values[metric_name].append(value)

    correlations: dict[str, dict[str, Any]] = {}
    for spec in _WEIGHT_COMPONENT_SPECS:
        metric_name = str(spec["metric_name"])
        direction = str(spec["direction"])
        values = metric_values.get(metric_name, [])
        value_range = (max(values) - min(values)) if values else 0.0
        variable = len(values) >= 2 and value_range > 0.0
        score_range = (max(scores) - min(scores)) if scores else 0.0
        score_variable = len(scores) >= 2 and score_range > 0.0
        correlation: float | None
        aligned: bool | None
        if variable and score_variable:
            correlation = float(spearman_correlation(scores, values))
            aligned = correlation > 0.0 if direction == "positive" else correlation < 0.0
        else:
            correlation = None
            aligned = None
        correlations[metric_name] = {
            "weight_name": str(spec["weight_name"]),
            "direction": direction,
            "spearman": correlation,
            "variable": bool(variable),
            "aligned_with_expected_direction": aligned,
        }
    return correlations


def compute_planner_snqi_ordering(
    episodes: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Compute planner-level SNQI ordering from episode records.

    Returns:
        Sorted planner rows with ``rank``, ``mean_snqi``, and ``episode_count`` fields.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        metrics = episode.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        score = _episode_score(metrics, weights=weights, baseline=baseline)
        if not _is_finite(score):
            continue
        planner_key = str(episode.get("planner_key", "unknown"))
        kinematics = str(episode.get("kinematics", "unknown"))
        key = _episode_planner_key(episode)
        bucket = grouped.setdefault(
            key,
            {
                "planner_key": planner_key,
                "kinematics": kinematics,
                "episode_count": 0,
                "score_sum": 0.0,
            },
        )
        bucket["episode_count"] = int(bucket["episode_count"]) + 1
        bucket["score_sum"] = float(bucket["score_sum"]) + float(score)

    rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        episode_count = int(bucket["episode_count"])
        mean_snqi = float(bucket["score_sum"]) / episode_count if episode_count > 0 else 0.0
        rows.append(
            {
                "planner_key": str(bucket["planner_key"]),
                "kinematics": str(bucket["kinematics"]),
                "episode_count": episode_count,
                "mean_snqi": mean_snqi,
            }
        )
    rows.sort(
        key=lambda row: (-float(row["mean_snqi"]), str(row["planner_key"]), str(row["kinematics"]))
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def compute_weight_sensitivity(
    episodes: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Measure how ablation of each weight perturbs SNQI conclusions.

    Returns:
        Per-weight ablation rows sorted by descending sensitivity impact on planner and
        episode ranking stability.
    """
    base_scores: list[float] = []
    scored_episodes: list[Mapping[str, Any]] = []
    for episode in episodes:
        metrics = episode.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        score = _episode_score(metrics, weights=weights, baseline=baseline)
        if not _is_finite(score):
            continue
        scored_episodes.append(episode)
        base_scores.append(float(score))

    base_ordering = compute_planner_snqi_ordering(
        scored_episodes, weights=weights, baseline=baseline
    )
    base_planner_order = [f"{row['planner_key']}::{row['kinematics']}" for row in base_ordering]
    base_planner_scores = [float(row["mean_snqi"]) for row in base_ordering]
    rows: list[dict[str, Any]] = []
    dominance = compute_component_dominance(scored_episodes, weights=weights, baseline=baseline)
    total_weight = sum(
        float(weights.get(str(spec["weight_name"]), 0.0)) for spec in _WEIGHT_COMPONENT_SPECS
    )

    for spec in _WEIGHT_COMPONENT_SPECS:
        weight_name = str(spec["weight_name"])
        contribution_name = str(spec["contribution_name"])
        ablated_weights = dict(weights)
        ablated_weights[weight_name] = 0.0
        ablated_scores: list[float] = []
        for episode in scored_episodes:
            metrics = episode.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            ablated_scores.append(
                _episode_score(metrics, weights=ablated_weights, baseline=baseline)
            )
        episode_rank_corr = (
            float(spearman_correlation(base_scores, ablated_scores))
            if len(base_scores) >= 2 and len(ablated_scores) == len(base_scores)
            else 1.0
        )
        planner_ordering = compute_planner_snqi_ordering(
            scored_episodes,
            weights=ablated_weights,
            baseline=baseline,
        )
        planner_order = [f"{row['planner_key']}::{row['kinematics']}" for row in planner_ordering]
        planner_scores = [float(row["mean_snqi"]) for row in planner_ordering]
        planner_rank_corr = (
            float(spearman_correlation(base_planner_scores, planner_scores))
            if len(base_planner_scores) >= 2 and len(planner_scores) == len(base_planner_scores)
            else 1.0
        )
        rows.append(
            {
                "weight_name": weight_name,
                "metric_name": str(spec["metric_name"]),
                "direction": str(spec["direction"]),
                "configured_weight": float(weights.get(weight_name, 0.0)),
                "configured_weight_share": (
                    float(weights.get(weight_name, 0.0)) / total_weight
                    if total_weight > 0.0
                    else 0.0
                ),
                "mean_abs_contribution": float(dominance.get(contribution_name, 0.0)),
                "mean_abs_score_delta_if_ablated": (
                    sum(
                        abs(base - ablated)
                        for base, ablated in zip(base_scores, ablated_scores, strict=False)
                    )
                    / len(base_scores)
                    if base_scores
                    else 0.0
                ),
                "episode_rank_correlation_if_ablated": episode_rank_corr,
                "planner_rank_correlation_if_ablated": planner_rank_corr,
                "planner_order_changed_if_ablated": planner_order != base_planner_order,
            }
        )

    rows.sort(
        key=lambda row: (
            float(row["planner_rank_correlation_if_ablated"]),
            float(row["episode_rank_correlation_if_ablated"]),
            -float(row["mean_abs_score_delta_if_ablated"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["sensitivity_rank"] = index
    return rows


def build_positioning_recommendation(
    component_correlations: Mapping[str, Mapping[str, Any]],
    planner_ordering: Sequence[Mapping[str, Any]],
    weight_sensitivity: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize whether SNQI is defensible as an operational aggregate.

    Returns:
        Recommendation payload with claim scope, caveats, and simple evidence counts.
    """
    aligned_metrics = 0
    variable_metrics = 0
    degenerate_metrics: list[str] = []
    for metric_name, payload in component_correlations.items():
        if bool(payload.get("variable")):
            variable_metrics += 1
            if payload.get("aligned_with_expected_direction") is True:
                aligned_metrics += 1
        else:
            degenerate_metrics.append(metric_name)
    planner_names = [str(row.get("planner_key", "unknown")) for row in planner_ordering]
    ordering_is_informative = len(planner_names) >= 2 and len(set(planner_names)) >= 2
    order_changes = sum(
        1 for row in weight_sensitivity if bool(row.get("planner_order_changed_if_ablated"))
    )
    recommendation = (
        "strengthen_as_operational_multi_objective_aggregation"
        if aligned_metrics >= max(3, variable_metrics - 1) and ordering_is_informative
        else "downgrade_to_appendix_or_implementation_aid"
    )
    caveats = []
    if degenerate_metrics:
        caveats.append(
            "Degenerate metrics on this slice cannot be used as independent validation signals: "
            + ", ".join(sorted(degenerate_metrics))
        )
    if order_changes <= 0:
        caveats.append(
            "Planner ordering is stable under one-at-a-time weight ablation on this slice."
        )
    return {
        "recommendation": recommendation,
        "claim_scope": "benchmark aggregate, not a universal ground-truth utility",
        "aligned_metric_count": aligned_metrics,
        "variable_metric_count": variable_metrics,
        "degenerate_metrics": sorted(degenerate_metrics),
        "planner_ordering_informative": ordering_is_informative,
        "ablation_order_change_count": order_changes,
        "caveats": caveats,
    }


def calibrate_weights(
    planner_rows: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    *,
    baseline: Mapping[str, Mapping[str, float]],
    seed: int = 123,
    trials: int = 3000,
    min_weight: float = 0.05,
    max_weight: float = 0.35,
) -> dict[str, Any]:
    """Calibrate SNQI weights via deterministic random search over simplex.

    Returns:
        Recommendation payload including calibrated weights and optimization metrics.
    """
    rng = random.Random(seed)
    thresholds = SnqiContractThresholds(
        rank_alignment_warn=-1.0,
        rank_alignment_fail=-1.1,
        outcome_separation_warn=-1e9,
        outcome_separation_fail=-1e9,
    )
    best_eval: SnqiContractEvaluation | None = None
    best_weights: dict[str, float] | None = None
    best_metrics: dict[str, float] | None = None

    for _ in range(max(1, int(trials))):
        draws = [rng.gammavariate(1.0, 1.0) for _ in WEIGHT_NAMES]
        total = sum(draws)
        if total <= 0.0:
            continue
        base = {name: draw / total for name, draw in zip(WEIGHT_NAMES, draws, strict=False)}
        floor_budget = min_weight * len(WEIGHT_NAMES)
        if floor_budget >= 1.0:
            floor_budget = 0.0
        candidate = {name: min_weight + base[name] * (1.0 - floor_budget) for name in WEIGHT_NAMES}
        if any(value > max_weight for value in candidate.values()):
            continue
        evaluation = evaluate_snqi_contract(
            planner_rows,
            episodes,
            weights=candidate,
            baseline=baseline,
            thresholds=thresholds,
        )
        if best_eval is None or evaluation.objective_score > best_eval.objective_score:
            best_eval = evaluation
            best_weights = candidate
            best_metrics = {
                "rank_alignment_spearman": evaluation.rank_alignment_spearman,
                "outcome_separation": evaluation.outcome_separation,
                "objective_score": evaluation.objective_score,
            }

    if best_weights is None or best_metrics is None:
        best_weights = default_weight_mapping()
        best_metrics = {
            "rank_alignment_spearman": 0.0,
            "outcome_separation": 0.0,
            "objective_score": 0.0,
        }

    return {
        "weights": best_weights,
        "metrics": best_metrics,
        "seed": int(seed),
        "trials": int(trials),
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
    }


def collect_episodes_from_campaign_runs(
    run_entries: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Load episode records referenced by campaign run entries for diagnostics.

    Returns:
        Flattened list of episode records with planner/kinematics tags.
    """
    episodes: list[dict[str, Any]] = []
    for entry in run_entries:
        episodes_path = entry.get("episodes_path")
        if not isinstance(episodes_path, str):
            continue
        planner = entry.get("planner") if isinstance(entry.get("planner"), Mapping) else {}
        planner_key = str(planner.get("key", "unknown"))
        kinematics = str(planner.get("kinematics", "differential_drive"))
        path = (repo_root / episodes_path).resolve()
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                payload["planner_key"] = planner_key
                payload["kinematics"] = kinematics
                episodes.append(payload)
    return episodes


__all__ = [
    "SnqiContractEvaluation",
    "SnqiContractThresholds",
    "build_positioning_recommendation",
    "calibrate_weights",
    "collect_episodes_from_campaign_runs",
    "compute_baseline_stats_from_episodes",
    "compute_component_correlations",
    "compute_component_dominance",
    "compute_planner_snqi_ordering",
    "compute_weight_sensitivity",
    "evaluate_snqi_contract",
    "resolve_weight_mapping",
    "sanitize_baseline_stats",
    "spearman_correlation",
]
