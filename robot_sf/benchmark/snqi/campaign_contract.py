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


@dataclass(frozen=True)
class SnqiContractThresholds:
    """Threshold configuration for SNQI campaign contract evaluation."""

    rank_alignment_warn: float = 0.5
    rank_alignment_fail: float = 0.3
    outcome_separation_warn: float = 0.05
    outcome_separation_fail: float = 0.0


@dataclass(frozen=True)
class SnqiContractEvaluation:
    """Computed contract metrics and resolved status."""

    status: str
    rank_alignment_spearman: float
    outcome_separation: float
    objective_score: float


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

    objective = rank_alignment + 0.25 * outcome_separation

    if (
        rank_alignment < thresholds.rank_alignment_fail
        or outcome_separation < thresholds.outcome_separation_fail
    ):
        status = "fail"
    elif (
        rank_alignment < thresholds.rank_alignment_warn
        or outcome_separation < thresholds.outcome_separation_warn
    ):
        status = "warn"
    else:
        status = "pass"

    return SnqiContractEvaluation(
        status=status,
        rank_alignment_spearman=float(rank_alignment),
        outcome_separation=float(outcome_separation),
        objective_score=float(objective),
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
    "calibrate_weights",
    "collect_episodes_from_campaign_runs",
    "compute_baseline_stats_from_episodes",
    "compute_component_dominance",
    "evaluate_snqi_contract",
    "resolve_weight_mapping",
    "sanitize_baseline_stats",
    "spearman_correlation",
]
