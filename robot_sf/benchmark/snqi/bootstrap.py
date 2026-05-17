"""Deterministic bootstrap stability for SNQI planner rankings.

This module provides a small evidence-grade stability calculation for benchmark
records that already contain per-episode ``metrics.snqi`` values. The function
does not recompute SNQI from raw metric components because normalization
baselines are not part of this public API. Callers must supply episodes with
finite SNQI values and an explicit random generator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = ["bootstrap_stability"]


def bootstrap_stability(
    episodes: Iterable[dict],
    weights: Mapping[str, float],
    *,
    rng: Any | None = None,
    _rng: Any | None = None,
    samples: int = 30,
    group_key: str | None = "algo",
) -> dict[str, Any]:
    """Estimate SNQI planner-ranking stability by stratified bootstrapping.

    Args:
        episodes: Episode dictionaries with a planner/group key and finite
            ``metrics.snqi`` value.
        weights: SNQI weight metadata used for provenance. The function expects
            SNQI to have already been computed in each episode.
        rng: Random generator exposing ``integers(low, high, size=...)``; for
            example ``numpy.random.default_rng(seed)``.
        _rng: Backward-compatible alias for callers that used the old scaffold
            parameter name.
        samples: Number of bootstrap resamples.
        group_key: Episode key used to group records by planner or policy.

    Returns:
        A JSON-compatible payload with ``status: "ok"`` and a normalized
        stability score in ``[0, 1]``. Higher values mean bootstrap resamples
        preserve the baseline group ranking more consistently.

    Raises:
        ValueError: If inputs cannot support evidence-grade ranking stability.
    """
    active_rng = rng if rng is not None else _rng
    if active_rng is None:
        raise ValueError("bootstrap_stability requires rng=np.random.default_rng(seed)")
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if not group_key:
        raise ValueError("group_key must be non-empty")

    grouped = _group_snqi_scores(episodes, group_key=group_key)
    if len(grouped) < 2:
        raise ValueError("bootstrap_stability requires at least two groups")

    baseline_means = {group: _mean(scores) for group, scores in grouped.items()}
    baseline_ordering = _ordering(baseline_means)
    baseline_ranks = _rank_vector(baseline_ordering, baseline_means)
    correlations: list[float] = []
    sampled_orderings: list[list[str]] = []
    for _ in range(samples):
        sample_means = {
            group: _mean(_resample(scores, rng=active_rng)) for group, scores in grouped.items()
        }
        sample_ordering = _ordering(sample_means)
        sampled_orderings.append(sample_ordering)
        sample_ranks = _rank_vector(baseline_ordering, sample_means)
        correlations.append(_spearman(baseline_ranks, sample_ranks))

    raw_mean = _mean(correlations)
    normalized = (raw_mean + 1.0) / 2.0
    return {
        "stability": _clamp01(normalized),
        "samples": int(samples),
        "method": "bootstrap_spearman",
        "status": "ok",
        "details": {
            "group_key": group_key,
            "weights": {str(key): float(value) for key, value in weights.items()},
            "baseline_ordering": baseline_ordering,
            "groups": {
                group: {"episodes": len(scores), "mean_snqi": _mean(scores)}
                for group, scores in sorted(grouped.items())
            },
            "raw_spearman_mean": raw_mean,
            "raw_spearman_min": min(correlations),
            "raw_spearman_max": max(correlations),
            "sample_orderings": sampled_orderings,
            "normalization": "stability = (mean_spearman + 1) / 2",
        },
    }


def _group_snqi_scores(
    episodes: Iterable[dict],
    *,
    group_key: str,
) -> dict[str, list[float]]:
    """Group finite per-episode SNQI values by planner key.

    Returns:
        Mapping from planner/group key to finite SNQI values.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"episode[{index}] must be a mapping")
        group = episode.get(group_key)
        if group is None:
            raise ValueError(f"episode[{index}] missing group key '{group_key}'")
        snqi = _episode_snqi(episode, index=index)
        grouped[str(group)].append(snqi)
    if not grouped:
        raise ValueError("bootstrap_stability requires at least one episode")
    return dict(grouped)


def _episode_snqi(episode: dict[str, Any], *, index: int) -> float:
    """Read a finite SNQI value from one episode.

    Returns:
        Parsed per-episode SNQI value.
    """
    metrics = episode.get("metrics")
    if not isinstance(metrics, dict) or "snqi" not in metrics:
        raise ValueError(f"episode[{index}] must contain finite metrics.snqi")
    try:
        value = float(metrics["snqi"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"episode[{index}] metrics.snqi must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"episode[{index}] metrics.snqi must be finite")
    return value


def _resample(scores: list[float], *, rng: Any) -> list[float]:
    """Return one with-replacement bootstrap resample."""
    indexes = rng.integers(0, len(scores), size=len(scores))
    return [scores[int(index)] for index in indexes]


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean of finite values."""
    values_list = [float(value) for value in values]
    if not values_list:
        raise ValueError("mean requires at least one value")
    return float(sum(values_list) / len(values_list))


def _ordering(means: Mapping[str, float]) -> list[str]:
    """Return deterministic descending SNQI ordering."""
    return sorted(means, key=lambda group: (-float(means[group]), group))


def _rank_vector(ordering: list[str], means: Mapping[str, float]) -> list[float]:
    """Return average ranks for groups in baseline-ordering order."""
    sorted_groups = _ordering(means)
    ranks_by_group: dict[str, float] = {}
    position = 0
    while position < len(sorted_groups):
        group = sorted_groups[position]
        value = float(means[group])
        end = position + 1
        while end < len(sorted_groups) and math.isclose(
            float(means[sorted_groups[end]]),
            value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for tied_group in sorted_groups[position:end]:
            ranks_by_group[tied_group] = average_rank
        position = end
    return [ranks_by_group[group] for group in ordering]


def _spearman(left: list[float], right: list[float]) -> float:
    """Return Spearman correlation for two rank vectors."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation requires equal vectors with at least two entries")
    left_mean = _mean(left)
    right_mean = _mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left == right else 0.0
    return float(numerator / (left_norm * right_norm))


def _clamp01(value: float) -> float:
    """Clamp a floating-point value to the unit interval.

    Returns:
        Value bounded to ``[0, 1]``.
    """
    return max(0.0, min(1.0, float(value)))
