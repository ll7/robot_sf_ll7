"""Effect size calculations for rate and continuous metrics (T032).

Implements the research decisions (see `research.md`):
  * Rate metrics (collision_rate, success_rate): absolute difference Δp and Cohen's h
    between reference density (cfg.effect_size_reference_density) and each *other* density
    inside the same archetype.
  * Continuous metrics (time_to_goal, path_efficiency, snqi, avg_speed): difference
    of means Δμ (other - reference) and Glass' Δ = (μ_other - μ_ref) / σ_ref where σ_ref
    estimated as sample standard deviation of underlying episode values for that metric
    in the reference density group. If σ_ref == 0 we set standardized=0.0.

Input ``groups`` is the list produced by ``aggregate_metrics`` (AggregateMetricsGroup).
We expect each group.metrics[metric_name] to expose ``mean``. For Glass Δ we also need
the raw values; since aggregation currently does not keep them, we cannot derive σ_ref
directly. To keep T032 focused and unblock tests, we approximate σ_ref using the width
of the bootstrap mean CI when available:

    σ_ref ≈ (ci_high - ci_low) / (2 * z)

with z ~ 1.96 for 95% CI. This derives from mean CI ≈ μ ± z * (σ / sqrt(n)). Rearranged:
σ ≈ half_width * sqrt(n) / z. We recover n from group.count. If CI missing, fall back to
standardized = 0.0 (conservative) with a log note.

Returned structure: list of EffectSizeReport (dataclasses defined inline for now).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from loguru import logger

RATE_METRICS = {"collision_rate", "success_rate"}
CONT_METRICS = {"time_to_goal", "path_efficiency", "snqi", "avg_speed"}

__all__ = ["EffectSizeEntry", "EffectSizeReport", "compute_effect_sizes"]


@dataclass
class EffectSizeEntry:  # matches data model
    """TODO docstring. Document this class."""

    metric: str
    density_low: str
    density_high: str
    diff: float  # high - low (or other - reference for continuous)
    standardized: float  # Cohen's h or Glass Δ


@dataclass
class EffectSizeReport:
    """TODO docstring. Document this class."""

    archetype: str
    comparisons: list[EffectSizeEntry]


def compute_effect_sizes(groups, cfg):  # T032
    """TODO docstring. Document this function.

    Args:
        groups: TODO docstring.
        cfg: TODO docstring.
    """
    by_arch = _group_by_archetype(groups)
    reference_density = getattr(cfg, "effect_size_reference_density", "low")
    reports: list[EffectSizeReport] = []
    for _, density_map in sorted(by_arch.items()):
        ref_group = density_map.get(reference_density)
        if not ref_group:
            logger.debug("Skipping archetype: reference density '{}' absent", reference_density)
            continue
        comparisons = _build_comparisons_for_archetype(ref_group, density_map, reference_density)
        if comparisons:
            arch_name = getattr(ref_group, "archetype", "unknown")
            reports.append(EffectSizeReport(archetype=arch_name, comparisons=comparisons))
    return reports


def _group_by_archetype(groups) -> dict[str, dict[str, object]]:
    """TODO docstring. Document this function.

    Args:
        groups: TODO docstring.

    Returns:
        TODO docstring.
    """
    result: dict[str, dict[str, object]] = {}
    for g in groups:
        result.setdefault(g.archetype, {})[g.density] = g
    return result


def _build_comparisons_for_archetype(ref_group, density_map, reference_density):
    """TODO docstring. Document this function.

    Args:
        ref_group: TODO docstring.
        density_map: TODO docstring.
        reference_density: TODO docstring.
    """
    comparisons: list[EffectSizeEntry] = []
    for density_key, group in sorted(density_map.items()):
        if density_key == reference_density:
            continue
        comparisons.extend(_rate_effect_sizes(ref_group, group, reference_density, density_key))
        comparisons.extend(
            _continuous_effect_sizes(ref_group, group, reference_density, density_key),
        )
    return comparisons


def _rate_effect_sizes(ref_group, other_group, ref_density, other_density):
    """TODO docstring. Document this function.

    Args:
        ref_group: TODO docstring.
        other_group: TODO docstring.
        ref_density: TODO docstring.
        other_density: TODO docstring.
    """
    entries: list[EffectSizeEntry] = []
    for metric in RATE_METRICS:
        ref_metric = ref_group.metrics.get(metric)
        other_metric = other_group.metrics.get(metric)
        if not ref_metric or not other_metric:
            continue
        p_ref = ref_metric.mean
        p_other = other_metric.mean
        try:
            h = 2 * math.asin(math.sqrt(p_other)) - 2 * math.asin(math.sqrt(p_ref))
        except ValueError:
            h = math.nan
        entries.append(
            EffectSizeEntry(
                metric=metric,
                density_low=ref_density,
                density_high=other_density,
                diff=p_other - p_ref,
                standardized=h,
            ),
        )
    return entries


def _continuous_effect_sizes(ref_group, other_group, ref_density, other_density):
    """TODO docstring. Document this function.

    Args:
        ref_group: TODO docstring.
        other_group: TODO docstring.
        ref_density: TODO docstring.
        other_density: TODO docstring.
    """
    entries: list[EffectSizeEntry] = []
    for metric in CONT_METRICS:
        ref_metric = ref_group.metrics.get(metric)
        other_metric = other_group.metrics.get(metric)
        if not ref_metric or not other_metric:
            continue
        mu_ref = ref_metric.mean
        mu_other = other_metric.mean
        diff = mu_other - mu_ref
        glass_delta = _glass_delta(ref_metric, diff, getattr(ref_group, "count", 0))
        entries.append(
            EffectSizeEntry(
                metric=metric,
                density_low=ref_density,
                density_high=other_density,
                diff=diff,
                standardized=glass_delta,
            ),
        )
    return entries


def _glass_delta(ref_metric, diff, n):
    """TODO docstring. Document this function.

    Args:
        ref_metric: TODO docstring.
        diff: TODO docstring.
        n: TODO docstring.
    """
    ci = getattr(ref_metric, "mean_ci", None)
    if not ci or n <= 1:
        if not ci:
            logger.debug(
                "No CI for metric {}; Glass Δ set 0",
                getattr(ref_metric, "name", "unknown"),
            )
        return 0.0
    half_width = (ci[1] - ci[0]) / 2.0
    z = 1.959963984540054
    sigma_ref = half_width * math.sqrt(n) / z if z > 0 else 0.0
    if sigma_ref <= 0:
        return 0.0
    return diff / sigma_ref
