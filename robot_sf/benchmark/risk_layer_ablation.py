"""Layered-risk ablation for planner evaluation (issue #5832).

Scores each planner under progressively richer risk models and reports
per-layer ranking deltas with bootstrap confidence intervals.

The ablation axis mirrors the research motivation in issue #5832:

* **L0 geometry-only** — static obstacles / contact geometry. The current
  baseline behaviour where only hard collisions and obstacles matter.
* **L1 +dynamics** — moving pedestrians with velocity / intent exposure.
  Near-field and motion-comfort terms become active.
* **L2 +semantic risk** — zone / context weighting (crossings, doorways,
  high-density zones weighted in the risk field). The semantic-risk term is
  read from ``metrics.semantic_risk_exposure`` when a scenario carries zone
  metadata; otherwise the layer reports the term ``unavailable`` and the
  planner score is computed from the remaining active terms only.

This module is pure analysis / reporting tooling built on top of the existing
SNQI metric stack. It never runs a benchmark campaign and never promotes a
benchmark claim: it consumes episode records that already carry metrics and
produces per-layer tables plus a JSON / Markdown report artifact.

Metric mapping
--------------
Each risk layer is a fixed-schema SNQI weight configuration (see
``RISK_LAYERS``). Activating a higher layer enables additional weight terms;
deactivating a layer term sets its weight to ``0.0``. Because
``compute_snqi_v0`` uses the zero-weight contract, a missing metric under a
``0.0`` weight never collapses the score, and a missing *semantic* metric
under a non-zero L2 weight is reported as unavailable rather than imputed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_sf.benchmark.rank_metrics import rank_by, spearman
from robot_sf.benchmark.snqi.compute import compute_snqi_v0

RISK_LAYER_ABLATION_SCHEMA = "risk_layer_ablation.v1"
RISK_LAYER_VERSION = "risk-layers.v1"

# Semantic-risk exposure metric produced by zone-aware scenarios. Absent for
# scenario families without zone metadata; the L2 layer degrades gracefully.
SEMANTIC_RISK_METRIC_KEY = "semantic_risk_exposure"

# Canonical SNQI components used by the risk layers (subset of WEIGHT_NAMES in
# robot_sf.benchmark.snqi.compute).
RISK_WEIGHT_NAMES = (
    "w_success",
    "w_collisions",
    "w_force_exceed",
    "w_near",
    "w_comfort",
    "w_time",
    "w_semantic_risk",
)


@dataclass(frozen=True, slots=True)
class RiskLayerDefinition:
    """One risk-model layer in the progressive ablation axis."""

    name: str
    level: int
    description: str
    # Names of the hazard dimensions this layer adds relative to the previous one.
    adds: tuple[str, ...]


# Progressive risk-model definitions. L0 is the current baseline; each higher
# layer strictly adds hazard dimensions.
RISK_LAYERS: tuple[RiskLayerDefinition, ...] = (
    RiskLayerDefinition(
        name="L0_geometry_only",
        level=0,
        description="Static obstacles and hard contact geometry only.",
        adds=("geometry",),
    ),
    RiskLayerDefinition(
        name="L1_plus_dynamics",
        level=1,
        description="L0 plus moving-pedestrian dynamics: near-field exposure and motion comfort.",
        adds=("dynamics",),
    ),
    RiskLayerDefinition(
        name="L2_plus_semantic_risk",
        level=2,
        description="L1 plus semantic/zone risk weighting from scenario zone metadata.",
        adds=("semantic_risk",),
    ),
)


# Mapping from layer name to its SNQI weight configuration. Activating a layer
# enables the cumulative set of weight terms; deactivating keeps a ``0.0``
# weight so the zero-weight contract keeps the score finite.
RISK_LAYER_WEIGHTS: dict[str, dict[str, float]] = {
    "L0_geometry_only": {
        "w_success": 2.0,
        "w_collisions": 2.0,
        "w_force_exceed": 1.0,
        "w_near": 0.0,
        "w_comfort": 0.0,
        "w_time": 0.0,
        "w_semantic_risk": 0.0,
    },
    "L1_plus_dynamics": {
        "w_success": 2.0,
        "w_collisions": 2.0,
        "w_force_exceed": 1.0,
        "w_near": 1.5,
        "w_comfort": 1.0,
        "w_time": 0.0,
        "w_semantic_risk": 0.0,
    },
    "L2_plus_semantic_risk": {
        "w_success": 2.0,
        "w_collisions": 2.0,
        "w_force_exceed": 1.0,
        "w_near": 1.5,
        "w_comfort": 1.0,
        "w_time": 0.5,
        "w_semantic_risk": 1.5,
    },
}


@dataclass
class RiskLayerAblationRow:
    """Per-planner row across the risk-layer ablation axis."""

    planner: str
    per_layer_rank: dict[str, int]
    per_layer_score: dict[str, float]
    per_layer_support: dict[str, int]
    # layer_name -> rank delta relative to L0 (positive = worse than at L0).
    rank_delta_from_l0: dict[str, float]
    # True when the planner's rank changed between L0 and the richest layer.
    rank_changed: bool = field(default=False)


def _get_nested(record: Mapping[str, Any], dotted: str, default: Any | None = None) -> Any:
    """Return a dotted-path value from a mapping."""
    current: Any = record
    for part in dotted.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


def _resolve_group_key(record: Mapping[str, Any], group_by: str, fallback_group_by: str) -> str:
    """Resolve a planner/group key from an episode record.

    Returns:
        Planner/group key string, or ``"unknown"`` when both keys are absent.
    """
    value = _get_nested(record, group_by)
    if value is None:
        value = _get_nested(record, fallback_group_by)
    return str(value) if value is not None else "unknown"


def _episode_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized metrics view for one episode."""
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    return dict(metrics)


def _score_episode(
    record: Mapping[str, Any],
    weights: Mapping[str, float],
    baseline_stats: Mapping[str, Mapping[str, float]],
) -> float:
    """Score one episode under a risk-layer weight configuration.

    Returns:
        SNQI score for the episode under the supplied weights.
    """
    return float(compute_snqi_v0(_episode_metrics(record), weights, baseline_stats))


def _semantic_risk_available(records: Iterable[Mapping[str, Any]]) -> bool:
    """Return whether any episode carries a finite semantic-risk metric."""
    for record in records:
        value = _episode_metrics(record).get(SEMANTIC_RISK_METRIC_KEY)
        if isinstance(value, bool | int | float) and math.isfinite(float(value)):
            return True
    return False


def compute_layer_scores(
    records: Iterable[Mapping[str, Any]],
    *,
    layer_weights: Mapping[str, float],
    baseline_stats: Mapping[str, Mapping[str, float]],
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "algo",
) -> dict[str, dict[str, Any]]:
    """Compute per-planner mean SNQI scores for one risk layer.

    Returns:
        Mapping from planner id to ``{"mean": float, "support": int}``.
    """
    groups: dict[str, list[float]] = {}
    for record in records:
        key = _resolve_group_key(record, group_by, fallback_group_by)
        try:
            score = _score_episode(record, layer_weights, baseline_stats)
        except (ValueError, TypeError, KeyError):
            continue
        if not math.isfinite(score):
            continue
        groups.setdefault(key, []).append(score)
    return {
        planner: {"mean": sum(vals) / len(vals), "support": len(vals)}
        for planner, vals in groups.items()
        if vals
    }


def compute_risk_layer_ablation(
    records: Iterable[Mapping[str, Any]],
    *,
    baseline_stats: Mapping[str, Mapping[str, float]] | None = None,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "algo",
    layer_names: Sequence[str] | None = None,
    weights_override: Mapping[str, Mapping[str, float]] | None = None,
) -> list[RiskLayerAblationRow]:
    """Score planners under each risk layer and compute per-layer rank deltas.

    Args:
        records: Episode records carrying ``metrics`` and a planner/group key.
        baseline_stats: Optional SNQI baseline normalization stats. When absent,
            the v0 zero-baseline fallback is used (all terms normalized to 0).
        group_by: Dotted key used to group episodes by planner.
        fallback_group_by: Fallback grouping key when ``group_by`` is absent.
        layer_names: Ordered subset of risk-layer names to evaluate. Defaults to
            all ``RISK_LAYERS`` in level order.
        weights_override: Optional per-layer weight configurations overriding
            ``RISK_LAYER_WEIGHTS`` (used by tests and custom configs).

    Returns:
        List of per-planner rows ordered by L0 rank (best first).
    """
    record_list = [dict(record) for record in records]
    baseline: Mapping[str, Mapping[str, float]] = baseline_stats or {}
    names = list(layer_names) if layer_names is not None else [layer.name for layer in RISK_LAYERS]

    weights_by_layer: dict[str, dict[str, float]] = {}
    for name in names:
        if weights_override is not None and name in weights_override:
            weights_by_layer[name] = dict(weights_override[name])
        elif name in RISK_LAYER_WEIGHTS:
            weights_by_layer[name] = dict(RISK_LAYER_WEIGHTS[name])
        else:
            raise ValueError(f"unknown risk layer: {name!r}")

    per_layer_means: dict[str, dict[str, float]] = {}
    per_layer_support: dict[str, dict[str, int]] = {}
    for name in names:
        scored = compute_layer_scores(
            record_list,
            layer_weights=weights_by_layer[name],
            baseline_stats=baseline,
            group_by=group_by,
            fallback_group_by=fallback_group_by,
        )
        per_layer_means[name] = {planner: info["mean"] for planner, info in scored.items()}
        per_layer_support[name] = {planner: info["support"] for planner, info in scored.items()}

    if not per_layer_means or not names:
        return []

    base_layer = names[0]
    base_ranks = rank_by(
        per_layer_means[base_layer],
        higher_is_better=True,
        tie_abs_tol=1e-12,
    )

    rows_by_planner: dict[str, RiskLayerAblationRow] = {}
    for name in names:
        ranks = rank_by(
            per_layer_means[name],
            higher_is_better=True,
            tie_abs_tol=1e-12,
        )
        for planner, rank in ranks.items():
            row = rows_by_planner.get(planner)
            if row is None:
                row = RiskLayerAblationRow(
                    planner=planner,
                    per_layer_rank={},
                    per_layer_score={},
                    per_layer_support={},
                    rank_delta_from_l0={},
                )
                rows_by_planner[planner] = row
            row.per_layer_rank[name] = round(rank)
            row.per_layer_score[name] = float(per_layer_means[name][planner])
            row.per_layer_support[name] = int(per_layer_support[name].get(planner, 0))
            row.rank_delta_from_l0[name] = float(rank - base_ranks.get(planner, rank))

    for row in rows_by_planner.values():
        richest = names[-1]
        row.rank_changed = int(row.per_layer_rank.get(richest, 0)) != int(
            row.per_layer_rank.get(base_layer, 0)
        )

    ordered = sorted(
        rows_by_planner.values(),
        key=lambda r: r.per_layer_rank.get(base_layer, len(rows_by_planner) + 1),
    )
    return ordered


def bootstrap_layer_rank_stability(
    records: Iterable[Mapping[str, Any]],
    *,
    layer_name: str,
    rng: Any | None = None,
    samples: int = 200,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "algo",
    weights_override: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Estimate bootstrap stability of a layer's planner ranking.

    Stratified resampling preserves each planner's episode count and recomputes
    the mean SNQI per planner, then measures how often the resampled ranking
    preserves the baseline ordering (Spearman correlation). Higher values mean
    the layer's ranking is robust to episode resampling.

    Args:
        records: Episode records with a planner/group key and metrics.
        layer_name: Risk layer whose ranking stability is estimated.
        rng: Random generator exposing ``integers(...)`` (e.g.
            ``numpy.random.default_rng(seed)``).
        samples: Number of bootstrap resamples.
        group_by: Planner grouping key.
        fallback_group_by: Fallback grouping key.
        weights_override: Optional per-layer weight overrides.

    Returns:
        JSON-compatible payload with ``status: "ok"`` and a normalized
        ``stability`` score in ``[0, 1]`` plus per-layer rank CIs.
    """
    if rng is None:
        raise ValueError("bootstrap_layer_rank_stability requires rng=default_rng(seed)")
    if samples < 1:
        raise ValueError("samples must be >= 1")

    weights = (
        weights_override[layer_name]
        if weights_override is not None and layer_name in weights_override
        else RISK_LAYER_WEIGHTS[layer_name]
    )
    grouped: dict[str, list[float]] = {}
    for record in records:
        key = _resolve_group_key(record, group_by, fallback_group_by)
        try:
            score = _score_episode(record, weights, {})
        except (ValueError, TypeError, KeyError):
            continue
        if math.isfinite(score):
            grouped.setdefault(key, []).append(score)

    if len(grouped) < 2:
        raise ValueError("bootstrap_layer_rank_stability requires at least two planners")

    baseline_ordering = sorted(
        grouped,
        key=lambda g: (-_mean(grouped[g]), g),
    )
    baseline_ranks = rank_by(
        {g: _mean(grouped[g]) for g in grouped},
        higher_is_better=True,
        tie_abs_tol=1e-12,
    )
    baseline_rank_vec = [baseline_ranks[g] for g in baseline_ordering]

    correlations: list[float] = []
    rank_samples: dict[str, list[float]] = {g: [] for g in grouped}
    for _ in range(samples):
        sample_means = {g: _mean(_resample(grouped[g], rng=rng)) for g in grouped}
        sample_ranks = rank_by(sample_means, higher_is_better=True, tie_abs_tol=1e-12)
        for g in grouped:
            rank_samples[g].append(float(sample_ranks[g]))
        sample_vec = [sample_ranks[g] for g in baseline_ordering]
        correlation = spearman(baseline_rank_vec, sample_vec, degenerate=None, tie_abs_tol=1e-12)
        correlations.append(
            float(correlation)
            if correlation is not None
            else (1.0 if sample_vec == baseline_rank_vec else 0.0)
        )

    raw_mean = _mean(correlations)
    stability = (raw_mean + 1.0) / 2.0
    return {
        "status": "ok",
        "layer": layer_name,
        "stability": _clamp01(stability),
        "samples": int(samples),
        "method": "bootstrap_spearman",
        "baseline_ordering": baseline_ordering,
        "rank_cis": {
            g: {
                "mean_rank": _mean(rank_samples[g]),
                "min_rank": float(min(rank_samples[g])),
                "max_rank": float(max(rank_samples[g])),
                "ci95_low": _percentile(rank_samples[g], 2.5),
                "ci95_high": _percentile(rank_samples[g], 97.5),
            }
            for g in baseline_ordering
        },
    }


def build_risk_layer_report(
    records: Iterable[Mapping[str, Any]],
    *,
    baseline_stats: Mapping[str, Mapping[str, float]] | None = None,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "algo",
    layer_names: Sequence[str] | None = None,
    weights_override: Mapping[str, Mapping[str, float]] | None = None,
    bootstrap: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full risk-layer ablation report artifact.

    Args:
        records: Episode records carrying metrics and a planner/group key.
        baseline_stats: Optional SNQI baseline normalization stats.
        group_by: Planner grouping key.
        fallback_group_by: Fallback grouping key.
        layer_names: Ordered subset of risk layers to evaluate.
        weights_override: Optional per-layer weight overrides.
        bootstrap: Optional ``{"seed": int, "samples": int}`` enabling bootstrap
            rank-CI estimation on every evaluated layer.

    Returns:
        JSON-serializable report with per-layer tables, per-planner deltas, and
        optional bootstrap stability blocks.
    """
    record_list = [dict(record) for record in records]
    rows = compute_risk_layer_ablation(
        record_list,
        baseline_stats=baseline_stats,
        group_by=group_by,
        fallback_group_by=fallback_group_by,
        layer_names=layer_names,
        weights_override=weights_override,
    )
    names = list(layer_names) if layer_names is not None else [layer.name for layer in RISK_LAYERS]

    semantic_available = _semantic_risk_available(record_list)
    bootstrap_blocks: dict[str, Any] = {}
    if bootstrap is not None and record_list:
        rng = np.random.default_rng(int(bootstrap.get("seed", 0)))
        for name in names:
            try:
                bootstrap_blocks[name] = bootstrap_layer_rank_stability(
                    record_list,
                    layer_name=name,
                    rng=rng,
                    samples=int(bootstrap.get("samples", 200)),
                    group_by=group_by,
                    fallback_group_by=fallback_group_by,
                    weights_override=weights_override,
                )
            except ValueError:
                bootstrap_blocks[name] = {"status": "skipped", "reason": "insufficient_planners"}

    return {
        "schema_version": RISK_LAYER_ABLATION_SCHEMA,
        "risk_layer_version": RISK_LAYER_VERSION,
        "status": "artifact_only",
        "claim_boundary": (
            "Computes per-layer planner rankings from supplied episode records "
            "only; does not run a benchmark campaign or establish a benchmark claim."
        ),
        "layers": [
            {
                "name": layer.name,
                "level": layer.level,
                "description": layer.description,
                "adds": list(layer.adds),
                "weights": weights_override[layer.name]
                if weights_override is not None and layer.name in weights_override
                else dict(RISK_LAYER_WEIGHTS[layer.name]),
            }
            for layer in RISK_LAYERS
            if layer.name in names
        ],
        "semantic_risk_available": semantic_available,
        "group_by": group_by,
        "fallback_group_by": fallback_group_by,
        "n_episodes": len(record_list),
        "rows": [
            {
                "planner": row.planner,
                "per_layer_rank": dict(row.per_layer_rank),
                "per_layer_score": dict(row.per_layer_score),
                "per_layer_support": dict(row.per_layer_support),
                "rank_delta_from_l0": dict(row.rank_delta_from_l0),
                "rank_changed": row.rank_changed,
            }
            for row in rows
        ],
        "bootstrap": bootstrap_blocks,
    }


def format_risk_layer_markdown(report: Mapping[str, Any]) -> str:
    """Render the risk-layer ablation report as a Markdown table.

    Returns:
        Markdown string with a per-planner delta table.
    """
    rows = report.get("rows", [])
    if not rows:
        return "_no planner rows_"
    layer_names = [layer["name"] for layer in report.get("layers", [])]
    header_cells = ["Planner", "L0 rank"] + [f"Δrank {name}" for name in layer_names[1:]]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(["---:"] * len(header_cells)) + "|",
    ]
    for row in rows:
        l0 = row["per_layer_rank"].get(layer_names[0], "")
        deltas = [f"{row['rank_delta_from_l0'].get(name, 0.0):+.0f}" for name in layer_names[1:]]
        lines.append("| " + " | ".join([str(row["planner"]), str(l0), *deltas]) + " |")
    semantic = report.get("semantic_risk_available")
    if semantic is False:
        lines.append("")
        lines.append(
            "_Note: L2 semantic-risk term unavailable on supplied records "
            "(no zone metadata); L2 scored on remaining active terms._"
        )
    return "\n".join(lines) + "\n"


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean of finite values.

    Returns:
        Mean value of the inputs.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("mean requires at least one value")
    return float(math.fsum(vals) / len(vals))


def _resample(values: list[float], *, rng: Any) -> list[float]:
    """One with-replacement bootstrap resample.

    Returns:
        Resampled list with the same length as ``values``.
    """
    indexes = rng.integers(0, len(values), size=len(values))
    return [values[int(index)] for index in indexes]


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolation percentile (0-100) over sorted values.

    Returns:
        Percentile value at ``quantile`` (0-100 scale).
    """
    vals = sorted(float(v) for v in values)
    if not vals:
        raise ValueError("percentile requires at least one value")
    if len(vals) == 1:
        return vals[0]
    rank = (quantile / 100.0) * (len(vals) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return vals[low]
    frac = rank - low
    return vals[low] * (1.0 - frac) + vals[high] * frac


def _clamp01(value: float) -> float:
    """Clamp a floating-point value to ``[0, 1]``.

    Returns:
        Value bounded to ``[0, 1]``.
    """
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "RISK_LAYERS",
    "RISK_LAYER_ABLATION_SCHEMA",
    "RISK_LAYER_VERSION",
    "RISK_LAYER_WEIGHTS",
    "RISK_WEIGHT_NAMES",
    "SEMANTIC_RISK_METRIC_KEY",
    "RiskLayerAblationRow",
    "RiskLayerDefinition",
    "bootstrap_layer_rank_stability",
    "build_risk_layer_report",
    "compute_layer_scores",
    "compute_risk_layer_ablation",
    "format_risk_layer_markdown",
]
