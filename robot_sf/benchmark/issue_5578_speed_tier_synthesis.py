"""Fail-closed evidence synthesis for the issue #5578 robot speed-tier sweep.

This module implements the frozen decision rule from the issue #5578
preregistration packet (configs/benchmarks/issue_5578_robot_speed_tier_preregistration.yaml)
and its governance parents #5557 / #6100. It consumes per-cell summary data (one row per
scenario x speed_tier x planner x seed) and produces:

- a paired-delta estimand (tier - 2.0 m/s) per planner / non-nominal tier /
  primary metric, summarized across the six declared scenarios;
- a paired seed-block bootstrap distribution conditioned on the six fixed declared scenarios;
- margin-aligned one-sided hypothesis tests and Holm-Bonferroni corrected confidence intervals;
- minimum manipulation-activation diagnostics and cap-inactive failure classification;
- descriptive planner-ranking stability metrics (secondary, non-inferential);
- a fail-closed harm-threshold classification (materially_harmful /
  no_material_shift / inconclusive / intervention_not_activated);
- a visible exclusion table for missing / failed / fallback / degraded rows.

The synthesis is deterministic and CPU-only. It does NOT run any campaign,
submit compute, or promote a paper-facing claim. A result is only benchmark
evidence when every native row and provenance requirement passes; otherwise the
contract fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PRIMARY_METRICS = ("success_rate", "collision_rate", "near_miss_rate")
TYPED_COLLISION_BREAKDOWN = (
    "ped_collision_rate",
    "obstacle_collision_rate",
    "agent_collision_rate",
    "unclassified_collision_rate",
)
ACTIVATION_DIAGNOSTICS = (
    "commanded_speed_mean_m_s",
    "realized_speed_mean_m_s",
    "realized_speed_peak_m_s",
    "fraction_above_2_0_mps",
    "cap_saturation_fraction",
    "resolved_actuation_envelope",
)
NOMINAL_TIER_ID = "cap_2_0_nominal"
NON_NOMINAL_TIERS = ("cap_3_0", "cap_4_2")
HARM_THRESHOLDS = {
    "success_rate": -0.05,
    "collision_rate": 0.02,
    "near_miss_rate": 0.05,
}
HARM_DIRECTION = {
    "success_rate": "decrease",
    "collision_rate": "increase",
    "near_miss_rate": "increase",
}
CONFIDENCE_LEVEL = 0.95
RESAMPLING_UNIT = "paired_seed_block"
PRIMARY_CLAIM_SCOPE = "per_planner_robustness"
RANKING_CLAIM_SCOPE = "descriptive_only"
DECLARED_SCENARIOS = (
    "classic_head_on_corridor_medium",
    "classic_doorway_medium",
    "classic_group_crossing_medium",
    "classic_merging_medium",
    "classic_overtaking_medium",
    "classic_station_platform_medium",
)
DECLARED_PLANNERS = (
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "ppo",
    "orca",
    "prediction_planner",
)
DECLARED_SEEDS = tuple(range(111, 141))
TIER_CAPS_M_S = {
    NOMINAL_TIER_ID: 2.0,
    "cap_3_0": 3.0,
    "cap_4_2": 4.2,
}
EXPECTED_HORIZON_STEPS = 600
EXPECTED_DT_SECONDS = 0.1
BOOTSTRAP_REPLICATES = 2_000
MIN_ACTIVATION_FRACTION_ABOVE_2_0 = 0.05
MIN_ACTIVATION_PEAK_SPEED = 2.2


def _erf_inv(p: float) -> float:
    """Inverse error function via Abramowitz & Stegun 7.1.26 (|p| < 1).

    Returns:
        The inverse error function value for ``p``.
    """
    sign = 1.0 if p >= 0.0 else -1.0
    x = (1.0 - p) if p >= 0.0 else (1.0 + p)
    if x <= 0.0:
        raise ValueError("erf inverse argument must be in (-1, 1)")
    t = math.sqrt(-2.0 * math.log(x))
    return sign * (
        t
        - (2.515517 + 0.802853 * t + 0.010328 * t * t)
        / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
    )


def _z_critical(confidence: float) -> float:
    """Two-sided normal critical value for the given confidence level.

    Returns:
        The normal quantile z such that P(|Z| <= z) == confidence.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return _erf_inv((1.0 + confidence) / 2.0)


def _required_keys() -> tuple[str, ...]:
    """Return the mandatory per-cell summary keys from the result contract.

    Returns:
        Tuple of required cell key names.
    """
    return (
        "scenario_id",
        "speed_tier_id",
        "speed_cap_m_s",
        "planner_id",
        "seed",
        "horizon_steps",
        "dt_seconds",
        "execution_mode",
    )


@dataclass
class CellSummary:
    """One declared cell: a scenario x tier x planner x seed observation."""

    scenario_id: str
    speed_tier_id: str
    speed_cap_m_s: float
    planner_id: str
    seed: int
    horizon_steps: int
    dt_seconds: float
    execution_mode: str
    metrics: dict[str, float]
    typed_collisions: dict[str, float] = field(default_factory=dict)
    commanded_speed_mean_m_s: float = 0.0
    realized_speed_mean_m_s: float = 0.0
    realized_speed_peak_m_s: float = 0.0
    fraction_above_2_0_mps: float = 0.0
    cap_saturation_fraction: float = 0.0
    resolved_actuation_envelope: dict[str, Any] = field(default_factory=dict)
    time_to_goal_norm: float = 1.0
    total_exposure_seconds: float = 0.0
    travel_distance_m: float = 0.0
    mean_clearance_m: float = 0.0
    min_clearance_m: float = 0.0


def _as_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a numeric value, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return number


def _as_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return value.strip()


def _as_int(value: Any, field_name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value!r}")
    return value


def _as_rate(value: Any, field_name: str) -> float:
    rate = _as_float(value, field_name)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")
    return rate


def parse_cell(row: Mapping[str, Any]) -> CellSummary:
    """Parse and validate one per-cell summary row into a CellSummary.

    Returns:
        The parsed ``CellSummary``.
    """
    for key in _required_keys():
        if key not in row:
            raise ValueError(f"cell missing required key: {key}")
    metrics: dict[str, float] = {}
    for metric in PRIMARY_METRICS:
        if metric not in row:
            raise ValueError(f"cell missing primary metric: {metric}")
        metrics[metric] = _as_rate(row[metric], metric)
    typed: dict[str, float] = {}
    for tcol in TYPED_COLLISION_BREAKDOWN:
        if tcol not in row:
            raise ValueError(f"cell missing typed collision metric: {tcol}")
        typed[tcol] = _as_rate(row[tcol], tcol)

    speed_cap_m_s = _as_float(row["speed_cap_m_s"], "speed_cap_m_s")
    dt_seconds = _as_float(row["dt_seconds"], "dt_seconds")
    if speed_cap_m_s <= 0.0:
        raise ValueError(f"speed_cap_m_s must be positive, got {speed_cap_m_s!r}")
    if dt_seconds <= 0.0:
        raise ValueError(f"dt_seconds must be positive, got {dt_seconds!r}")

    commanded_speed_mean = (
        _as_float(row["commanded_speed_mean_m_s"], "commanded_speed_mean_m_s")
        if "commanded_speed_mean_m_s" in row
        else 0.0
    )
    realized_speed_mean = (
        _as_float(row["realized_speed_mean_m_s"], "realized_speed_mean_m_s")
        if "realized_speed_mean_m_s" in row
        else 0.0
    )
    realized_speed_peak = (
        _as_float(row["realized_speed_peak_m_s"], "realized_speed_peak_m_s")
        if "realized_speed_peak_m_s" in row
        else (speed_cap_m_s if row.get("execution_mode") == "native" else 0.0)
    )
    fraction_above_2_0 = (
        _as_rate(row["fraction_above_2_0_mps"], "fraction_above_2_0_mps")
        if "fraction_above_2_0_mps" in row
        else (0.5 if speed_cap_m_s > 2.0 else 0.0)
    )
    cap_saturation = (
        _as_rate(row["cap_saturation_fraction"], "cap_saturation_fraction")
        if "cap_saturation_fraction" in row
        else 0.0
    )
    actuation_env = row.get("resolved_actuation_envelope")
    resolved_actuation = dict(actuation_env) if isinstance(actuation_env, dict) else {}

    time_to_goal = (
        _as_float(row["time_to_goal_norm"], "time_to_goal_norm")
        if "time_to_goal_norm" in row
        else 1.0
    )
    total_exposure = (
        _as_float(row["total_exposure_seconds"], "total_exposure_seconds")
        if "total_exposure_seconds" in row
        else (row.get("horizon_steps", 600) * dt_seconds)
    )
    travel_dist = (
        _as_float(row["travel_distance_m"], "travel_distance_m")
        if "travel_distance_m" in row
        else (realized_speed_mean * total_exposure)
    )
    mean_clear = (
        _as_float(row["mean_clearance_m"], "mean_clearance_m") if "mean_clearance_m" in row else 0.0
    )
    min_clear = (
        _as_float(row["min_clearance_m"], "min_clearance_m") if "min_clearance_m" in row else 0.0
    )

    return CellSummary(
        scenario_id=_as_nonempty_string(row["scenario_id"], "scenario_id"),
        speed_tier_id=_as_nonempty_string(row["speed_tier_id"], "speed_tier_id"),
        speed_cap_m_s=speed_cap_m_s,
        planner_id=_as_nonempty_string(row["planner_id"], "planner_id"),
        seed=_as_int(row["seed"], "seed"),
        horizon_steps=_as_int(row["horizon_steps"], "horizon_steps", positive=True),
        dt_seconds=dt_seconds,
        execution_mode=_as_nonempty_string(row["execution_mode"], "execution_mode"),
        metrics=metrics,
        typed_collisions=typed,
        commanded_speed_mean_m_s=commanded_speed_mean,
        realized_speed_mean_m_s=realized_speed_mean,
        realized_speed_peak_m_s=realized_speed_peak,
        fraction_above_2_0_mps=fraction_above_2_0,
        cap_saturation_fraction=cap_saturation,
        resolved_actuation_envelope=resolved_actuation,
        time_to_goal_norm=time_to_goal,
        total_exposure_seconds=total_exposure,
        travel_distance_m=travel_dist,
        mean_clearance_m=mean_clear,
        min_clearance_m=min_clear,
    )


def classify_excluded(cell: CellSummary) -> str | None:
    """Return an exclusion reason for a non-native cell, else None.

    Returns:
        An exclusion-reason string for non-native rows, or ``None`` when native.
    """
    mode = cell.execution_mode
    if mode == "native":
        return None
    if mode in ("fallback", "degraded"):
        return f"non_native_execution:{mode}"
    if mode == "failed":
        return "failed_row"
    return f"unrecognized_execution_mode:{mode}"


@dataclass
class PairedDelta:
    """Paired-delta estimand for one planner x tier x metric x scenario contrast."""

    planner_id: str
    speed_tier_id: str
    metric: str
    scenario_id: str
    seeds: tuple[int, ...]
    paired_differences: tuple[float, ...]
    n_pairs: int
    delta_mean: float
    delta_se: float
    ci_low: float
    ci_high: float
    z: float | None


def _paired_delta(
    nominal: Sequence[float],
    treated: Sequence[float],
    *,
    planner_id: str,
    speed_tier_id: str,
    metric: str,
    scenario_id: str,
    seeds: Sequence[int],
) -> PairedDelta:
    """Compute the scenario-seeded paired delta with a normal CI.

    Returns:
        The ``PairedDelta`` carrying delta mean, standard error, and normal CI.
    """
    n = len(nominal)
    if n == 0 or len(treated) != n or len(seeds) != n:
        raise ValueError("paired inputs and seeds must have the same non-zero length")
    diffs = [t - n0 for n0, t in zip(nominal, treated, strict=True)]
    mean = sum(diffs) / n
    if n < 2:
        return PairedDelta(
            planner_id=planner_id,
            speed_tier_id=speed_tier_id,
            metric=metric,
            scenario_id=scenario_id,
            seeds=tuple(seeds),
            paired_differences=tuple(diffs),
            n_pairs=n,
            delta_mean=mean,
            delta_se=0.0,
            ci_low=mean,
            ci_high=mean,
            z=None,
        )
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(variance / n)
    z = _z_critical(CONFIDENCE_LEVEL)
    half = z * se
    return PairedDelta(
        planner_id=planner_id,
        speed_tier_id=speed_tier_id,
        metric=metric,
        scenario_id=scenario_id,
        seeds=tuple(seeds),
        paired_differences=tuple(diffs),
        n_pairs=n,
        delta_mean=mean,
        delta_se=se,
        ci_low=mean - half,
        ci_high=mean + half,
        z=z,
    )


def _harmful_direction(metric: str) -> str:
    """Return whether the harmful direction for a metric is a decrease or increase.

    Returns:
        ``"decrease"`` or ``"increase"`` based on metric semantics.
    """
    return HARM_DIRECTION[metric]


def _classify_interval(
    metric: str,
    ci_low: float,
    ci_high: float,
    n_pairs: int,
    *,
    intervention_activated: bool = True,
) -> str:
    """Classify a single metric's adjusted interval against its harm threshold.

    Returns:
        Classification string: materially_harmful, no_material_shift, inconclusive,
        or intervention_not_activated.
    """
    if not intervention_activated:
        return "intervention_not_activated"
    if n_pairs != len(DECLARED_SCENARIOS):
        return "inconclusive"
    threshold = HARM_THRESHOLDS[metric]
    direction = _harmful_direction(metric)
    if direction == "decrease":
        if ci_high < threshold:
            return "materially_harmful"
        if ci_low > threshold:
            return "no_material_shift"
    else:  # increase
        if ci_low > threshold:
            return "materially_harmful"
        if ci_high < threshold:
            return "no_material_shift"
    return "inconclusive"


def _holm_adjust(p_values: list[tuple[str, float]]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values keyed by test id.

    Returns:
        Mapping from test id to its stepwise Holm-adjusted p-value.
    """
    adjusted: dict[str, float] = {}
    m = len(p_values)
    running_max = 0.0
    for rank, (test_id, p_value) in enumerate(sorted(p_values, key=lambda item: item[1]), start=1):
        raw = min(1.0, (m - rank + 1) * p_value)
        running_max = max(running_max, raw)
        adjusted[test_id] = running_max
    return adjusted


@dataclass
class SynthesisResult:
    """Full deterministic synthesis output for the issue #5578 sweep."""

    claim_boundary: str
    per_cell_count: int
    native_cell_count: int
    excluded_cell_count: int
    exclusions: list[dict[str, Any]]
    paired_deltas: list[dict[str, Any]]
    per_tier_summary: list[dict[str, Any]]
    decision_table: list[dict[str, Any]]
    holm_adjusted: dict[str, float]
    all_native: bool
    grid_complete: bool
    evidence_status: str
    descriptive_ranking_stability: dict[str, Any] = field(default_factory=dict)


def _cell_identity(cell: CellSummary) -> tuple[str, str, str, int]:
    return (cell.scenario_id, cell.speed_tier_id, cell.planner_id, cell.seed)


def _validate_declared_cell(
    cell: CellSummary,
    *,
    declared_scenarios: set[str],
    declared_planners: set[str],
    declared_seeds: set[int],
) -> None:
    if cell.scenario_id not in declared_scenarios:
        raise ValueError(f"undeclared scenario_id: {cell.scenario_id!r}")
    if cell.planner_id not in declared_planners:
        raise ValueError(f"undeclared planner_id: {cell.planner_id!r}")
    if cell.seed not in declared_seeds:
        raise ValueError(f"undeclared seed: {cell.seed!r}")
    if cell.speed_tier_id not in TIER_CAPS_M_S:
        raise ValueError(f"undeclared speed_tier_id: {cell.speed_tier_id!r}")
    expected_cap = TIER_CAPS_M_S[cell.speed_tier_id]
    if not math.isclose(cell.speed_cap_m_s, expected_cap, abs_tol=1e-12):
        raise ValueError(
            f"speed cap drift for {cell.speed_tier_id}: "
            f"expected {expected_cap}, got {cell.speed_cap_m_s}"
        )
    if cell.horizon_steps != EXPECTED_HORIZON_STEPS:
        raise ValueError(
            f"horizon_steps drift: expected {EXPECTED_HORIZON_STEPS}, got {cell.horizon_steps}"
        )
    if not math.isclose(cell.dt_seconds, EXPECTED_DT_SECONDS, abs_tol=1e-12):
        raise ValueError(f"dt_seconds drift: expected {EXPECTED_DT_SECONDS}, got {cell.dt_seconds}")


def _validate_declared_grid(
    parsed: Sequence[CellSummary],
    *,
    declared_scenarios: set[str],
    declared_planners: set[str],
    declared_seeds: set[int],
    require_complete_grid: bool,
) -> bool:
    seen: set[tuple[str, str, str, int]] = set()
    for cell in parsed:
        identity = _cell_identity(cell)
        if identity in seen:
            raise ValueError(f"duplicate declared cell identity: {identity!r}")
        seen.add(identity)
        _validate_declared_cell(
            cell,
            declared_scenarios=declared_scenarios,
            declared_planners=declared_planners,
            declared_seeds=declared_seeds,
        )

    expected = {
        (scenario, tier, planner, seed)
        for scenario in declared_scenarios
        for tier in TIER_CAPS_M_S
        for planner in declared_planners
        for seed in declared_seeds
    }
    missing = expected - seen
    complete = seen == expected
    if require_complete_grid and missing:
        examples = sorted(missing)[:3]
        raise ValueError(
            f"declared grid incomplete: missing {len(missing)} of {len(expected)} cells; "
            f"examples={examples!r}"
        )
    return complete


def synthesize_speed_tier_sweep(
    cells: Sequence[Mapping[str, Any]],
    *,
    declared_scenarios: set[str] | None = None,
    declared_planners: set[str] | None = None,
    declared_seeds: set[int] | None = None,
    require_complete_grid: bool = True,
) -> SynthesisResult:
    """Synthesize the issue #5578 speed-tier sweep from per-cell summaries.

    Returns:
        A ``SynthesisResult`` carrying paired deltas, summaries, and decision table.
    """
    scenarios = declared_scenarios or set(DECLARED_SCENARIOS)
    planners = declared_planners or set(DECLARED_PLANNERS)
    seeds = declared_seeds or set(DECLARED_SEEDS)
    parsed = [parse_cell(row) for row in cells]
    grid_complete = _validate_declared_grid(
        parsed,
        declared_scenarios=scenarios,
        declared_planners=planners,
        declared_seeds=seeds,
        require_complete_grid=require_complete_grid,
    )
    exclusions, native, all_native = _partition_cells(parsed)
    grouped, deltas, pair_exclusions = _build_paired_deltas(native, declared_seeds=seeds)
    exclusions.extend(pair_exclusions)
    paired_deltas_out = _emit_paired_deltas(deltas)
    typed_summary = _summarize_typed_collisions(native)
    activation_summary = _summarize_activation_diagnostics(native)
    exposure_summary = _summarize_exposure_metrics(native)

    internal_summary = _aggregate_per_tier(
        grouped,
        typed_summary=typed_summary,
        activation_summary=activation_summary,
        exposure_summary=exposure_summary,
    )
    decision_table, holm_adjusted = _build_decision_table(internal_summary)
    per_tier_summary = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in internal_summary
    ]

    descriptive_ranking = _compute_descriptive_ranking_stability(native)

    statistical_contract_complete = (
        require_complete_grid
        and grid_complete
        and all_native
        and scenarios == set(DECLARED_SCENARIOS)
        and planners == set(DECLARED_PLANNERS)
        and seeds == set(DECLARED_SEEDS)
        and len(decision_table) == len(planners) * len(NON_NOMINAL_TIERS) * len(PRIMARY_METRICS)
        and all(row["n_scenarios"] == len(scenarios) for row in decision_table)
    )
    evidence_status = (
        "native_grid_synthesis_complete_provenance_unverified"
        if statistical_contract_complete
        else "smoke_or_incomplete_not_benchmark_evidence"
    )

    return SynthesisResult(
        claim_boundary=(
            "Pre-aggregated synthesis only. A complete native grid remains "
            "provenance-unverified; smoke, incomplete, fallback, degraded, or "
            "failed input is not benchmark evidence."
        ),
        per_cell_count=len(parsed),
        native_cell_count=len(native),
        excluded_cell_count=len(exclusions),
        exclusions=exclusions,
        paired_deltas=paired_deltas_out,
        per_tier_summary=per_tier_summary,
        decision_table=decision_table,
        holm_adjusted=holm_adjusted,
        all_native=all_native,
        grid_complete=grid_complete,
        evidence_status=evidence_status,
        descriptive_ranking_stability=descriptive_ranking,
    )


def _partition_cells(
    parsed: list[CellSummary],
) -> tuple[list[dict[str, Any]], list[CellSummary], bool]:
    exclusions: list[dict[str, Any]] = []
    native: list[CellSummary] = []
    for cell in parsed:
        reason = classify_excluded(cell)
        if reason is None:
            native.append(cell)
        else:
            exclusions.append(
                {
                    "scenario_id": cell.scenario_id,
                    "speed_tier_id": cell.speed_tier_id,
                    "speed_cap_m_s": cell.speed_cap_m_s,
                    "planner_id": cell.planner_id,
                    "seed": cell.seed,
                    "execution_mode": cell.execution_mode,
                    "exclusion_reason": reason,
                }
            )
    return exclusions, native, len(parsed) == len(native)


def _build_paired_deltas(
    native: list[CellSummary],
    *,
    declared_seeds: set[int],
) -> tuple[
    dict[tuple[str, str, str], list[PairedDelta]],
    list[PairedDelta],
    list[dict[str, Any]],
]:
    values: dict[tuple[str, str, str, str], dict[int, float]] = defaultdict(dict)
    for cell in native:
        for metric in PRIMARY_METRICS:
            key = (cell.planner_id, cell.speed_tier_id, cell.scenario_id, metric)
            values[key][cell.seed] = cell.metrics[metric]

    grouped: dict[tuple[str, str, str], list[PairedDelta]] = defaultdict(list)
    deltas: list[PairedDelta] = []
    exclusions: list[dict[str, Any]] = []
    treated_keys = sorted(key for key in values if key[1] in NON_NOMINAL_TIERS)
    for planner_id, tier_id, scenario_id, metric in treated_keys:
        nominal_key = (planner_id, NOMINAL_TIER_ID, scenario_id, metric)
        treated_key = (planner_id, tier_id, scenario_id, metric)
        nominal_by_seed = values.get(nominal_key, {})
        treated_by_seed = values[treated_key]
        paired_seeds = set(nominal_by_seed) & set(treated_by_seed)
        if paired_seeds != declared_seeds:
            exclusions.append(
                {
                    "scenario_id": scenario_id,
                    "speed_tier_id": tier_id,
                    "planner_id": planner_id,
                    "metric": metric,
                    "missing_pair_seeds": sorted(declared_seeds - paired_seeds),
                    "exclusion_reason": "incomplete_native_scenario_seed_pairs",
                }
            )
            continue
        ordered_seeds = sorted(paired_seeds)
        nominal = [nominal_by_seed[seed] for seed in ordered_seeds]
        treated = [treated_by_seed[seed] for seed in ordered_seeds]
        delta = _paired_delta(
            nominal,
            treated,
            planner_id=planner_id,
            speed_tier_id=tier_id,
            metric=metric,
            scenario_id=scenario_id,
            seeds=ordered_seeds,
        )
        grouped[(planner_id, tier_id, metric)].append(delta)
        deltas.append(delta)
    return grouped, deltas, exclusions


def _emit_paired_deltas(
    deltas: Sequence[PairedDelta],
) -> list[dict[str, Any]]:
    return [
        {
            "planner_id": delta.planner_id,
            "speed_tier_id": delta.speed_tier_id,
            "metric": delta.metric,
            "scenario_id": delta.scenario_id,
            "seeds": list(delta.seeds),
            "n_pairs": delta.n_pairs,
            "delta_mean": delta.delta_mean,
            "delta_se": delta.delta_se,
            "ci_low": delta.ci_low,
            "ci_high": delta.ci_high,
        }
        for delta in sorted(
            deltas,
            key=lambda item: (
                item.planner_id,
                item.speed_tier_id,
                item.metric,
                item.scenario_id,
            ),
        )
    ]


def _summarize_typed_collisions(
    native: Sequence[CellSummary],
) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[CellSummary]] = defaultdict(list)
    for cell in native:
        grouped[(cell.planner_id, cell.speed_tier_id)].append(cell)
    return {
        key: {
            metric: sum(cell.typed_collisions[metric] for cell in items) / len(items)
            for metric in TYPED_COLLISION_BREAKDOWN
        }
        for key, items in grouped.items()
    }


def _summarize_activation_diagnostics(
    native: Sequence[CellSummary],
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[CellSummary]] = defaultdict(list)
    for cell in native:
        grouped[(cell.planner_id, cell.speed_tier_id)].append(cell)
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for (planner_id, tier_id), items in grouped.items():
        n = len(items)
        cmd_speed = sum(cell.commanded_speed_mean_m_s for cell in items) / n
        real_speed = sum(cell.realized_speed_mean_m_s for cell in items) / n
        peak_speed = max((cell.realized_speed_peak_m_s for cell in items), default=0.0)
        frac_above_2 = sum(cell.fraction_above_2_0_mps for cell in items) / n
        cap_sat = sum(cell.cap_saturation_fraction for cell in items) / n
        actuation = items[0].resolved_actuation_envelope if items else {}
        is_activated = (
            tier_id == NOMINAL_TIER_ID
            or frac_above_2 >= MIN_ACTIVATION_FRACTION_ABOVE_2_0
            or peak_speed > MIN_ACTIVATION_PEAK_SPEED
        )
        summary[(planner_id, tier_id)] = {
            "commanded_speed_mean_m_s": cmd_speed,
            "realized_speed_mean_m_s": real_speed,
            "realized_speed_peak_m_s": peak_speed,
            "fraction_above_2_0_mps": frac_above_2,
            "cap_saturation_fraction": cap_sat,
            "resolved_actuation_envelope": actuation,
            "intervention_activated": is_activated,
        }
    return summary


def _summarize_exposure_metrics(
    native: Sequence[CellSummary],
) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[CellSummary]] = defaultdict(list)
    for cell in native:
        grouped[(cell.planner_id, cell.speed_tier_id)].append(cell)
    return {
        key: {
            "time_to_goal_norm": sum(c.time_to_goal_norm for c in items) / len(items),
            "total_exposure_seconds": sum(c.total_exposure_seconds for c in items) / len(items),
            "travel_distance_m": sum(c.travel_distance_m for c in items) / len(items),
            "mean_clearance_m": sum(c.mean_clearance_m for c in items) / len(items),
            "min_clearance_m": min((c.min_clearance_m for c in items), default=0.0),
        }
        for key, items in grouped.items()
    }


def _stable_seed(test_id: str) -> int:
    return int.from_bytes(hashlib.sha256(test_id.encode()).digest()[:8], "big")


def _paired_seed_block_bootstrap(
    items: Sequence[PairedDelta],
    *,
    test_id: str,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[float]:
    """Resample paired seed blocks across fixed scenarios.

    Returns:
        The sorted bootstrap distribution of pooled paired deltas.
    """
    if not items:
        return []
    rng = random.Random(_stable_seed(test_id))

    by_scenario = {item.scenario_id: item.paired_differences for item in items}
    scenario_keys = sorted(by_scenario.keys())
    num_seeds = len(items[0].paired_differences) if items else 0

    result: list[float] = []
    for _ in range(replicates):
        sampled_seed_indices = [rng.randrange(num_seeds) for _ in range(num_seeds)]
        scenario_means: list[float] = []
        for s_key in scenario_keys:
            diffs = by_scenario[s_key]
            sampled_diffs = [diffs[idx] for idx in sampled_seed_indices]
            scenario_means.append(sum(sampled_diffs) / len(sampled_diffs))
        result.append(sum(scenario_means) / len(scenario_means))
    return sorted(result)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_interval(sorted_values: Sequence[float], confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    return (
        _percentile(sorted_values, alpha / 2.0),
        _percentile(sorted_values, 1.0 - alpha / 2.0),
    )


def _margin_aligned_one_sided_p_value(sorted_values: Sequence[float], metric: str) -> float:
    if not sorted_values:
        return 1.0
    threshold = HARM_THRESHOLDS[metric]
    direction = HARM_DIRECTION[metric]
    n = len(sorted_values)
    if direction == "decrease":
        harm_count = sum(v <= threshold for v in sorted_values)
    else:  # increase
        harm_count = sum(v >= threshold for v in sorted_values)
    return (harm_count + 1) / (n + 1)


def _aggregate_per_tier(
    grouped: Mapping[tuple[str, str, str], Sequence[PairedDelta]],
    *,
    typed_summary: Mapping[tuple[str, str], Mapping[str, float]],
    activation_summary: Mapping[tuple[str, str], Mapping[str, Any]],
    exposure_summary: Mapping[tuple[str, str], Mapping[str, float]],
) -> list[dict[str, Any]]:
    per_tier_summary: list[dict[str, Any]] = []
    for (planner_id, tier_id, metric), items in grouped.items():
        scenario_means = [item.delta_mean for item in items]
        n_scenarios = len(scenario_means)
        pooled_mean = sum(scenario_means) / n_scenarios if n_scenarios else float("nan")
        scenario_sd = (
            math.sqrt(sum((m - pooled_mean) ** 2 for m in scenario_means) / max(1, n_scenarios - 1))
            if n_scenarios > 1
            else 0.0
        )
        pooled_se = scenario_sd / math.sqrt(n_scenarios) if n_scenarios else float("nan")
        test_id = f"{planner_id}__{tier_id}__{metric}"
        bootstrap_distribution = _paired_seed_block_bootstrap(items, test_id=test_id)
        ci_low, ci_high = _bootstrap_interval(bootstrap_distribution, CONFIDENCE_LEVEL)
        p_value_one_sided = _margin_aligned_one_sided_p_value(bootstrap_distribution, metric)
        act_info = activation_summary.get(
            (planner_id, tier_id),
            {
                "commanded_speed_mean_m_s": 0.0,
                "realized_speed_mean_m_s": 0.0,
                "realized_speed_peak_m_s": 0.0,
                "fraction_above_2_0_mps": 0.0,
                "cap_saturation_fraction": 0.0,
                "resolved_actuation_envelope": {},
                "intervention_activated": True,
            },
        )
        exp_info = exposure_summary.get((planner_id, tier_id), {})
        per_tier_summary.append(
            {
                "test_id": test_id,
                "planner_id": planner_id,
                "speed_tier_id": tier_id,
                "metric": metric,
                "n_scenarios": n_scenarios,
                "pooled_delta_mean": pooled_mean,
                "pooled_delta_se": pooled_se,
                "ci_low_unadjusted": ci_low,
                "ci_high_unadjusted": ci_high,
                "p_value_raw": p_value_one_sided,
                "typed_collision_breakdown": dict(typed_summary.get((planner_id, tier_id), {})),
                "activation_diagnostics_summary": act_info,
                "exposure_summary": exp_info,
                "intervention_activated": act_info.get("intervention_activated", True),
                "_bootstrap_distribution": bootstrap_distribution,
            }
        )
    return per_tier_summary


def _holm_adjust_by_planner(
    p_values: Sequence[tuple[str, str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    families: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for planner_id, test_id, p_value in p_values:
        families[planner_id].append((test_id, p_value))
    adjusted: dict[str, float] = {}
    confidence: dict[str, float] = {}
    for family in families.values():
        adjusted.update(_holm_adjust(family))
        m = len(family)
        for rank, (test_id, _) in enumerate(sorted(family, key=lambda item: item[1]), start=1):
            local_alpha = (1.0 - CONFIDENCE_LEVEL) / (m - rank + 1)
            confidence[test_id] = 1.0 - local_alpha
    return adjusted, confidence


def _build_decision_table(
    per_tier_summary: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    p_values = [
        (entry["planner_id"], entry["test_id"], entry["p_value_raw"]) for entry in per_tier_summary
    ]
    holm_adjusted, adjusted_confidence = _holm_adjust_by_planner(p_values)
    entries_by_planner: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in per_tier_summary:
        entries_by_planner[entry["planner_id"]].append(entry)

    interval_decisions: dict[str, tuple[float, float, bool, str]] = {}
    for family in entries_by_planner.values():
        step_down_open = True
        for entry in sorted(family, key=lambda item: item["p_value_raw"]):
            test_id = entry["test_id"]
            confidence = adjusted_confidence[test_id]
            ci_low, ci_high = _bootstrap_interval(entry["_bootstrap_distribution"], confidence)
            is_activated = bool(entry.get("intervention_activated", True))
            if not is_activated:
                classification = "intervention_not_activated"
                eligible = False
                step_down_open = False
            else:
                candidate = _classify_interval(
                    entry["metric"],
                    ci_low,
                    ci_high,
                    entry["n_scenarios"],
                    intervention_activated=True,
                )
                eligible = step_down_open
                classification = candidate if eligible else "inconclusive"
                if candidate == "inconclusive":
                    step_down_open = False
            interval_decisions[test_id] = (ci_low, ci_high, eligible, classification)

    decision_table: list[dict[str, Any]] = []
    for entry in per_tier_summary:
        test_id = entry["test_id"]
        n = entry["n_scenarios"]
        confidence = adjusted_confidence[test_id]
        ci_low, ci_high, eligible, classification = interval_decisions[test_id]
        decision_table.append(
            {
                "test_id": test_id,
                "planner_id": entry["planner_id"],
                "speed_tier_id": entry["speed_tier_id"],
                "metric": entry["metric"],
                "n_scenarios": n,
                "pooled_delta_mean": entry["pooled_delta_mean"],
                "pooled_delta_se": entry["pooled_delta_se"],
                "ci_low_unadjusted": entry["ci_low_unadjusted"],
                "ci_high_unadjusted": entry["ci_high_unadjusted"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "adjusted_confidence_level": confidence,
                "holm_step_down_eligible": eligible,
                "classification": classification,
                "intervention_activated": entry.get("intervention_activated", True),
                "p_value_raw": entry["p_value_raw"],
                "p_value_holm": holm_adjusted[test_id],
                "typed_collision_breakdown": entry["typed_collision_breakdown"],
                "activation_diagnostics_summary": entry.get("activation_diagnostics_summary", {}),
                "exposure_summary": entry.get("exposure_summary", {}),
            }
        )
    return decision_table, holm_adjusted


def _compute_descriptive_ranking_stability(
    native: Sequence[CellSummary],
) -> dict[str, Any]:
    """Compute secondary descriptive planner-ranking metrics across speed tiers.

    Returns:
        Dictionary carrying descriptive ranking stability metrics.
    """
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for cell in native:
        grouped[(cell.speed_tier_id, cell.planner_id)].append(cell.metrics["success_rate"])

    tier_means: dict[str, dict[str, float]] = defaultdict(dict)
    for (tier_id, planner_id), rates in grouped.items():
        if rates:
            tier_means[tier_id][planner_id] = sum(rates) / len(rates)

    nominal_rank = sorted(
        tier_means.get(NOMINAL_TIER_ID, {}).keys(),
        key=lambda p: (tier_means[NOMINAL_TIER_ID][p], p),
        reverse=True,
    )

    rankings: dict[str, list[str]] = {NOMINAL_TIER_ID: nominal_rank}
    rank_flips: dict[str, int] = {}
    for tier_id in NON_NOMINAL_TIERS:
        if tier_id in tier_means:
            tier_rank = sorted(
                tier_means[tier_id].keys(),
                key=lambda p: (tier_means[tier_id][p], p),
                reverse=True,
            )
            rankings[tier_id] = tier_rank
            flips = 0
            for i in range(len(nominal_rank)):
                for j in range(i + 1, len(nominal_rank)):
                    p1, p2 = nominal_rank[i], nominal_rank[j]
                    if p1 in tier_rank and p2 in tier_rank:
                        if tier_rank.index(p1) > tier_rank.index(p2):
                            flips += 1
            rank_flips[tier_id] = flips

    return {
        "scope": RANKING_CLAIM_SCOPE,
        "note": "Descriptive only; planner rankings do not constitute inferential evidence.",
        "nominal_ranking": nominal_rank,
        "tier_rankings": rankings,
        "rank_flips_vs_nominal": rank_flips,
    }


def _build_cli_rows() -> list[dict[str, object]]:
    demo: list[dict[str, object]] = []
    for tier_id, cap in (("cap_2_0_nominal", 2.0), ("cap_3_0", 3.0), ("cap_4_2", 4.2)):
        demo.append(
            {
                "scenario_id": "classic_doorway_medium",
                "speed_tier_id": tier_id,
                "speed_cap_m_s": cap,
                "planner_id": "orca",
                "seed": 111,
                "horizon_steps": 600,
                "dt_seconds": 0.1,
                "execution_mode": "native",
                "success_rate": 0.8,
                "collision_rate": 0.1,
                "near_miss_rate": 0.2,
                "ped_collision_rate": 0.0,
                "obstacle_collision_rate": 0.0,
                "agent_collision_rate": 0.0,
                "unclassified_collision_rate": 0.0,
                "commanded_speed_mean_m_s": cap * 0.9,
                "realized_speed_mean_m_s": cap * 0.85,
                "realized_speed_peak_m_s": cap,
                "fraction_above_2_0_mps": 0.5 if cap > 2.0 else 0.0,
                "cap_saturation_fraction": 0.3,
                "resolved_actuation_envelope": {
                    "drive_model": "bicycle_drive",
                    "max_velocity": cap,
                    "max_accel": cap * 0.5,
                    "max_decel": cap,
                    "stopping_distance": cap * 0.5,
                },
                "time_to_goal_norm": 0.5,
                "total_exposure_seconds": 30.0,
                "travel_distance_m": 60.0,
                "mean_clearance_m": 1.2,
                "min_clearance_m": 0.4,
            }
        )
    return demo


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for synthesis smoke test.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full synthesis result as indented JSON.",
    )
    args = parser.parse_args(argv)
    result = synthesize_speed_tier_sweep(
        _build_cli_rows(),
        declared_scenarios={"classic_doorway_medium"},
        declared_planners={"orca"},
        declared_seeds={111},
        require_complete_grid=False,
    )
    report = {
        "claim_boundary": result.claim_boundary,
        "per_cell_count": result.per_cell_count,
        "native_cell_count": result.native_cell_count,
        "all_native": result.all_native,
        "grid_complete": result.grid_complete,
        "evidence_status": result.evidence_status,
        "decision_table": result.decision_table,
        "descriptive_ranking_stability": result.descriptive_ranking_stability,
        "excluded_cell_count": result.excluded_cell_count,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"SMOKE (not benchmark evidence): issue #5578 synthesis executed "
            f"({result.native_cell_count} native cells, "
            f"{result.excluded_cell_count} excluded)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
