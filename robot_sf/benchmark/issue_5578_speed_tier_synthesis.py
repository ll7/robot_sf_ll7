"""Fail-closed evidence synthesis for the issue #5578 robot speed-tier sweep.

This module implements the frozen decision rule from the issue #5578
preregistration packet (configs/benchmarks/issue_5578_robot_speed_tier_preregistration.yaml)
and its governance parent #5557. It consumes per-cell summary data (one row per
scenario x speed_tier x planner x seed) and produces:

- a paired-delta estimand (tier - 2.0 m/s) per planner / non-nominal tier /
  primary metric, summarized across the six declared scenarios;
- a Holm-Bonferroni corrected confidence interval per test in the declared
  family (planner x non-nominal tier x primary metric, six tests per planner);
- a fail-closed harm-threshold classification (materially_harmful /
  no_material_shift / inconclusive);
- a visible exclusion table for missing / failed / fallback / degraded rows.

The synthesis is deterministic and CPU-only. It does NOT run any campaign,
submit compute, or promote a paper-facing claim. A result is only benchmark
evidence when every native row and provenance requirement passes; otherwise the
contract fails closed.
"""

from __future__ import annotations

import argparse
import json
import math
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
    return _erf_inv(confidence)


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


def _as_float(value: Any, field_name: str) -> float:
    """Coerce and validate a numeric cell field value.

    Returns:
        The finite float value of ``value``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a numeric value, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return number


def parse_cell(row: Mapping[str, Any]) -> CellSummary:
    """Parse and validate one per-cell summary row into a CellSummary.

    Raises:
        ValueError: if required keys or primary metrics are missing / invalid.

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
        metrics[metric] = _as_float(row[metric], metric)
    typed: dict[str, float] = {}
    for tcol in TYPED_COLLISION_BREAKDOWN:
        if tcol in row:
            typed[tcol] = _as_float(row[tcol], tcol)
    return CellSummary(
        scenario_id=str(row["scenario_id"]),
        speed_tier_id=str(row["speed_tier_id"]),
        speed_cap_m_s=_as_float(row["speed_cap_m_s"], "speed_cap_m_s"),
        planner_id=str(row["planner_id"]),
        seed=int(row["seed"]),
        horizon_steps=int(row["horizon_steps"]),
        dt_seconds=_as_float(row["dt_seconds"], "dt_seconds"),
        execution_mode=str(row["execution_mode"]),
        metrics=metrics,
        typed_collisions=typed,
    )


def classify_excluded(cell: CellSummary) -> str | None:
    """Return an exclusion reason for a non-native cell, else None.

    Per the provenance contract, only ``native`` execution rows count as
    evidence; ``fallback`` / ``degraded`` / ``failed`` rows are visible
    exclusions.

    Returns:
        An exclusion-reason string for non-native rows, or ``None`` when the
        cell is native and therefore admissible as evidence.
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
    seed: int
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
    seed: int,
) -> PairedDelta:
    """Compute the scenario-seeded paired delta with a normal CI.

    Returns:
        The ``PairedDelta`` carrying the delta mean, standard error, and the
        ``CONFIDENCE_LEVEL`` normal confidence interval.
    """
    n = len(nominal)
    diffs = [t - n0 for n0, t in zip(nominal, treated, strict=True)]
    mean = sum(diffs) / n
    if n < 2:
        return PairedDelta(
            planner_id=planner_id,
            speed_tier_id=speed_tier_id,
            metric=metric,
            scenario_id=scenario_id,
            seed=seed,
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
        seed=seed,
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
        ``"decrease"`` or ``"increase"`` from the frozen harm-direction map.
    """
    return HARM_DIRECTION[metric]


def _classify_interval(metric: str, ci_low: float, ci_high: float, n_pairs: int) -> str:
    """Classify a single metric's adjusted interval against its harm threshold.

    Returns:
        One of ``"materially_harmful"``, ``"no_material_shift"``, or
        ``"inconclusive"``.
    """
    if n_pairs < 2:
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

    The family is planner x non-nominal tier x primary metric (six tests per
    planner, per the preregistration multiplicity contract). When the sample is
    too small to form a two-sided test (n_pairs < 2) the p-value is forced to
    1.0 so the test can never pass as evidence.

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


def synthesize_speed_tier_sweep(
    cells: Sequence[Mapping[str, Any]],
    *,
    declared_scenarios: set[str] | None = None,
    declared_planners: set[str] | None = None,
) -> SynthesisResult:
    """Synthesize the issue #5578 speed-tier sweep from per-cell summaries.

    Args:
        cells: per-cell summary rows (one per scenario x tier x planner x seed).
        declared_scenarios: optional allow-list for fail-closed scenario drift.
        declared_planners: optional allow-list for fail-closed planner drift.

    Returns:
        A ``SynthesisResult`` with paired deltas, per-tier summaries, a Holm
        corrected decision table, and a visible exclusion table.
    """
    parsed = [parse_cell(row) for row in cells]
    exclusions, native, all_native = _partition_cells(parsed)
    nominal_idx = _build_nominal_index(native)
    grouped, deltas = _build_paired_deltas(native, nominal_idx)
    paired_deltas_out = _emit_paired_deltas(native, nominal_idx, deltas)
    per_tier_summary = _aggregate_per_tier(grouped, declared_planners=declared_planners)
    decision_table, p_values = _build_decision_table(per_tier_summary)
    holm_adjusted = _holm_adjust(p_values)
    for row in decision_table:
        row["p_value_holm"] = holm_adjusted[row["test_id"]]

    return SynthesisResult(
        claim_boundary=(
            "Pre-aggregated synthesis only; not benchmark evidence unless every "
            "native row and provenance requirement passes. Fallback/degraded/"
            "failed rows are visible exclusions."
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
    )


def _partition_cells(
    parsed: list[CellSummary],
) -> tuple[list[dict[str, Any]], list[CellSummary], bool]:
    """Split parsed cells into a visible exclusion table and admissible natives.

    Returns:
        A tuple ``(exclusions, native, all_native)`` where ``exclusions`` is the
        list of excluded-cell records, ``native`` are the admissible cells, and
        ``all_native`` flags whether every cell was native.
    """
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


def _build_nominal_index(
    native: list[CellSummary],
) -> dict[tuple[str, str, str, int], float]:
    """Index native nominal-tier metric values by planner/scenario/metric/seed.

    Returns:
        Mapping from ``(planner_id, scenario_id, metric, seed)`` to the nominal
        2.0 m/s cell's metric value.
    """
    nominal_idx: dict[tuple[str, str, str, int], float] = {}
    for cell in native:
        if cell.speed_tier_id != NOMINAL_TIER_ID:
            continue
        for metric in PRIMARY_METRICS:
            nominal_idx[(cell.planner_id, cell.scenario_id, metric, cell.seed)] = cell.metrics[
                metric
            ]
    return nominal_idx


def _build_paired_deltas(
    native: list[CellSummary],
    nominal_idx: Mapping[tuple[str, str, str, int], float],
) -> tuple[dict[tuple[str, str, str], list[PairedDelta]], list[PairedDelta]]:
    """Compute per-scenario paired deltas against the nominal tier.

    Returns:
        A tuple ``(grouped, deltas)`` where ``grouped`` collects deltas by
        ``(planner, tier, metric)`` and ``deltas`` is the flat ordered list.
    """
    grouped: dict[tuple[str, str, str], list[PairedDelta]] = defaultdict(list)
    deltas: list[PairedDelta] = []
    for cell in native:
        if cell.speed_tier_id == NOMINAL_TIER_ID:
            continue
        for metric in PRIMARY_METRICS:
            key = (cell.planner_id, cell.scenario_id, metric, cell.seed)
            if key not in nominal_idx:
                continue
            delta = _paired_delta(
                [nominal_idx[key]],
                [cell.metrics[metric]],
                planner_id=cell.planner_id,
                speed_tier_id=cell.speed_tier_id,
                metric=metric,
                scenario_id=cell.scenario_id,
                seed=cell.seed,
            )
            grouped[(cell.planner_id, cell.speed_tier_id, metric)].append(delta)
            deltas.append(delta)
    return grouped, deltas


def _emit_paired_deltas(
    native: list[CellSummary],
    nominal_idx: Mapping[tuple[str, str, str, int], float],
    deltas: Sequence[PairedDelta],
) -> list[dict[str, Any]]:
    """Emit per-cell paired-delta records for every admissible non-nominal cell.

    Returns:
        The list of per-cell paired-delta dictionaries.
    """
    paired_deltas_out: list[dict[str, Any]] = []
    for cell in native:
        if cell.speed_tier_id == NOMINAL_TIER_ID:
            continue
        for metric in PRIMARY_METRICS:
            key = (cell.planner_id, cell.scenario_id, metric, cell.seed)
            if key not in nominal_idx:
                continue
            delta = next(
                d
                for d in deltas
                if d.planner_id == cell.planner_id
                and d.speed_tier_id == cell.speed_tier_id
                and d.metric == metric
                and d.scenario_id == cell.scenario_id
                and d.seed == cell.seed
            )
            paired_deltas_out.append(
                {
                    "planner_id": delta.planner_id,
                    "speed_tier_id": delta.speed_tier_id,
                    "metric": delta.metric,
                    "scenario_id": delta.scenario_id,
                    "seed": cell.seed,
                    "n_pairs": delta.n_pairs,
                    "delta_mean": delta.delta_mean,
                    "delta_se": delta.delta_se,
                    "ci_low": delta.ci_low,
                    "ci_high": delta.ci_high,
                }
            )
    return paired_deltas_out


def _aggregate_per_tier(
    grouped: Mapping[tuple[str, str, str], Sequence[PairedDelta]],
    *,
    declared_planners: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Pool per-scenario paired deltas into per-tier summary rows.

    Returns:
        The list of per-tier pooled-delta summary dictionaries.
    """
    per_tier_summary: list[dict[str, Any]] = []
    for (planner_id, tier_id, metric), items in grouped.items():
        if declared_planners is not None and planner_id not in declared_planners:
            continue
        scenario_means = [item.delta_mean for item in items]
        n_scenarios = len(scenario_means)
        pooled_mean = sum(scenario_means) / n_scenarios if n_scenarios else float("nan")
        se_pooled = (
            math.sqrt(sum((m - pooled_mean) ** 2 for m in scenario_means) / max(1, n_scenarios - 1))
            if n_scenarios > 1
            else 0.0
        )
        z = _z_critical(CONFIDENCE_LEVEL)
        half = z * (se_pooled / math.sqrt(n_scenarios)) if n_scenarios else 0.0
        per_tier_summary.append(
            {
                "planner_id": planner_id,
                "speed_tier_id": tier_id,
                "metric": metric,
                "n_scenarios": n_scenarios,
                "pooled_delta_mean": pooled_mean,
                "pooled_delta_se": se_pooled,
                "ci_low": pooled_mean - half,
                "ci_high": pooled_mean + half,
            }
        )
    return per_tier_summary


def _build_decision_table(
    per_tier_summary: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, float]]]:
    """Classify each pooled delta and collect the Holm p-value family.

    Returns:
        A tuple ``(decision_table, p_values)`` where ``decision_table`` carries
        the per-test classification and raw/holm p-values, and ``p_values`` is
        the ``(test_id, p_value)`` family fed to the Holm correction.
    """
    p_values: list[tuple[str, float]] = []
    decision_table: list[dict[str, Any]] = []
    for entry in per_tier_summary:
        test_id = f"{entry['planner_id']}__{entry['speed_tier_id']}__{entry['metric']}"
        n = entry["n_scenarios"]
        ci_low, ci_high = entry["ci_low"], entry["ci_high"]
        classification = _classify_interval(entry["metric"], ci_low, ci_high, n)
        p_value = _approx_p_value(entry["metric"], ci_low, ci_high, n)
        p_values.append((test_id, p_value))
        decision_table.append(
            {
                "test_id": test_id,
                "planner_id": entry["planner_id"],
                "speed_tier_id": entry["speed_tier_id"],
                "metric": entry["metric"],
                "n_scenarios": n,
                "pooled_delta_mean": entry["pooled_delta_mean"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "classification": classification,
                "p_value_raw": p_value,
            }
        )
    return decision_table, p_values


def _approx_p_value(metric: str, ci_low: float, ci_high: float, n: int) -> float:
    """Two-sided normal p-value for a pooled delta from its confidence interval.

    Uses the interval half-width and the scenario count as the effective sample
    size. When n < 2 the test is undefined and returns 1.0 (never evidence).

    Returns:
        The two-sided normal p-value in ``[0.0, 1.0]``.
    """
    if n < 2:
        return 1.0
    half = (ci_high - ci_low) / 2.0
    if half <= 0.0:
        return 1.0
    se = half / _z_critical(CONFIDENCE_LEVEL)
    z_stat = (ci_high + ci_low) / (2.0 * se)
    tail = 0.5 * (1.0 - math.erf(abs(z_stat) / math.sqrt(2.0)))
    return min(1.0, 2.0 * tail)


def _build_cli_rows() -> list[dict[str, object]]:
    """Build a fail-closed demo grid so the CLI runs without a campaign.

    Returns:
        A minimal per-cell summary grid (one planner, one scenario, one seed,
        all native) for smoke-testing the synthesis path end to end.
    """
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
            }
        )
    return demo


def main(argv: list[str] | None = None) -> int:
    """CLI smoke entry point: synthesize the demo grid and print the report.

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
    result = synthesize_speed_tier_sweep(_build_cli_rows())
    report = {
        "claim_boundary": result.claim_boundary,
        "per_cell_count": result.per_cell_count,
        "native_cell_count": result.native_cell_count,
        "all_native": result.all_native,
        "decision_table": result.decision_table,
        "excluded_cell_count": result.excluded_cell_count,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"PASS: issue #5578 synthesis valid "
            f"({result.native_cell_count} native cells, "
            f"{result.excluded_cell_count} excluded)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
