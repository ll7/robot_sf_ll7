"""Hierarchical paired release analysis engine for issue #5351.

This is the statistical layer the benchmark protocol already declares but the
``0.0.2`` release did not deliver: paired estimands on matched
planner-scenario-seed cells, scenario-family cluster (hierarchical) bootstrap
intervals, a predeclared multiplicity policy, practical-effect thresholds,
timeout-censored completion-time summaries, and exposure-normalized near-miss
rates.  It is analysis ON TOP of frozen metrics: it does not redefine any
benchmark metric, and it cannot promote a benchmark, release, paper, or
dissertation claim on its own.

The engine is pure and deterministic given a seeded RNG.  It is designed to run
over the typed-ledger successor rows produced by the #4364 release cut
(``EpisodeEventLedger.v2`` rows).  Until those rows exist it fail-closes: the
runner refuses placeholder/empty input and, even when real rows are supplied,
the emitted claim gate stays ``blocked_analysis_not_run`` until the analysis is
actually executed and reviewed.  Synthetic-fixture unit tests prove the
estimators compute the intended values; they are not benchmark evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.event_ledger import (
    EPISODE_EVENT_LEDGER_SCHEMA_VERSION,
)
from robot_sf.benchmark.finite_checks import require_finite_scalar
from robot_sf.benchmark.hierarchical_paired_release_inputs import (
    INPUTS_READY_ANALYSIS_NOT_RUN,
    evaluate_hierarchical_paired_release_inputs,
    validate_hierarchical_paired_release_input_manifest,
)
from robot_sf.errors import RobotSfError

if TYPE_CHECKING:
    from pathlib import Path

HIERARCHICAL_PAIRED_RELEASE_ANALYSIS_REPORT_SCHEMA = (
    "hierarchical_paired_release_analysis_report.v1"
)
CLAIM_GATE_BLOCKED_ANALYSIS_NOT_RUN = "blocked_analysis_not_run"
CLAIM_GATE_BLOCKED_REVIEW_PENDING = "blocked_review_pending"
EVIDENCE_STATUS_NOT_BENCHMARK_EVIDENCE = "not_benchmark_evidence"
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
# Practical-effect thresholds are predeclared per the protocol.  A risk
# difference below ``min_risk_difference`` is treated as practically null even
# when its interval excludes zero, so a statistically separable but tiny effect
# cannot be reported as a meaningful safety improvement.
DEFAULT_MIN_RISK_DIFFERENCE = 0.02
# Near-miss exposure is normalized per unit interaction opportunity (one
# matched cell's exposure window) by default; callers may override.
DEFAULT_EXPOSURE_OPPORTUNITY = 1.0
EXPOSURE_DIMENSIONS = ("time", "distance", "opportunity")


class HierarchicalPairedReleaseAnalysisError(RobotSfError, ValueError):
    """Raised when the #5351 analysis is invoked on unsafe or incomplete input."""


@dataclass(frozen=True, slots=True)
class MatchedCell:
    """One paired planner-scenario-seed cell built from successor ledger rows.

    Attributes:
        scenario_id: Scenario identifier shared by both planner arms.
        scenario_family: Coarser family used as the cluster unit for the
            hierarchical bootstrap.  Families cluster scenarios; seeds do not.
        seed: Shared random seed so the two arms face identical conditions.
        planner_a/planner_b: Paired planner names.
        collision_a/collision_b: Exact collision outcome (0/1) for each arm.
        near_miss_a/near_miss_b: Surrogate near-miss outcome (0/1) per arm.
        timeout_a/timeout_b: Exact timeout outcome (0/1) per arm.
        completion_time_a/completion_time_b: Observed completion time.  When
            the arm timed out the time is treated as right-censored at the
            horizon and ``censored_a``/``censored_b`` is set.
        exposure_a/exposure_b: Labeled interaction exposure dimensions over
            which near-miss counts arose.
    """

    scenario_id: str
    scenario_family: str
    seed: int
    planner_a: str
    planner_b: str
    collision_a: int
    collision_b: int
    near_miss_a: int
    near_miss_b: int
    timeout_a: int
    timeout_b: int
    completion_time_a: float
    completion_time_b: float
    censored_a: bool
    censored_b: bool
    exposure_a: dict[str, float]
    exposure_b: dict[str, float]


@dataclass(frozen=True, slots=True)
class AnalysisPolicy:
    """Predeclared analysis policy: multiplicity, effect, and resampling settings.

    Thresholds are fixed up front so post-hoc tuning cannot manufacture a
    separable effect.  ``min_risk_difference`` must be strictly positive: a zero
    practical-effect threshold collapses the statistical and practical
    boundaries onto the same interval-overlap test the release already
    (mis)used.
    """

    confidence: float = DEFAULT_CONFIDENCE
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    bootstrap_seed: int = 20260718
    min_risk_difference: float = DEFAULT_MIN_RISK_DIFFERENCE
    exposure_opportunity: float = DEFAULT_EXPOSURE_OPPORTUNITY


@dataclass(frozen=True, slots=True)
class PairedEffect:
    """Paired risk-difference and odds-risk-ratio estimand for one comparison.

    The risk difference is the per-cell difference in the binary outcome rate;
    the odds-risk ratio is the ratio of arm-A to arm-B odds.  Both are computed
    by exact arithmetic over the matched cells (no asymptotic approximation),
    and their intervals come from the hierarchical bootstrap.
    """

    comparison: str
    n_cells: int
    risk_a: float
    risk_b: float
    risk_difference: float
    risk_difference_ci_low: float
    risk_difference_ci_high: float
    odds_a: float
    odds_b: float
    odds_ratio: float
    odds_ratio_ci_low: float
    odds_ratio_ci_high: float


@dataclass(frozen=True, slots=True)
class MultiplicityDecision:
    """Holm step-down adjusted p-value and rejection verdict for one comparison."""

    comparison: str
    raw_p_value: float
    adjusted_p_value: float
    rejected: bool


@dataclass(frozen=True, slots=True)
class CompletionTimeSummary:
    """Timeout-aware (right-censored) completion-time summary per planner.

    Timeouts are failures, not slow successes: a timed-out arm contributes its
    horizon as a lower bound and is flagged censored so the mean is not reported
    as if the episode completed.
    """

    planner: str
    n_total: int
    n_observed: int
    n_censored: int
    mean_observed: float
    median_observed: float
    censoring_rate: float


@dataclass(frozen=True, slots=True)
class NearMissExposureSummary:
    """Exposure-normalized near-miss rate per planner.

    Raw near-miss counts are not comparable across arms with differing
    exposure; this reports near-miss per unit opportunity so an arm that simply
    ran longer is not penalized for accumulating more events.
    """

    planner: str
    dimension: str
    total_near_miss: int
    total_exposure: float
    normalized_rate: float


def build_matched_cells_from_ledger_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    planner_pair: tuple[str, str],
    family_of: Mapping[str, str] | None = None,
) -> list[MatchedCell]:
    """Pair ``EpisodeEventLedger.v2`` rows into matched planner-scenario-seed cells.

    Each scenario-seed must have exactly one row per planner arm.  Unmatched or
    duplicate rows raise so a partial release cannot silently shrink the
    denominator.

    Args:
        rows: Typed-ledger successor rows (each an ``EpisodeEventLedger.v2`` mapping).
        planner_pair: The ``(planner_a, planner_b)`` arms to pair.
        family_of: Optional mapping from ``scenario_id`` to ``scenario_family``.
            When omitted the scenario id is its own family.

    Returns:
        Sorted list of paired cells.
    """

    planner_a, planner_b = planner_pair
    if planner_a == planner_b:
        raise HierarchicalPairedReleaseAnalysisError(
            "planner_pair arms must differ for a paired comparison"
        )
    if not rows:
        raise HierarchicalPairedReleaseAnalysisError(
            "cannot build matched cells from empty successor row set"
        )
    buckets: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        _validate_ledger_row(row, index=index)
        planner = str(row["planner"])
        if planner not in planner_pair:
            continue
        key = (str(row["scenario_id"]), _as_int(row["seed"]), planner)
        if key in buckets:
            raise HierarchicalPairedReleaseAnalysisError(
                f"duplicate ledger row for scenario/seed/planner {key}"
            )
        buckets[key] = row
    cells: list[MatchedCell] = []
    arm_a = {
        (sid, seed): row for (sid, seed, planner), row in buckets.items() if planner == planner_a
    }
    arm_b = {
        (sid, seed): row for (sid, seed, planner), row in buckets.items() if planner == planner_b
    }
    common = sorted(set(arm_a) & set(arm_b))
    missing_a = sorted(set(arm_b) - set(arm_a))
    missing_b = sorted(set(arm_a) - set(arm_b))
    if missing_a or missing_b:
        raise HierarchicalPairedReleaseAnalysisError(
            "unmatched successor rows prevent pairing: "
            f"missing {planner_a} for {missing_a[:3]}, missing {planner_b} for {missing_b[:3]}"
        )
    family_resolver = dict(family_of) if family_of else {}
    for scenario_id, seed in common:
        row_a = arm_a[(scenario_id, seed)]
        row_b = arm_b[(scenario_id, seed)]
        cells.append(
            MatchedCell(
                scenario_id=scenario_id,
                scenario_family=family_resolver.get(scenario_id, scenario_id),
                seed=seed,
                planner_a=planner_a,
                planner_b=planner_b,
                collision_a=_binary_outcome(row_a, "collision"),
                collision_b=_binary_outcome(row_b, "collision"),
                near_miss_a=_count_outcome(row_a, "near_miss"),
                near_miss_b=_count_outcome(row_b, "near_miss"),
                timeout_a=_binary_outcome(row_a, "timeout"),
                timeout_b=_binary_outcome(row_b, "timeout"),
                completion_time_a=_completion_time(row_a),
                completion_time_b=_completion_time(row_b),
                censored_a=bool(_binary_outcome(row_a, "timeout")),
                censored_b=bool(_binary_outcome(row_b, "timeout")),
                exposure_a=_exposure(row_a),
                exposure_b=_exposure(row_b),
            )
        )
    return cells


def estimate_paired_effects(
    cells: Sequence[MatchedCell],
    *,
    outcome: str,
    policy: AnalysisPolicy | None = None,
) -> PairedEffect:
    """Estimate the paired risk-difference and odds-risk-ratio for one outcome.

    The point estimates are exact; the confidence interval is a scenario-family
    cluster bootstrap (nonparametric, resampling whole families with
    replacement) so within-family correlation from shared scenarios does not
    artificially narrow the interval.

    Args:
        cells: Matched planner-scenario-seed cells.
        outcome: ``"collision"``, ``"near_miss"``, or ``"timeout"``.
        policy: Predeclared analysis policy.  Defaults to :class:`AnalysisPolicy`.

    Returns:
        The paired effect estimand with risk-difference and odds-risk-ratio point
        estimates and hierarchical-bootstrap confidence intervals.
    """

    resolved = policy or AnalysisPolicy()
    families = _ordered_families(cells)
    return _estimate_paired_effects_with_clusters(
        cells,
        outcome=outcome,
        policy=resolved,
        clusters=families,
    )


def _estimate_paired_effects_with_clusters(
    cells: Sequence[MatchedCell],
    *,
    outcome: str,
    policy: AnalysisPolicy,
    clusters: Sequence[Sequence[int]],
) -> PairedEffect:
    """Estimate one outcome using an explicitly selected bootstrap cluster unit.

    Returns:
        The paired effect with confidence intervals from the selected clusters.
    """

    outcomes_a, outcomes_b = _select_outcome(cells, outcome)
    _require_paired_outcomes(outcomes_a, outcomes_b, outcome=outcome)
    comparison = f"{cells[0].planner_a}:{cells[0].planner_b}:{outcome}"
    diff_samples, ratio_samples = _cluster_bootstrap_paired(
        outcomes_a=outcomes_a,
        outcomes_b=outcomes_b,
        families=clusters,
        policy=policy,
    )
    ci_low, ci_high = _percentile_interval(diff_samples, policy.confidence)
    ratio_low, ratio_high = _percentile_interval(ratio_samples, policy.confidence)
    risk_a = float(np.mean(outcomes_a))
    risk_b = float(np.mean(outcomes_b))
    odds_a = _empirical_odds(risk_a)
    odds_b = _empirical_odds(risk_b)
    odds_ratio = _odds_ratio(risk_a, risk_b)
    return PairedEffect(
        comparison=comparison,
        n_cells=len(cells),
        risk_a=risk_a,
        risk_b=risk_b,
        risk_difference=risk_a - risk_b,
        risk_difference_ci_low=ci_low,
        risk_difference_ci_high=ci_high,
        odds_a=odds_a,
        odds_b=odds_b,
        odds_ratio=odds_ratio,
        odds_ratio_ci_low=ratio_low,
        odds_ratio_ci_high=ratio_high,
    )


def _sensitivity_analysis(
    cells: Sequence[MatchedCell],
    *,
    outcomes: Sequence[str],
    policy: AnalysisPolicy,
) -> list[dict[str, Any]]:
    """Emit seed-level and family-level bootstrap sensitivity rows.

    The primary interval is family-clustered.  This explicit companion table
    makes the declared seed-vs-family sensitivity protocol auditable without
    changing the primary estimand or treating the sensitivity result as
    benchmark evidence.

    Returns:
        One row per outcome with separate seed-level and family-level summaries.
    """

    cluster_levels = {
        "seed": _ordered_clusters(cells, key=lambda cell: str(cell.seed)),
        "family": _ordered_families(cells),
    }
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        level_results: dict[str, dict[str, Any]] = {}
        for level, clusters in cluster_levels.items():
            effect = _estimate_paired_effects_with_clusters(
                cells,
                outcome=outcome,
                policy=policy,
                clusters=clusters,
            )
            level_results[level] = {
                "n_clusters": len(clusters),
                "risk_difference": effect.risk_difference,
                "risk_difference_ci": [
                    effect.risk_difference_ci_low,
                    effect.risk_difference_ci_high,
                ],
                "odds_ratio": effect.odds_ratio,
                "odds_ratio_ci": [effect.odds_ratio_ci_low, effect.odds_ratio_ci_high],
            }
        rows.append(
            {
                "outcome": outcome,
                "planner_pair": [cells[0].planner_a, cells[0].planner_b],
                "seed_level": level_results["seed"],
                "family_level": level_results["family"],
            }
        )
    return rows


def holm_multiplicity(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> list[MultiplicityDecision]:
    """Apply the Holm step-down multiplicity correction to a family of p-values.

    The family is the set of declared comparisons; this controls the
    family-wise error rate without Bonferroni's conservatism.  Adjusted
    p-values are monotonized so the order of the raw values cannot flip a
    rejection.

    Args:
        p_values: Raw per-comparison p-values forming one family.
        alpha: Family-wise significance level.

    Returns:
        One :class:`MultiplicityDecision` per input p-value, in input order.
    """

    if not 0.0 < alpha < 1.0:
        raise HierarchicalPairedReleaseAnalysisError("alpha must be in the open interval (0, 1)")
    if not p_values:
        return []
    entries = sorted(
        ((float(p), index) for index, p in enumerate(p_values)),
        key=lambda item: item[0],
    )
    m = len(entries)
    decisions: list[tuple[int, MultiplicityDecision]] = []
    running_max = 0.0
    for rank, (raw_p, original_index) in enumerate(entries):
        adjusted = min(1.0, raw_p * (m - rank))
        running_max = max(running_max, adjusted)
        decisions.append(
            (
                original_index,
                MultiplicityDecision(
                    comparison=f"comparison_{original_index}",
                    raw_p_value=raw_p,
                    adjusted_p_value=running_max,
                    rejected=raw_p <= alpha / (m - rank) if rank == 0 else running_max <= alpha,
                ),
            )
        )
    # Recompute rejection flags from the final monotonized adjusted values so
    # the step-down sequence is internally consistent regardless of input order.
    finalized: list[MultiplicityDecision] = [None] * m  # type: ignore[list-item]
    for original_index, decision in decisions:
        rejected = decision.adjusted_p_value <= alpha
        finalized[original_index] = MultiplicityDecision(
            comparison=decision.comparison,
            raw_p_value=decision.raw_p_value,
            adjusted_p_value=decision.adjusted_p_value,
            rejected=rejected,
        )
    return finalized


def censored_completion_time(
    cells: Sequence[MatchedCell],
    *,
    horizon: float,
) -> list[CompletionTimeSummary]:
    """Summarize completion time per planner, treating timeouts as right-censored.

    Args:
        cells: Matched cells.
        horizon: The episode time cap.  Censored times are clamped to this value
            so a timed-out run cannot inflate the observed-time distribution.

    Returns:
        One :class:`CompletionTimeSummary` per planner arm (A then B).
    """

    require_finite_scalar("horizon", horizon)
    if horizon <= 0.0:
        raise HierarchicalPairedReleaseAnalysisError("horizon must be positive")
    summaries: list[CompletionTimeSummary] = []
    for planner, time_attr, censored_attr in (
        (cells[0].planner_a, "completion_time_a", "censored_a"),
        (cells[0].planner_b, "completion_time_b", "censored_b"),
    ):
        observed_times = []
        n_censored = 0
        for cell in cells:
            raw_time = float(getattr(cell, time_attr))
            is_censored = bool(getattr(cell, censored_attr))
            if is_censored:
                n_censored += 1
            else:
                observed_times.append(max(min(raw_time, horizon), 0.0))
        observed = np.asarray(observed_times, dtype=np.float64)
        n_total = len(cells)
        n_observed = n_total - n_censored
        summaries.append(
            CompletionTimeSummary(
                planner=planner,
                n_total=n_total,
                n_observed=n_observed,
                n_censored=n_censored,
                mean_observed=float(np.mean(observed)) if observed.size else 0.0,
                median_observed=float(np.median(observed)) if observed.size else 0.0,
                censoring_rate=n_censored / n_total if n_total else 0.0,
            )
        )
    return summaries


def normalized_near_miss_exposure(
    cells: Sequence[MatchedCell],
    *,
    policy: AnalysisPolicy | None = None,
) -> list[NearMissExposureSummary]:
    """Report near-miss counts normalized per unit interaction exposure.

    Exposure is the matched-cell opportunity window (time, distance, or
    interaction count) summed across cells; the rate is total near-miss events
    divided by total exposure, so arms with longer runs are not over-penalized.

    Args:
        cells: Matched cells.
        policy: Predeclared analysis policy supplying the exposure opportunity.

    Returns:
        One :class:`NearMissExposureSummary` per planner arm (A then B).
    """

    resolved = policy or AnalysisPolicy()
    opportunity = resolved.exposure_opportunity
    require_finite_scalar("exposure_opportunity", opportunity)
    if opportunity <= 0.0:
        raise HierarchicalPairedReleaseAnalysisError("exposure_opportunity must be positive")
    summaries: list[NearMissExposureSummary] = []
    for planner, miss_attr, exposure_attr in (
        (cells[0].planner_a, "near_miss_a", "exposure_a"),
        (cells[0].planner_b, "near_miss_b", "exposure_b"),
    ):
        total_near_miss = sum(int(getattr(cell, miss_attr)) for cell in cells)
        for dimension in EXPOSURE_DIMENSIONS:
            exposures = [float(getattr(cell, exposure_attr)[dimension]) for cell in cells]
            if any(not math.isfinite(exposure) or exposure <= 0.0 for exposure in exposures):
                raise HierarchicalPairedReleaseAnalysisError(
                    f"{planner} has missing, non-finite, or non-positive {dimension} exposure"
                )
            total_exposure = sum(exposures)
            if not math.isfinite(total_exposure) or total_exposure <= 0.0:
                raise HierarchicalPairedReleaseAnalysisError(
                    f"{planner} has invalid total {dimension} exposure"
                )
            rate = total_near_miss / total_exposure
            summaries.append(
                NearMissExposureSummary(
                    planner=planner,
                    dimension=dimension,
                    total_near_miss=total_near_miss,
                    total_exposure=float(total_exposure),
                    normalized_rate=rate / opportunity,
                )
            )
    return summaries


def practical_effect_classification(
    effect: PairedEffect,
    *,
    policy: AnalysisPolicy | None = None,
) -> dict[str, Any]:
    """Classify a paired effect against the predeclared practical-effect threshold.

    Returns a machine-readable record distinguishing statistical separation
    (interval excludes zero) from practical meaning (interval clears the
    predeclared minimum risk difference).  A statistically separable but
    practically null effect is reported as ``statistically_separable_practically_null``.

    Args:
        effect: The paired effect estimand to classify.
        policy: Predeclared analysis policy supplying the practical-effect threshold.

    Returns:
        A mapping of the practical-effect verdict and its boolean components.
    """

    resolved = policy or AnalysisPolicy()
    min_diff = resolved.min_risk_difference
    require_finite_scalar("min_risk_difference", min_diff)
    if min_diff <= 0.0:
        raise HierarchicalPairedReleaseAnalysisError("min_risk_difference must be positive")
    interval_excludes_zero = effect.risk_difference_ci_low > 0.0 or (
        effect.risk_difference_ci_high < 0.0
    )
    interval_clears_threshold = effect.risk_difference_ci_low >= min_diff or (
        effect.risk_difference_ci_high <= -min_diff
    )
    if interval_clears_threshold:
        verdict = "practically_separable"
    elif interval_excludes_zero:
        verdict = "statistically_separable_practically_null"
    else:
        verdict = "not_separable"
    return {
        "comparison": effect.comparison,
        "min_risk_difference": min_diff,
        "risk_difference": effect.risk_difference,
        "risk_difference_ci": [effect.risk_difference_ci_low, effect.risk_difference_ci_high],
        "statistically_separable": interval_excludes_zero,
        "practically_separable": interval_clears_threshold,
        "verdict": verdict,
    }


def run_hierarchical_paired_release_analysis(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
    successor_rows: Sequence[Mapping[str, Any]],
    planner_pairs: Sequence[tuple[str, str]],
    family_of: Mapping[str, str] | None = None,
    policy: AnalysisPolicy | None = None,
    outcomes: Sequence[str] = ("collision", "near_miss", "timeout"),
    horizon: float | None = None,
) -> dict[str, Any]:
    """Run the full hierarchical paired release analysis and emit a claim-gate report.

    The report is machine-readable and deterministic given the policy.  Even on
    valid successor rows the claim gate stays blocked pending human review:
    this engine produces the analysis artifact, it does not promote a claim.

    Args:
        manifest: The validated #5351 input manifest.
        repo_root: Repository root used to verify durable successor-release inputs.
        successor_rows: Typed-ledger successor rows to analyse.
        planner_pairs: Planner arms to compare; each pair contributes one
            family of comparisons to the multiplicity correction.
        family_of: Optional scenario-id to scenario-family mapping.
        policy: Predeclared analysis policy.
        outcomes: Binary outcomes to estimate per pair (default collision,
            near-miss, timeout).
        horizon: Episode time cap for censored completion-time summaries.

    Returns:
        A ``hierarchical_paired_release_analysis_report.v1`` mapping.
    """

    normalized_manifest = validate_hierarchical_paired_release_input_manifest(manifest)
    _require_ready_successor_release_inputs(normalized_manifest, repo_root=repo_root)
    resolved_policy = policy or AnalysisPolicy()
    if not successor_rows:
        raise HierarchicalPairedReleaseAnalysisError(
            "cannot run analysis on empty successor row set"
        )
    if not planner_pairs:
        raise HierarchicalPairedReleaseAnalysisError("planner_pairs must not be empty")
    if not outcomes:
        raise HierarchicalPairedReleaseAnalysisError("outcomes must not be empty")
    paired_effects: list[dict[str, Any]] = []
    multiplicity_inputs: list[float] = []
    multiplicity_labels: list[str] = []
    completion_summaries: list[dict[str, Any]] | None = [] if horizon is not None else None
    near_miss_summaries: list[dict[str, Any]] = []
    sensitivity_analyses: list[dict[str, Any]] = []
    for pair in planner_pairs:
        cells = build_matched_cells_from_ledger_rows(
            successor_rows, planner_pair=pair, family_of=family_of
        )
        if not cells:
            raise HierarchicalPairedReleaseAnalysisError(
                f"planner pair {pair} produced no matched cells from the successor rows"
            )
        for outcome in outcomes:
            effect = estimate_paired_effects(cells, outcome=outcome, policy=resolved_policy)
            practical = practical_effect_classification(effect, policy=resolved_policy)
            p_value = _paired_mcnemar_p_value(cells, outcome=outcome)
            paired_effects.append(
                {
                    "planner_pair": list(pair),
                    "outcome": outcome,
                    "n_cells": effect.n_cells,
                    "risk_difference": effect.risk_difference,
                    "risk_difference_ci": [
                        effect.risk_difference_ci_low,
                        effect.risk_difference_ci_high,
                    ],
                    "odds_ratio": effect.odds_ratio,
                    "odds_ratio_ci": [effect.odds_ratio_ci_low, effect.odds_ratio_ci_high],
                    "raw_p_value": p_value,
                    "practical_effect": practical,
                }
            )
            multiplicity_inputs.append(p_value)
            multiplicity_labels.append(f"{pair[0]}:{pair[1]}:{outcome}")
        if horizon is not None:
            completion_summaries.extend(
                {
                    "planner_pair": list(pair),
                    **_dataclass_to_dict(summary),
                }
                for summary in censored_completion_time(cells, horizon=horizon)
            )
        sensitivity_analyses.extend(
            _sensitivity_analysis(cells, outcomes=outcomes, policy=resolved_policy)
        )
        near_miss_summaries.extend(
            {
                "planner_pair": list(pair),
                **_dataclass_to_dict(summary),
            }
            for summary in normalized_near_miss_exposure(cells, policy=resolved_policy)
        )
    multiplicity = holm_multiplicity(multiplicity_inputs)
    for label, decision in zip(multiplicity_labels, multiplicity, strict=True):
        paired_effects_with_adj = [
            entry
            for entry in paired_effects
            if f"{entry['planner_pair'][0]}:{entry['planner_pair'][1]}:{entry['outcome']}" == label
        ]
        if paired_effects_with_adj:
            paired_effects_with_adj[0]["holm_adjusted_p_value"] = decision.adjusted_p_value
            paired_effects_with_adj[0]["rejected_at_family_wise_alpha"] = decision.rejected
    return {
        "schema_version": HIERARCHICAL_PAIRED_RELEASE_ANALYSIS_REPORT_SCHEMA,
        "issue": 5351,
        "claim_boundary": normalized_manifest["claim_boundary"],
        "evidence_status": EVIDENCE_STATUS_NOT_BENCHMARK_EVIDENCE,
        "analysis_executed": True,
        "policy": _dataclass_to_dict(resolved_policy),
        "planner_pairs": [list(pair) for pair in planner_pairs],
        "paired_effects": paired_effects,
        "multiplicity": {
            "method": "holm_step_down",
            "family_size": len(multiplicity_inputs),
            "decisions": [_dataclass_to_dict(decision) for decision in multiplicity],
        },
        "censored_completion_time": completion_summaries,
        "normalized_near_miss_exposure": near_miss_summaries,
        "sensitivity_analyses": sensitivity_analyses,
        "protocol_conformance": _protocol_conformance(
            manifest=normalized_manifest,
            analysis_executed=True,
            completion_time_delivered=completion_summaries is not None,
            near_miss_exposure_delivered=bool(near_miss_summaries),
        ),
        "claim_gate": {
            "status": CLAIM_GATE_BLOCKED_REVIEW_PENDING,
            "reason": (
                "analysis executed over successor rows; claim promotion requires "
                "human review of the deterministic report"
            ),
        },
        "semantics": {
            "benchmark_metrics_changed": False,
            "analysis_executed": True,
            "claim_promotion": "none",
        },
    }


def _require_ready_successor_release_inputs(
    manifest: Mapping[str, Any], *, repo_root: str | Path
) -> None:
    """Reject execution until release provenance and durable rows pass the input gate."""

    input_report = evaluate_hierarchical_paired_release_inputs(manifest, repo_root=repo_root)
    if (
        input_report["status"] != INPUTS_READY_ANALYSIS_NOT_RUN
        or input_report["blocking_prerequisites"]
    ):
        raise HierarchicalPairedReleaseAnalysisError(
            "cannot run analysis until successor-release input readiness is satisfied: "
            f"{input_report['blocking_prerequisites']}"
        )


def fail_closed_analysis_from_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Return the blocked analysis-side report when successor rows are unavailable.

    This complements the input gate: it reports the *analysis* protocol
    conformance (what the analysis would deliver) as blocked, without claiming
    any result.  Used by the CLI to emit a reviewable report on a pre-release
    repository.
    """

    normalized_manifest = validate_hierarchical_paired_release_input_manifest(manifest)
    input_report = evaluate_hierarchical_paired_release_inputs(
        normalized_manifest, repo_root=repo_root
    )
    return {
        "schema_version": HIERARCHICAL_PAIRED_RELEASE_ANALYSIS_REPORT_SCHEMA,
        "issue": 5351,
        "claim_boundary": normalized_manifest["claim_boundary"],
        "evidence_status": EVIDENCE_STATUS_NOT_BENCHMARK_EVIDENCE,
        "analysis_executed": False,
        "paired_effects": [],
        "multiplicity": {"method": "holm_step_down", "family_size": 0, "decisions": []},
        "censored_completion_time": None,
        "normalized_near_miss_exposure": None,
        "sensitivity_analyses": [],
        "protocol_conformance": _protocol_conformance(
            manifest=normalized_manifest, analysis_executed=False
        ),
        "claim_gate": {
            "status": CLAIM_GATE_BLOCKED_ANALYSIS_NOT_RUN,
            "reason": input_report["claim_gate"]["reason"],
        },
        "semantics": {
            "benchmark_metrics_changed": False,
            "analysis_executed": False,
            "claim_promotion": "none",
        },
    }


def _validate_ledger_row(row: Mapping[str, Any], *, index: int) -> None:
    """Validate that a successor row is an ``EpisodeEventLedger.v2`` mapping."""

    if not isinstance(row, Mapping):
        raise HierarchicalPairedReleaseAnalysisError(
            f"successor row[{index}] must be a mapping, got {type(row).__name__}"
        )
    schema = row.get("schema_version")
    if schema != EPISODE_EVENT_LEDGER_SCHEMA_VERSION:
        raise HierarchicalPairedReleaseAnalysisError(
            f"successor row[{index}] schema_version must be "
            f"{EPISODE_EVENT_LEDGER_SCHEMA_VERSION!r}, got {schema!r}"
        )
    for required in ("scenario_id", "seed", "planner", "exact_events", "surrogate_events"):
        if required not in row:
            raise HierarchicalPairedReleaseAnalysisError(
                f"successor row[{index}] missing required field {required!r}"
            )
    exact_events = row["exact_events"]
    if not isinstance(exact_events, Mapping):
        raise HierarchicalPairedReleaseAnalysisError(
            f"successor row[{index}] exact_events must be a mapping"
        )
    for field in ("collision", "goal_reached", "timeout", "invalid_run"):
        if not isinstance(exact_events.get(field), bool):
            raise HierarchicalPairedReleaseAnalysisError(
                f"successor row[{index}] exact_events.{field} must be a bool"
            )
    surrogate_events = row["surrogate_events"]
    if not isinstance(surrogate_events, Mapping) or not isinstance(
        surrogate_events.get("near_miss"), bool
    ):
        raise HierarchicalPairedReleaseAnalysisError(
            f"successor row[{index}] surrogate_events.near_miss must be a bool"
        )


def _select_outcome(cells: Sequence[MatchedCell], outcome: str) -> tuple[list[int], list[int]]:
    """Return the per-arm binary outcome lists for the named outcome."""

    field_map = {
        "collision": ("collision_a", "collision_b"),
        "near_miss": ("near_miss_a", "near_miss_b"),
        "timeout": ("timeout_a", "timeout_b"),
    }
    if outcome not in field_map:
        raise HierarchicalPairedReleaseAnalysisError(
            f"outcome must be one of {sorted(field_map)}, got {outcome!r}"
        )
    attr_a, attr_b = field_map[outcome]
    return [getattr(c, attr_a) for c in cells], [getattr(c, attr_b) for c in cells]


def _require_paired_outcomes(
    outcomes_a: Sequence[int], outcomes_b: Sequence[int], *, outcome: str
) -> None:
    """Reject degenerate or mismatched outcome vectors before estimation."""

    if len(outcomes_a) != len(outcomes_b) or not outcomes_a:
        raise HierarchicalPairedReleaseAnalysisError(
            f"paired outcome {outcome!r} requires equal-length non-empty arms"
        )


def _ordered_families(cells: Sequence[MatchedCell]) -> list[list[int]]:
    """Group cell indices by scenario-family for the cluster bootstrap.

    Returns:
        Lists of cell indices, one per distinct scenario-family.
    """

    return _ordered_clusters(cells, key=lambda cell: cell.scenario_family)


def _ordered_clusters(cells: Sequence[MatchedCell], *, key: Any) -> list[list[int]]:
    """Group cell indices by a stable cluster key for bootstrap resampling.

    Returns:
        Lists of cell indices, one per distinct cluster key.
    """

    groups: dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        groups.setdefault(str(key(cell)), []).append(index)
    return list(groups.values())


def _cluster_bootstrap_paired(
    *,
    outcomes_a: Sequence[int],
    outcomes_b: Sequence[int],
    families: Sequence[Sequence[int]],
    policy: AnalysisPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample whole scenario-families to build paired risk-difference samples.

    Returns arrays of bootstrap risk-difference and odds-ratio samples.  The
    cluster unit is the family, so within-family correlation from shared
    scenario structure is preserved rather than treated as independent.

    Returns:
        ``(diff_samples, ratio_samples)`` arrays of per-draw risk differences and odds ratios.
    """

    if isinstance(policy.bootstrap_samples, bool) or not isinstance(policy.bootstrap_samples, int):
        raise HierarchicalPairedReleaseAnalysisError("bootstrap_samples must be a positive integer")
    if policy.bootstrap_samples <= 0:
        raise HierarchicalPairedReleaseAnalysisError("bootstrap_samples must be a positive integer")
    a = np.asarray(outcomes_a, dtype=np.float64)
    b = np.asarray(outcomes_b, dtype=np.float64)
    family_blocks = [np.asarray(block, dtype=np.int64) for block in families]
    rng = np.random.default_rng(policy.bootstrap_seed)
    diff_samples = np.empty(policy.bootstrap_samples, dtype=np.float64)
    ratio_samples = np.empty(policy.bootstrap_samples, dtype=np.float64)
    family_count = len(family_blocks)
    for draw in range(policy.bootstrap_samples):
        sampled_families = rng.integers(0, family_count, size=family_count)
        indices = np.concatenate([family_blocks[i] for i in sampled_families])
        if indices.size == 0:
            diff_samples[draw] = 0.0
            ratio_samples[draw] = 1.0
            continue
        rate_a = float(np.mean(a[indices]))
        rate_b = float(np.mean(b[indices]))
        diff_samples[draw] = rate_a - rate_b
        ratio_samples[draw] = _odds_ratio(rate_a, rate_b)
    return diff_samples, ratio_samples


def _percentile_interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    """Return the equal-tailed percentile confidence interval from samples.

    Returns:
        The ``(lower, upper)`` percentile bounds of the bootstrap samples.
    """

    if not 0.0 < confidence < 1.0:
        raise HierarchicalPairedReleaseAnalysisError(
            "confidence must be in the open interval (0, 1)"
        )
    if samples.size == 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    lower = float(np.percentile(samples, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def _empirical_odds(rate: float) -> float:
    """Return p/(1-p) odds with a finite fallback for degenerate rates.

    Returns:
        The empirical odds, clamped to a ``1e6`` sentinel at rate 1.0 and ``0.0`` at rate 0.0.
    """

    clamped = min(max(rate, 0.0), 1.0)
    if clamped >= 1.0:
        return 1e6
    if clamped <= 0.0:
        return 0.0
    return clamped / (1.0 - clamped)


def _odds_ratio(risk_a: float, risk_b: float) -> float:
    """Return a finite odds ratio with one shared degenerate-rate convention."""

    odds_a = _empirical_odds(risk_a)
    odds_b = _empirical_odds(risk_b)
    if odds_a == 0.0 and odds_b == 0.0:
        return 1.0
    if odds_b == 0.0:
        return 1e6
    return odds_a / odds_b


def _paired_mcnemar_p_value(cells: Sequence[MatchedCell], *, outcome: str) -> float:
    """Exact two-sided McNemar p-value for the discordant paired cells.

    Uses the exact binomial form on the discordant count so small matched
    samples remain valid; this feeds the multiplicity correction rather than
    relying on interval overlap.

    Returns:
        The two-sided exact McNemar p-value in ``[0.0, 1.0]``.
    """

    outcomes_a, outcomes_b = _select_outcome(cells, outcome)
    # Discordant directions: b = arm A had the event and B did not;
    # c = arm B had the event and A did not.  Under the paired null each
    # discordant cell is equally likely to fall either way (binomial, p=0.5).
    b_count = 0
    c_count = 0
    for a, b in zip(outcomes_a, outcomes_b, strict=True):
        a_event = int(a) >= 1
        b_event = int(b) >= 1
        if a_event and not b_event:
            b_count += 1
        elif b_event and not a_event:
            c_count += 1
    discordant = b_count + c_count
    if discordant == 0:
        return 1.0
    larger = max(b_count, c_count)
    # Upper-tail probability of observing at least ``larger`` events in one
    # direction under Binomial(discordant, 0.5); double it for the two-sided
    # test and cap at 1.0.
    upper_tail = sum(math.comb(discordant, k) for k in range(larger, discordant + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * upper_tail)


def _protocol_conformance(
    *,
    manifest: Mapping[str, Any],
    analysis_executed: bool = False,
    completion_time_delivered: bool = False,
    near_miss_exposure_delivered: bool = False,
) -> list[dict[str, Any]]:
    """Map each declared protocol element to its delivery status in this report.

    Returns:
        One conformance row per declared protocol element with its status.
    """

    delivered = {
        "paired_effects": True,
        "hierarchical_intervals": True,
        "sensitivity_analyses": True,
        "multiplicity_control": True,
        "practical_effect_reporting": True,
        "censored_completion_time": completion_time_delivered,
        "normalized_near_miss_exposure": near_miss_exposure_delivered,
        "claim_gate_and_conformance": True,
    }
    conformance: list[dict[str, Any]] = []
    for item in manifest["protocol"]:
        element_id = item["id"]
        is_delivered = analysis_executed and delivered.get(element_id, False)
        conformance.append(
            {
                "id": element_id,
                "declared_delivery": item["declared_delivery"],
                "status": "delivered_analysis_pending_human_review"
                if is_delivered
                else "declared_pending_analysis",
            }
        )
    return conformance


def _binary_outcome(row: Mapping[str, Any], field: str) -> int:
    """Read a binary exact_events field as 0/1.

    Returns:
        ``1`` when the named exact event is truthy, otherwise ``0``.
    """

    exact = row.get("exact_events")
    if not isinstance(exact, Mapping):
        return 0
    return 1 if bool(exact.get(field)) else 0


def _count_outcome(row: Mapping[str, Any], field: str) -> int:
    """Read a surrogate near-miss count from the ledger row.

    Returns:
        ``1`` when the surrogate event is present, otherwise ``0``.
    """

    surrogate = row.get("surrogate_events")
    if isinstance(surrogate, Mapping) and surrogate.get(field):
        return 1
    exact = row.get("exact_events")
    if isinstance(exact, Mapping):
        return int(bool(exact.get(field)))
    return 0


def _completion_time(row: Mapping[str, Any]) -> float:
    """Read a required finite non-negative completion time from a ledger row.

    Raises:
        HierarchicalPairedReleaseAnalysisError: If completion time is absent,
            non-numeric, non-finite, or negative.

    Returns:
        The validated completion time.
    """

    provenance = row.get("provenance")
    if isinstance(provenance, Mapping) and "completion_time" in provenance:
        candidate = provenance["completion_time"]
        if not _valid_completion_time(candidate):
            raise HierarchicalPairedReleaseAnalysisError(
                "completion_time must be a finite non-negative provenance.completion_time "
                "or metrics.completion_time.value"
            )
        return float(candidate)
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping) and "completion_time" in metrics:
        metric = metrics["completion_time"]
        if not isinstance(metric, Mapping) or "value" not in metric:
            raise HierarchicalPairedReleaseAnalysisError(
                "completion_time must be a finite non-negative provenance.completion_time "
                "or metrics.completion_time.value"
            )
        candidate = metric["value"]
        if not _valid_completion_time(candidate):
            raise HierarchicalPairedReleaseAnalysisError(
                "completion_time must be a finite non-negative provenance.completion_time "
                "or metrics.completion_time.value"
            )
        return float(candidate)
    raise HierarchicalPairedReleaseAnalysisError(
        "completion_time must be a finite non-negative provenance.completion_time "
        "or metrics.completion_time.value"
    )


def _valid_completion_time(value: Any) -> bool:
    """Return whether ``value`` is a finite non-negative numeric duration."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _exposure(row: Mapping[str, Any]) -> dict[str, float]:
    """Read labeled positive interaction-exposure dimensions from provenance.

    Raises:
        HierarchicalPairedReleaseAnalysisError: If exposure is absent, non-finite,
            or not strictly positive.

    Returns:
        The validated, strictly positive exposure values by dimension.
    """

    provenance = row.get("provenance")
    if isinstance(provenance, Mapping):
        candidate = provenance.get("exposure")
        if isinstance(candidate, Mapping) and set(candidate) == set(EXPOSURE_DIMENSIONS):
            exposure: dict[str, float] = {}
            for dimension in EXPOSURE_DIMENSIONS:
                value = candidate[dimension]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    break
                numeric = float(value)
                if not math.isfinite(numeric) or numeric <= 0.0:
                    break
                exposure[dimension] = numeric
            else:
                return exposure
    raise HierarchicalPairedReleaseAnalysisError(
        "exposure must be a mapping with finite positive time, distance, and opportunity values"
    )


def _as_int(value: Any) -> int:
    """Coerce a ledger seed to int, raising on non-integer values.

    Returns:
        The seed as an ``int``.
    """

    if isinstance(value, bool):
        raise HierarchicalPairedReleaseAnalysisError(f"seed must be an int, got bool {value}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise HierarchicalPairedReleaseAnalysisError(f"seed must be an int, got {value!r}")


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Return a shallow dict view of a frozen dataclass for JSON serialization.

    Returns:
        A dict mapping field name to value for dataclasses and mappings.
    """

    if hasattr(obj, "__dataclass_fields__"):
        return {field_name: getattr(obj, field_name) for field_name in obj.__dataclass_fields__}
    if isinstance(obj, Mapping):
        return dict(obj)
    raise HierarchicalPairedReleaseAnalysisError(f"cannot serialize {type(obj).__name__}")


__all__ = [
    "CLAIM_GATE_BLOCKED_ANALYSIS_NOT_RUN",
    "CLAIM_GATE_BLOCKED_REVIEW_PENDING",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_EXPOSURE_OPPORTUNITY",
    "DEFAULT_MIN_RISK_DIFFERENCE",
    "EVIDENCE_STATUS_NOT_BENCHMARK_EVIDENCE",
    "HIERARCHICAL_PAIRED_RELEASE_ANALYSIS_REPORT_SCHEMA",
    "AnalysisPolicy",
    "CompletionTimeSummary",
    "HierarchicalPairedReleaseAnalysisError",
    "MatchedCell",
    "MultiplicityDecision",
    "NearMissExposureSummary",
    "PairedEffect",
    "build_matched_cells_from_ledger_rows",
    "censored_completion_time",
    "estimate_paired_effects",
    "fail_closed_analysis_from_manifest",
    "holm_multiplicity",
    "normalized_near_miss_exposure",
    "practical_effect_classification",
    "run_hierarchical_paired_release_analysis",
]
