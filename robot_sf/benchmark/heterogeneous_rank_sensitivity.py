"""Rank-order sensitivity and bootstrap planner comparison for issue #3574.

Evaluates whether planner ranking remains stable (or undergoes reversals) when
moving from a homogeneous mean-matched population to a heterogeneous mixture population.
Uses paired bootstrap resampling over seeds to compute confidence bounds and P(A beats B).

This module also provides :func:`pre_specified_rank_reversal_test`, a *preregistered*
hypothesis test on rank reversal that is distinct from the descriptive disagreement flag
emitted by :func:`compute_bootstrap_rank_sensitivity`. The descriptive flag records any
ranking-list disagreement; the preregistered test only declares a reversal when a planner
pair has a bootstrap-determined ordering in *both* arms with *opposite* signs, so a
reversal cannot be triggered by sampling noise. The test parameters (significance level
``alpha``, percentile-CI method, and decision rule) are recorded as declared-before-results
so the comparison is auditable when real campaign episode records arrive.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def compute_bootstrap_rank_sensitivity(  # noqa: C901,PLR0912,PLR0915
    records: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    planners: Sequence[str],
    higher_is_safer: bool = True,
    num_bootstrap: int = 1000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate planner rank sensitivity using paired bootstrap resampling.

    Args:
        records: Episode records containing planner, seed, population_arm, and metrics.
        metric_key: Key under ``metrics`` to evaluate (e.g. "clearance_m", "collisions").
        planners: List of planner keys to compare.
        higher_is_safer: True if larger metric values are better/safer.
        num_bootstrap: Number of bootstrap iterations.
        seed: RNG seed for bootstrap determinism.

    Returns:
        Structured dictionary with pairwise probabilities, ranking lists, and reversals.
    """
    if len(planners) < 2:
        raise ValueError("At least 2 planners are required to compare ranks")

    # Group records by (arm, planner, seed)
    data: dict[str, dict[str, dict[int, float]]] = {}
    for rec in records:
        # ``scenario_params`` may be absent or explicitly ``None`` (issue #4618 R1);
        # coerce to an empty mapping so the fallback lookups never raise AttributeError.
        scenario_params = rec.get("scenario_params") or {}
        arm = rec.get("population_arm")
        if arm is None:
            arm = scenario_params.get("population_arm")
        planner = rec.get("planner")
        if planner is None:
            planner = scenario_params.get("planner")
        # coerce to int seed
        s_val = rec.get("seed")
        if s_val is None:
            s_val = scenario_params.get("seed")
        if arm is None or planner is None or s_val is None:
            continue
        try:
            s = int(s_val)
        except (TypeError, ValueError):
            continue

        metrics = rec.get("metrics")
        if not isinstance(metrics, Mapping) or metric_key not in metrics:
            continue
        val = metrics[metric_key]
        if val is None or not math.isfinite(float(val)):
            continue

        arm_str = str(arm).strip()
        p_str = str(planner).strip()
        if p_str not in planners:
            continue

        data.setdefault(arm_str, {}).setdefault(p_str, {})[s] = float(val)

    # Collect available arms
    arms = sorted(data.keys())
    if not arms:
        return {
            "schema_version": "heterogeneous_rank_sensitivity.v1",
            "metric_key": metric_key,
            "status": "blocked",
            "blockers": ["No valid episode data matches configuration"],
        }

    # Find the set of common seeds across all planners and arms
    all_seed_sets = []
    for arm in arms:
        for planner in planners:
            all_seed_sets.append(set(data[arm].get(planner, {}).keys()))

    common_seeds = sorted(set.intersection(*all_seed_sets)) if all_seed_sets else []
    if len(common_seeds) < 2:
        return {
            "schema_version": "heterogeneous_rank_sensitivity.v1",
            "metric_key": metric_key,
            "status": "blocked",
            "blockers": [
                f"Insufficient paired seeds across planners/arms (found {len(common_seeds)}, need >=2)"
            ],
        }

    # Convert to array: shape (num_arms, num_planners, num_seeds)
    # Map index
    arm_map = {name: idx for idx, name in enumerate(arms)}
    p_map = {name: idx for idx, name in enumerate(planners)}

    arr = np.zeros((len(arms), len(planners), len(common_seeds)), dtype=float)
    for arm_name in arms:
        for p_name in planners:
            for s_idx, s in enumerate(common_seeds):
                arr[arm_map[arm_name], p_map[p_name], s_idx] = data[arm_name][p_name][s]

    rng = np.random.default_rng(seed)
    num_seeds = len(common_seeds)

    # Run bootstrap
    # shape: (num_bootstrap, num_arms, num_planners)
    bootstrap_means = np.zeros((num_bootstrap, len(arms), len(planners)), dtype=float)
    for b in range(num_bootstrap):
        # sample seed indices with replacement
        indices = rng.choice(num_seeds, size=num_seeds, replace=True)
        # mean over seeds
        bootstrap_means[b] = np.mean(arr[:, :, indices], axis=2)

    # Calculate actual observed means
    observed_means = np.mean(arr, axis=2)

    arm_results = {}
    for a_idx, arm_name in enumerate(arms):
        # actual observed ranks
        p_means = observed_means[a_idx]
        sorted_p_indices = np.argsort(p_means)
        if higher_is_safer:
            sorted_p_indices = sorted_p_indices[::-1]

        ranking = [planners[idx] for idx in sorted_p_indices]

        # pairwise comparisons
        pairwise = {}
        for i, p_a in enumerate(planners):
            for j, p_b in enumerate(planners):
                if i == j:
                    continue
                # P(A beats B)
                diffs = bootstrap_means[:, a_idx, i] - bootstrap_means[:, a_idx, j]
                if higher_is_safer:
                    p_val = float(np.count_nonzero(diffs > 0) / num_bootstrap)
                else:
                    p_val = float(np.count_nonzero(diffs < 0) / num_bootstrap)
                pairwise[f"{p_a}_beats_{p_b}"] = p_val

        arm_results[arm_name] = {
            "observed_means": {planners[i]: float(p_means[i]) for i in range(len(planners))},
            "ranking": ranking,
            "pairwise_probabilities": pairwise,
        }

    # Detect rank reversals
    # A reversal is defined if the ranking differs between arms
    reversals = []
    if "heterogeneous" in arm_results and "mean_matched_homogeneous" in arm_results:
        rank_het = arm_results["heterogeneous"]["ranking"]
        rank_hom = arm_results["mean_matched_homogeneous"]["ranking"]
        if rank_het != rank_hom:
            reversals.append(
                {
                    "type": "rank_order_disagreement",
                    "heterogeneous_ranking": rank_het,
                    "mean_matched_homogeneous_ranking": rank_hom,
                    "description": f"Heterogeneous ranking {rank_het} differs from homogeneous {rank_hom}",
                }
            )

    return {
        "schema_version": "heterogeneous_rank_sensitivity.v1",
        "status": "ready",
        "metric_key": metric_key,
        "higher_is_safer": higher_is_safer,
        "common_seeds_count": num_seeds,
        "planners": list(planners),
        "arms": arm_results,
        "reversals": reversals,
    }


RANK_REVERSAL_TEST_SCHEMA = "heterogeneous_rank_reversal_test.v1"
_DEFAULT_HETEROGENEOUS_ARM = "heterogeneous"
_DEFAULT_MEAN_MATCHED_ARM = "mean_matched_homogeneous"


def pre_specified_rank_reversal_test(
    records: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    planners: Sequence[str],
    higher_is_safer: bool = True,
    alpha: float = 0.05,
    num_bootstrap: int = 1000,
    seed: int | None = None,
    arms: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Preregistered rank-reversal hypothesis test across population compositions.

    This is the *pre-specified rank-reversal test* requested by issue #3574's Definition of
    Done ("Rank-order sensitivity across >=3 planners per population composition reported with
    bootstrap P(A beats B) and a pre-specified rank-reversal test"). It complements
    :func:`compute_bootstrap_rank_sensitivity`, which already reports bootstrap P(A beats B)
    and a descriptive ranking-list disagreement flag. Where the descriptive flag fires on
    *any* ordering difference (including differences within bootstrap sampling noise), this
    test only declares a reversal when a planner pair has a bootstrap-determined ordering in
    *both* arms with *opposite* signs.

    Null hypothesis ``H0``: rank order is stable across the two population compositions (no
    reversal). The test rejects ``H0`` iff at least one planner pair is determined in both
    arms with opposite signs. The significance level ``alpha``, the percentile bootstrap CI
    method, and this exact decision rule are declared-before-results and echoed in the
    ``pre_registration`` block so the comparison is auditable once real campaign episode
    records arrive.

    Args:
        records: Episode records of the same shape consumed by
            :func:`compute_bootstrap_rank_sensitivity` (``population_arm`` / ``planner`` /
            ``seed`` / ``metrics[metric_key]``, with ``scenario_params`` fallback).
        metric_key: Metric under ``metrics`` to test (e.g. ``"clearance_m"``).
        planners: Planner keys to compare; at least two are required.
        higher_is_safer: When true, larger metric values rank a planner higher.
        alpha: Two-sided significance level in ``(0, 1)`` (default ``0.05``); the percentile
            CI is computed at level ``1 - alpha``.
        num_bootstrap: Number of bootstrap resamples of the seed axis.
        seed: RNG seed for deterministic resampling.
        arms: Optional ``(heterogeneous_arm, mean_matched_homogeneous_arm)`` name override; the
            two names must differ. Defaults to the canonical #3574 arm names.

    Returns:
        Versioned (``heterogeneous_rank_reversal_test.v1``) report. ``status`` is
        ``"ready"`` when both arms are present with >=2 common paired seeds across all
        planners, else ``"blocked"`` with precise fail-closed blockers. A ready report
        carries the preregistration block, the per-arm pairwise bootstrap CIs, the per-pair
        reversal verdicts, the overall decision, and an explicit claim boundary.
    """
    planners, heterogeneous_arm, mean_matched_arm = _validate_test_arguments(
        planners=planners, arms=arms, alpha=alpha, num_bootstrap=num_bootstrap
    )

    # The decision specification is fixed here, before any results are inspected, so the
    # echoed ``pre_registration`` block is an ex-ante declaration rather than a post-hoc fit.
    decision_rule = (
        "Reject H0 (rank stability) iff some planner pair (i, j) has a bootstrap "
        "(1-alpha) percentile CI on the per-seed paired mean difference that excludes "
        "zero in BOTH arms with opposite signs (i ahead of j in one arm, j ahead of i in "
        "the other)."
    )
    pre_registration = {
        "declared_before_results": True,
        "significance_level_alpha": float(alpha),
        "ci_method": "percentile_bootstrap",
        "ci_level": float(1.0 - alpha),
        "num_bootstrap": int(num_bootstrap),
        "higher_is_safer": bool(higher_is_safer),
        "heterogeneous_arm": heterogeneous_arm,
        "mean_matched_arm": mean_matched_arm,
        "null_hypothesis": "rank order is stable across population compositions (no reversal)",
        "decision_rule": decision_rule,
    }

    grouped = _group_records_by_arm_planner_seed(records, metric_key=metric_key)

    blockers: list[str] = []
    if heterogeneous_arm not in grouped:
        blockers.append(f"no valid episode records for arm {heterogeneous_arm!r}")
    if mean_matched_arm not in grouped:
        blockers.append(f"no valid episode records for arm {mean_matched_arm!r}")
    if blockers:
        return _blocked_test_result(metric_key, blockers, pre_registration)

    arm_arrays, arm_common_seeds, arm_blockers = _paired_performance_arrays(
        grouped,
        planners=planners,
        arms=(heterogeneous_arm, mean_matched_arm),
    )
    if arm_blockers:
        return _blocked_test_result(metric_key, arm_blockers, pre_registration)

    het_arr = arm_arrays[heterogeneous_arm]
    hom_arr = arm_arrays[mean_matched_arm]
    common_seeds_count = arm_common_seeds[heterogeneous_arm]

    rng = np.random.default_rng(seed)

    pair_results: list[dict[str, Any]] = []
    reversals: list[dict[str, Any]] = []
    for i, planner_i in enumerate(planners):
        for j in range(i + 1, len(planners)):
            pair_record, reversal = _evaluate_planner_pair(
                arrays=(het_arr, hom_arr),
                i=i,
                j=j,
                planner_i=planner_i,
                planner_j=planners[j],
                higher_is_safer=higher_is_safer,
                num_bootstrap=num_bootstrap,
                rng=rng,
            )
            pair_results.append(pair_record)
            if reversal is not None:
                reversals.append(reversal)

    decision = "reject_null_rank_stability" if reversals else "fail_to_reject_null_rank_stability"
    return {
        "schema_version": RANK_REVERSAL_TEST_SCHEMA,
        "status": "ready",
        "metric_key": metric_key,
        "planners": list(planners),
        "arms": [heterogeneous_arm, mean_matched_arm],
        "common_seeds_count": common_seeds_count,
        "pre_registration": pre_registration,
        "pairwise": pair_results,
        "reversals": reversals,
        "decision": decision,
        "reversal_count": len(reversals),
        "claim_boundary": (
            "Preregistered rank-reversal test primitive. Reports a reversal only when a "
            "planner pair is bootstrap-determined in both arms with opposite signs. No "
            "benchmark campaign, rank-order, realism, or sim-to-real claim is established "
            "here; a campaign conclusion requires real paired episode records."
        ),
    }


def _validate_test_arguments(
    *,
    planners: Sequence[str],
    arms: tuple[str, str] | None,
    alpha: float,
    num_bootstrap: int,
) -> tuple[list[str], str, str]:
    """Validate test arguments and resolve the arm-name pair.

    Returns:
        ``(planners_list, heterogeneous_arm, mean_matched_arm)``.

    Raises:
        ValueError: If fewer than two planners, identical arm names, or invalid alpha / count.
    """
    if len(planners) < 2:
        raise ValueError("At least 2 planners are required to compare ranks")
    heterogeneous_arm, mean_matched_arm = arms or (
        _DEFAULT_HETEROGENEOUS_ARM,
        _DEFAULT_MEAN_MATCHED_ARM,
    )
    if heterogeneous_arm == mean_matched_arm:
        raise ValueError("heterogeneous_arm and mean_matched_arm must differ")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if num_bootstrap < 1:
        raise ValueError("num_bootstrap must be >= 1")
    return list(planners), heterogeneous_arm, mean_matched_arm


def _blocked_test_result(
    metric_key: str,
    blockers: list[str],
    pre_registration: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a fail-closed blocked test result that still echoes the preregistration.

    Returns:
        Versioned blocked report with the precise blockers.
    """
    return {
        "schema_version": RANK_REVERSAL_TEST_SCHEMA,
        "status": "blocked",
        "metric_key": metric_key,
        "blockers": blockers,
        "pre_registration": pre_registration,
    }


def _evaluate_planner_pair(
    *,
    arrays: tuple[np.ndarray, np.ndarray],
    i: int,
    j: int,
    planner_i: str,
    planner_j: str,
    higher_is_safer: bool,
    num_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Compute the per-pair verdict, pair record, and optional reversal for one pair.

    Returns:
        ``(pair_record, reversal)`` where ``reversal`` is ``None`` when the pair is not a
        determined reversal (stable or indeterminate).
    """
    het_array, hom_array = arrays
    het_diff = _seed_pair_difference(het_array, i, j, higher_is_safer=higher_is_safer)
    hom_diff = _seed_pair_difference(hom_array, i, j, higher_is_safer=higher_is_safer)
    het_ci = _bootstrap_mean_percentile_ci(het_diff, num_bootstrap, rng)
    hom_ci = _bootstrap_mean_percentile_ci(hom_diff, num_bootstrap, rng)
    het_sign = _determined_sign(het_ci)
    hom_sign = _determined_sign(hom_ci)

    if het_sign != 0 and hom_sign not in {0, het_sign}:
        verdict = "reversal_detected"
        is_reversal = True
    elif het_sign != 0 and hom_sign == het_sign:
        verdict = "stable_determined"
        is_reversal = False
    else:
        # At least one arm's CI straddles zero -> the ordering is not statistically
        # determined; we do not claim a reversal and we do not claim confirmed
        # stability. ``indeterminate`` is the honest pre-specified outcome.
        verdict = "indeterminate"
        is_reversal = False

    ahead_i_het = het_sign > 0
    ahead_i_hom = hom_sign > 0
    pair_record = {
        "planners": [planner_i, planner_j],
        "verdict": verdict,
        "reversal": is_reversal,
        "heterogeneous": {
            "mean_difference": float(np.mean(het_diff)),
            "ci_lower": float(het_ci[0]),
            "ci_upper": float(het_ci[1]),
            "determined_sign": _sign_label(het_sign, planner_i=planner_i, planner_j=planner_j),
            f"{planner_i}_ahead": bool(ahead_i_het) if het_sign != 0 else None,
        },
        "mean_matched_homogeneous": {
            "mean_difference": float(np.mean(hom_diff)),
            "ci_lower": float(hom_ci[0]),
            "ci_upper": float(hom_ci[1]),
            "determined_sign": _sign_label(hom_sign, planner_i=planner_i, planner_j=planner_j),
            f"{planner_i}_ahead": bool(ahead_i_hom) if hom_sign != 0 else None,
        },
    }
    if not is_reversal:
        return pair_record, None
    het_leader = planner_i if ahead_i_het else planner_j
    hom_leader = planner_i if ahead_i_hom else planner_j
    reversal = {
        "planners": [planner_i, planner_j],
        "heterogeneous_leader": het_leader,
        "mean_matched_homogeneous_leader": hom_leader,
        "description": (
            f"Pair ({planner_i}, {planner_j}) ordering is determined in both arms with "
            f"opposite signs: heterogeneous leader {het_leader} vs mean-matched leader "
            f"{hom_leader}."
        ),
    }
    return pair_record, reversal


def _group_records_by_arm_planner_seed(
    records: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
) -> dict[str, dict[str, dict[int, float]]]:
    """Group finite episode metric values by arm, planner, and integer seed.

    Mirrors the parsing contract of :func:`compute_bootstrap_rank_sensitivity`: arm, planner,
    and seed fall back to ``scenario_params``; the seed is coerced to ``int``; non-finite or
    missing metrics are dropped. Unknown planners (not in any caller's list) are kept here
    and filtered by the caller so this helper stays reusable.

    Returns:
        Nested mapping ``arm -> planner -> seed -> metric value``.
    """
    grouped: dict[str, dict[str, dict[int, float]]] = {}
    for rec in records:
        parsed = _parse_rank_record(rec, metric_key=metric_key)
        if parsed is None:
            continue
        arm, planner, seed_int, numeric = parsed
        grouped.setdefault(arm, {}).setdefault(planner, {})[seed_int] = numeric
    return grouped


def _parse_rank_record(
    record: Mapping[str, Any],
    *,
    metric_key: str,
) -> tuple[str, str, int, float] | None:
    """Parse one episode record into ``(arm, planner, seed, value)`` or drop it.

    Returns:
        The parsed tuple, or ``None`` when the record is missing required fields, has a
        non-integer seed, or carries a missing/null/non-finite metric.
    """
    scenario_params = record.get("scenario_params") or {}
    arm = record.get("population_arm")
    if arm is None:
        arm = scenario_params.get("population_arm")
    planner = record.get("planner")
    if planner is None:
        planner = scenario_params.get("planner")
    seed_value = record.get("seed")
    if seed_value is None:
        seed_value = scenario_params.get("seed")
    if arm is None or planner is None or seed_value is None:
        return None
    try:
        seed_int = int(seed_value)
    except (TypeError, ValueError):
        return None
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or metric_key not in metrics:
        return None
    value = metrics[metric_key]
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return str(arm).strip(), str(planner).strip(), seed_int, numeric


def _paired_performance_arrays(
    grouped: Mapping[str, Mapping[str, Mapping[int, float]]],
    *,
    planners: Sequence[str],
    arms: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, int], list[str]]:
    """Build ``(planner, seed)`` performance arrays per arm over shared common seeds.

    Returns:
        The per-arm arrays, the common-seed count, and a list of fail-closed blockers
        (missing planner, or fewer than two paired seeds across planners within an arm).
    """
    arrays: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    blockers: list[str] = []
    for arm in arms:
        arm_data = grouped.get(arm, {})
        seed_sets = []
        for planner in planners:
            if planner not in arm_data:
                blockers.append(f"arm {arm!r} has no records for planner {planner!r}")
            seed_sets.append(set(arm_data.get(planner, {}).keys()))
        if blockers:
            continue
        common = sorted(set.intersection(*seed_sets)) if seed_sets else []
        if len(common) < 2:
            blockers.append(
                f"arm {arm!r} has {len(common)} paired seed(s) across planners (need >=2)"
            )
            continue
        arr = np.zeros((len(planners), len(common)), dtype=float)
        for planner_index, planner in enumerate(planners):
            for seed_index, seed_int in enumerate(common):
                arr[planner_index, seed_index] = arm_data[planner][seed_int]
        arrays[arm] = arr
        counts[arm] = len(common)
    return arrays, counts, blockers


def _seed_pair_difference(
    arm_array: np.ndarray,
    i: int,
    j: int,
    *,
    higher_is_safer: bool,
) -> np.ndarray:
    """Per-seed paired difference of planner ``i`` minus planner ``j``.

    When ``higher_is_safer`` is false (lower is better), the difference is negated so a
    positive value consistently means ``i`` is preferred, keeping the sign logic arm-agnostic.

    Returns:
        The 1-D per-seed paired difference (positive favors planner ``i``).
    """
    diff = arm_array[i] - arm_array[j]
    return diff if higher_is_safer else -diff


def _bootstrap_mean_percentile_ci(
    differences: np.ndarray,
    num_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of a 1-D paired-difference array.

    Returns:
        The ``(lower, upper)`` 2.5th / 97.5th percentile bounds on the bootstrapped mean.
    """
    n = int(differences.size)
    if n == 0:
        raise ValueError("differences must be non-empty")
    indices = rng.integers(0, n, size=(num_bootstrap, n))
    resampled = differences[indices]
    means = resampled.mean(axis=1)
    lower = float(np.percentile(means, 2.5))
    upper = float(np.percentile(means, 97.5))
    return lower, upper


def _determined_sign(ci: tuple[float, float]) -> int:
    """Return +1 / -1 when the CI excludes zero, else 0 (indeterminate ordering).

    Returns:
        ``+1`` if the CI is entirely positive, ``-1`` if entirely negative, else ``0``.
    """
    lower, upper = ci
    if lower > 0.0:
        return 1
    if upper < 0.0:
        return -1
    return 0


def _sign_label(sign: int, *, planner_i: str, planner_j: str) -> str:
    """Human-readable leader label for a determined sign (``indeterminate`` when 0).

    Returns:
        ``planner_i`` when sign is positive, ``planner_j`` when negative, else
        ``"indeterminate"``.
    """
    if sign > 0:
        return planner_i
    if sign < 0:
        return planner_j
    return "indeterminate"
