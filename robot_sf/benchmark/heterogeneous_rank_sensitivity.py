"""Rank-order sensitivity and bootstrap planner comparison for issue #3574.

Evaluates whether planner ranking remains stable (or undergoes reversals) when
moving from a homogeneous mean-matched population to a heterogeneous mixture population.
Uses paired bootstrap resampling over seeds to compute confidence bounds and P(A beats B).
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
        arm = rec.get("population_arm") or rec.get("scenario_params", {}).get("population_arm")
        planner = rec.get("planner") or rec.get("scenario_params", {}).get("planner")
        # coerce to int seed
        s_val = rec.get("seed") or rec.get("scenario_params", {}).get("seed")
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
