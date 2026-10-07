"""Opt-in #10190 profiles and paired dev-seed sensitivity summaries.

This study owner consumes native episode metrics; it changes no benchmark formula.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from typing import Any

import numpy as np
from scipy.stats import kendalltau

from robot_sf.benchmark.runtime_seed_guard import check_simulation_seed
from robot_sf.sim.sim_config import SimulationSettings

METRICS = ("success", "collision", "near_miss", "time_to_goal")


def require_dev_seeds(seeds: list[int]) -> None:
    """Refuse all seeds outside 1001..1030, including None, booleans and duplicates."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("development seeds must be nonempty and unique")
    for seed in seeds:
        check_simulation_seed(seed, boundary="pedsens", dev_only=True)


def profile_scenario(scenario: dict, profile: dict, seed: int) -> dict:
    """Bind profile through the normal scenario override path and check dependency.

    Returns:
        A copied scenario with explicit profile and dev RNG overrides.
    """
    require_dev_seeds([seed])
    if "obstacle_force_profile" in profile:
        available = {item.name for item in fields(SimulationSettings)}
        available.update(getattr(SimulationSettings, "__dataclass_fields__", {}))
        if "obstacle_force_profile" not in available:
            raise ValueError("wall/combined requires PR #10073; no legacy substitution allowed")
    result = deepcopy(scenario)
    result.pop("seeds", None)
    result["seed"] = seed
    simulation = result.setdefault("simulation_config", {})
    simulation.update(profile)
    # Every stochastic pedestrian stream uses the paired dev seed, even baseline.
    for key in ("route_spawn_seed", "archetype_seed", "response_law_seed"):
        simulation[key] = seed
    if simulation.get("desired_speed_mean") is not None:
        simulation["desired_speed_seed"] = seed
    return result


def metric_values(record: dict) -> dict[str, float | None]:
    """Extract native rates and success-only seconds, rejecting missing metrics.

    Returns:
        Bernoulli outcomes (episode contains any collision/near miss), and seconds.
    """
    metrics = record["metrics"]
    values = {
        "success": float(metrics["success"]),
        "collision": float(float(metrics["collisions"]) > 0),
        "near_miss": float(float(metrics["near_misses"]) > 0),
        "time_to_goal": metrics.get("time_to_goal"),
    }
    for key in METRICS:
        raw = metrics.get({"collision": "collisions", "near_miss": "near_misses"}.get(key, key))
        if key != "time_to_goal" and (raw is None or not np.isfinite(float(raw))):
            raise ValueError(f"invalid native {key}")
    if values["success"] and values["time_to_goal"] is None:
        # Native core metrics export normalized goal time, not always raw seconds.
        # This is exactly the native time_to_goal identity: norm * horizon * dt.
        values["time_to_goal"] = (
            float(metrics["time_to_goal_norm_success_only"])
            * int(record["effective_budget_steps"])
            * float(record["scenario_params"]["run_dt"])
        )
    if values["success"] and (
        values["time_to_goal"] is None or not np.isfinite(float(values["time_to_goal"]))
    ):
        raise ValueError("successful native row has unavailable time-to-goal")
    values["time_to_goal"] = (
        float(values["time_to_goal"])
        if values["success"]
        and values["time_to_goal"] is not None
        and np.isfinite(float(values["time_to_goal"]))
        else None
    )
    return values


def summarize(  # noqa: C901
    rows: list[dict], *, samples: int = 2000, confidence: float = 0.95, bootstrap_seed: int = 1001
) -> dict[str, Any]:
    """Paired seed-cluster bootstrap over a fixed scenario matrix and roster.

    Each resampled dev seed retains all scenarios, planners and factors. The
    time delta conditions on both arms succeeding (and reports its pair count).
    Tied success ranks use tau-b; degenerate draws are reported, never zero-filled.

    Returns:
        Ranking correlation and per-planner deltas with percentile CIs.
    """
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("positive bootstrap samples and confidence in (0, 1) required")
    require_dev_seeds([bootstrap_seed])
    table = {}
    sources = {}
    for row in rows:
        require_dev_seeds([row["seed"]])
        key = (row["factor"], row["planner"], row["scenario"], row["seed"])
        if key in table:
            raise ValueError(f"duplicate paired row: {key}")
        for metric in METRICS:
            value = row["values"][metric]
            if value is None and metric == "time_to_goal":
                continue
            if value is None or not np.isfinite(value):
                raise ValueError(f"invalid {metric}: {key}")
            if metric != "time_to_goal" and value not in (0, 1):
                raise ValueError(f"rate must be a Bernoulli outcome: {key}")
        table[key] = row["values"]
        sources[key] = {
            "episode_path": row.get("episode_path"),
            "failure_evidence": row.get("failure_evidence"),
        }
    factors = sorted({key[0] for key in table})
    planners = sorted({key[1] for key in table})
    scenarios = sorted({key[2] for key in table})
    seeds = sorted({key[3] for key in table})
    if "legacy" not in factors or len(planners) < 2:
        raise ValueError("legacy baseline and at least two planners required")
    expected = len(factors) * len(planners) * len(scenarios) * len(seeds)
    if len(table) != expected:
        raise ValueError("incomplete paired factor/planner/scenario/seed grid")
    alpha = (1 - confidence) / 2
    draws = np.random.default_rng(bootstrap_seed).choice(seeds, (samples, len(seeds)))

    def interval(point, values):
        finite = [float(value) for value in values if value is not None and np.isfinite(value)]
        return {
            "estimate": point,
            "ci": list(np.quantile(finite, [alpha, 1 - alpha])) if finite else None,
            "valid_draws": len(finite),
            "undefined_draws": samples - len(finite),
        }

    def scores(factor, selected):
        return [
            np.mean(
                [
                    table[factor, planner, scenario, seed]["success"]
                    for seed in selected
                    for scenario in scenarios
                ]
            )
            for planner in planners
        ]

    def tau(factor, selected):
        value = float(kendalltau(scores("legacy", selected), scores(factor, selected)).statistic)
        return value if np.isfinite(value) else None

    output = {
        "schema_version": "pedsens-summary.v1",
        "evidence_status": "diagnostic-only",
        "ranking_metric": "success",
        "tau_variant": "b",
        "confidence": confidence,
        "resampling_unit": "paired seed cluster; scenarios fixed",
        "time_to_goal_estimand": "mean paired seconds among successes in both arms",
        "seeds": seeds,
        "scenarios": scenarios,
        "planners": planners,
        "episode_count": len(rows),
        "factors": {},
        "new_failures": [],
    }
    for factor in factors:
        deltas = {}
        for planner in planners:
            deltas[planner] = {}
            for metric in METRICS:

                def differences(selected):
                    result = []
                    for seed in selected:
                        for scenario in scenarios:
                            base = table["legacy", planner, scenario, seed][metric]
                            variant = table[factor, planner, scenario, seed][metric]
                            if base is not None and variant is not None:
                                result.append(variant - base)
                    return result

                observed = differences(seeds)
                deltas[planner][metric] = interval(
                    float(np.mean(observed)) if observed else None,
                    [
                        float(np.mean(values)) if (values := differences(draw)) else None
                        for draw in draws
                    ],
                )
                deltas[planner][metric]["paired_n"] = len(observed)
            for scenario in scenarios:
                for seed in seeds:
                    base = table["legacy", planner, scenario, seed]
                    variant = table[factor, planner, scenario, seed]
                    if (
                        base["success"] > variant["success"]
                        or base["collision"] < variant["collision"]
                    ):
                        output["new_failures"].append(
                            {
                                "factor": factor,
                                "planner": planner,
                                "scenario": scenario,
                                "seed": seed,
                                "class": "new_contact"
                                if base["collision"] < variant["collision"]
                                else "lost_success",
                                "mechanism": "unclassified_requires_trace_review",
                                "gate_admitted": False,
                                "baseline_source": sources["legacy", planner, scenario, seed],
                                "variant_source": sources[factor, planner, scenario, seed],
                                "baseline": base,
                                "variant": variant,
                            }
                        )
        output["factors"][factor] = {
            "kendall_tau": interval(tau(factor, seeds), [tau(factor, draw) for draw in draws]),
            "deltas": deltas,
        }
    return output
