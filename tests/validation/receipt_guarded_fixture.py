"""Complete guarded fixture metadata emitted by the actual native shield writer."""

import copy
from functools import cache

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.planner.guarded_ppo import (
    GuardedPPOAdapter,
    GuardedPPOConfig,
    build_guarded_ppo_fallback,
)
from robot_sf.planner.safety_shield import (
    new_shield_stats,
    shield_contract_metadata,
    update_shield_stats,
)


@cache
def _metadata(label="ppo_safe"):
    fallback = build_guarded_ppo_fallback({})
    adapter = GuardedPPOAdapter(config=GuardedPPOConfig(), fallback_adapter=fallback)
    decision = adapter._shield_decision(
        ppo_command=(1.0, 0.0),
        filtered_command=(1.0, 0.0),
        label=label,
        reason="fixture safe command",
        ppo_eval={"safe": True},
        selected_eval={"safe": True},
        fallback_policy="stop" if label in {"stop_safe", "stop_best_effort"} else None,
        hard_constraint_violation=label in {"stop_best_effort", "fallback_best_effort"},
    )
    metadata = enrich_algorithm_metadata(algo="guarded_ppo")
    metadata["status"] = "ok"
    metadata["safety_shield_contract"] = shield_contract_metadata(
        shield_name="GuardedPPOAdapter",
        prediction_source="short_horizon_rollout",
        fallback_policy=type(fallback).__name__,
    )
    metadata["shield_stats"] = update_shield_stats(new_shield_stats(), decision)
    metadata["planner_runtime"] = {**adapter.diagnostics(), "last_decision": decision.to_metadata()}
    metadata["paired_effect_metric_producer"] = {"pairing": {"key": {"planner": "guarded_ppo"}}}
    return metadata


def guarded_episode(row, *, label="ppo_safe"):
    """Complete a synthetic row with source-emitted guarded identity and diagnostics."""
    row = copy.deepcopy(row)
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    row["algo"] = "guarded_ppo"
    row["algorithm_metadata"] = {**copy.deepcopy(_metadata(label)), "simulation_step_trace": trace}
    row["event_ledger"] = {**row.get("event_ledger", {}), "planner": "guarded_ppo"}
    row["scenario_params"] = {**row.get("scenario_params", {}), "algo": "guarded_ppo"}
    for step in trace["steps"]:
        planner = step["planner"]
        for key in ("selected_action", "applied_environment_action"):
            planner[key] = {"linear_velocity": 1.0, "angular_velocity": 0.0}
        planner["action_conversion"] = {
            "schema_version": "policy-action-conversion.v1",
            "kind": "unicycle_velocity_to_acceleration",
        }
    return row
