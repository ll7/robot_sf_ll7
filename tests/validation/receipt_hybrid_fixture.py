"""Source-built hybrid adapter metadata for fresh synthetic producer rows."""

import copy
from functools import cache

from robot_sf.benchmark.map_runner_policies.rule_and_grid import build


@cache
def _metadata():
    _policy, metadata = build(
        "hybrid_rule_local_planner", {}, robot_kinematics="differential_drive"
    )
    metadata["paired_effect_metric_producer"] = {
        "pairing": {"key": {"planner": "hybrid_rule_local_planner"}}
    }
    return metadata


def hybrid_episode(row):
    """Complete the fixture before normal sealing; no planner episode is executed."""
    row = copy.deepcopy(row)
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    row["algo"] = "hybrid_rule_local_planner"
    row["algorithm_metadata"] = {**copy.deepcopy(_metadata()), "simulation_step_trace": trace}
    row["event_ledger"] = {**row.get("event_ledger", {}), "planner": row["algo"]}
    row["scenario_params"] = {**row.get("scenario_params", {}), "algo": row["algo"]}
    for step in trace["steps"]:
        planner = step["planner"]
        for name in ("selected_action", "applied_environment_action"):
            planner[name] = {"linear_velocity": 1.0, "angular_velocity": 0.0}
        planner["action_conversion"] = {
            "schema_version": "policy-action-conversion.v1",
            "kind": "unicycle_velocity_to_acceleration",
        }
    return row
