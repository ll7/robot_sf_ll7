"""Audit full producer census and within-block input contracts before inference."""

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path

import yaml

from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker


def paired_contract(r, m, k):
    """Select input contracts, omitting only arm-dependent provenance hashes."""
    params = copy.deepcopy(r["scenario_params"])
    params.pop("algo_config_hash", None)
    invariant = {
        field: r[field]
        for field in (
            "selected_map_identity",
            "development_pedestrian_model",
            "evaluation_pedestrian_model",
            "pedestrian_model",
            "observation_level",
            "observation_mode",
            "observation_noise",
            "observation_noise_hash",
            "tracking_precision",
            "tracking_precision_hash",
            "effective_budget_steps",
            "horizon",
            "metric_schema_version",
        )
    }
    invariant["scenario_params"] = params
    invariant["planner_contract"] = m["planner_contract"]
    invariant["planner_kinematics"] = k
    invariant["observation_spec"] = m["observation_spec"]
    invariant["observation_visibility"] = m["observation_visibility"]
    obstacle = copy.deepcopy(m["obstacle_force_law"])
    for site in obstacle.get("sites", {}).values():
        site.pop("config_hash", None)
    # The whole-config provenance digest varies by factor; force parameters do not.
    invariant["obstacle_force_law"] = {
        key: value
        for key, value in obstacle.items()
        if key not in ("configuration_hash", "configuration_sha256")
    }
    return invariant


def main():
    """Reject incomplete or mismatched full producer designs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = yaml.safe_load((args.packet / "inputs/study.yaml").read_text())
    cells = [cell for cells in study["groups"].values() for cell in cells]
    seeds = study["seeds"]
    expected = {(f"G{g}E{e}", c, s) for g in (0, 1) for e in (0, 1) for c in cells for s in seeds}
    slots = set()
    blocks = {}
    counts = collections.Counter()
    raw_success = collections.Counter()
    statuses = collections.Counter()
    sources = []
    paths = sorted(args.raw_root.glob("runs/*/episodes.jsonl"))
    assert len(paths) == 4
    for path in paths:
        arm = path.parent.name.split("__")[0].upper()
        n = 0
        with path.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                r = json.loads(line)
                m = r["algorithm_metadata"]
                n += 1
                key = (arm, r["scenario_id"], r["seed"])
                assert key in expected and key not in slots
                slots.add(key)
                assert r["git_hash"] == study["execution_commit"]
                assert m["status"] == "ok" and m["planner_runtime"]["fallback_count"] == 0
                statuses[str(r["status"])] += 1
                runtime_meta = dict(m)
                runtime_meta.pop("simulation_step_trace", None)
                runtime_meta.pop("paired_effect_metric_producer", None)
                assert (
                    runtime_fallback_or_degraded_marker(
                        runtime_meta, expected_algorithm="hybrid_rule_local_planner"
                    )
                    is None
                )
                k = m["planner_kinematics"]
                assert (
                    k["execution_mode"] == "adapter"
                    and k["adapter_name"] == "HybridRuleLocalPlannerAdapter"
                )
                assert (
                    k["robot_kinematics"] == "differential_drive"
                    and k["planner_command_space"] == "unicycle_vw"
                )
                counts["adapter"] += 1
                invariant = paired_contract(r, m, k)
                block = (r["scenario_id"], r["seed"])
                if block in blocks:
                    if blocks[block] != invariant:
                        differing = [
                            key for key in invariant if blocks[block][key] != invariant[key]
                        ]
                        raise ValueError(
                            "Nonfactor paired contracts differ: " + ", ".join(differing)
                        )
                else:
                    blocks[block] = invariant
                raw_success[arm] += bool(
                    r["outcome"]["route_complete"] and not r["outcome"]["collision_event"]
                )
        assert n == 540
        sources.append(
            {
                "path": str(path.relative_to(args.raw_root)),
                "rows": n,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    assert slots == expected and len(blocks) == 540
    args.output.write_text(
        json.dumps(
            {
                "schema": "guard-escape-replication.raw-audit.v1",
                "passed": True,
                "episodes": len(slots),
                "complete_seed_cell_blocks": len(blocks),
                "seed_blocks": 30,
                "actual_development_seeds": seeds,
                "duplicates": 0,
                "missing_slots": 0,
                "exclusions": 0,
                "execution_census": {
                    "adapter": counts["adapter"],
                    "native": 0,
                    "fallback": 0,
                    "degraded": 0,
                },
                "adapter_name": "HybridRuleLocalPlannerAdapter",
                "robot_kinematics": "differential_drive",
                "nonfactor_paired_contracts_equal": True,
                "producer_status_counts": dict(statuses),
                "raw_success_counts_per_arm": dict(raw_success),
                "producer_files": sources,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "PASS: 2160 producer rows; 540 paired input contracts; adapter binding; zero fallback/degraded"
    )


if __name__ == "__main__":
    main()
