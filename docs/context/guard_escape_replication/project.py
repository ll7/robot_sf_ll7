"""Project four replication producer files into compact, endpoint-complete evidence.

Optional custody audit: run in the pinned acquisition environment so its canonical
fallback parser is available. Numerical reproduction uses analyze.py instead.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import yaml
from analyze import PACKET, require

from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker


def main():
    """Reject unexpected inputs and bind every projected row to exact source bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources-output", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load((PACKET / "inputs/study.yaml").read_text())
    paths = sorted(args.raw_root.glob("runs/*/episodes.jsonl"))
    require(len(paths) == 4, "Expected exactly four full-run producer files")
    rows, sources = [], []
    constants = None
    for path in paths:
        arm = path.parent.name.split("__", 1)[0].upper()
        require(arm in {"G0E0", "G0E1", "G1E0", "G1E1"}, "Unknown factorial arm")
        source_path = str(path.relative_to(args.raw_root))
        source = {
            "path": source_path,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "rows": 0,
        }
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                require(
                    type(raw["seed"]) is int and raw["seed"] in spec["seeds"],
                    "Non-development seed rejected",
                )
                meta = raw["algorithm_metadata"]
                runtime_meta = dict(meta)
                runtime_meta.pop("simulation_step_trace", None)
                runtime_meta.pop("paired_effect_metric_producer", None)
                marker = runtime_fallback_or_degraded_marker(
                    runtime_meta, expected_algorithm="hybrid_rule_local_planner"
                )
                require(marker is None, "Fallback/degraded producer rejected")
                require(
                    meta["algorithm"] == "hybrid_rule_local_planner"
                    and meta["planner_contract"]["planner_id"] == "hybrid_rule_local_planner",
                    "Producer algorithm mismatch",
                )
                require(
                    meta["planner_kinematics"]["execution_mode"] == "adapter"
                    and meta["planner_kinematics"]["adapter_name"]
                    == "HybridRuleLocalPlannerAdapter"
                    and meta["planner_kinematics"]["robot_kinematics"] == "differential_drive"
                    and meta["planner_kinematics"]["planner_command_space"] == "unicycle_vw",
                    "Unexpected execution binding",
                )
                factors = ["ablation_disable_hard_guard", *spec["escape"]["bundle_switches"]]
                invariant = {
                    key: value for key, value in meta["config"].items() if key not in factors
                }
                if constants is None:
                    constants = invariant
                require(constants == invariant, "Non-factor controller settings differ")
                rows.append(
                    {
                        "planner_key": arm,
                        "scenario_id": raw["scenario_id"],
                        "seed": raw["seed"],
                        "git_hash": raw["git_hash"],
                        "mode": meta["planner_kinematics"]["execution_mode"],
                        "status": meta["status"],
                        "fallback_count": meta["planner_runtime"]["fallback_count"],
                        "fallback_or_degraded_marker": marker,
                        "factors": {key: meta["config"][key] for key in factors},
                        "effective_budget_steps": raw["effective_budget_steps"],
                        "steps": raw["steps"],
                        "termination_reason": raw["termination_reason"],
                        "outcome": raw["outcome"],
                        "metrics": {
                            key: raw["metrics"][key]
                            for key in (
                                "success",
                                "ped_collision_count",
                                "obstacle_collision_count",
                            )
                        },
                        "source_path": source_path,
                        "source_line": line_number,
                        "source_line_sha256": hashlib.sha256(line).hexdigest(),
                    }
                )
                source["rows"] += 1
        require(source["rows"] == 540, "Incomplete producer file")
        sources.append(source)
    require(len(rows) == 2160, "Incomplete full run")
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    args.sources_output.write_text(
        json.dumps(
            {
                "schema": "guard-escape-ablation.projection.v1",
                "execution_commit": spec["execution_commit"],
                "raw_root_role": "full/campaigns/exp_guardesc_replication",
                "files": sources,
                "rows": 2160,
                "non_factor_controller_settings_equal": True,
                "compact_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "row_digest_definition": "SHA-256 of exact producer line bytes, including its newline",
                "projection": "Endpoint metrics, outcome, arm/seed/cell, source/factor/execution identity, "
                "authored budget, termination and producer line identity; no trajectories.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
