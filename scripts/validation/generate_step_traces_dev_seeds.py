"""Generate simulation-step-trace rows on development seeds only (#9979).

Allows only DEV_SEEDS (1001-1030), excluding both evaluation bands. Rows include per-step goal and collision
fields for check_step_trace_invariants.py.

Usage:
    uv run python scripts/validation/generate_step_traces_dev_seeds.py --algo risk_dwa \
        --algo-config configs/algos/risk_dwa_camera_ready.yaml --seeds 1001 1002 \
        --empty-world --out traces.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from robot_sf.benchmark.classic_interactions_loader import load_classic_matrix
from robot_sf.benchmark.map_runner.map_runner import _run_map_episode
from robot_sf.benchmark.seed_bands import DEV_SEEDS
from robot_sf.scenario_certification.feasibility_diagnostics import make_actor_free_scenario
from robot_sf.scenario_certification.v1 import scenario_actor_source_census
from robot_sf.training.scenario_loader import build_robot_config_from_scenario


def main() -> int:
    """CLI entry point.

    Returns:
        0 on success, 2 when any non-development seed is requested.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="configs/scenarios/classic_interactions_francis2023.yaml")
    ap.add_argument("--algo", required=True)
    ap.add_argument("--algo-config", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--scenario-ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--empty-world", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: run scenarios where index %% n == i")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if any(s not in DEV_SEEDS for s in a.seeds):
        print("refusing non-development seed; only DEV_SEEDS 1001-1030 allowed", file=sys.stderr)
        return 2
    scen = load_classic_matrix(a.matrix)
    if a.scenario_ids:
        scen = [
            s
            for s in scen
            if s.get("name") in a.scenario_ids
            or s.get("scenario_id") in a.scenario_ids
            or s.get("id") in a.scenario_ids
        ]
    if a.limit:
        scen = scen[: a.limit]
    si, sn = (int(x) for x in a.shard.split("/"))
    scen = [s for i, s in enumerate(scen) if i % sn == si]
    with open(a.out, "w") as fh:
        for sc in scen:
            sc = copy.deepcopy(sc)
            if a.empty_world:
                sc = make_actor_free_scenario(sc)
                metadata = dict(sc.get("metadata") or {})
                if metadata.get("behavior") in {
                    "wait",
                    "join",
                    "leave",
                    "follow",
                    "lead",
                    "accompany",
                }:
                    metadata["empty_world_original_behavior"] = metadata["behavior"]
                    metadata["behavior"] = "none"
                    sc["metadata"] = metadata
                config = build_robot_config_from_scenario(sc, scenario_path=Path(a.matrix))
                census = scenario_actor_source_census(config)
                if census["verified_empty"] is not True:
                    raise RuntimeError(f"empty-world actor census failed: {census}")
            for seed in a.seeds:
                rec = _run_map_episode(
                    sc,
                    seed,
                    horizon=a.horizon,
                    dt=a.dt,
                    record_forces=False,
                    snqi_weights=None,
                    snqi_baseline=None,
                    algo=a.algo,
                    algo_config_path=a.algo_config,
                    scenario_path=Path(a.matrix),
                    record_simulation_step_trace=True,
                )
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                print(
                    sc.get("name") or sc.get("id"),
                    seed,
                    rec.get("termination_reason"),
                    rec.get("steps"),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
