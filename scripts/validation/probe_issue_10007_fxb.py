"""Diagnostic FXB map-runner probes; exclusively development seeds 1001–1010."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from robot_sf.benchmark.map_runner.map_runner import run_map_batch
from robot_sf.planner.socnav_orca import ORCAPlannerAdapter
from robot_sf.planner.socnav_social_force import SocialForcePlannerAdapter
from robot_sf.training.scenario_loader import load_scenarios

SCENARIOS = (
    "classic_doorway_medium",
    "classic_head_on_corridor_medium",
    "francis2023_frontal_approach",
)
SUITE = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")
TEMPLATE = Path(
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)


def main() -> None:
    """Run native release arms and preserve raw episode and action evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--algo", choices=("social_force", "orca"), required=True)
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=TEMPLATE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    logger.remove()
    logger.add(args.out / "runtime.log", level="WARNING")
    template = yaml.safe_load(args.manifest.read_text())
    arm = next(p for p in template["planners"] if p["key"] == args.algo)
    config_path = Path(arm["algo_config"]) if arm.get("algo_config") else None
    config = yaml.safe_load(config_path.read_text()) if config_path else {}
    if args.algo == "orca":
        config["orca_adapter_trace_enabled"] = True
    probe_config = args.out / "algo.yaml"
    probe_config.write_text(yaml.safe_dump(config))
    scenarios = load_scenarios(SUITE)
    provenance = {
        "evidence_status": "diagnostic-only",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"])).hexdigest(),
        "manifest": str(args.manifest),
        "algo_config": str(config_path) if config_path else None,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()
        if config_path
        else None,
        "horizon": 600,
        "dt": 0.1,
        "seeds": list(range(1001, 1011)),
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    cls = SocialForcePlannerAdapter if args.algo == "social_force" else ORCAPlannerAdapter
    original = cls.plan
    actions = []

    def observed_plan(self, observation):
        if not actions:
            np.savez_compressed(args.out / "last_reset_observation.npz", **observation)
        command = original(self, observation)
        robot, _, _ = self._socnav_fields(observation)
        row = {
            "command": list(command),
            "heading": float(np.asarray(robot["heading"]).ravel()[0]),
            "physical_speed": float(np.asarray(robot["speed"]).ravel()[0]),
            "flat_dt": float(np.asarray(observation.get("sim_timestep", [np.nan])).ravel()[0]),
            "nested_sim_present": "sim" in observation,
        }
        if args.algo == "social_force":
            row["resolved_dt"] = self._resolve_dt(observation)
        else:
            trace = self.adapter_trace()
            row["adapter_trace"] = trace[-1] if trace else None
        actions.append(row)
        return command

    cls.plan = observed_plan
    try:
        for name in SCENARIOS[1:2] if args.snapshot_only else SCENARIOS:
            scenario = copy.deepcopy(next(s for s in scenarios if s["name"] == name))
            for seed in range(1001, 1002) if args.snapshot_only else range(1001, 1011):
                actions.clear()
                scenario["seeds"] = [seed]
                target = args.out / f"{name}_{seed}.jsonl"
                result = run_map_batch(
                    [scenario],
                    target,
                    schema_path="robot_sf/benchmark/schemas/episode.schema.v1.json",
                    scenario_path=SUITE,
                    provenance_scenario_path=SUITE,
                    algo=args.algo,
                    algo_config_path=str(probe_config),
                    horizon=600,
                    dt=0.1,
                    workers=1,
                    resume=False,
                    socnav_missing_prereq_policy="fail-fast",
                )
                (args.out / f"{name}_{seed}_actions.json").write_text(json.dumps(actions))
                if result["failed_jobs"] or result["written"] != 1 or not actions:
                    raise RuntimeError(f"Probe failed: {name}/{seed}: {result}")
                print(
                    json.dumps({"scenario": name, "seed": seed, "steps": len(actions)}), flush=True
                )
    finally:
        cls.plan = original


if __name__ == "__main__":
    main()
