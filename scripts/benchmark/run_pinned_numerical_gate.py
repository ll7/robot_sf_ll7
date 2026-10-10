#!/usr/bin/env python3
"""Run a bounded diagnostic empty-world sweep for the pinned numerical policy.

Use fresh default and pinned interpreters, then compare retained native rows.
Only dev seeds 1001–1003 run; this gate does not admit release evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf._numerical_mode import bootstrap_numerical_mode, initialize_pinned_torch
from robot_sf._numerical_thread_env import pin_thread_env_for_determinism

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("default", "pinned"), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
pin_thread_env_for_determinism()
if args.mode == "pinned":
    bootstrap_numerical_mode("pinned_float64_v1")

import yaml  # noqa: E402
from loguru import logger  # noqa: E402

from robot_sf.benchmark.camera_ready._config import (  # noqa: E402
    _load_campaign_scenarios,
    load_campaign_config,
)
from robot_sf.benchmark.map_runner.map_runner import run_map_batch  # noqa: E402

initialize_pinned_torch()
logger.remove()
logger.add(lambda message: print(message, end=""), level="WARNING")
ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = (
    "classic_bottleneck_medium",
    "classic_cross_trap_medium",
    "classic_doorway_medium",
    "classic_group_crossing_medium",
    "classic_head_on_corridor_medium",
    "classic_overtaking_medium",
    "francis2023_frontal_approach",
    "francis2023_blind_corner",
    "francis2023_narrow_doorway",
    "francis2023_circular_crossing",
)
if __name__ == "__main__":
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_campaign_config(
        ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_1_0.yaml"
    )
    scenarios = [s for s in _load_campaign_scenarios(cfg) if s["name"] in SCENARIOS]
    if len(scenarios) != len(SCENARIOS):
        raise RuntimeError("Incomplete empty-world scenario inventory")
    for scenario in scenarios:
        scenario["seeds"] = [1001, 1002, 1003]
        scenario["_diagnostic_remove_pedestrian_actors"] = True
        scenario["map_file"] = str((ROOT / scenario["map_file"]).resolve())
    summaries = {}
    for planner in cfg.planners:
        algo_path = planner.algo_config_path
        if algo_path is not None and args.mode == "default":
            payload = yaml.safe_load(algo_path.read_text())
            payload.pop("numerical_mode", None)
            algo_path = args.output / f"{planner.key}.yaml"
            algo_path.write_text(yaml.safe_dump(payload))
        output = args.output / f"{planner.key}.jsonl"
        summaries[planner.key] = run_map_batch(
            scenarios,
            output,
            ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json",
            scenario_path=cfg.scenario_matrix_path,
            horizon=0,
            dt=0.1,
            algo=planner.algo,
            algo_config_path=str(algo_path) if algo_path else None,
            benchmark_profile=planner.benchmark_profile,
            socnav_missing_prereq_policy="fail-fast",
            adapter_impact_eval=True,
            workers=4,
            resume=False,
        )
        print(
            json.dumps(
                {
                    "arm": planner.key,
                    "written": summaries[planner.key].get("written"),
                    "total_jobs": summaries[planner.key].get("total_jobs"),
                }
            ),
            flush=True,
        )
    (args.output / "summaries.json").write_text(json.dumps(summaries, indent=2, default=str) + "\n")
