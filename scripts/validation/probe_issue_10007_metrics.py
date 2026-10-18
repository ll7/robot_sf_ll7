"""Reproduce F4/F5/F7/F8 on development episodes, one simulation at a time.

Run this same script from the base and corrected checkouts, with distinct output
folders. Outputs are diagnostic evidence; they admit no release or paper claim.
The captures include reset/terminal references and raw positions, so path and
failure-distance changes can be checked independently of reported metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from loguru import logger

from robot_sf.benchmark.map_runner import map_runner as runner
from robot_sf.benchmark.map_runner import map_runner_episode as episode
from robot_sf.training.scenario_loader import load_scenarios


def main() -> None:
    """Run twelve diagnostic episodes on seeds 1001/1002 and save reviewable bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    args.output.mkdir(parents=True, exist_ok=True)
    logger.remove()
    path = root / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    scenarios = {s["name"]: s for s in load_scenarios(path)}
    scenarios["fxm_open_dev"] = {
        "name": "fxm_open_dev",
        "map_file": str(root / "maps/svg_maps/planner_sanity_open.svg"),
        "simulation_config": {"ped_density": 0.0, "max_episode_steps": 600},
        "single_pedestrians": [],
        "seeds": [1002],
    }
    original_init = episode._init_step_loop_state
    original_post = episode._compute_post_loop_metrics
    capture = {}

    def initialize(**kwargs):
        state = original_init(**kwargs)
        capture["reset"] = {
            "start": state.initial_robot_pos.tolist(),
            "terminal_goal": list(kwargs["env"].simulator.robot_navs[0].waypoints[-1]),
            "trace_goal": state.goal_vec.tolist(),
            "initial_goal_distance_m": state.initial_goal_distance,
        }
        return state

    def compute(**kwargs):
        result = original_post(**kwargs)
        capture.update(
            shortest_path_m=result.shortest_path,
            metric_goal=kwargs["goal_vec"].tolist(),
            positions=result.robot_pos_arr.tolist(),
            accelerations=result.robot_acc_arr.tolist(),
            reached_goal_step=kwargs["reached_goal_step"],
        )
        return result

    episode._init_step_loop_state = initialize
    episode._compute_post_loop_metrics = compute
    try:
        for name in (
            "classic_doorway_low",
            "classic_merging_low",
            "francis2023_frontal_approach",
            "fxm_open_dev",
        ):
            seed = 1002 if name == "fxm_open_dev" else 1001
            for planner in ("goal", "orca", "hybrid_rule_local_planner"):
                capture.clear()
                candidate = (
                    "configs/policy_search/candidates/"
                    "hybrid_rule_v4_fast_progress_static_escape_s30_h600_release.yaml"
                    if planner == "hybrid_rule_local_planner"
                    else None
                )
                row = runner._run_map_episode(
                    copy.deepcopy(scenarios[name]),
                    seed,
                    horizon=600,
                    dt=0.1,
                    record_forces=True,
                    snqi_weights=None,
                    snqi_baseline=None,
                    algo=planner,
                    scenario_path=path,
                    algo_config_path=candidate,
                    record_simulation_step_trace=True,
                )
                trace = row["algorithm_metadata"]["simulation_step_trace"]
                capture["first_segment_m"] = math.dist(
                    capture["reset"]["start"], capture["positions"][0]
                )
                capture["final_trace_progress_m"] = trace["initial_goal_distance_m"] - math.dist(
                    capture["positions"][-1], capture["reset"]["trace_goal"]
                )
                row["fxm_probe"] = copy.deepcopy(capture)
                target = args.output / f"{name}__{planner}__{seed}.json"
                target.write_text(json.dumps(row, default=np.ndarray.tolist) + "\n")
                print(target.name, row["steps"], row["outcome"], flush=True)
    finally:
        episode._init_step_loop_state = original_init
        episode._compute_post_loop_metrics = original_post
    source_paths = [
        "robot_sf/benchmark/metrics.py",
        "robot_sf/benchmark/map_runner/map_runner_episode.py",
        "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml",
        "scripts/validation/probe_issue_10007_metrics.py",
    ]
    identity = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "seeds": [1001, 1002],
        "simulations_concurrent": 1,
        "evidence": "diagnostic-only; no calibration, held-out or release evaluation",
        "source_sha256": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths
        },
    }
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
