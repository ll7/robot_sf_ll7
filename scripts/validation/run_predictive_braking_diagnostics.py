#!/usr/bin/env python3
"""Native, dev-only paired stopping-bound diagnostics; never release evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger

from robot_sf.benchmark.map_runner.map_runner import (
    _build_env_config,
    _build_policy,
    _policy_command_to_env_action,
    _scenario_with_episode_seed_defaults,
)
from robot_sf.benchmark.termination_reason import route_complete_success
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_empty_world_sweep import assert_dev_seeds, remove_pedestrians
from scripts.validation.run_hybrid_feasibility_diagnostics import MAIN_MATRIX, ROOT, hybrid_config


def run_cell(task):  # noqa: PLR0915 -- native episode custody stays in one try/finally
    """Execute the canonical native policy/environment loop and record actual outcomes."""
    scenario, seed, predictive, empty, horizon = task
    logger.remove()
    logger.add(os.sys.stderr, level="ERROR")
    scenario = _scenario_with_episode_seed_defaults(dict(scenario, seeds=[seed]), seed=seed)
    if empty:
        scenario = remove_pedestrians(scenario, [seed])
    if horizon:
        scenario["simulation_config"] = dict(
            scenario.get("simulation_config") or {}, max_episode_steps=horizon
        )
    cfg = _build_env_config(scenario, scenario_path=MAIN_MATRIX)
    pcfg = hybrid_config(scenario)
    pcfg["debug_candidate_evaluator"] = False
    pcfg["v4_predictive_braking_enabled"] = predictive
    pcfg["v4_prediction_speed_error"] = 0.2
    policy, _ = _build_policy(
        "hybrid_rule_local_planner", pcfg, robot_kinematics="differential_drive"
    )
    env = make_robot_env(config=cfg, seed=seed, debug=False)
    steps = collisions = near_events = near_steps = 0
    previous_near = False
    minimum = float("inf")
    travel = 0.0
    success = False
    tube_checks = tube_violations = 0
    maximum_error = 0.0
    try:
        obs, _ = env.reset(seed=seed)
        policy._planner_bind_env(env)
        policy._planner_reset(seed=seed)
        initial = float(
            np.linalg.norm(np.array(env.simulator.goal_pos[0]) - env.simulator.robot_pos[0])
        )
        while True:
            pre = np.array(env.simulator.robot_pos[0])
            ped_pre = env.simulator.ped_pos.copy()
            ped_velocity = env.simulator.ped_vel.copy()
            command = policy(obs)
            action = _policy_command_to_env_action(env=env, config=cfg, command=command)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if predictive and ped_pre.shape == env.simulator.ped_pos.shape and ped_pre.size:
                dt = env.simulator.config.time_per_step_in_secs
                errors = np.linalg.norm(env.simulator.ped_pos - ped_pre - ped_velocity * dt, axis=1)
                tube_checks += len(errors)
                tube_violations += int(np.count_nonzero(errors > 0.2 * dt + 1e-9))
                maximum_error = max(maximum_error, float(np.max(errors)) / dt)
            travel += float(np.linalg.norm(env.simulator.robot_pos[0] - pre))
            meta = info.get("meta", {})
            contact = any(
                meta.get(k, False)
                for k in ("is_pedestrian_collision", "is_obstacle_collision", "is_robot_collision")
            )
            collisions += bool(contact)
            near = bool(meta.get("near_misses", 0))
            near_steps += near
            near_events += near and not previous_near
            previous_near = near
            minimum = min(minimum, float(meta.get("min_distance", float("inf"))))
            success = route_complete_success(info)
            if terminated or truncated or success:
                break
        runtime = policy._planner_adapter.diagnostics()
        if runtime.get("fallback_count", 0) or runtime.get("degraded_count", 0):
            raise RuntimeError("Fallback/degraded execution is not diagnostic evidence")
        result = {
            "scenario": scenario["name"],
            "seed": seed,
            "predictive": predictive,
            "empty": empty,
            "outcome": "collision" if collisions else "success" if success else "timeout",
            "steps": steps,
            "duration_s": steps * env.simulator.config.time_per_step_in_secs,
            "initial_goal_distance_m": initial,
            "final_goal_distance_m": float(
                np.linalg.norm(np.array(env.simulator.goal_pos[0]) - env.simulator.robot_pos[0])
            ),
            "travel_m": travel,
            "min_center_separation_m": minimum if np.isfinite(minimum) else None,
            "near_miss_events": near_events,
            "near_miss_steps": near_steps,
            "collision_steps": collisions,
            "one_step_tube_checks": tube_checks,
            "one_step_tube_violations": tube_violations,
            "maximum_one_step_prediction_error_m_s": maximum_error,
            "protective_stop_count": runtime.get("protective_stop_count", 0),
            "policy_config_sha256": hashlib.sha256(
                json.dumps(pcfg, sort_keys=True).encode()
            ).hexdigest(),
        }
    finally:
        env.close()
    print(f"{result['scenario']} {seed} predictive={predictive} {result['outcome']}", flush=True)
    return result


def main():
    """Resolve all seeds before execution and preserve provenance plus one row per cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1001, 1002])
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--predictive", action="store_true")
    parser.add_argument("--empty", action="store_true")
    parser.add_argument(
        "--horizon", type=int, default=0, help="0 retains authored scenario budgets"
    )
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = assert_dev_seeds(args.seeds)
    scenarios = load_scenarios(MAIN_MATRIX)
    if args.scenarios:
        scenarios = [s for s in scenarios if s["name"] in args.scenarios]
        if {s["name"] for s in scenarios} != set(args.scenarios):
            parser.error("unknown scenario")
    args.output.mkdir(parents=True, exist_ok=True)
    files = [MAIN_MATRIX, Path(__file__), ROOT / "robot_sf/planner/hybrid_rule_local_planner.py"]
    manifest = {
        "status": "diagnostic-only",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seeds": seeds,
        "scenarios": [s["name"] for s in scenarios],
        "predictive": args.predictive,
        "prediction_speed_error_m_s": 0.2,
        "empty": args.empty,
        "horizon_override": args.horizon,
        "workers": args.workers,
        "sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tasks = [
        (s, seed, args.predictive, args.empty, args.horizon) for s in scenarios for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        with (args.output / "episodes.csv").open("w", newline="") as stream:
            writer = None
            for row in pool.map(run_cell, tasks):
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
    print(f"Completed {len(tasks)} episodes", flush=True)


if __name__ == "__main__":
    main()
