"""Matched native dev-only validation for the opt-in guarded-progress identity.

Resolve frozen profiles before renaming scenarios. Keep the authored budgets,
native plant and reset matched, and change only the planner variant. Never use
release seed schedules. Results are diagnostics, not release evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from robot_sf.benchmark.map_runner.map_runner import (
    _build_env_config,
    _build_policy,
    _policy_command_to_env_action,
    _scenario_with_episode_seed_defaults,
)
from robot_sf.benchmark.termination_reason import collision_event, route_complete_success
from robot_sf.common.hybrid_defaults import defaults_for_source
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.planner.hybrid_rule_local_planner import HYBRID_RULE_V4_GUARDED_PROGRESS_VARIANT
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_empty_world_sweep import remove_pedestrians
from scripts.validation.run_policy_search_candidate import _effective_candidate_runtime_for_scenario
from scripts.validation.run_policy_search_step_diagnostics import _json_ready

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
SCHEDULE = ROOT / "configs/benchmarks/horizon_schedules/release_0_0_8_authored_v1.yaml"
CANDIDATES = ROOT / "configs/policy_search/candidates"
PROFILES = {
    "static": "hybrid_rule_v4_fast_progress_static_escape",
    "continuous": "hybrid_rule_v4_fast_progress_static_escape_continuous",
    "adaptive_bottleneck": "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4",
    "adaptive_collision": "scenario_adaptive_hybrid_orca_v2_collision_guard_v4",
}
DEV_SCENARIOS = (
    "francis2023_join_group",
    "francis2023_leave_group",
    "francis2023_crowd_navigation",
    "classic_station_platform_medium",
    "classic_merging_low",
    "francis2023_perpendicular_traffic",
)


def digest(value) -> str:
    """Return a deterministic digest of JSON-ready diagnostic state."""
    return hashlib.sha256(json.dumps(_json_ready(value), sort_keys=True).encode()).hexdigest()


def episode(task) -> dict:  # noqa: PLR0915 - keep the native control loop in execution order
    """Run one authored-budget episode and return native outcomes and progress."""
    name, seed, profile, variant, empty = task
    assert 1001 <= seed <= 1030
    logger.remove()
    logger.add(os.sys.stderr, level="ERROR")
    source = copy.deepcopy(next(s for s in load_scenarios(MATRIX) if s["name"] == name))
    source["seeds"] = [seed]
    path = CANDIDATES / (PROFILES[profile] + "_s30_h600_release_0_0_8_frozen.yaml")
    manifest = yaml.safe_load(path.read_text())
    cfg = yaml.safe_load((ROOT / manifest["base_config_path"]).read_text())
    cfg.update(manifest.get("params", {}))
    algo, cfg = _effective_candidate_runtime_for_scenario(
        manifest,
        cfg,
        source,
        default_algo=manifest["algo"],
        config_anchor=path.parent,
    )
    # Adaptive leave-group is ORCA, so it is an unchanged native control.
    if variant == "guarded_progress" and algo == "hybrid_rule_local_planner":
        cfg["planner_variant"] = HYBRID_RULE_V4_GUARDED_PROGRESS_VARIANT
    if empty:
        source = remove_pedestrians(source, [seed])
    horizon = yaml.safe_load(SCHEDULE.read_text())["scenarios"][name]["recommended_horizon_steps"]
    source["name"] = "own10291_dev_" + name
    source.setdefault("simulation_config", {}).update(
        max_episode_steps=horizon,
        goal_completion_policy="goal_zone_entry_v1",
        social_force_kernel_version="wrapped_v2",
    )
    source = _scenario_with_episode_seed_defaults(source, seed=seed)
    with defaults_for_source(MATRIX):
        ecfg = _build_env_config(source, scenario_path=MATRIX)
        ecfg.sim_config.episode_step_limit = horizon
        ecfg.sim_config.sim_time_in_secs = horizon * 0.1
        env = make_robot_env(config=ecfg, seed=seed, debug=False)
    with defaults_for_source(path):
        policy, meta = _build_policy(algo, cfg, robot_kinematics="differential_drive")
    contacts = Counter()
    gates = Counter()
    streak = longest = 0
    path_length = 0.0
    success = collided = False
    trace = []
    try:
        obs, _ = env.reset(seed=seed)
        policy._planner_bind_env(env)
        policy._planner_reset(seed=seed)
        final_goal = np.asarray(env.simulator.robot_navs[0].waypoints[-1]).copy()
        reset_digest = digest(
            {
                "position": env.simulator.robot_pos,
                "peds": env.simulator.ped_pos,
                "final_goal": final_goal,
            }
        )
        if empty:
            assert len(env.simulator.ped_pos) == 0, "empty-world pedestrian residue"
        planner = getattr(policy, "_planner_adapter", None)
        for step in range(horizon):
            pre = np.asarray(env.simulator.robot_pos[0]).copy()
            command = policy(obs)
            decision = (
                planner.last_decision() if planner and algo == "hybrid_rule_local_planner" else {}
            )
            action = _policy_command_to_env_action(env=env, config=ecfg, command=command)
            obs, _, terminated, truncated, info = env.step(action)
            end = np.asarray(env.simulator.robot_pos[0]).copy()
            distance = float(np.linalg.norm(end - pre))
            path_length += distance
            collided |= collision_event(info)
            for key in ("is_pedestrian_collision", "is_obstacle_collision", "is_robot_collision"):
                contacts[key] += bool(info.get("meta", {}).get(key))
            success = route_complete_success(info)
            streak = streak + 1 if distance / 0.1 < 0.05 and not success else 0
            longest = max(longest, streak)
            safety = (decision or {}).get("speed_safety") or {}
            gates["zero_cap_steps"] += safety.get("speed_cap", float("inf")) <= 1e-9
            gates.update((decision or {}).get("rejection_counts", {}))
            trace.append(
                {
                    "step": step,
                    "speed": distance / 0.1,
                    "command": _json_ready(command),
                    "goal_distance": float(np.linalg.norm(end - final_goal)),
                    "speed_safety": safety,
                    "rejections": (decision or {}).get("rejection_counts", {}),
                }
            )
            if terminated or truncated or success:
                break
        runtime = planner.diagnostics() if planner else meta
        fallback = bool(runtime.get("fallback") or runtime.get("fallback_count", 0))
        degraded = bool(runtime.get("degraded") or runtime.get("degraded_count", 0))
        return _json_ready(
            {
                "scenario": name,
                "seed": seed,
                "profile": profile,
                "variant": variant,
                "empty": empty,
                "algo": algo,
                "config": cfg,
                "config_digest": digest(cfg),
                "reset_digest": reset_digest,
                "authored_horizon": horizon,
                "steps": len(trace),
                "outcome": "collision" if collided else "success" if success else "timeout",
                "contact": collided,
                "contacts": dict(contacts),
                "gates": dict(gates),
                "longest_stationary_s": longest * 0.1,
                "path_length_m": path_length,
                "final_goal_distance_m": trace[-1]["goal_distance"],
                "execution_mode": "degraded" if degraded else "fallback" if fallback else "native",
                "runtime": runtime,
                "trace": trace,
            }
        )
    finally:
        env.close()


def main() -> None:
    """Run paired dev episodes with bounded workers and explicit source identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--empty", action="store_true")
    parser.add_argument("--witness", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.workers <= 8
    scenarios = [s["name"] for s in load_scenarios(MATRIX)] if args.empty else DEV_SCENARIOS
    seeds = range(1001, 1011 if args.empty else 1031)
    profiles = ["static"] if args.empty else PROFILES
    if args.witness:
        scenarios, seeds, profiles = ["francis2023_join_group"], [1001], ["static"]
    tasks = [
        (n, s, p, v, args.empty)
        for n in scenarios
        for s in seeds
        for p in profiles
        for v in ("frozen_v4", "guarded_progress")
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "job": os.getenv("SLURM_JOB_ID"),
        "workers": args.workers,
        "tasks": tasks,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(episode, tasks, chunksize=1):
            stem = "__".join(str(result[k]) for k in ("scenario", "seed", "profile", "variant"))
            (args.output / (stem + ".json")).write_text(json.dumps(result, allow_nan=False))
            print(
                json.dumps(
                    {k: result[k] for k in ("scenario", "seed", "profile", "variant", "outcome")}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
