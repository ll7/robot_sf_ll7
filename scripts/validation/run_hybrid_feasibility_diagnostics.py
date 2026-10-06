#!/usr/bin/env python3
"""Diagnostic-only dev-seed hybrid/ORCA feasibility traces for issue #10092.

Uses the benchmark's environment, policy and action conversion. ORCA replay
checks actual executed endpoints with the hybrid's hard predicates, and reports
separately a constant-command horizon rejection (not an executed-path rejection).
No release campaign or evaluation seed set is executed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
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
from robot_sf.benchmark.termination_reason import route_complete_success
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.planner.hybrid_rule_local_planner import HybridRuleCandidate
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_empty_world_sweep import assert_dev_seeds, remove_pedestrians
from scripts.validation.run_policy_search_candidate import (
    _DEFAULT_REGISTRY,
    _effective_candidate_runtime_for_scenario,
    load_candidate_definition,
)
from scripts.validation.run_policy_search_step_diagnostics import _json_ready

ROOT = Path(__file__).resolve().parents[2]
MAIN_MATRIX = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
WIDTH_MATRIX = (
    ROOT / "configs/scenarios/francis2023_narrow_doorway_three_width_release_0_0_8_v1.yaml"
)
CANDIDATE = "hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release"
TARGETS = (
    "classic_station_platform_medium",
    "classic_cross_trap_high",
    "classic_t_intersection_medium",
    "francis2023_narrow_doorway_width_2p20",
)
ARM_SWITCHES = {
    "off": (False, False, False),
    "static_only": (True, False, False),
    "sensor_only": (False, False, True),
    "goal_validity_with_sensor": (False, True, True),
    "static_plus_goal_validity": (True, True, True),
    "current_defaults": (True, True, True),
    "orca": (False, False, False),
}


def load_cells(names):
    """Resolve explicit scenarios without resolving any release seed policy."""
    cells = {}
    for matrix in (MAIN_MATRIX, WIDTH_MATRIX):
        for scenario in load_scenarios(matrix):
            if scenario["name"] in names:
                cells[scenario["name"]] = (scenario, matrix)
    if set(names) != set(cells):
        raise ValueError(f"Missing scenarios: {set(names) - set(cells)}")
    return cells


def hybrid_config(scenario, enabled=False, goal_validity=False):
    """Load the named hybrid candidate including its scenario overrides."""
    _, payload, cfg, path = load_candidate_definition(ROOT / _DEFAULT_REGISTRY, CANDIDATE)
    algo, cfg = _effective_candidate_runtime_for_scenario(
        payload, cfg, scenario, default_algo="hybrid_rule_local_planner", config_anchor=path.parent
    )
    assert algo == "hybrid_rule_local_planner"
    cfg["debug_candidate_evaluator"] = True
    cfg["physical_static_exclusion_enabled"] = bool(enabled)
    cfg["goal_next_validity_enabled"] = bool(goal_validity)
    return cfg


def replay_segment(planner, obs, state, candidate, endpoint):
    """Evaluate one actually executed plant segment, without predicting its endpoint."""
    ctx = planner._prepare_evaluation_context(
        candidate=candidate, state=state, strict_static_clearance=False
    )
    start_clearance = planner._min_obstacle_clearance(state["robot_pos"], obs)
    result, _, _ = planner._rollout_step_static_check(
        candidate=candidate,
        observation=obs,
        ctx=ctx,
        robot_pos=endpoint,
        initial_static_clearance=start_clearance,
        min_static_clearance=float("inf"),
        static_clearance_exception_terms=set(),
        strict_static_clearance=False,
        progress_windows={},
        t=state["dt"],
    )
    if result is None:
        result, _ = planner._check_dynamic_collision(
            candidate=candidate,
            ped_pos=state["ped_pos"],
            ped_vel=state["ped_vel"],
            robot_pos=endpoint,
            collision_radius=ctx["collision_radius"],
            min_dynamic_clearance=float("inf"),
            t=state["dt"],
        )
    if result is None:
        return None
    name, key, threshold = planner._debug_rejection_constraint(result)
    body_collision = planner._continuous_static_collision(endpoint, state["robot_radius"])
    return {
        "constraint": name,
        "threshold_name": key,
        "threshold": threshold,
        "evaluation": planner._rejection_diagnostic(result),
        "start": state["robot_pos"].tolist(),
        "end": endpoint.tolist(),
        "robot_radius_m": state["robot_radius"],
        "physical_static_contact": body_collision,
    }


def motion_metrics(rows, dt):
    """Measure stationary time and sustained low displacement (diagnostic units: seconds)."""
    stopped = sum(r["displacement_m"] / dt <= 0.05 for r in rows)
    forced = sum(
        r["debug"]["feasible_moving_count"] == 0
        for r in rows
        if (r.get("debug") or {}).get("candidate_count", 0) > 0
    )
    longest = current = 0
    # A robot that moves less than 0.5 m net over 10 s is stuck/oscillating.
    width = int(np.floor(10 / dt)) + 1  # strictly more than 10 s
    for row in rows:
        current = current + 1 if row["displacement_m"] / dt <= 0.05 else 0
        longest = max(longest, current)
    oscillating = any(
        np.linalg.norm(np.array(rows[i]["position"]) - np.array(rows[i - width]["position"])) < 0.5
        for i in range(width, len(rows))
    )
    return {
        "freezing": longest * dt > 10 or oscillating,
        "stopped_time_fraction": stopped / len(rows) if rows else 0,
        "no_feasible_moving_s": forced * dt,
        "longest_low_progress_s": longest * dt,
    }


def evaluate_orca_step(shadow, obs, state, command, end):
    """Keep actual-segment and hypothetical constant-command evidence separate."""
    candidate = HybridRuleCandidate(float(command[0]), float(command[1]), "orca_executed")
    rejected = replay_segment(shadow, obs, state, candidate, end)
    full = shadow._evaluate_candidate(
        candidate=candidate,
        observation=obs,
        state=state,
        speed_cap=shadow._v4_human_speed_cap(state),
        nearest_ped=shadow._nearest_ped_distance(state["robot_pos"], state["ped_pos"]),
    )
    return rejected, full


def missing_candidate_probe(planner, obs, state, command):
    """Find an admissible forward action excluded by the scalar proximity speed cap."""
    if planner._last_v4_speed_safety is None:
        return None
    cap = planner._last_v4_speed_safety["speed_cap"]
    _, reachable, _, _ = planner._dynamic_window(
        state["current_speed"], planner._v4_effective_max_speed()
    )
    if reachable <= cap + 1e-9:
        return None
    candidate = HybridRuleCandidate(reachable, float(command[1]), "unsampled_forward_probe")
    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=obs,
        state=state,
        speed_cap=cap,
        nearest_ped=planner._nearest_ped_distance(state["robot_pos"], state["ped_pos"]),
    )
    return {
        "command": [candidate.linear, candidate.angular],
        "generated_speed_cap_m_s": cap,
        "accepted": evaluation["accepted"],
        "evaluation": planner._candidate_diagnostic(evaluation)
        if evaluation["accepted"]
        else planner._rejection_diagnostic(evaluation),
    }


def run_cell(task):  # noqa: C901, PLR0915 -- native episode custody stays within one try/finally
    """Run one native episode and persist complete step diagnostics atomically."""
    name, seed, arm, empty, output, horizon = task
    assert_dev_seeds([seed])
    logger.remove()
    logger.add(os.sys.stderr, level="ERROR")
    scenario, matrix = load_cells([name])[name]
    scenario = _scenario_with_episode_seed_defaults(dict(scenario, seeds=[seed]), seed=seed)
    if empty:
        scenario = remove_pedestrians(scenario, [seed])
    # Fixed 60 s comparison budget from #10092, independent of release authored budgets.
    scenario["simulation_config"] = dict(
        scenario.get("simulation_config") or {}, max_episode_steps=horizon
    )
    cfg = _build_env_config(scenario, scenario_path=matrix)
    static, validity, sensor = ARM_SWITCHES[arm]
    hcfg = hybrid_config(
        scenario,
        enabled=static,
        goal_validity=validity,
    )
    if arm == "current_defaults":
        hcfg.pop("physical_static_exclusion_enabled")
        hcfg.pop("goal_next_validity_enabled")
    else:
        cfg.include_goal_next_valid = sensor
    algo = "orca" if arm == "orca" else "hybrid_rule_local_planner"
    pcfg = (
        yaml.safe_load((ROOT / "configs/algos/orca_release_v0_0_8.yaml").read_text())
        if arm == "orca"
        else hcfg
    )
    policy, metadata = _build_policy(algo, pcfg, robot_kinematics="differential_drive")
    planner = policy._planner_adapter
    env = make_robot_env(config=cfg, seed=seed, debug=False)
    shadow = None
    rows = []
    first_segment = first_horizon = None
    try:
        obs, _ = env.reset(seed=seed)
        policy._planner_bind_env(env)
        policy._planner_reset(seed=seed)
        next_valid_observed = "next_valid" in planner._socnav_fields(obs)[1]
        if planner.config.goal_next_validity_enabled and not next_valid_observed:
            raise RuntimeError("Enabled successor validity did not reach the planner")
        if arm == "orca":
            shadow_policy, _ = _build_policy(
                "hybrid_rule_local_planner", hcfg, robot_kinematics="differential_drive"
            )
            shadow = shadow_policy._planner_adapter
            shadow.bind_env(env)
            shadow.reset(seed=seed)
        success = collision = False
        info = {}
        for step in range(horizon):
            pre = np.array(env.simulator.robot_pos[0], dtype=float)
            command = policy(obs)
            evaluator = shadow or planner
            if shadow is not None:
                shadow.plan(obs)
            decision = evaluator.last_decision()
            state = evaluator._extract_state(obs)
            debug = decision.get("candidate_evaluator_debug") if decision else None
            probe = (
                missing_candidate_probe(evaluator, obs, state, command) if shadow is None else None
            )
            action = _policy_command_to_env_action(env=env, config=cfg, command=command)
            next_obs, _, terminated, truncated, info = env.step(action)
            end = np.array(env.simulator.robot_pos[0], dtype=float)
            meta = info.get("meta", {})
            contact = any(
                meta.get(k, False)
                for k in ("is_pedestrian_collision", "is_obstacle_collision", "is_robot_collision")
            )
            collision |= contact
            success = route_complete_success(info)
            row = {
                "step": step,
                "position": pre.tolist(),
                "end_position": end.tolist(),
                "command": _json_ready(command),
                "displacement_m": float(np.linalg.norm(end - pre)),
                "goal_distance_m": float(np.linalg.norm(env.simulator.goal_pos[0] - end)),
                "collision": contact,
                "collision_types": [
                    k
                    for k in (
                        "is_pedestrian_collision",
                        "is_obstacle_collision",
                        "is_robot_collision",
                    )
                    if meta.get(k, False)
                ],
                "pedestrian_separation_m": (
                    float(meta["min_distance"])
                    if np.isfinite(meta.get("min_distance", float("nan")))
                    else None
                ),
                "pedestrian_surface_gap_m": (
                    float(meta["min_clearance"])
                    if np.isfinite(meta.get("min_clearance", float("nan")))
                    else None
                ),
                "near_miss": bool(meta.get("near_misses", 0)),
                "debug": debug,
                "decision": decision,
                "missing_candidate_probe": probe,
            }
            if shadow is not None:
                rejected, full = evaluate_orca_step(shadow, obs, state, command, end)
                row["executed_segment_rejection"] = rejected
                if rejected is not None and first_segment is None:
                    first_segment = dict(step=step, **rejected)
                if not full["accepted"] and first_horizon is None:
                    name_, key, value = shadow._debug_rejection_constraint(full)
                    first_horizon = {
                        "step": step,
                        "constraint": name_,
                        "threshold_name": key,
                        "threshold": value,
                        "evaluation": shadow._rejection_diagnostic(full),
                    }
            rows.append(row)
            obs = next_obs
            if terminated or truncated or success:
                break
        runtime = planner.diagnostics() if hasattr(planner, "diagnostics") else {}
        if runtime.get("fallback_count", 0) or runtime.get("degraded_count", 0):
            raise RuntimeError(f"Fallback/degraded execution: {runtime}")
        result = {
            "execution_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "next_valid_field_observed": next_valid_observed,
            "effective_switches": {
                "physical_static_exclusion_enabled": bool(
                    planner.config.physical_static_exclusion_enabled
                ),
                "goal_next_validity_enabled": bool(planner.config.goal_next_validity_enabled),
                "include_goal_next_valid": bool(cfg.include_goal_next_valid),
            },
            "scenario": name,
            "seed": seed,
            "arm": arm,
            "empty": empty,
            "outcome": "collision" if collision else "success" if success else "timeout",
            "steps": len(rows),
            "duration_s": len(rows) * 0.1,
            "first_rejected_executed_segment": first_segment,
            "first_rejected_constant_command_horizon": first_horizon,
            "metrics": motion_metrics(rows, 0.1),
            "algorithm_metadata": metadata,
            "runtime": runtime,
            "final_goal_distance_m": rows[-1]["goal_distance_m"],
            "admissible_unsampled_steps": sum(
                bool(r.get("missing_candidate_probe") and r["missing_candidate_probe"]["accepted"])
                for r in rows
            ),
        }
        separations = [r["pedestrian_separation_m"] for r in rows]
        separations = [v for v in separations if v is not None and np.isfinite(v)]
        result["metrics"].update(
            minimum_pedestrian_separation_m=min(separations) if separations else None,
            near_miss_steps=sum(r["near_miss"] for r in rows),
            near_miss_events=sum(
                r["near_miss"] and (i == 0 or not rows[i - 1]["near_miss"])
                for i, r in enumerate(rows)
            ),
        )
    finally:
        env.close()
    path = Path(output) / f"{name}__{seed}__{arm}__{'empty' if empty else 'crowd'}"
    with gzip.open(path.with_suffix(".jsonl.gz"), "wt") as stream:
        for row in rows:
            stream.write(json.dumps(_json_ready(row), allow_nan=False) + "\n")
    path.with_suffix(".json").write_text(
        json.dumps(_json_ready(result), indent=2, allow_nan=False) + "\n"
    )
    print(f"{name} {seed} {arm} {result['outcome']}", flush=True)
    return result


def main():
    """Execute only an explicitly resolved development-seed task matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", default=list(TARGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1001, 1011)))
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["off", "static_only", "static_plus_goal_validity", "orca"],
        default=["off", "orca"],
    )
    parser.add_argument("--empty", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = assert_dev_seeds(args.seeds)
    limit = 32 if os.environ.get("SLURM_JOB_ID") else 2
    if not 1 <= args.workers <= limit:
        parser.error(f"workers must be 1..{limit}")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"RESOLVED DEV SEEDS: {seeds}", flush=True)
    load_cells(args.scenarios)
    tasks = [
        (n, s, a, args.empty, str(args.output), args.horizon)
        for n in args.scenarios
        for s in seeds
        for a in args.arms
    ]
    files = [
        Path(__file__),
        ROOT / "robot_sf/planner/hybrid_rule_local_planner.py",
        ROOT / "robot_sf/planner/grid_route.py",
        ROOT / "robot_sf/planner/socnav_occupancy.py",
        ROOT / "robot_sf/benchmark/map_runner/map_runner_observations.py",
        ROOT / "robot_sf/sensor/socnav_observation.py",
        ROOT / "robot_sf/gym_env/unified_config.py",
        MAIN_MATRIX,
        WIDTH_MATRIX,
    ]
    manifest = {
        "status": "diagnostic-only",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seeds": seeds,
        "scenarios": args.scenarios,
        "arms": args.arms,
        "arm_flags": {
            a: {
                "physical_static_exclusion_enabled": a
                in {"static_only", "static_plus_goal_validity"},
                "goal_next_validity_enabled": a == "static_plus_goal_validity",
                "include_goal_next_valid": a == "static_plus_goal_validity",
            }
            for a in args.arms
            if a != "orca"
        },
        "workers": args.workers,
        "horizon": args.horizon,
        "empty": args.empty,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    reuse = args.output / "reuse-provenance.json"
    if reuse.exists():
        manifest["reused_episode_sources"] = json.loads(reuse.read_text())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pending = [
        t
        for t in tasks
        if not (
            Path(t[4]) / f"{t[0]}__{t[1]}__{t[2]}__{'empty' if t[3] else 'crowd'}.json"
        ).exists()
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_cell, pending))
    print(
        f"Completed {len(results)} new episodes; {len(tasks) - len(pending)} retained", flush=True
    )


if __name__ == "__main__":
    main()
