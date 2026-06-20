#!/usr/bin/env python3
"""Run a bounded diagnostic runtime smoke for density coverage-entropy candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.map_runner_actions import (
    policy_command_to_env_action,
    vel_and_acc,
)
from robot_sf.benchmark.map_runner_env import build_env_config
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics
from robot_sf.benchmark.termination_reason import resolve_termination_reason
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import load_scenarios

SCHEMA_VERSION = "density_runtime_smoke.v1"


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    """Return the stable scenario identifier for matching coverage rows."""
    for key in ("name", "scenario_id", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def select_candidate_rows(
    coverage_report: Mapping[str, Any],
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select top novelty rows plus redundant or lowest-novelty comparator rows."""
    rows_raw = coverage_report.get("scenario_rows")
    rows = [dict(row) for row in rows_raw if isinstance(row, Mapping)]
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not rows:
        raise ValueError("coverage report has no scenario_rows")

    novel = sorted(
        (row for row in rows if row.get("recommendation") == "retain_or_investigate"),
        key=lambda row: (-float(row.get("novelty_score", 0.0)), str(row.get("scenario_id", ""))),
    )[:top_k]
    if len(novel) < top_k:
        raise ValueError("coverage report does not contain enough novel candidates")

    novel_ids = {str(row.get("scenario_id")) for row in novel}
    redundant = sorted(
        (row for row in rows if row.get("recommendation") == "merge_or_drop"),
        key=lambda row: (float(row.get("novelty_score", 0.0)), str(row.get("scenario_id", ""))),
    )[:top_k]
    comparator_source = "merge_or_drop"
    if len(redundant) < top_k:
        comparator_source = "lowest_novelty_fallback"
        redundant = sorted(
            (row for row in rows if str(row.get("scenario_id")) not in novel_ids),
            key=lambda row: (
                float(row.get("novelty_score", 0.0)),
                str(row.get("scenario_id", "")),
            ),
        )[:top_k]
    if len(redundant) < top_k:
        raise ValueError("coverage report does not contain enough comparator candidates")

    selected: list[dict[str, Any]] = []
    for group, group_rows in (("novel", novel), ("redundant_comparator", redundant)):
        for row in group_rows:
            payload = dict(row)
            payload["selection_group"] = group
            payload["comparator_source"] = comparator_source if group != "novel" else "top_novelty"
            selected.append(payload)

    summary = coverage_report.get("summary")
    summary_map = summary if isinstance(summary, Mapping) else {}
    return selected, {
        "top_k": top_k,
        "novel_count": len(novel),
        "comparator_count": len(redundant),
        "comparator_source": comparator_source,
        "coverage_redundant_count": int(summary_map.get("redundant_count", 0)),
        "caveat": (
            "No merge_or_drop rows were available; used lowest-novelty rows as comparators."
            if comparator_source == "lowest_novelty_fallback"
            else None
        ),
    }


def classify_failure_semantics(
    *,
    success: bool,
    collision: bool,
    near_misses: float,
    progress_ratio: float,
    progress_stall_threshold: float,
) -> str:
    """Map runtime outcomes into issue #3200's diagnostic row labels."""
    if collision:
        return "collision"
    if success:
        return "success"
    if near_misses > 0.0:
        return "near-miss"
    if progress_ratio <= progress_stall_threshold:
        return "progress_stall"
    return "horizon_exhausted"


def _goal_command(env: Any, goal: np.ndarray, *, max_speed: float) -> tuple[float, float]:
    """Return a simple goal-directed unicycle command."""
    robot = env.simulator.robots[0]
    robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
    vec = goal - robot_pos
    distance = float(np.linalg.norm(vec))
    if distance < 1e-6:
        return 0.0, 0.0
    heading = float(robot.pose[1])
    desired_heading = float(math.atan2(vec[1], vec[0]))
    heading_error = math.atan2(
        math.sin(desired_heading - heading),
        math.cos(desired_heading - heading),
    )
    linear = float(min(max_speed, distance) * max(0.0, 1.0 - abs(heading_error) / math.pi))
    angular = float(np.clip(heading_error, -1.0, 1.0))
    return linear, angular


def _stack_ped_positions(traj: list[np.ndarray]) -> np.ndarray:
    """Stack variable pedestrian counts into an EpisodeData-compatible tensor."""
    if not traj:
        return np.zeros((0, 0, 2), dtype=float)
    first_shape = traj[0].shape
    if all(arr.shape == first_shape for arr in traj):
        return np.stack(traj).astype(float, copy=False)
    max_count = max(arr.shape[0] for arr in traj)
    stacked = np.full((len(traj), max_count, 2), np.nan, dtype=float)
    for index, arr in enumerate(traj):
        if arr.size:
            stacked[index, : arr.shape[0]] = arr
    return stacked


def run_smoke_episode(
    scenario: Mapping[str, Any],
    *,
    matrix_path: Path,
    seed: int,
    horizon: int,
    dt: float,
    max_speed: float,
    progress_stall_threshold: float,
) -> dict[str, Any]:
    """Run one diagnostic goal-policy episode for one scenario row."""
    scenario_payload = dict(scenario)
    scenario_id = _scenario_id(scenario_payload)
    config = build_env_config(scenario_payload, scenario_path=matrix_path)
    config.sim_config.time_per_step_in_secs = float(dt)

    env = make_robot_env(config=config, seed=int(seed), debug=False)
    robot_positions: list[np.ndarray] = []
    ped_positions: list[np.ndarray] = []
    collision_seen = False
    success_seen = False
    timeout_seen = False
    termination_reason = "max_steps"
    try:
        _obs, _info = env.reset(seed=int(seed))
        goal = np.asarray(env.simulator.goal_pos[0], dtype=float)
        initial_robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
        initial_goal_distance = float(np.linalg.norm(goal - initial_robot_pos))

        for step_index in range(int(horizon)):
            command = _goal_command(env, goal, max_speed=max_speed)
            action = policy_command_to_env_action(env=env, config=config, command=command)
            _obs, _reward, terminated, truncated, info = env.step(action)
            robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
            peds = np.asarray(env.simulator.ped_pos, dtype=float)
            robot_positions.append(robot_pos.copy())
            ped_positions.append(peds.copy())

            meta = info.get("meta", {}) if isinstance(info, Mapping) else {}
            step_collision = bool(info.get("collision")) if isinstance(info, Mapping) else False
            step_success = bool(info.get("success")) if isinstance(info, Mapping) else False
            step_timeout = bool(meta.get("is_timesteps_exceeded", False))
            collision_seen = collision_seen or step_collision
            success_seen = success_seen or (step_success and not step_collision)
            timeout_seen = timeout_seen or step_timeout
            if success_seen or collision_seen or terminated or truncated:
                termination_reason = resolve_termination_reason(
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    success=success_seen,
                    collision=collision_seen,
                    reached_max_steps=False,
                )
                break
            if step_index == int(horizon) - 1:
                timeout_seen = True
    finally:
        env.close()

    robot_pos_arr = np.asarray(robot_positions, dtype=float)
    final_robot_pos = robot_pos_arr[-1] if robot_pos_arr.size else initial_robot_pos
    final_goal_distance = float(np.linalg.norm(goal - final_robot_pos))
    progress_m = max(0.0, initial_goal_distance - final_goal_distance)
    progress_ratio = progress_m / initial_goal_distance if initial_goal_distance > 1e-9 else 0.0
    robot_vel_arr, robot_acc_arr = vel_and_acc(robot_pos_arr, float(dt))
    ped_pos_arr = _stack_ped_positions(ped_positions)
    ped_forces_arr = np.zeros_like(ped_pos_arr, dtype=float)
    episode = EpisodeData(
        robot_pos=robot_pos_arr,
        robot_vel=robot_vel_arr,
        robot_acc=robot_acc_arr,
        peds_pos=ped_pos_arr,
        ped_forces=ped_forces_arr,
        obstacles=None,
        goal=goal,
        dt=float(dt),
        reached_goal_step=0 if success_seen else None,
        robot_radius=float(getattr(config.robot_config, "radius", 1.0)),
        ped_radius=float(getattr(config.sim_config, "ped_radius", 0.4)),
    )
    metrics = compute_all_metrics(episode, horizon=int(horizon))
    if collision_seen:
        metrics["collisions"] = max(float(metrics.get("collisions", 0.0)), 1.0)
    if success_seen:
        metrics["success"] = 1.0
    near_miss_count = float(metrics.get("near_misses", 0.0) or 0.0)
    failure_semantics = classify_failure_semantics(
        success=success_seen,
        collision=collision_seen,
        near_misses=near_miss_count,
        progress_ratio=progress_ratio,
        progress_stall_threshold=progress_stall_threshold,
    )
    return {
        "scenario_id": scenario_id,
        "seed": int(seed),
        "status": "completed",
        "row_status": "diagnostic_only",
        "counts_as_success_evidence": False,
        "termination_reason": termination_reason,
        "failure_semantics": failure_semantics,
        "outcome": {
            "success": bool(success_seen),
            "collision": bool(collision_seen),
            "timeout": bool(timeout_seen or failure_semantics == "horizon_exhausted"),
        },
        "metrics": {
            "success": float(metrics.get("success", 0.0) or 0.0),
            "collisions": float(metrics.get("collisions", 0.0) or 0.0),
            "near_misses": near_miss_count,
            "min_distance": float(metrics.get("min_distance", float("nan"))),
            "progress_m": progress_m,
            "progress_ratio": progress_ratio,
            "steps": int(robot_pos_arr.shape[0]),
        },
        "caveat": "one-seed goal-policy diagnostic smoke; not benchmark evidence",
    }


def run_density_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run the selected density runtime smoke and return a JSON payload."""
    coverage = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    selected_rows, selection_summary = select_candidate_rows(coverage, top_k=int(args.top_k))
    scenarios = {
        _scenario_id(scenario): dict(scenario)
        for scenario in load_scenarios(args.matrix, base_dir=args.matrix)
    }

    row_payloads: list[dict[str, Any]] = []
    missing: list[str] = []
    for selected in selected_rows:
        scenario_id = str(selected.get("scenario_id"))
        scenario = scenarios.get(scenario_id)
        if scenario is None:
            missing.append(scenario_id)
            continue
        runtime = run_smoke_episode(
            scenario,
            matrix_path=args.matrix,
            seed=int(args.seed),
            horizon=int(args.horizon),
            dt=float(args.dt),
            max_speed=float(args.max_speed),
            progress_stall_threshold=float(args.progress_stall_threshold),
        )
        row_payloads.append(
            {
                "selection": {
                    "group": selected["selection_group"],
                    "comparator_source": selected["comparator_source"],
                    "novelty_score": selected.get("novelty_score"),
                    "nearest_neighbor": selected.get("nearest_neighbor"),
                    "recommendation": selected.get("recommendation"),
                },
                **runtime,
            }
        )

    status = "completed" if row_payloads and not missing else "blocked"
    if missing:
        status = "blocked"
    failure_counts: dict[str, int] = {}
    for row in row_payloads:
        key = str(row.get("failure_semantics", "unknown"))
        failure_counts[key] = failure_counts.get(key, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 3200,
        "status": status,
        "evidence_status": "diagnostic-only",
        "claim_boundary": "bounded same-seed runtime smoke; not benchmark or paper evidence",
        "matrix": str(args.matrix),
        "coverage_json": str(args.coverage_json),
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "dt": float(args.dt),
        "selection_summary": selection_summary,
        "failure_semantics_counts": failure_counts,
        "missing_scenarios": missing,
        "rows": row_payloads,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--coverage-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3200)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--progress-stall-threshold", type=float, default=0.05)
    return parser


def main() -> int:
    """Run CLI."""
    args = _build_parser().parse_args()
    payload = run_density_smoke(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "status": payload["status"],
                "rows": len(payload["rows"]),
                "failure_semantics_counts": payload["failure_semantics_counts"],
                "output_json": str(args.output_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
