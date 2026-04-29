#!/usr/bin/env python3
"""Run step-level diagnostics for a policy-search candidate on one scenario/seed."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.map_runner import (
    _build_env_config,
    _build_policy,
    _policy_command_to_env_action,
)
from robot_sf.benchmark.termination_reason import route_complete_success
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.policy_search_common import infer_scenario_family
from scripts.validation.run_policy_search_candidate import (
    _DEFAULT_FUNNEL,
    _DEFAULT_REGISTRY,
    _deep_merge,
    _load_stage_scenarios,
    _load_yaml,
    _resolve_path,
    load_candidate_definition,
)


def _json_ready(value: Any) -> Any:
    """Convert nested values into JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--stage",
        default="smoke",
        choices=(
            "smoke",
            "nominal_sanity",
            "stress_slice",
            "full_matrix",
            "robustness_extension",
        ),
    )
    parser.add_argument("--candidate-registry", type=Path, default=_DEFAULT_REGISTRY)
    parser.add_argument("--funnel-config", type=Path, default=_DEFAULT_FUNNEL)
    parser.add_argument("--scenario-name", type=str, default=None)
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _scenario_id(scenario: dict[str, Any]) -> str:
    return str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )


def _select_scenario(scenarios: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Select one scenario by name or index."""
    if args.scenario_name:
        for scenario in scenarios:
            if _scenario_id(scenario) == args.scenario_name:
                return dict(scenario)
        raise KeyError(f"Scenario '{args.scenario_name}' not found in stage selection.")
    if not scenarios:
        raise RuntimeError("No scenarios resolved for diagnostics run.")
    return dict(scenarios[int(args.scenario_index)])


def _seed_list(scenario: dict[str, Any]) -> list[int]:
    """Return explicit scenario seeds or a conservative default."""
    seeds = scenario.get("seeds")
    if isinstance(seeds, list) and seeds:
        return [int(seed) for seed in seeds]
    return [0]


def _obs_min_robot_ped_distance(obs: dict[str, Any]) -> float | None:
    """Compute min robot-pedestrian distance from observation payload."""
    robot = obs.get("robot", {})
    peds = obs.get("pedestrians", {})
    robot_pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=float).reshape(-1)[:2]
    ped_pos = np.asarray(peds.get("positions", []), dtype=float)
    if ped_pos.ndim == 1:
        ped_pos = ped_pos.reshape(-1, 2) if ped_pos.size % 2 == 0 else np.zeros((0, 2), dtype=float)
    ped_count = int(np.asarray(peds.get("count", [ped_pos.shape[0]]), dtype=float).reshape(-1)[0])
    ped_count = max(0, min(ped_count, ped_pos.shape[0]))
    ped_pos = ped_pos[:ped_count]
    if ped_pos.size == 0:
        return None
    return float(np.min(np.linalg.norm(ped_pos - robot_pos.reshape(1, 2), axis=1)))


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    """Execute one step-trace diagnostics run and write trace/report artifacts."""
    args = parse_args()

    funnel = _load_yaml(args.funnel_config)
    stages = funnel.get("stages")
    if not isinstance(stages, dict) or args.stage not in stages:
        raise KeyError(f"Unknown stage '{args.stage}' in {args.funnel_config}")
    stage_cfg = stages[args.stage]
    if not isinstance(stage_cfg, dict):
        raise TypeError(f"Stage config must be a mapping: {args.stage}")

    stage_matrix = _resolve_path(args.funnel_config.parent, stage_cfg.get("scenario_matrix"))
    if stage_matrix is None:
        raise ValueError(f"Stage '{args.stage}' is missing a resolvable scenario_matrix")
    seed_manifest = _resolve_path(args.funnel_config.parent, stage_cfg.get("seed_manifest"))

    entry, candidate_payload, algo_cfg, _config_path = load_candidate_definition(
        args.candidate_registry,
        args.candidate,
    )
    algo = candidate_payload.get("algo") or entry.get("algo")
    if not isinstance(algo, str) or not algo.strip():
        raise ValueError(f"Candidate '{args.candidate}' is missing a valid algo field")

    loaded = _load_stage_scenarios(stage_matrix, seed_manifest)
    scenarios = load_scenarios(loaded) if isinstance(loaded, Path) else [dict(s) for s in loaded]
    scenario = _select_scenario(scenarios, args)
    family = infer_scenario_family(scenario)
    family_overrides = candidate_payload.get("family_overrides")
    effective_cfg = deepcopy(algo_cfg)
    if isinstance(family_overrides, dict):
        override = family_overrides.get(family)
        if isinstance(override, dict):
            effective_cfg = _deep_merge(effective_cfg, override)

    scenario_seed_list = _seed_list(scenario)
    seed = (
        int(args.seed) if args.seed is not None else int(scenario_seed_list[int(args.seed_index)])
    )
    horizon = int(args.horizon or stage_cfg.get("horizon", 0) or 300)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("output")
            / "policy_search"
            / args.candidate
            / "step_diagnostics"
            / args.stage
            / "latest"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = _build_env_config(scenario, scenario_path=stage_matrix)
    policy_fn, algo_meta = _build_policy(
        algo,
        dict(effective_cfg),
        robot_kinematics="differential_drive",
    )
    planner_adapter = getattr(policy_fn, "_planner_adapter", None)
    planner_reset = getattr(policy_fn, "_planner_reset", None)
    planner_bind_env = getattr(policy_fn, "_planner_bind_env", None)
    planner_native_action = getattr(policy_fn, "_planner_native_env_action", False)

    env = make_robot_env(config=env_config, seed=seed, debug=False)
    trace_rows: list[dict[str, Any]] = []
    done_info: dict[str, Any] = {}
    try:
        obs, _ = env.reset(seed=seed)
        if callable(planner_bind_env):
            planner_bind_env(env)
        if callable(planner_reset):
            planner_reset(seed=seed)

        for step_idx in range(horizon):
            robot_pos = np.array(env.simulator.robot_pos[0], dtype=float, copy=True)
            goal_pos = np.array(env.simulator.goal_pos[0], dtype=float, copy=True)
            goal_distance = float(np.linalg.norm(goal_pos - robot_pos))
            min_robot_ped_dist = _obs_min_robot_ped_distance(obs)

            policy_command = policy_fn(obs)
            step_is_native = getattr(policy_fn, "_last_step_native", planner_native_action)
            if step_is_native:
                env_action = np.asarray(policy_command, dtype=np.float32)
            else:
                env_action = _policy_command_to_env_action(
                    env=env,
                    config=env_config,
                    command=policy_command,
                )

            planner_decision = None
            if planner_adapter is not None:
                last_decision = getattr(planner_adapter, "last_decision", None)
                if callable(last_decision):
                    planner_decision = last_decision()

            obs, reward, terminated, truncated, info = env.step(env_action)
            meta = info.get("meta", {}) if isinstance(info, dict) else {}
            is_success = route_complete_success(info if isinstance(info, dict) else {})
            trace_rows.append(
                {
                    "step": int(step_idx),
                    "policy_command": _json_ready(policy_command),
                    "env_action": _json_ready(env_action),
                    "reward": float(reward),
                    "goal_distance": goal_distance,
                    "min_robot_ped_distance": min_robot_ped_dist,
                    "planner_decision": _json_ready(planner_decision),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "is_success": bool(is_success),
                    "is_pedestrian_collision": bool(meta.get("is_pedestrian_collision", False)),
                    "is_obstacle_collision": bool(meta.get("is_obstacle_collision", False)),
                    "is_robot_collision": bool(meta.get("is_robot_collision", False)),
                }
            )
            if terminated or truncated or is_success:
                done_info = {
                    "step": int(step_idx),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success": bool(is_success),
                    "meta": _json_ready(meta),
                    "family": family,
                }
                break
    finally:
        planner_summary = None
        if planner_adapter is not None:
            diagnostics = getattr(planner_adapter, "diagnostics", None)
            if callable(diagnostics):
                planner_summary = diagnostics()
        env.close()

    trace_payload = {
        "candidate": args.candidate,
        "stage": args.stage,
        "scenario_id": _scenario_id(scenario),
        "family": family,
        "seed": seed,
        "horizon": horizon,
        "algo": algo,
        "algo_config": _json_ready(effective_cfg),
        "algorithm_metadata": _json_ready(algo_meta),
        "planner_summary": _json_ready(planner_summary),
        "done_info": _json_ready(done_info),
        "steps": trace_rows,
    }
    trace_path = output_dir / "trace.json"
    trace_path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")

    decision_counter = Counter()
    selected_head_counter = Counter()
    for row in trace_rows:
        decision = row.get("planner_decision") or {}
        if isinstance(decision, dict):
            if isinstance(decision.get("decision"), str):
                decision_counter[str(decision["decision"])] += 1
            if isinstance(decision.get("selected_head"), str):
                selected_head_counter[str(decision["selected_head"])] += 1

    report_lines = [
        f"# Step Diagnostics: {args.candidate} ({args.stage})",
        "",
        f"- Scenario: `{_scenario_id(scenario)}`",
        f"- Family: `{family}`",
        f"- Seed: `{seed}`",
        f"- Horizon: `{horizon}`",
        f"- Trace JSON: `{trace_path}`",
        f"- Decision counts: `{dict(decision_counter)}`",
        f"- Selected heads: `{dict(selected_head_counter)}`",
        "",
        "## Outcome",
        "",
        f"- Done info: `{_json_ready(done_info)}`",
        "",
        "## Step Summary",
        "",
        "| Step | Decision | Head | Goal Dist | Min Robot-Ped Dist | Command | Success |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for row in trace_rows:
        decision = row.get("planner_decision") or {}
        decision_name = decision.get("decision") if isinstance(decision, dict) else None
        selected_head = decision.get("selected_head") if isinstance(decision, dict) else None
        goal_distance = row.get("goal_distance")
        min_robot_ped_dist = row.get("min_robot_ped_distance")
        report_lines.append(
            "| "
            f"{row['step']} | {decision_name} | {selected_head} | "
            f"{goal_distance if goal_distance is not None else 'n/a'} | "
            f"{min_robot_ped_dist if min_robot_ped_dist is not None else 'n/a'} | "
            f"{row['policy_command']} | {row['is_success']} |"
        )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "trace": str(trace_path),
                "report": str(report_path),
                "scenario_id": _scenario_id(scenario),
                "family": family,
                "seed": seed,
                "decision_counts": dict(decision_counter),
                "selected_head_counts": dict(selected_head_counter),
                "done_info": _json_ready(done_info),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
