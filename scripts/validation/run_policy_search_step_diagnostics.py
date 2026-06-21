#!/usr/bin/env python3
"""Run step-level diagnostics for a policy-search candidate on one scenario/seed."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.map_runner import (
    _build_env_config,
    _build_policy,
    _policy_command_to_env_action,
    _scenario_with_episode_seed_defaults,
)
from robot_sf.benchmark.observation_perturbation import (
    EVIDENCE_IDEAL,
    ObservationPerturbationSpec,
    ObservationPerturbationState,
    perturb_ground_truth,
)
from robot_sf.benchmark.termination_reason import route_complete_success
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.policy_search_common import infer_scenario_family
from scripts.validation.run_policy_search_candidate import (
    _DEFAULT_FUNNEL,
    _DEFAULT_REGISTRY,
    _effective_candidate_runtime_for_scenario,
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


def _optional_float(value: Any) -> float | None:
    """Return a float for numeric diagnostic values, preserving missing values."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _format_summary_float(value: Any) -> str:
    """Format optional floats for the Markdown report."""
    number = _optional_float(value)
    return "n/a" if number is None else f"{number:.4f}"


def _empty_trace_progress_summary() -> dict[str, Any]:
    """Return the stable progress-summary shape for an empty trace."""
    return {
        "steps_observed": 0,
        "initial_goal_distance": None,
        "final_goal_distance": None,
        "best_goal_distance": None,
        "net_goal_progress": None,
        "best_goal_progress": None,
        "progress_step_count": 0,
        "regression_step_count": 0,
        "stagnant_step_count": 0,
        "longest_stagnant_run": 0,
        "closest_robot_ped_distance": None,
        "closest_robot_ped_step": None,
        "collision_flag_counts": {"pedestrian": 0, "obstacle": 0, "robot": 0},
    }


def _progress_bucket(
    pre_goal: float | None,
    post_goal: float | None,
    *,
    stagnation_epsilon: float,
) -> str | None:
    """Classify one step as progress, regression, stagnant, or unavailable."""
    if pre_goal is None or post_goal is None:
        return None
    step_progress = pre_goal - post_goal
    if step_progress > stagnation_epsilon:
        return "progress"
    if step_progress < -stagnation_epsilon:
        return "regression"
    return "stagnant"


def _step_index(row: dict[str, Any]) -> int | None:
    """Return a row's integer step index when available."""
    step = row.get("step")
    return step if isinstance(step, int) and not isinstance(step, bool) else None


def _collision_flag_counts(trace_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count per-step collision flags emitted by the environment."""
    return {
        "pedestrian": sum(1 for row in trace_rows if row.get("is_pedestrian_collision")),
        "obstacle": sum(1 for row in trace_rows if row.get("is_obstacle_collision")),
        "robot": sum(1 for row in trace_rows if row.get("is_robot_collision")),
    }


def _goal_distance_summary(trace_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Summarize initial, final, best, and derived goal-distance progress."""
    goal_distance_values: list[float] = []
    initial_goal_distance: float | None = None
    final_goal_distance: float | None = None

    for row in trace_rows:
        pre_goal = _optional_float(row.get("goal_distance"))
        post_goal = _optional_float(row.get("post_step_goal_distance"))
        if initial_goal_distance is None:
            initial_goal_distance = pre_goal if pre_goal is not None else post_goal
        if pre_goal is not None:
            goal_distance_values.append(pre_goal)
        if post_goal is not None:
            goal_distance_values.append(post_goal)
        last_goal_distance = post_goal if post_goal is not None else pre_goal
        if last_goal_distance is not None:
            final_goal_distance = last_goal_distance

    best_goal_distance = min(goal_distance_values) if goal_distance_values else None
    net_goal_progress = (
        initial_goal_distance - final_goal_distance
        if initial_goal_distance is not None and final_goal_distance is not None
        else None
    )
    best_goal_progress = (
        initial_goal_distance - best_goal_distance
        if initial_goal_distance is not None and best_goal_distance is not None
        else None
    )
    return {
        "initial_goal_distance": initial_goal_distance,
        "final_goal_distance": final_goal_distance,
        "best_goal_distance": best_goal_distance,
        "net_goal_progress": net_goal_progress,
        "best_goal_progress": best_goal_progress,
    }


def _progress_step_summary(
    trace_rows: list[dict[str, Any]],
    *,
    stagnation_epsilon: float,
) -> dict[str, int]:
    """Count step-level progress, regression, and stagnation streaks."""
    progress_step_count = 0
    regression_step_count = 0
    stagnant_step_count = 0
    current_stagnant_run = 0
    longest_stagnant_run = 0

    for row in trace_rows:
        bucket = _progress_bucket(
            _optional_float(row.get("goal_distance")),
            _optional_float(row.get("post_step_goal_distance")),
            stagnation_epsilon=stagnation_epsilon,
        )
        if bucket == "progress":
            progress_step_count += 1
            current_stagnant_run = 0
        elif bucket == "regression":
            regression_step_count += 1
            current_stagnant_run = 0
        elif bucket == "stagnant":
            stagnant_step_count += 1
            current_stagnant_run += 1
            longest_stagnant_run = max(longest_stagnant_run, current_stagnant_run)
        else:
            current_stagnant_run = 0

    return {
        "progress_step_count": progress_step_count,
        "regression_step_count": regression_step_count,
        "stagnant_step_count": stagnant_step_count,
        "longest_stagnant_run": longest_stagnant_run,
    }


def _closest_robot_ped_summary(trace_rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Find the closest robot-pedestrian clearance observed in a trace."""
    closest_robot_ped_distance: float | None = None
    closest_robot_ped_step: int | None = None

    for row in trace_rows:
        step_index = _step_index(row)
        for field in ("min_robot_ped_distance", "post_step_min_robot_ped_distance"):
            distance = _optional_float(row.get(field))
            if distance is None:
                continue
            if closest_robot_ped_distance is None or distance < closest_robot_ped_distance:
                closest_robot_ped_distance = distance
                closest_robot_ped_step = step_index

    return {
        "closest_robot_ped_distance": closest_robot_ped_distance,
        "closest_robot_ped_step": closest_robot_ped_step,
    }


def _trace_progress_summary(
    trace_rows: list[dict[str, Any]],
    *,
    stagnation_epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Summarize per-step progress, stagnation, clearance, and collision flags."""
    if not trace_rows:
        return _empty_trace_progress_summary()

    return {
        "steps_observed": len(trace_rows),
        **_goal_distance_summary(trace_rows),
        **_progress_step_summary(
            trace_rows,
            stagnation_epsilon=stagnation_epsilon,
        ),
        **_closest_robot_ped_summary(trace_rows),
        "collision_flag_counts": _collision_flag_counts(trace_rows),
    }


def _format_planner_summary_lines(planner_summary: Any) -> list[str]:
    """Return Markdown lines for the aggregate planner diagnostics summary."""
    summary = _json_ready(planner_summary)
    lines = ["## Planner Summary", ""]
    if summary is None:
        return [*lines, "- Planner summary: `null`"]
    if not isinstance(summary, dict):
        return [*lines, f"- Planner summary: `{summary}`"]
    if not summary:
        return [*lines, "- Planner summary: `{}`"]
    for key, value in sorted(summary.items()):
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, sort_keys=True)
        else:
            rendered = str(value)
        lines.append(f"- `{key}`: `{rendered}`")
    return lines


def _diagnostics_stdout_payload(
    *,
    metadata: dict[str, Any],
    progress_summary: dict[str, Any],
    planner_summary: Any,
    done_info: dict[str, Any],
) -> dict[str, Any]:
    """Return the stable machine-readable diagnostics CLI payload."""
    return {
        **_json_ready(metadata),
        "progress_summary": _json_ready(progress_summary),
        "planner_summary": _json_ready(planner_summary),
        "done_info": _json_ready(done_info),
    }


def _planner_fallback_degraded_status(planner_summary: Any) -> dict[str, Any]:
    """Return a compact fallback/degraded status from planner diagnostics."""
    summary = _json_ready(planner_summary)
    if summary is None:
        return {
            "source": "planner_adapter_diagnostics",
            "available": False,
            "reported_fallback_or_degraded": None,
        }
    rendered = json.dumps(summary, sort_keys=True).lower()
    return {
        "source": "planner_adapter_diagnostics",
        "available": True,
        "reported_fallback_or_degraded": "fallback" in rendered or "degraded" in rendered,
    }


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
    parser.add_argument("--observation-noise-std-m", type=float, default=0.0)
    parser.add_argument("--observation-noise-bound-m", type=float, default=0.0)
    parser.add_argument("--missed-detection-probability", type=float, default=0.0)
    parser.add_argument("--occlusion-distance-m", type=float, default=None)
    parser.add_argument("--false-positive-actor-count", type=int, default=0)
    parser.add_argument("--false-positive-offset-x-m", type=float, default=1.0)
    parser.add_argument("--false-positive-offset-y-m", type=float, default=0.0)
    parser.add_argument("--false-positive-spacing-y-m", type=float, default=0.5)
    parser.add_argument("--observation-delay-steps", type=int, default=0)
    parser.add_argument("--observation-perturbation-seed", type=int, default=None)
    return parser.parse_args()


def _scenario_id(scenario: dict[str, Any]) -> str:
    """Resolve a scenario identifier from common manifest fields.

    Returns:
        str: Scenario identifier, or ``"unknown"``.
    """
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


def _sim_min_robot_ped_distance(env: Any) -> float | None:
    """Compute min robot-pedestrian distance from simulator state."""
    robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float).reshape(-1)[:2]
    ped_pos = np.asarray(getattr(env.simulator, "ped_pos", []), dtype=float)
    if ped_pos.ndim == 1:
        ped_pos = ped_pos.reshape(-1, 2) if ped_pos.size % 2 == 0 else np.zeros((0, 2), dtype=float)
    if ped_pos.size == 0:
        return None
    return float(np.min(np.linalg.norm(ped_pos[:, :2] - robot_pos.reshape(1, 2), axis=1)))


def _pedestrian_state_from_sim(env: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return current simulator pedestrian positions, velocities, and stable synthetic IDs."""
    ped_pos = np.asarray(getattr(env.simulator, "ped_pos", []), dtype=float)
    if ped_pos.ndim == 1:
        ped_pos = ped_pos.reshape(-1, 2) if ped_pos.size % 2 == 0 else np.zeros((0, 2), dtype=float)
    ped_pos = ped_pos[:, :2].copy() if ped_pos.size else np.zeros((0, 2), dtype=float)

    ped_vel = np.asarray(getattr(env.simulator, "ped_vel", []), dtype=float)
    if ped_vel.ndim == 1:
        ped_vel = ped_vel.reshape(-1, 2) if ped_vel.size % 2 == 0 else np.zeros((0, 2), dtype=float)
    if ped_vel.shape[0] != ped_pos.shape[0]:
        ped_vel = np.zeros_like(ped_pos)
    else:
        ped_vel = ped_vel[:, :2].copy()

    actor_ids = [f"ped_{idx}" for idx in range(ped_pos.shape[0])]
    return ped_pos, ped_vel, actor_ids


def _fixture_first_visible_step(scenario: Mapping[str, Any]) -> int | None:
    """Return a scenario fixture first-visible step when explicitly configured."""
    metadata = scenario.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    fixture = metadata.get("fixture_contract")
    if not isinstance(fixture, Mapping):
        return None
    value = fixture.get("first_visible_step")
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _empty_observation_snapshot() -> dict[str, Any]:
    """Return an empty observation snapshot for delay-buffer warmup."""
    return {
        "positions": np.zeros((0, 2), dtype=float),
        "velocities": np.zeros((0, 2), dtype=float),
        "ids": [],
    }


def _fixture_visibility_mask(
    *,
    actor_count: int,
    step: int,
    first_visible_step: int | None,
) -> np.ndarray | None:
    """Return a mask that hides all actors before a fixture first-visible step."""
    if first_visible_step is None or step >= first_visible_step:
        return None
    return np.zeros(max(0, int(actor_count)), dtype=bool)


def _occlusion_mask_by_distance(
    env: Any,
    ped_pos: np.ndarray,
    *,
    occlusion_distance_m: float | None,
) -> np.ndarray | None:
    """Return a simple range-limited occlusion mask for diagnostic stress tests."""
    if occlusion_distance_m is None:
        return None
    robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float).reshape(-1)[:2]
    if ped_pos.size == 0:
        return np.zeros((0,), dtype=bool)
    distances = np.linalg.norm(ped_pos - robot_pos.reshape(1, 2), axis=1)
    return distances > float(occlusion_distance_m)


def _false_positive_actor_state_from_args(
    args: argparse.Namespace,
    env: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str] | None]:
    """Return deterministic observed-only actors requested by CLI flags."""
    count = int(args.false_positive_actor_count)
    if count < 0:
        raise ValueError("false_positive_actor_count must be >= 0")
    if count == 0:
        return None, None, None
    robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float).reshape(-1)[:2]
    offsets = np.zeros((count, 2), dtype=float)
    offsets[:, 0] = float(args.false_positive_offset_x_m)
    offsets[:, 1] = float(args.false_positive_offset_y_m) + (
        np.arange(count, dtype=float) * float(args.false_positive_spacing_y_m)
    )
    positions = robot_pos.reshape(1, 2) + offsets
    velocities = np.zeros_like(positions)
    ids = [f"false_positive_{idx}" for idx in range(count)]
    return positions, velocities, ids


def _observation_perturbation_spec(
    args: argparse.Namespace,
    env: Any,
    *,
    visibility_mask: np.ndarray | None = None,
) -> ObservationPerturbationSpec:
    """Build a per-step observation perturbation spec from CLI flags."""
    ped_pos, _ped_vel, _actor_ids = _pedestrian_state_from_sim(env)
    false_pos, false_vel, false_ids = _false_positive_actor_state_from_args(args, env)
    return ObservationPerturbationSpec(
        position_noise_std_m=float(args.observation_noise_std_m),
        position_noise_bound_m=float(args.observation_noise_bound_m),
        missed_detection_probability=float(args.missed_detection_probability),
        occlusion_mask=_occlusion_mask_by_distance(
            env,
            ped_pos,
            occlusion_distance_m=args.occlusion_distance_m,
        ),
        false_positive_positions=false_pos,
        false_positive_velocities=false_vel,
        false_positive_ids=false_ids,
        delay_steps=int(args.observation_delay_steps),
        visibility_mask=visibility_mask,
        seed=args.observation_perturbation_seed,
    )


def _apply_observed_pedestrians_to_policy_obs(
    obs: Any,
    perturbation: dict[str, Any],
) -> Any:
    """Return an observation copy whose pedestrian payload uses perturbed state."""
    if not isinstance(obs, dict):
        return obs
    policy_obs = dict(obs)
    pedestrians = dict(policy_obs.get("pedestrians", {}))
    observed = perturbation["observed"]
    observed_positions = np.asarray(observed["positions"], dtype=np.float32)
    observed_velocities = np.asarray(observed["velocities"], dtype=np.float32)
    observed_count = np.asarray([observed_positions.shape[0]], dtype=np.float32)
    pedestrians["positions"] = observed_positions
    pedestrians["velocities"] = observed_velocities
    pedestrians["count"] = observed_count
    policy_obs["pedestrians"] = pedestrians
    if "pedestrians_positions" in policy_obs:
        policy_obs["pedestrians_positions"] = observed_positions
    if "pedestrians_velocities" in policy_obs:
        policy_obs["pedestrians_velocities"] = observed_velocities
    if "pedestrians_count" in policy_obs:
        policy_obs["pedestrians_count"] = observed_count
    return policy_obs


def _trace_observation_payload(perturbation: dict[str, Any]) -> dict[str, Any]:
    """Return the compact trace payload for ground-truth and observed pedestrians."""
    metadata = perturbation["metadata"]
    ground_truth = perturbation["ground_truth"]
    observed = perturbation["observed"]
    return {
        "ground_truth_observation": {
            "positions": _json_ready(ground_truth["positions"]),
            "velocities": _json_ready(ground_truth["velocities"]),
            "ids": _json_ready(ground_truth["ids"]),
            "evidence_class": EVIDENCE_IDEAL,
        },
        "observed_observation": {
            "positions": _json_ready(observed["positions"]),
            "velocities": _json_ready(observed["velocities"]),
            "ids": _json_ready(observed["ids"]),
            "missing_ids": _json_ready(perturbation["missing_ids"]),
            "evidence_class": metadata["evidence_class"],
            "noise_profile": metadata["noise_profile"],
        },
        "observation_perturbation": _json_ready(metadata),
    }


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

    entry, candidate_payload, algo_cfg, config_path = load_candidate_definition(
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
    algo, effective_cfg = _effective_candidate_runtime_for_scenario(
        candidate_payload,
        algo_cfg,
        scenario,
        default_algo=algo.strip().lower(),
        config_anchor=config_path.parent,
    )

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

    scenario = _scenario_with_episode_seed_defaults(scenario, seed=seed)
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
    default_execution_mode = "native_env_action" if planner_native_action else "command_adapter"

    env = make_robot_env(config=env_config, seed=seed, debug=False)
    trace_rows: list[dict[str, Any]] = []
    done_info: dict[str, Any] = {}
    observation_state = (
        ObservationPerturbationState(delay_steps=int(args.observation_delay_steps))
        if int(args.observation_delay_steps) > 0
        else None
    )
    first_visible_step = _fixture_first_visible_step(scenario)
    if observation_state is not None and first_visible_step is not None:
        observation_state.reset(initial_obs=_empty_observation_snapshot())
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
            min_robot_ped_dist = _sim_min_robot_ped_distance(env)
            if min_robot_ped_dist is None:
                min_robot_ped_dist = _obs_min_robot_ped_distance(obs)

            ped_pos, ped_vel, actor_ids = _pedestrian_state_from_sim(env)
            visibility_mask = _fixture_visibility_mask(
                actor_count=len(actor_ids),
                step=step_idx,
                first_visible_step=first_visible_step,
            )
            perturbation_spec = _observation_perturbation_spec(
                args,
                env,
                visibility_mask=visibility_mask,
            )
            perturbation = perturb_ground_truth(
                ped_pos,
                ped_vel,
                actor_ids,
                spec=perturbation_spec,
                step=step_idx,
                state=observation_state,
            )
            policy_obs = _apply_observed_pedestrians_to_policy_obs(obs, perturbation)
            policy_command = policy_fn(policy_obs)
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
            post_step_min_robot_ped_dist = _sim_min_robot_ped_distance(env)
            post_step_goal_distance = float(
                np.linalg.norm(
                    np.array(env.simulator.goal_pos[0], dtype=float)
                    - np.array(env.simulator.robot_pos[0], dtype=float)
                )
            )
            is_success = route_complete_success(info if isinstance(info, dict) else {})
            trace_rows.append(
                {
                    "step": int(step_idx),
                    "policy_command": _json_ready(policy_command),
                    "env_action": _json_ready(env_action),
                    "reward": float(reward),
                    "goal_distance": goal_distance,
                    "post_step_goal_distance": post_step_goal_distance,
                    "min_robot_ped_distance": min_robot_ped_dist,
                    "post_step_min_robot_ped_distance": post_step_min_robot_ped_dist,
                    "meta": _json_ready(meta),
                    "planner_decision": _json_ready(planner_decision),
                    "planner_execution_mode": (
                        "native_env_action" if step_is_native else "command_adapter"
                    ),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "is_success": bool(is_success),
                    "is_pedestrian_collision": bool(meta.get("is_pedestrian_collision", False)),
                    "is_obstacle_collision": bool(meta.get("is_obstacle_collision", False)),
                    "is_robot_collision": bool(meta.get("is_robot_collision", False)),
                    **_trace_observation_payload(perturbation),
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

    progress_summary = _trace_progress_summary(trace_rows)
    fallback_degraded_status = _planner_fallback_degraded_status(planner_summary)
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
        "planner_execution_mode": default_execution_mode,
        "fallback_degraded_status": fallback_degraded_status,
        "observation_perturbation_config": {
            "position_noise_std_m": float(args.observation_noise_std_m),
            "position_noise_bound_m": float(args.observation_noise_bound_m),
            "missed_detection_probability": float(args.missed_detection_probability),
            "occlusion_distance_m": args.occlusion_distance_m,
            "false_positive_actor_count": int(args.false_positive_actor_count),
            "false_positive_offset_x_m": float(args.false_positive_offset_x_m),
            "false_positive_offset_y_m": float(args.false_positive_offset_y_m),
            "false_positive_spacing_y_m": float(args.false_positive_spacing_y_m),
            "delay_steps": int(args.observation_delay_steps),
            "seed": args.observation_perturbation_seed,
            "fixture_first_visible_step": first_visible_step,
        },
        "planner_summary": _json_ready(planner_summary),
        "progress_summary": _json_ready(progress_summary),
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
        f"- Algorithm: `{algo}`",
        f"- Planner execution mode: `{default_execution_mode}`",
        f"- Fallback/degraded status: `{fallback_degraded_status}`",
        f"- Trace JSON: `{trace_path}`",
        f"- Decision counts: `{dict(decision_counter)}`",
        f"- Selected heads: `{dict(selected_head_counter)}`",
        f"- Observation perturbation: `{trace_payload['observation_perturbation_config']}`",
        "",
        "## Outcome",
        "",
        f"- Done info: `{_json_ready(done_info)}`",
        "",
        *_format_planner_summary_lines(planner_summary),
        "",
        "## Progress/Risk Summary",
        "",
        f"- Net goal progress: `{_format_summary_float(progress_summary['net_goal_progress'])}`",
        f"- Best goal progress: `{_format_summary_float(progress_summary['best_goal_progress'])}`",
        f"- Initial goal distance: `{_format_summary_float(progress_summary['initial_goal_distance'])}`",
        f"- Final goal distance: `{_format_summary_float(progress_summary['final_goal_distance'])}`",
        f"- Best goal distance: `{_format_summary_float(progress_summary['best_goal_distance'])}`",
        f"- Progress/regression/stagnant steps: "
        f"`{progress_summary['progress_step_count']}` / "
        f"`{progress_summary['regression_step_count']}` / "
        f"`{progress_summary['stagnant_step_count']}`",
        f"- Longest stagnant run: `{progress_summary['longest_stagnant_run']}`",
        f"- Closest robot-ped distance: "
        f"`{_format_summary_float(progress_summary['closest_robot_ped_distance'])}` "
        f"at step `{progress_summary['closest_robot_ped_step']}`",
        f"- Collision flag counts: `{progress_summary['collision_flag_counts']}`",
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

    payload = _diagnostics_stdout_payload(
        metadata={
            "trace": trace_path,
            "report": report_path,
            "scenario_id": _scenario_id(scenario),
            "family": family,
            "seed": seed,
            "decision_counts": dict(decision_counter),
            "selected_head_counts": dict(selected_head_counter),
        },
        progress_summary=progress_summary,
        planner_summary=planner_summary,
        done_info=done_info,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
