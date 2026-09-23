#!/usr/bin/env python3
"""Narrow-doorway PPO crash-vs-wait diagnostic for issue #9545.

Runs the bound released PPO checkpoint in the canonical narrow-doorway
scenario (400-step cap, seeds 225/226/227), instruments per-step reward
decomposition under the bound evaluation objective (route_completion_v3,
final-stage weights), replays matched hold/stop counterfactuals from a
pre-contact state, and runs a bounded eval-time sensitivity sweep.

Outputs (all under --output-dir):
- binding.json: checkpoint/objective/configuration binding record
- trace_seed{SEED}.jsonl: per-step measured trace (policy continuation)
- counterfactual_seed{SEED}_t{T}.jsonl: matched hold/stop continuation
- return_table.csv: crash-vs-wait discounted/undiscounted comparison
- sensitivity.csv: bounded reward/discount sensitivity (eval-time replay)
- figure_trajectory.png + figure_reward_timeline.png + figure_mechanism_timeline.png:
  deterministic diagnostic figures
- evidence_note.md is written separately (docs/analysis/...).

Evidence tiers: measured replay traces are diagnostic-only evidence, not
benchmark evidence. Counterfactual continuations diverge from the measured
episode where pedestrian reactions differ; divergence is documented per row.
No retraining is performed; sensitivity varies eval-time replay weights only.

Claim scope: bound checkpoint
  ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417
in francis2023_narrow_doorway only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.models import resolve_model_path
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]

SCENARIO_YAML = "configs/scenarios/single/francis2023_narrow_doorway.yaml"
CANONICAL_SEEDS = (225, 226, 227)
MODEL_ID = "ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417"
PREDICTIVE_MODEL_ID = "predictive_proxy_selected_v2_full"
DEFAULT_HOLD_START_OFFSET = 20
DEFAULT_HOLD_STEPS = 60
TRAINING_CONFIG = (
    "configs/training/ppo/ablations/"
    "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity.yaml"
)
TRAINING_BASE_CONFIG = (
    "configs/training/ppo/ablations/"
    "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_base.yaml"
)

FINAL_STAGE_WEIGHTS = {
    "progress": 1.1,
    "living": -0.015,
    "collision": -15.0,
    "near_miss": -1.5,
    "ttc_risk": -1.2,
    "comfort": -0.5,
    "smoothness": -0.18,
    "timeout": -6.5,
    "stagnation": -0.8,
    "terminal_bonus": 20.0,
}

BASE_ALGO_CONFIG = {
    "model_id": MODEL_ID,
    "device": "cpu",
    "deterministic": True,
    "obs_mode": "dict",
    "action_space": "unicycle",
    "v_max": 2.0,
    "omega_max": 1.0,
    "fallback_to_goal": False,
    "predictive_foresight_enabled": True,
    "predictive_foresight_model_id": PREDICTIVE_MODEL_ID,
    "predictive_foresight_device": "cpu",
}

REWARD_GIT_PATH = "robot_sf/gym_env/reward.py"
PEDESTRIAN_TTC_SCHEMA = "physical-pedestrian-ttc.v1"
PEDESTRIAN_TTC_DEFINITION = (
    "First t >= 0 satisfying ||(pedestrian_position - robot_position) + "
    "(pedestrian_velocity - robot_velocity) * t|| <= robot_radius + pedestrian_radius, "
    "under constant observed velocities; no acceleration model or reward proxy is used."
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _portable_path(path: Path) -> str:
    """Render a repository-local artifact path without leaking worktree roots."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _checkpoint_gamma() -> tuple[float | None, str]:
    """Read embedded gamma from the SB3 checkpoint data blob without torch."""
    for cand in (
        resolve_model_path(MODEL_ID),
        REPO_ROOT / f"output/model_cache/{MODEL_ID}/{MODEL_ID}-model.zip",
    ):
        try:
            if cand.exists():
                import re
                import zipfile

                raw = zipfile.ZipFile(cand).read("data")
                m = re.search(rb'"gamma": ([0-9.]+)', raw)
                if m:
                    return float(m.group(1)), str(cand)
        except (OSError, ValueError, KeyError, RuntimeError):
            continue
    return None, "unavailable"


def _git_head() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, ValueError, RuntimeError):
        return "unavailable"


def _reward_git_sha() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD", "--", REWARD_GIT_PATH],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        lines = out.stdout.strip().splitlines()
        blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{REWARD_GIT_PATH}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        _ = lines
        return blob.stdout.strip()
    except (OSError, ValueError, RuntimeError):
        return "unavailable"


def _build_diagnostic_env(seed: int, reward_weights: dict | None = None):
    """Build the narrow-doorway env with instrumented reward decomposition."""
    from robot_sf.gym_env.environment_factory import make_robot_env

    scenarios = load_scenarios(SCENARIO_YAML)
    scenario = next(s for s in scenarios if s.get("name") == "francis2023_narrow_doorway")
    config = build_env_config(scenario, scenario_path=Path(SCENARIO_YAML))
    weights = dict(FINAL_STAGE_WEIGHTS if reward_weights is None else reward_weights)
    env = make_robot_env(
        config=config,
        seed=seed,
        reward_name="route_completion_v3",
        reward_kwargs={"weights": weights},
    )
    return env, scenario, config


def _min_obstacle_clearance(env) -> float:
    """Surface clearance from robot edge to nearest static wall segment."""
    sim = env.simulator
    rx, ry = float(sim.robot_pos[0][0]), float(sim.robot_pos[0][1])
    radius = float(getattr(env.env_config.robot_config, "radius", 1.0))
    best = float("inf")
    pts = np.asarray([rx, ry], dtype=float)
    segments = getattr(sim, "iter_obstacle_segments", None)
    if callable(segments):
        obstacle_segments = segments()
    else:
        # MapDefinition.obstacles_pysf is the legacy [x1, x2, y1, y2]
        # contract. Convert it to endpoint pairs before projecting; reshaping
        # each row as two points fabricates diagonal segments.
        try:
            raw_obstacles = np.asarray(sim.map_def.obstacles_pysf, dtype=float).reshape(-1, 4)
        except (AttributeError, TypeError, ValueError):
            return float("nan")
        obstacle_segments = [((row[0], row[2]), (row[1], row[3])) for row in raw_obstacles]
    for start, end in obstacle_segments:
        a = np.asarray(start, dtype=float).reshape(-1)[:2]
        b = np.asarray(end, dtype=float).reshape(-1)[:2]
        if a.size != 2 or b.size != 2:
            return float("nan")
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = float(np.dot(pts - a, ab) / denom) if denom > 0 else 0.0
        t = min(1.0, max(0.0, t))
        proj = a + t * ab
        best = min(best, float(np.linalg.norm(pts - proj)) - radius)
    if not math.isfinite(best):
        return float("nan")
    return best


def _min_ped_distance(env) -> float:
    sim = env.simulator
    rx, ry = float(sim.robot_pos[0][0]), float(sim.robot_pos[0][1])
    try:
        peds = np.asarray(sim.ped_pos, dtype=float).reshape(-1, 2)
    except (AttributeError, TypeError, ValueError):
        return float("nan")
    if peds.size == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(peds - np.array([rx, ry]), axis=1)))


def _serialize_env_action(action) -> list[float]:
    """Serialize the exact one-dimensional action vector passed to ``env.step``."""
    try:
        values = np.asarray(action, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("env action must be a numeric vector for trace serialization") from exc
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("env action must be a finite one-dimensional vector")
    return values.tolist()


def _trace_xy_rows(value) -> np.ndarray | None:
    """Coerce an observed actor position/velocity collection without inventing rows."""
    if value is None:
        return None
    try:
        rows = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if rows.size == 0:
        return np.empty((0, 2), dtype=float)
    if rows.ndim == 1 and rows.shape == (2,):
        rows = rows.reshape(1, 2)
    if rows.ndim != 2 or rows.shape[1] != 2 or not np.all(np.isfinite(rows)):
        return None
    return rows


def _trace_xy_vector(value) -> np.ndarray | None:
    """Coerce one observed two-dimensional position or velocity."""
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _trace_radius(value) -> float | None:
    """Return a finite positive radius, or ``None`` when unavailable."""
    try:
        radius = float(value)
    except (TypeError, ValueError):
        return None
    return radius if math.isfinite(radius) and radius > 0.0 else None


def _disc_contact_time(relative_position: np.ndarray, relative_velocity: np.ndarray, radius: float):
    """Return first contact for constant relative velocity and its classification.

    The horizon is unbounded (t >= 0), so every finite nonzero closing speed
    remains eligible for an estimate. Normalize velocity before solving the
    distance-to-contact quadratic to avoid a speed-squared cutoff.
    """
    c = float(np.dot(relative_position, relative_position) - radius**2)
    if c <= 1e-12:
        return 0.0, "overlapping"
    relative_speed = math.hypot(float(relative_velocity[0]), float(relative_velocity[1]))
    if relative_speed == 0.0:
        return None, None
    direction = relative_velocity / relative_speed
    b = 2.0 * float(np.dot(relative_position, direction))
    discriminant = b * b - 4.0 * c
    if discriminant < -1e-12:
        return None, None
    root = math.sqrt(max(0.0, discriminant))
    stable_root = -0.5 * (b + math.copysign(root, b))
    if stable_root == 0.0:
        contact_distances = [-b / 2.0]
    else:
        contact_distances = [stable_root, c / stable_root]
    future_times = [distance / relative_speed for distance in contact_distances if distance >= 0.0]
    return (min(future_times), "estimated") if future_times else (None, None)


def _physical_pedestrian_ttc(
    *,
    robot_position,
    robot_velocity,
    robot_radius,
    pedestrian_positions,
    pedestrian_velocities,
    pedestrian_radius,
) -> dict:
    """Estimate first disc contact under constant observed robot/pedestrian velocities.

    This geometric estimate is deliberately independent of ``ttc_risk`` reward
    decomposition. Missing or malformed simulator inputs produce a null estimate with
    an explicit status; non-intersecting trajectories are not encoded as infinity.
    """
    result = {
        "schema_version": PEDESTRIAN_TTC_SCHEMA,
        "definition": PEDESTRIAN_TTC_DEFINITION,
        "estimate_s": None,
        "status": "unavailable",
        "reason": None,
        "pedestrian_index": None,
        "robot_radius_m": None,
        "pedestrian_radius_m": None,
        "combined_radius_m": None,
    }
    ped_positions = _trace_xy_rows(pedestrian_positions)
    if ped_positions is None:
        result["reason"] = "pedestrian positions are unavailable or malformed"
        return result
    if ped_positions.shape[0] == 0:
        result.update(status="no_pedestrians", reason="simulator reports no pedestrians")
        return result

    ped_velocities = _trace_xy_rows(pedestrian_velocities)
    robot_pos = _trace_xy_vector(robot_position)
    robot_vel = _trace_xy_vector(robot_velocity)
    robot_radius_value = _trace_radius(robot_radius)
    ped_radius_value = _trace_radius(pedestrian_radius)
    if ped_velocities is None or ped_velocities.shape != ped_positions.shape:
        result["reason"] = "pedestrian velocities are unavailable or do not match positions"
        return result
    if robot_pos is None:
        result["reason"] = "robot position is unavailable or malformed"
        return result
    if robot_vel is None:
        result["reason"] = "robot velocity is unavailable or malformed"
        return result
    if robot_radius_value is None or ped_radius_value is None:
        result["reason"] = "robot or pedestrian collision radius is unavailable or invalid"
        return result

    combined_radius = robot_radius_value + ped_radius_value
    result.update(
        robot_radius_m=robot_radius_value,
        pedestrian_radius_m=ped_radius_value,
        combined_radius_m=combined_radius,
    )
    candidates: list[tuple[float, int, str]] = []
    for index, (ped_pos, ped_vel) in enumerate(zip(ped_positions, ped_velocities, strict=True)):
        relative_position = ped_pos - robot_pos
        relative_velocity = ped_vel - robot_vel
        estimate, status = _disc_contact_time(relative_position, relative_velocity, combined_radius)
        if estimate is not None and status is not None:
            candidates.append((estimate, index, status))

    if not candidates:
        result.update(
            status="no_intercept",
            reason=(
                "no pedestrian's constant-velocity relative trajectory intersects the "
                "combined-radius disc for t >= 0"
            ),
        )
        return result
    estimate_s, pedestrian_index, status = min(candidates)
    result.update(
        estimate_s=float(estimate_s),
        status=status,
        pedestrian_index=pedestrian_index,
    )
    return result


def _step_trace_state(simulator) -> dict:
    """Capture the observed kinematics and physical pedestrian TTC for a trace row."""
    ped_positions = _trace_xy_rows(getattr(simulator, "ped_pos", None))
    ped_velocities = _trace_xy_rows(getattr(simulator, "ped_vel", None))
    robot_position = None
    robot_velocity = None
    robot_radius = None
    try:
        robot = simulator.robots[0]
        pose = robot.pose
        robot_position = _trace_xy_vector(pose[0])
        heading = float(pose[1])
        current_speed = np.asarray(robot.current_speed, dtype=float).reshape(-1)
        if (
            current_speed.size >= 1
            and math.isfinite(heading)
            and np.all(np.isfinite(current_speed))
        ):
            linear_speed = float(current_speed[0])
            robot_velocity = np.asarray(
                [linear_speed * math.cos(heading), linear_speed * math.sin(heading)],
                dtype=float,
            )
        robot_radius = _trace_radius(getattr(getattr(robot, "config", None), "radius", None))
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    pedestrian_radius = _trace_radius(
        getattr(getattr(simulator, "config", None), "ped_radius", None)
    )
    ttc = _physical_pedestrian_ttc(
        robot_position=robot_position,
        robot_velocity=robot_velocity,
        robot_radius=robot_radius,
        pedestrian_positions=ped_positions,
        pedestrian_velocities=ped_velocities,
        pedestrian_radius=pedestrian_radius,
    )
    return {
        "ped_positions": None if ped_positions is None else ped_positions.tolist(),
        "ped_velocities_mps": None if ped_velocities is None else ped_velocities.tolist(),
        "robot_velocity_mps": None if robot_velocity is None else robot_velocity.tolist(),
        "pedestrian_ttc": ttc,
    }


def _normalize_runner_obs(step_obs):
    """Normalize a runner observation through the canonical map-runner path."""
    from robot_sf.benchmark.map_runner.map_runner_observations import normalize_map_observation

    return normalize_map_observation(step_obs)


def _rollout_policy(env, planner, max_steps: int, obs):
    """Step the bound policy to termination; return per-step measured rows.

    Uses the canonical map-runner command path: planner (v, omega) velocity
    command converted to env acceleration action via
    policy_command_to_env_action, matching benchmark replay exactly.
    """
    from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
        policy_command_to_env_action as _to_env_action,
    )

    config = env.env_config
    rows: list[dict] = []
    done = False
    step_idx = 0
    while not done and step_idx < max_steps:
        # Preserve the complete canonical environment dict.  The PPO adapter
        # aligns its native MultiInput observation and predictive features from
        # this payload; lossy reconstruction silently backfills model inputs.
        step_obs = _normalize_runner_obs(obs)
        action_dict = planner.step(step_obs)
        _assert_predictive_foresight_loaded(planner)
        v = float(action_dict.get("v", action_dict.get("linear_velocity", 0.0)))
        w = float(action_dict.get("omega", action_dict.get("angular_velocity", 0.0)))
        env_action = np.asarray(_to_env_action(env=env, config=config, command=(v, w)))
        action_trace = _serialize_env_action(env_action)
        obs, reward, terminated, truncated, info = env.step(env_action)
        meta = dict(info.get("meta", {}))
        terms = {k: float(vv) for k, vv in dict(meta.get("reward_terms", {})).items()}
        rx, ry = float(env.simulator.robot_pos[0][0]), float(env.simulator.robot_pos[0][1])
        rows.append(
            {
                "step": step_idx,
                "robot_x": rx,
                "robot_y": ry,
                "heading": float(env.simulator.robot_poses[0][1]),
                "cmd_v": v,
                "cmd_w": w,
                "env_action": action_trace,
                **_step_trace_state(env.simulator),
                "reward": float(reward),
                "reward_terms": terms,
                "is_obstacle_collision": bool(meta.get("is_obstacle_collision", False)),
                "is_pedestrian_collision": bool(meta.get("is_pedestrian_collision", False)),
                "is_route_complete": bool(meta.get("is_route_complete", False)),
                "is_timesteps_exceeded": bool(meta.get("is_timesteps_exceeded", False)),
                "distance_to_goal": float(meta.get("distance_to_goal", float("nan"))),
                "min_obstacle_clearance_m": _min_obstacle_clearance(env),
                "min_ped_distance_m": _min_ped_distance(env),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        done = bool(terminated or truncated)
        step_idx += 1
    return rows


def _make_planner():
    """Build the bound PPO planner adapter for diagnostic replay."""
    from robot_sf.baselines.ppo import PPOPlanner

    planner = PPOPlanner(dict(BASE_ALGO_CONFIG))
    metadata = planner.get_metadata()
    if metadata.get("status") != "ok":
        raise RuntimeError(f"PPO planner did not load successfully: {metadata}")
    return planner


def _assert_predictive_foresight_loaded(planner) -> None:
    """Reject replay when the configured predictive checkpoint degraded."""
    diagnostics = planner.foresight_diagnostics()
    provenance = diagnostics.get("foresight_prediction", {})
    if (
        provenance.get("load_status") != "loaded"
        or provenance.get("effective_prediction_mode") != "predictive_foresight"
        or provenance.get("fallback_used") is True
    ):
        raise RuntimeError(
            "Diagnostic replay requires the verified predictive checkpoint; "
            f"observed provenance={provenance}"
        )


def _discounted_return(rewards: list[float], gamma: float) -> float:
    """Accumulate discounted return over a reward sequence."""
    total = 0.0
    disc = 1.0
    for r in rewards:
        total += disc * float(r)
        disc *= gamma
    return total


def run_diagnostic(
    seeds: tuple[int, ...],
    out_dir: Path,
    gamma: float,
    hold_start_offset: int = DEFAULT_HOLD_START_OFFSET,
    hold_steps: int = DEFAULT_HOLD_STEPS,
    sensitivity_gammas: tuple[float, ...] = (0.95, 0.99, 0.995),
    sensitivity_collisions: tuple[float, ...] = (-5.0, -15.0, -25.0),
) -> dict:
    """Run measured replays, matched counterfactuals, and sensitivity sweep."""
    out_dir.mkdir(parents=True, exist_ok=True)
    planner = _make_planner()
    binding = build_binding(out_dir, gamma)
    max_steps = binding["scenario_cap_steps"]
    summary_rows: list[dict] = []
    all_traces: dict[int, list[dict]] = {}
    for seed in seeds:
        env, _scenario, _config = _build_diagnostic_env(seed)
        obs, _info = env.reset(seed=seed)
        rows = _rollout_policy(env, planner, max_steps, obs)
        all_traces[seed] = rows
        with (out_dir / f"trace_seed{seed}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        contact_idx = next(
            (i for i, r in enumerate(rows) if r["is_obstacle_collision"] or r["terminated"]),
            None,
        )
        summary_rows.append(
            {
                "seed": seed,
                "steps": len(rows),
                "contact_step": contact_idx,
                "undiscounted_return": sum(r["reward"] for r in rows),
                "discounted_return": _discounted_return([r["reward"] for r in rows], gamma),
                "outcome": "collision"
                if any(r["is_obstacle_collision"] for r in rows)
                else ("success" if any(r["is_route_complete"] for r in rows) else "timeout"),
            }
        )
        env.close()

    counter_rows: list[dict] = []
    for seed in seeds:
        rows = all_traces[seed]
        contact_idx = next(
            (i for i, r in enumerate(rows) if r["is_obstacle_collision"] or r["terminated"]),
            len(rows) - 1,
        )
        start_idx = max(0, contact_idx - hold_start_offset)
        for mode in ("hold_stop", "policy_continuation"):
            branch = _replay_branch_from_prefix(
                seed, rows, start_idx, mode=mode, horizon=hold_steps, gamma=gamma
            )
            with (out_dir / f"counterfactual_seed{seed}_t{start_idx}_{mode}.jsonl").open(
                "w", encoding="utf-8"
            ) as fh:
                for row in branch:
                    fh.write(json.dumps(row) + "\n")
            counter_rows.append(
                {
                    "seed": seed,
                    "fork_step": start_idx,
                    "contact_step": contact_idx,
                    "mode": mode,
                    "steps": len(branch),
                    "undiscounted_return": sum(r["reward"] for r in branch),
                    "discounted_return": _discounted_return([r["reward"] for r in branch], gamma),
                    "ends_in_contact": any(r["is_obstacle_collision"] for r in branch),
                    "divergence_note": "open-loop prefix replay to fork state; pedestrian "
                    "reactions after fork may differ from measured episode",
                }
            )

    sens_rows: list[dict] = []
    for seed in seeds:
        rows = all_traces[seed]
        contact_idx = next(
            (i for i, r in enumerate(rows) if r["is_obstacle_collision"] or r["terminated"]),
            len(rows) - 1,
        )
        start_idx = max(0, contact_idx - hold_start_offset)
        for g in sensitivity_gammas:
            for c in sensitivity_collisions:
                weights = dict(FINAL_STAGE_WEIGHTS)
                weights["collision"] = float(c)
                crash_branch = _replay_branch_from_prefix(
                    seed,
                    rows,
                    start_idx,
                    mode="policy_continuation",
                    horizon=hold_steps,
                    gamma=g,
                    reward_weights=weights,
                )
                wait_branch = _replay_branch_from_prefix(
                    seed,
                    rows,
                    start_idx,
                    mode="hold_stop",
                    horizon=hold_steps,
                    gamma=g,
                    reward_weights=weights,
                )
                crash_ret = _discounted_return([r["reward"] for r in crash_branch], g)
                wait_ret = _discounted_return([r["reward"] for r in wait_branch], g)
                sens_rows.append(
                    {
                        "seed": seed,
                        "fork_step": start_idx,
                        "gamma": g,
                        "collision_weight": c,
                        "crash_discounted": crash_ret,
                        "wait_discounted": wait_ret,
                        "prefers_crash": bool(crash_ret > wait_ret),
                    }
                )

    with (out_dir / "return_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(counter_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(counter_rows)
    with (out_dir / "sensitivity.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sens_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sens_rows)
    with (out_dir / "episode_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    _render_trajectory_figure(all_traces, out_dir / "figure_trajectory.png")
    _render_reward_timeline(all_traces, out_dir / "figure_reward_timeline.png", contact_note=True)
    _render_mechanism_timeline(all_traces, out_dir / "figure_mechanism_timeline.png")
    return {"binding": binding, "episodes": summary_rows, "counterfactuals": counter_rows}


def _replay_branch_from_prefix(
    seed: int,
    measured_rows: list[dict],
    start_idx: int,
    *,
    mode: str,
    horizon: int,
    gamma: float,
    reward_weights: dict | None = None,
) -> list[dict]:
    """Replay env to start_idx with policy actions, then apply branch mode.

    Prefix replay is open-loop (recorded cmd_v/cmd_w converted through the
    canonical policy_command_to_env_action path) to recover the pre-contact
    state deterministically. The hold branch commands (v=0, w=0) velocity,
    which decelerates the robot to rest via the same conversion; it never
    freezes the simulator. Rewards are recomputed live by the env reward
    function so decomposition stays bound to the evaluation objective.
    """
    from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
        policy_command_to_env_action as _to_env_action,
    )

    env, _scenario, _config = _build_diagnostic_env(seed, reward_weights=reward_weights)
    try:
        obs, _info = env.reset(seed=seed)
        config = env.env_config
        prefix = measured_rows[:start_idx]
        for row in prefix:
            action = np.asarray(
                _to_env_action(env=env, config=config, command=(row["cmd_v"], row["cmd_w"]))
            )
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        branch: list[dict] = []
        planner = _make_planner() if mode == "policy_continuation" else None
        for k in range(horizon):
            if mode == "hold_stop" or planner is None:
                cmd_v, cmd_w = 0.0, 0.0
                action = np.asarray(_to_env_action(env=env, config=config, command=(0.0, 0.0)))
            else:
                act = planner.step(_normalize_runner_obs(obs))
                _assert_predictive_foresight_loaded(planner)
                cmd_v = float(act.get("v", act.get("linear_velocity", 0.0)))
                cmd_w = float(act.get("omega", act.get("angular_velocity", 0.0)))
                action = np.asarray(_to_env_action(env=env, config=config, command=(cmd_v, cmd_w)))
            action_trace = _serialize_env_action(action)
            obs, reward, terminated, truncated, info = env.step(action)
            meta = dict(info.get("meta", {}))
            terms = {kk: float(vv) for kk, vv in dict(meta.get("reward_terms", {})).items()}
            rx, ry = float(env.simulator.robot_pos[0][0]), float(env.simulator.robot_pos[0][1])
            branch.append(
                {
                    "branch_step": k,
                    "fork_step": start_idx,
                    "mode": mode,
                    "robot_x": rx,
                    "robot_y": ry,
                    "cmd_v": cmd_v,
                    "cmd_w": cmd_w,
                    "env_action": action_trace,
                    **_step_trace_state(env.simulator),
                    "reward": float(reward),
                    "reward_terms": terms,
                    "is_obstacle_collision": bool(meta.get("is_obstacle_collision", False)),
                    "is_route_complete": bool(meta.get("is_route_complete", False)),
                    "is_timesteps_exceeded": bool(meta.get("is_timesteps_exceeded", False)),
                    "distance_to_goal": float(meta.get("distance_to_goal", float("nan"))),
                    "min_obstacle_clearance_m": _min_obstacle_clearance(env),
                    "min_ped_distance_m": _min_ped_distance(env),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            if terminated or truncated:
                break
        return branch
    finally:
        env.close()


def build_binding(out_dir: Path, gamma: float) -> dict:
    """Record the immutable checkpoint/objective/configuration binding."""
    scenarios = load_scenarios(SCENARIO_YAML)
    scenario = next(s for s in scenarios if s.get("name") == "francis2023_narrow_doorway")
    cap = int(scenario["simulation_config"]["max_episode_steps"])
    model_path = resolve_model_path(MODEL_ID)
    predictive_model_path = resolve_model_path(PREDICTIVE_MODEL_ID)
    base_cfg = yaml.safe_load((REPO_ROOT / TRAINING_BASE_CONFIG).read_text(encoding="utf-8"))
    gamma_embedded, gamma_source = _checkpoint_gamma()
    binding = {
        "schema": "issue_9545_binding.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_head": _git_head(),
        "scenario": "francis2023_narrow_doorway",
        "scenario_file": SCENARIO_YAML,
        "scenario_cap_steps": cap,
        "scenario_seeds": list(scenario.get("seeds", [])),
        "checkpoint_model_id": MODEL_ID,
        "checkpoint_local_path": _portable_path(Path(model_path)),
        "checkpoint_sha256": _sha256_file(Path(model_path))
        if Path(model_path).exists()
        else "unavailable",
        "checkpoint_gamma_embedded": gamma_embedded,
        "checkpoint_gamma_source": f"SB3 data blob: {gamma_source}",
        "predictive_checkpoint_model_id": PREDICTIVE_MODEL_ID,
        "predictive_checkpoint_local_path": _portable_path(Path(predictive_model_path)),
        "predictive_checkpoint_sha256": _sha256_file(Path(predictive_model_path))
        if Path(predictive_model_path).exists()
        else "unavailable",
        "gamma_used_for_discounted_replay": gamma,
        "gamma_training_config_declared": None,
        "gamma_provenance_note": "Base training config declares no ppo_hyperparams.gamma; "
        "0.99 is checkpoint-embedded artifact metadata used for eval-time discounted "
        "replay only, not proof of the training-time objective.",
        "reward_impl": "robot_sf/gym_env/reward.py:route_completion_v3_reward",
        "reward_impl_git_blob_sha": _reward_git_sha(),
        "reward_name": "route_completion_v3",
        "reward_weights_final_stage": dict(FINAL_STAGE_WEIGHTS),
        "reward_curriculum": "2 stages; stage0 until_episodes=100 (progress/collision/timeout/terminal only); "
        "final stage = full weights above. Checkpoint at ~10M steps is past stage 0; "
        "stage-at-checkpoint is inferred, not logged.",
        "training_config": TRAINING_CONFIG,
        "training_base_config": TRAINING_BASE_CONFIG,
        "base_config_reward_block_sha256": hashlib.sha256(
            json.dumps(base_cfg.get("env_factory_kwargs", {}), sort_keys=True).encode()
        ).hexdigest(),
        "wrappers": {
            "benchmark_adapter": "ppo_action_to_unicycle (mixed), feasibility projection",
            "safety_wrapper": "disabled",
            "cbf_safety_filter": "disabled",
            "predictive_foresight": "enabled from the registry-pinned predictive checkpoint on CPU; "
            "replay fails closed if model loading degrades",
            "predictive_foresight_device": "cpu",
            "fallback_to_goal": False,
        },
        "observation_contract": {
            "status": "canonical_raw_grid_socnav_with_predictive_foresight",
            "source": "RobotEnv raw dict -> normalize_map_observation -> PPOPlanner native dict alignment",
            "occupancy_grid_preserved": True,
            "predictive_foresight_preserved": True,
            "fallback_or_degraded_execution": False,
        },
        "counterfactual_fork": {
            "hold_start_offset_steps": DEFAULT_HOLD_START_OFFSET,
            "hold_horizon_steps": DEFAULT_HOLD_STEPS,
            "rationale": "Fork early enough for the zero-command branch to decelerate before contact; "
            "the same measured prefix is replayed for both branches.",
        },
        "termination_semantics": "terminated = route_complete OR timeout(timestep>=max_sim_steps) OR "
        "ped/robot/obstacle collision; RobotEnv returns truncated=False always; "
        "reward timeout term fires only on non-collision, non-success timeout",
        "eval_reward_note": "Per-step rewards are recomputed live by route_completion_v3 with the "
        "final-stage weights; valid as evaluation-time replay under the bound objective, "
        "not as a recovered training-time return without training-log corroboration.",
    }
    with (out_dir / "binding.json").open("w", encoding="utf-8") as fh:
        json.dump(binding, fh, indent=2, sort_keys=True)
    return binding


def _render_trajectory_figure(traces: dict[int, list[dict]], path: Path) -> None:
    """Render per-seed trajectory panels against doorway geometry."""
    fig, axes = plt.subplots(1, len(traces), figsize=(4 * len(traces), 4), sharey=True)
    if len(traces) == 1:
        axes = [axes]
    for ax, (seed, rows) in zip(axes, sorted(traces.items()), strict=False):
        ax.add_patch(Rectangle((0.0, 0.0), 30.0, 1.0, color="0.75", alpha=0.7))
        ax.add_patch(Rectangle((15.0, 1.0), 1.0, 3.0, color="0.75", alpha=0.7))
        ax.add_patch(Rectangle((15.0, 6.0), 1.0, 3.0, color="0.75", alpha=0.7))
        xs = [r["robot_x"] for r in rows]
        ys = [r["robot_y"] for r in rows]
        ax.plot(xs, ys, "-o", ms=2, label=f"seed {seed}")
        ax.set_title(f"seed {seed}: {len(rows)} steps to contact")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(2, 18)
        ax.set_ylim(0, 8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Narrow-doorway PPO replay: bottom boundary y=1 and doorway walls x=15..16 (gap y=4..6)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_reward_timeline(traces: dict[int, list[dict]], path: Path, contact_note: bool) -> None:
    """Render per-step reward and wall-clearance timelines to contact."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for seed, rows in sorted(traces.items()):
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [r["reward"] for r in rows], label=f"seed {seed}")
        axes[1].plot(steps, [r["min_obstacle_clearance_m"] for r in rows], label=f"seed {seed}")
    axes[0].set_ylabel("route_completion_v3 step reward")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("wall clearance (m, edge)")
    axes[1].set_xlabel("env step")
    axes[1].grid(alpha=0.3)
    if contact_note:
        fig.suptitle("Per-step reward + wall clearance to contact (measured replay)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_mechanism_timeline(traces: dict[int, list[dict]], path: Path) -> None:
    """Render the measured progress, command, and pedestrian-distance traces."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for seed, rows in sorted(traces.items()):
        steps = [r["step"] for r in rows]
        progress = [r["reward_terms"].get("progress", 0.0) for r in rows]
        axes[0].plot(steps, progress, label=f"seed {seed}")
        axes[1].plot(steps, [r["cmd_v"] for r in rows], label=f"seed {seed}")
        axes[2].plot(steps, [r["min_ped_distance_m"] for r in rows], label=f"seed {seed}")
    axes[0].set_ylabel("progress term")
    axes[1].set_ylabel("cmd_v (m/s)")
    axes[2].set_ylabel("min ped distance (m)")
    axes[2].set_xlabel("env step")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Narrow-doorway PPO mechanism trace (canonical measured replay)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Build the diagnostic CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="225,226,227")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hold-start-offset", type=int, default=DEFAULT_HOLD_START_OFFSET)
    parser.add_argument("--hold-steps", type=int, default=DEFAULT_HOLD_STEPS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the crash-vs-wait diagnostic from CLI arguments."""
    args = build_parser().parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    for s in seeds:
        if s not in CANONICAL_SEEDS:
            print(f"WARNING: seed {s} is outside the canonical {CANONICAL_SEEDS}", file=sys.stderr)
    result = run_diagnostic(
        seeds,
        args.output_dir,
        gamma=float(args.gamma),
        hold_start_offset=int(args.hold_start_offset),
        hold_steps=int(args.hold_steps),
    )
    print(json.dumps(result["episodes"], indent=2))
    print(json.dumps(result["counterfactuals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
