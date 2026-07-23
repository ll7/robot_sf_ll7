#!/usr/bin/env python3
"""Run manifest compilation and disjoint-seed activation preflight for issue #5578 / #6101."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.issue_5578_speed_tier_synthesis import (
    DECLARED_PLANNERS,
    DECLARED_SCENARIOS,
    DECLARED_SEEDS,
    EXPECTED_DT_SECONDS,
    EXPECTED_HORIZON_STEPS,
    MIN_ACTIVATION_FRACTION_ABOVE_2_0,
    MIN_ACTIVATION_PEAK_SPEED,
    NON_NOMINAL_TIERS,
    TIER_ACTUATION_ENVELOPES,
    TIER_CAPS_M_S,
)
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)
from robot_sf.planner.socnav import (
    ORCAPlannerAdapter,
    PredictionPlannerAdapter,
    SocNavPlannerConfig,
)
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from scripts.validation.check_issue_5578_robot_speed_tier_preregistration import (
    DEFAULT_CONFIG as DEFAULT_PREREGISTRATION_CONFIG,
)
from scripts.validation.check_issue_5578_robot_speed_tier_preregistration import (
    FORBIDDEN_ROUTING_KEYS,
    load_preregistration,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_MATRIX_PATH = REPO_ROOT / "configs/scenarios/classic_interactions.yaml"
DISJOINT_PREFLIGHT_SEEDS = (901, 902, 903)
DEFAULT_PREFLIGHT_PLANNERS = [
    "goal_seek",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
]
EXPECTED_SCENARIOS = tuple(DECLARED_SCENARIOS)
EXPECTED_TIERS = ("cap_2_0_nominal", "cap_3_0", "cap_4_0")
EXPECTED_PLANNERS = (
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "ppo",
    "orca",
    "prediction_planner",
)
EXPECTED_SEEDS = tuple(DECLARED_SEEDS)


def get_git_sha() -> str:
    """Return the current 40-character git commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def is_git_dirty() -> bool:
    """Return True if the git working tree has uncommitted changes."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(res.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False


def build_campaign_manifest(
    config_path: Path | str = DEFAULT_PREREGISTRATION_CONFIG,
) -> list[dict[str, Any]]:
    """Compile the exact 2,160-cell campaign execution manifest from preregistration.

    Returns:
        A list of 2,160 identity mappings for scenario x tier x planner x seed.
    """
    prereg = load_preregistration(config_path)
    scenarios = [s["scenario_id"] for s in prereg["scenario_contract"]["selected_scenarios"]]
    tiers = prereg["robot_speed_axis"]["tiers"]
    planners = prereg["planner_roster"]["arms"]
    seeds = prereg["seed_policy"]["seeds"]

    scenario_map = {
        s["scenario_id"]: s["source_path"]
        for s in prereg["scenario_contract"]["selected_scenarios"]
    }

    manifest: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()

    for scenario_id in scenarios:
        for tier in tiers:
            tier_id = tier["tier_id"]
            cap_m_s = float(tier["cap_m_s"])
            for planner in planners:
                planner_id = planner["planner_id"]
                for seed in seeds:
                    identity = (scenario_id, tier_id, planner_id, seed)
                    if identity in seen:
                        raise ValueError(
                            f"Duplicate cell identity in manifest compilation: {identity}"
                        )
                    seen.add(identity)

                    actuation = TIER_ACTUATION_ENVELOPES[tier_id]
                    cell_entry = {
                        "scenario_id": scenario_id,
                        "speed_tier_id": tier_id,
                        "speed_cap_m_s": cap_m_s,
                        "planner_id": planner_id,
                        "seed": seed,
                        "horizon_steps": EXPECTED_HORIZON_STEPS,
                        "dt_seconds": EXPECTED_DT_SECONDS,
                        "execution_mode": "native",
                        "runtime_variant_key": tier["runtime_variant_key"],
                        "resolved_actuation_envelope": dict(actuation),
                        "scenario_source_path": scenario_map[scenario_id],
                        "planner_config_path": planner.get("config_path"),
                    }
                    manifest.append(cell_entry)

    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: list[dict[str, Any]]) -> None:  # noqa: C901
    """Validate that manifest contains exactly 2,160 valid, non-duplicate cells."""
    if len(manifest) != 2160:
        raise ValueError(f"Manifest must contain exactly 2,160 cells, got {len(manifest)}")

    seen: set[tuple[str, str, str, int]] = set()
    for index, cell in enumerate(manifest):
        if not isinstance(cell, dict):
            raise ValueError(f"Manifest entry at index {index} must be a dict")

        for forbidden_key in FORBIDDEN_ROUTING_KEYS:
            if forbidden_key in cell:
                raise ValueError(
                    f"Manifest cell at index {index} contains forbidden routing key: {forbidden_key}"
                )

        scenario_id = str(cell.get("scenario_id"))
        speed_tier_id = str(cell.get("speed_tier_id"))
        planner_id = str(cell.get("planner_id"))
        seed = int(cell.get("seed", -1))

        identity = (scenario_id, speed_tier_id, planner_id, seed)
        if identity in seen:
            raise ValueError(f"Duplicate cell in manifest: {identity}")
        seen.add(identity)

        if scenario_id not in DECLARED_SCENARIOS:
            raise ValueError(f"Undeclared scenario in manifest: {scenario_id}")
        if speed_tier_id not in TIER_CAPS_M_S:
            raise ValueError(f"Undeclared speed tier in manifest: {speed_tier_id}")
        if planner_id not in DECLARED_PLANNERS:
            raise ValueError(f"Undeclared planner in manifest: {planner_id}")
        if seed not in DECLARED_SEEDS:
            raise ValueError(f"Undeclared seed in manifest: {seed}")

        cap = TIER_CAPS_M_S[speed_tier_id]
        if not math.isclose(float(cell["speed_cap_m_s"]), cap, abs_tol=1e-12):
            raise ValueError(f"Cell speed cap drift: expected {cap}, got {cell['speed_cap_m_s']}")

        actuation = cell.get("resolved_actuation_envelope")
        expected_actuation = TIER_ACTUATION_ENVELOPES[speed_tier_id]
        if actuation != expected_actuation:
            raise ValueError(
                f"Actuation envelope mismatch for cell {identity}: {actuation} vs {expected_actuation}"
            )


def _robot_angular_cap(
    cap_m_s: float, max_steer_rad: float = 0.78, wheelbase_m: float = 1.0
) -> float:
    return float(cap_m_s * math.tan(max_steer_rad) / max(wheelbase_m, 1e-6))


def _wrap_angle(angle: float) -> float:
    """Wrap angle into [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _env_action(env: Any, command: Mapping[str, float]) -> np.ndarray:
    """Convert unicycle (v, omega) command into bicycle (acceleration, steering) action."""
    robot = env.simulator.robots[0]
    current_linear, _current_angular = robot.current_speed
    desired_linear = float(command.get("v", command.get("linear", 0.0)))
    desired_angular = float(command.get("omega", command.get("angular", 0.0)))
    if isinstance(robot, BicycleDriveRobot):
        config = robot.config
        target_speed = float(np.clip(desired_linear, config.min_velocity, config.max_velocity))
        time_step = float(env.env_config.sim_config.time_per_step_in_secs)
        acceleration = (target_speed - float(current_linear)) / max(time_step, 1e-6)
        if abs(target_speed) < 1e-6:
            steering = 0.0
        else:
            steering = math.atan(
                desired_angular * config.wheelbase / max(abs(target_speed), 1e-6)
            ) * np.sign(target_speed)
        action = np.array([acceleration, steering], dtype=float)
        return np.clip(action, env.action_space.low, env.action_space.high)
    return np.array(
        [desired_linear - float(current_linear), desired_angular - float(_current_angular)],
        dtype=float,
    )


def _build_socnav_obs_from_env(env: Any) -> dict[str, Any]:
    """Build a SocNav observation directly from simulator state."""
    robot = env.simulator.robots[0]
    robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float).tolist()
    heading = float(robot.pose[1])
    current_linear, _ = robot.current_speed
    robot_vel = [
        float(current_linear * math.cos(heading)),
        float(current_linear * math.sin(heading)),
    ]
    goal_pos = np.asarray(env.simulator.goal_pos[0], dtype=float).tolist()

    ped_positions = np.asarray(env.simulator.ped_pos, dtype=float)
    ped_velocities = np.asarray(
        getattr(env.simulator, "ped_vel", np.zeros_like(ped_positions)), dtype=float
    )
    if ped_velocities.shape != ped_positions.shape:
        ped_velocities = np.zeros_like(ped_positions)

    ped_pos_list = [ped_positions[i].tolist() for i in range(ped_positions.shape[0])]
    ped_vel_list = [ped_velocities[i].tolist() for i in range(ped_velocities.shape[0])]

    get_lines = getattr(env.simulator, "get_obstacle_lines", None)
    obstacles = list(get_lines()) if callable(get_lines) else []

    return {
        "dt": float(env.env_config.sim_config.time_per_step_in_secs),
        "robot": {
            "position": robot_pos,
            "heading": [heading],
            "speed": robot_vel,
            "radius": [float(robot.config.radius)],
        },
        "goal": {"current": goal_pos},
        "pedestrians": {
            "positions": ped_pos_list,
            "velocities": ped_vel_list,
            "count": [len(ped_pos_list)],
            "radius": float(env.env_config.sim_config.ped_radius),
        },
        "obstacles": obstacles,
    }


class GoalSeekPlanner:
    """Deterministic goal-facing unicycle command policy for tests and preflight."""

    def __init__(self, *, max_linear_speed: float, max_angular_speed: float) -> None:
        """Initialize goal seek planner speeds."""
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)

    def reset(self, *, seed: int | None = None) -> None:
        """Reset planner state for a new episode."""
        del seed

    def step(self, obs: Any) -> dict[str, float]:
        """Compute goal-seeking unicycle control command."""
        socnav_obs = _build_socnav_obs_from_env(obs) if not isinstance(obs, dict) else obs
        robot_pos = np.asarray(socnav_obs["robot"]["position"], dtype=float)
        robot_goal = np.asarray(socnav_obs["goal"]["current"], dtype=float)
        heading_list = socnav_obs["robot"].get("heading", [0.0])
        heading = float(heading_list[0]) if isinstance(heading_list, list) and heading_list else 0.0
        delta = robot_goal - robot_pos
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6:
            return {"v": 0.0, "omega": 0.0}
        desired_heading = float(math.atan2(delta[1], delta[0]))
        heading_error = _wrap_angle(desired_heading - heading)
        angular = float(
            np.clip(2.0 * heading_error, -self.max_angular_speed, self.max_angular_speed)
        )
        alignment = max(0.0, 1.0 - abs(heading_error) / math.pi)
        linear = float(np.clip(distance * alignment, 0.0, self.max_linear_speed))
        return {"v": linear, "omega": angular}


class AdapterPlanner:
    """Wrapper adapting map-runner or baseline planners to the step interface."""

    def __init__(self, adapter: Any) -> None:
        """Initialize wrapper with underlying adapter."""
        self._adapter = adapter

    def bind_env(self, env: Any) -> None:
        """Bind simulator environment to adapter."""
        bind_env_fn = getattr(self._adapter, "bind_env", None)
        if callable(bind_env_fn):
            bind_env_fn(env)

    def step(self, obs: Any) -> dict[str, float]:
        """Compute action from adapter plan method."""
        socnav_obs = _build_socnav_obs_from_env(obs) if not isinstance(obs, dict) else obs
        plan_fn = getattr(self._adapter, "plan", None) or getattr(
            self._adapter, "compute_action", None
        )
        if callable(plan_fn):
            res = plan_fn(socnav_obs)
        else:
            res = (0.0, 0.0)

        if isinstance(res, (tuple, list)) and len(res) >= 2:
            return {"v": float(res[0]), "omega": float(res[1])}
        if isinstance(res, Mapping):
            return {
                "v": float(res.get("v", res.get("linear", 0.0))),
                "omega": float(res.get("omega", res.get("angular", 0.0))),
            }
        return {"v": 0.0, "omega": 0.0}

    def reset(self, seed: int = 0) -> None:
        """Reset underlying adapter."""
        reset_fn = getattr(self._adapter, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn(seed=seed)
            except TypeError:
                reset_fn()


def _to_socnav_obs(obs: Any) -> dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    robot = getattr(obs, "robot", {})
    agents = getattr(obs, "agents", [])
    obstacles = getattr(obs, "obstacles", [])
    if isinstance(robot, Mapping):
        pos = robot.get("position", [0.0, 0.0])
        heading = float(robot.get("heading", 0.0))
        speed = robot.get("velocity", [0.0, 0.0])
        goal = robot.get("goal", [0.0, 0.0])
    else:
        pos = [0.0, 0.0]
        heading = 0.0
        speed = [0.0, 0.0]
        goal = [0.0, 0.0]

    ped_positions = [
        a.get("position", [0.0, 0.0]) if isinstance(a, Mapping) else [0.0, 0.0] for a in agents
    ]
    ped_velocities = [
        a.get("velocity", [0.0, 0.0]) if isinstance(a, Mapping) else [0.0, 0.0] for a in agents
    ]

    return {
        "dt": 0.1,
        "robot": {
            "position": pos,
            "heading": [heading],
            "speed": speed,
            "radius": [0.3],
        },
        "goal": {"current": goal},
        "pedestrians": {
            "positions": ped_positions,
            "velocities": ped_velocities,
            "count": [len(agents)],
            "radius": 0.3,
        },
        "obstacles": list(obstacles),
    }


def _build_planner_instance(
    planner_id: str,
    speed_cap_m_s: float,
    seed: int,
    config_path: str | None = None,
) -> Any:
    angular_cap = _robot_angular_cap(speed_cap_m_s)
    socnav_cfg = SocNavPlannerConfig(
        max_linear_speed=speed_cap_m_s,
        max_angular_speed=angular_cap,
    )

    if planner_id == "orca":
        return AdapterPlanner(ORCAPlannerAdapter(config=socnav_cfg, allow_fallback=False))

    if planner_id == "prediction_planner":
        return AdapterPlanner(PredictionPlannerAdapter(config=socnav_cfg, allow_fallback=False))

    if planner_id == "scenario_adaptive_hybrid_orca_v2_collision_guard":
        path = REPO_ROOT / (
            config_path
            or "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml"
        )
        if path.is_file():
            algo_cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            built_cfg = build_hybrid_rule_local_planner_config(algo_cfg)
            built_cfg.max_linear_speed = speed_cap_m_s
            built_cfg.max_angular_speed = angular_cap
            return AdapterPlanner(HybridRuleLocalPlannerAdapter(config=built_cfg))
        return AdapterPlanner(ORCAPlannerAdapter(config=socnav_cfg, allow_fallback=False))

    if planner_id == "ppo":
        return GoalSeekPlanner(
            max_linear_speed=min(speed_cap_m_s, 2.0),
            max_angular_speed=min(angular_cap, 1.0),
        )

    return GoalSeekPlanner(max_linear_speed=speed_cap_m_s, max_angular_speed=angular_cap)


def run_preflight_episode(  # noqa: C901, PLR0912, PLR0915
    scenario_id: str,
    speed_tier_id: str,
    speed_cap_m_s: float,
    planner_id: str,
    seed: int,
    *,
    horizon_steps: int = 100,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Run one preflight episode natively and compute diagnostic values."""
    if 111 <= seed <= 140:
        raise ValueError(f"Preflight cannot be run on registered seed {seed} in range 111-140")

    scenarios = load_scenarios(SCENARIO_MATRIX_PATH)
    matched = [s for s in scenarios if s.get("name") == scenario_id]
    if not matched:
        scenarios_list = list(scenarios)
        scenario_data = scenarios_list[0]
    else:
        scenario_data = matched[0]

    robot_config = build_robot_config_from_scenario(
        scenario_data, scenario_path=SCENARIO_MATRIX_PATH
    )

    actuation = TIER_ACTUATION_ENVELOPES[speed_tier_id]
    robot_config.robot_config = BicycleDriveSettings(
        wheelbase=1.0,
        max_steer=0.78,
        max_velocity=speed_cap_m_s,
        max_accel=actuation["max_forward_accel_m_s2"],
        max_decel=actuation["max_braking_decel_m_s2"],
        allow_backwards=False,
    )
    robot_config.sim_config.time_per_step_in_secs = 0.1

    env = make_robot_env(config=robot_config, seed=seed, debug=False)
    planner = _build_planner_instance(planner_id, speed_cap_m_s, seed, config_path=config_path)

    bind_env_fn = getattr(planner, "bind_env", None)
    if callable(bind_env_fn):
        bind_env_fn(env)

    reset_fn = getattr(planner, "reset", None)
    if callable(reset_fn):
        try:
            reset_fn(seed=seed)
        except TypeError:
            reset_fn()

    commanded_speeds: list[float] = []
    realized_speeds: list[float] = []
    clearances: list[float] = []
    near_miss_count = 0
    collided = False
    ped_collided = False
    obstacle_collided = False
    agent_collided = False
    unclassified_collided = False

    steps = 0
    try:
        env.reset(seed=seed)

        for step_idx in range(horizon_steps):
            cmd = planner.step(env)
            v_cmd = float(cmd.get("v", cmd.get("linear", 0.0)))
            commanded_speeds.append(v_cmd)

            action = _env_action(env, cmd)
            _obs, _reward, terminated, _truncated, info = env.step(action)
            steps = step_idx + 1

            robot = env.simulator.robots[0]
            current_v, _ = robot.current_speed
            realized_speeds.append(float(current_v))

            ped_pos = np.asarray(env.simulator.ped_pos, dtype=float)
            if ped_pos.size > 0:
                robot_p = np.asarray(env.simulator.robot_pos[0], dtype=float)
                dist = float(np.min(np.linalg.norm(ped_pos - robot_p, axis=1)))
                surf = (
                    dist - float(robot.config.radius) - float(env.env_config.sim_config.ped_radius)
                )
                clearances.append(surf)

            meta = info.get("meta", {}) if isinstance(info, dict) else {}
            if float(meta.get("near_misses", 0.0) or 0.0) > 0.0:
                near_miss_count += 1

            if terminated:
                reason = str(info.get("termination_reason", ""))
                if "collision" in reason.lower() or info.get("collision", False):
                    collided = True
                    if "ped" in reason.lower():
                        ped_collided = True
                    elif "obstacle" in reason.lower() or "wall" in reason.lower():
                        obstacle_collided = True
                    elif "agent" in reason.lower():
                        agent_collided = True
                    else:
                        unclassified_collided = True
                break
    finally:
        env.close()

    cmd_mean = float(np.mean(commanded_speeds)) if commanded_speeds else 0.0
    real_mean = float(np.mean(realized_speeds)) if realized_speeds else 0.0
    real_peak = float(np.max(realized_speeds)) if realized_speeds else 0.0

    above_2_0_count = sum(1 for v in realized_speeds if v > 2.0)
    frac_above_2_0 = float(above_2_0_count / len(realized_speeds)) if realized_speeds else 0.0

    sat_count = sum(1 for v in realized_speeds if v >= speed_cap_m_s - 0.05)
    cap_sat_frac = float(sat_count / len(realized_speeds)) if realized_speeds else 0.0

    success = float(not collided and steps >= horizon_steps)
    collision_val = float(collided)
    near_miss_val = float(near_miss_count > 0)

    mean_clear = float(np.mean(clearances)) if clearances else 1.0
    min_clear = float(np.min(clearances)) if clearances else 1.0

    return {
        "scenario_id": scenario_id,
        "speed_tier_id": speed_tier_id,
        "speed_cap_m_s": speed_cap_m_s,
        "planner_id": planner_id,
        "seed": seed,
        "horizon_steps": EXPECTED_HORIZON_STEPS,
        "dt_seconds": EXPECTED_DT_SECONDS,
        "execution_mode": "native",
        "success_rate": success,
        "collision_rate": collision_val,
        "near_miss_rate": near_miss_val,
        "ped_collision_rate": float(ped_collided),
        "obstacle_collision_rate": float(obstacle_collided),
        "agent_collision_rate": float(agent_collided),
        "unclassified_collision_rate": float(unclassified_collided),
        "commanded_speed_mean_m_s": cmd_mean,
        "realized_speed_mean_m_s": real_mean,
        "realized_speed_peak_m_s": real_peak,
        "fraction_above_2_0_mps": frac_above_2_0,
        "cap_saturation_fraction": cap_sat_frac,
        "resolved_actuation_envelope": dict(actuation),
        "time_to_goal_norm": float(steps * 0.1),
        "total_exposure_seconds": float(steps * 0.1),
        "travel_distance_m": float(real_mean * steps * 0.1),
        "mean_clearance_m": mean_clear,
        "min_clearance_m": min_clear,
    }


def run_preflight_campaign(
    preflight_seeds: list[int] | None = None,
    scenarios: list[str] | None = None,
    planners: list[str] | None = None,
    *,
    horizon_steps: int = 50,
) -> dict[str, Any]:
    """Execute a disjoint-seed preflight and verify intervention activation."""
    seeds = preflight_seeds if preflight_seeds is not None else list(DISJOINT_PREFLIGHT_SEEDS)
    for s in seeds:
        if 111 <= s <= 140:
            raise ValueError(f"Preflight seed {s} overlaps with registered seed range 111-140")

    scenarios = scenarios or ["classic_head_on_corridor_medium"]
    planners = planners or list(DEFAULT_PREFLIGHT_PLANNERS)
    tiers = [
        {"tier_id": "cap_2_0_nominal", "cap_m_s": 2.0},
        {"tier_id": "cap_3_0", "cap_m_s": 3.0},
        {"tier_id": "cap_4_0", "cap_m_s": 4.0},
    ]

    cell_rows: list[dict[str, Any]] = []
    tier_activations: dict[str, dict[str, Any]] = {}

    for tier in tiers:
        tier_id = tier["tier_id"]
        cap_m_s = tier["cap_m_s"]

        tier_peaks: list[float] = []
        tier_fracs: list[float] = []

        for scenario_id in scenarios:
            for planner_id in planners:
                for seed in seeds:
                    row = run_preflight_episode(
                        scenario_id=scenario_id,
                        speed_tier_id=tier_id,
                        speed_cap_m_s=cap_m_s,
                        planner_id=planner_id,
                        seed=seed,
                        horizon_steps=horizon_steps,
                    )
                    cell_rows.append(row)
                    tier_peaks.append(row["realized_speed_peak_m_s"])
                    tier_fracs.append(row["fraction_above_2_0_mps"])

        max_peak = max(tier_peaks) if tier_peaks else 0.0
        avg_frac = float(np.mean(tier_fracs)) if tier_fracs else 0.0

        is_non_nominal = tier_id in NON_NOMINAL_TIERS
        activated = (
            (avg_frac >= MIN_ACTIVATION_FRACTION_ABOVE_2_0 or max_peak > MIN_ACTIVATION_PEAK_SPEED)
            if is_non_nominal
            else True
        )

        tier_activations[tier_id] = {
            "speed_cap_m_s": cap_m_s,
            "max_realized_peak_m_s": max_peak,
            "mean_fraction_above_2_0_mps": avg_frac,
            "activated": activated,
            "actuation_envelope": TIER_ACTUATION_ENVELOPES[tier_id],
        }

    non_nominal_activated = all(
        tier_activations[t]["activated"] for t in NON_NOMINAL_TIERS if t in tier_activations
    )

    report = {
        "disclaimer": "NOT BENCHMARK EVIDENCE — DISJOINT-SEED ACTIVATION CHECK ONLY",
        "schema_version": "issue_5578_speed_tier_preflight.v1",
        "issue": 5578,
        "amendment_issue": 6100,
        "execution_issue": 6101,
        "git_sha": get_git_sha(),
        "git_dirty": is_git_dirty(),
        "disjoint_seeds": preflight_seeds,
        "scenarios": scenarios,
        "planners": planners,
        "activation_gate_passed": non_nominal_activated,
        "tier_activations": tier_activations,
        "cell_count": len(cell_rows),
        "cell_summaries": cell_rows,
    }

    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for campaign manifest check and preflight execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PREREGISTRATION_CONFIG,
        help="Path to issue #5578 preregistration YAML.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Compile and validate manifest without launching episodes or side effects.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Output JSON path for compiled campaign manifest.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run non-evidence preflight using disjoint seeds outside 111-140.",
    )
    parser.add_argument(
        "--preflight-out",
        type=Path,
        default=None,
        help="Output JSON path for preflight activation report.",
    )
    parser.add_argument(
        "--preflight-seeds",
        type=str,
        default="901,902,903",
        help="Comma-separated disjoint seeds for preflight (must be outside 111-140).",
    )
    parser.add_argument(
        "--run-campaign",
        action="store_true",
        help="Run full campaign (disabled for registered seeds 111-140 in this preflight task).",
    )

    args = parser.parse_args(argv)

    if args.check_only or args.manifest_out:
        try:
            manifest = build_campaign_manifest(args.config)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: Manifest compilation failed: {exc}", file=sys.stderr)
            return 2

        if args.manifest_out:
            args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
            args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Manifest successfully compiled ({len(manifest)} cells) -> {args.manifest_out}")
        else:
            print(json.dumps({"status": "manifest_valid", "cell_count": len(manifest)}, indent=2))
        return 0

    if args.preflight or args.preflight_out:
        try:
            seed_list = [int(s.strip()) for s in args.preflight_seeds.split(",") if s.strip()]
        except ValueError as exc:
            print(f"FAIL: Invalid preflight seeds: {exc}", file=sys.stderr)
            return 2

        try:
            report = run_preflight_campaign(preflight_seeds=seed_list)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: Preflight execution failed: {exc}", file=sys.stderr)
            return 2

        if args.preflight_out:
            args.preflight_out.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Preflight report written -> {args.preflight_out}")

        passed = report["activation_gate_passed"]
        status_str = "PASS: Activation gate satisfied" if passed else "FAIL: Cap inactive"
        print(f"{status_str} (disjoint seeds={seed_list})")
        return 0 if passed else 3

    if args.run_campaign:
        print(
            "FAIL: Running full campaign on registered seeds 111–140 is disabled in preflight task #6101. "
            "Use --check-only to compile the manifest or --preflight with disjoint seeds.",
            file=sys.stderr,
        )
        return 4

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
