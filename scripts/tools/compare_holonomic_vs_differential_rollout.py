#!/usr/bin/env python3
"""Compare paired rollout inputs between differential-drive and holonomic robots.

This script runs the same scenario and seed twice:

1. with a differential-drive robot configuration,
2. with a holonomic robot configuration.

It records the exact planner input contract at each step and reports when the
two runs begin to diverge, how large that divergence is, and which fields are
changing first.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # Matplotlib is optional for plots.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency fallback.
    plt = None  # type: ignore[assignment]

import yaml

from robot_sf.benchmark.map_runner import (
    _build_env_config,
    _build_policy,
    _policy_command_to_env_action,
)
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.robot.holonomic_drive import HolonomicDriveSettings
from robot_sf.training.scenario_loader import load_scenarios, select_scenario

DEFAULT_SCENARIO_FILE = Path("configs/scenarios/classic_interactions_francis2023.yaml")
DEFAULT_ALGO = "orca"
DEFAULT_DIFF_CONFIGS: dict[str, Path | None] = {
    "ppo": Path("configs/baselines/ppo_15m_grid_socnav.yaml"),
    "orca": None,
}
DEFAULT_HOLO_CONFIGS: dict[str, Path | None] = {
    "ppo": Path("configs/baselines/ppo_15m_grid_socnav_holonomic.yaml"),
    "orca": None,
}
DEFAULT_SEED = 111
DEFAULT_MAX_STEPS = 100
DEFAULT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class StepComparison:
    """One-step paired comparison between differential and holonomic rollouts."""

    step: int
    diff_done: bool
    holo_done: bool
    policy_input_l2: float
    robot_position_l2: float
    robot_velocity_l2: float
    robot_heading_abs: float
    goal_l2: float
    pedestrians_position_l2: float
    pedestrians_velocity_l2: float
    agent_count_diff: int
    diff_policy_status: str
    holo_policy_status: str


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _scenario_label(scenario: dict[str, Any]) -> str:
    return str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "scenario"
    )


def _structured_pedestrians(obs: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize structured or legacy pedestrian data into a list of dicts."""
    pedestrians = obs.get("pedestrians")
    if isinstance(pedestrians, dict):
        positions = np.asarray(pedestrians.get("positions", []), dtype=float).reshape(-1, 2)
        velocities = np.asarray(pedestrians.get("velocities", []), dtype=float).reshape(-1, 2)
        radii = np.asarray(pedestrians.get("radius", []), dtype=float).reshape(-1)
        count_value = pedestrians.get("count", [positions.shape[0]])
        count_array = np.asarray(count_value, dtype=float).reshape(-1)
        count = int(count_array[0]) if count_array.size else positions.shape[0]
        if positions.size == 0 and velocities.size == 0 and count <= 0:
            agents = obs.get("agents", [])
            if isinstance(agents, list):
                return [agent for agent in agents if isinstance(agent, dict)]
        agent_list: list[dict[str, Any]] = []
        for index in range(min(count, positions.shape[0])):
            agent_list.append(
                {
                    "position": positions[index].tolist(),
                    "velocity": velocities[index].tolist()
                    if index < velocities.shape[0]
                    else [0.0, 0.0],
                    "radius": [float(radii[index]) if index < radii.shape[0] else 0.3],
                }
            )
        return agent_list
    agents = obs.get("agents", [])
    if isinstance(agents, list):
        return [agent for agent in agents if isinstance(agent, dict)]
    return []


def _structured_robot_goal(obs: dict[str, Any]) -> np.ndarray:
    """Return the robot goal current position from structured or legacy observations."""
    goal = obs.get("goal", {})
    if isinstance(goal, dict) and "current" in goal:
        return np.asarray(goal.get("current", [0.0, 0.0]), dtype=float).reshape(-1)
    robot = obs.get("robot", {}) if isinstance(obs.get("robot"), dict) else {}
    return np.asarray(robot.get("goal", [0.0, 0.0]), dtype=float).reshape(-1)


def _robot_config_summary(config: Any) -> dict[str, Any]:
    robot_cfg = getattr(config, "robot_config", None)
    if robot_cfg is None:
        return {"robot_config": None}
    payload = {}
    for key in ("radius", "max_linear_speed", "max_speed", "max_angular_speed", "command_mode"):
        if hasattr(robot_cfg, key):
            payload[key] = getattr(robot_cfg, key)
    payload["robot_type"] = robot_cfg.__class__.__name__
    return payload


def _aligned_vectors(diff_vec: np.ndarray, holo_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad the shorter vector so comparisons stay defined when shapes drift."""
    if diff_vec.shape == holo_vec.shape:
        return diff_vec, holo_vec
    max_len = max(int(diff_vec.shape[0]), int(holo_vec.shape[0]))
    aligned_diff = np.zeros(max_len, dtype=float)
    aligned_holo = np.zeros(max_len, dtype=float)
    aligned_diff[: diff_vec.shape[0]] = diff_vec
    aligned_holo[: holo_vec.shape[0]] = holo_vec
    return aligned_diff, aligned_holo


def _override_robot_config(
    base_config: Any,
    *,
    kinematics: str,
) -> Any:
    config = deepcopy(base_config)
    robot_cfg = getattr(config, "robot_config", None)
    radius = float(getattr(robot_cfg, "radius", 1.0) or 1.0)
    max_linear_speed = float(
        getattr(robot_cfg, "max_linear_speed", getattr(robot_cfg, "max_speed", 2.0)) or 2.0
    )
    max_angular_speed = float(getattr(robot_cfg, "max_angular_speed", 1.0) or 1.0)
    if kinematics == "differential_drive":
        allow_backwards = bool(getattr(robot_cfg, "allow_backwards", False))
        config.robot_config = DifferentialDriveSettings(
            radius=radius,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
            allow_backwards=allow_backwards,
        )
        return config
    if kinematics == "holonomic":
        config.robot_config = HolonomicDriveSettings(
            radius=radius,
            max_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
            command_mode="vx_vy",
        )
        return config
    raise ValueError(f"Unsupported kinematics: {kinematics}")


def _extract_agent_contract(
    obs: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    robot = obs.get("robot", {}) if isinstance(obs.get("robot"), dict) else {}
    agents = _structured_pedestrians(obs)
    robot_pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=float).reshape(-1)
    robot_vel = np.asarray(robot.get("velocity", [0.0, 0.0]), dtype=float).reshape(-1)
    return agents, robot_pos, robot_vel


def _flatten_ppo_input(obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Flatten the exact PPO input contract into a numeric vector and components."""
    agents, robot_pos, robot_vel = _extract_agent_contract(obs)
    robot = obs.get("robot", {}) if isinstance(obs.get("robot"), dict) else {}
    robot_goal = _structured_robot_goal(obs)
    robot_radius = float(np.asarray(robot.get("radius", [0.0]), dtype=float).reshape(-1)[0])

    values: list[float] = []
    values.extend([float(obs.get("dt", 0.0))])
    values.extend(float(x) for x in robot_pos[:2])
    values.extend(float(x) for x in robot_vel[:2])
    values.extend(float(x) for x in robot_goal[:2])
    values.append(robot_radius)
    values.append(float(len(agents)))

    agent_positions: list[float] = []
    agent_velocities: list[float] = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        pos = np.asarray(agent.get("position", [0.0, 0.0]), dtype=float).reshape(-1)
        vel = np.asarray(agent.get("velocity", [0.0, 0.0]), dtype=float).reshape(-1)
        radius = float(np.asarray(agent.get("radius", [0.0]), dtype=float).reshape(-1)[0])
        values.extend(float(x) for x in pos[:2])
        values.extend(float(x) for x in vel[:2])
        values.append(radius)
        agent_positions.extend(float(x) for x in pos[:2])
        agent_velocities.extend(float(x) for x in vel[:2])

    components: dict[str, np.ndarray | float] = {
        "robot_position": robot_pos[:2],
        "robot_velocity": robot_vel[:2],
        "goal": robot_goal[:2],
        "agent_positions": np.asarray(agent_positions, dtype=float),
        "agent_velocities": np.asarray(agent_velocities, dtype=float),
        "agent_count": float(len(agents)),
    }
    return np.asarray(values, dtype=float), components


def _flatten_orca_input(obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Flatten the ORCA/Social-Navigation-PyEnvs input contract into a vector."""
    agents, robot_pos, robot_vel = _extract_agent_contract(obs)
    robot = obs.get("robot", {}) if isinstance(obs.get("robot"), dict) else {}
    robot_goal = _structured_robot_goal(obs)
    robot_heading = float(np.asarray(robot.get("heading", [0.0]), dtype=float).reshape(-1)[0])
    robot_radius = float(np.asarray(robot.get("radius", [0.0]), dtype=float).reshape(-1)[0])

    values: list[float] = []
    values.extend([float(obs.get("dt", 0.0))])
    values.extend(float(x) for x in robot_pos[:2])
    values.extend(float(x) for x in robot_vel[:2])
    values.extend(float(x) for x in robot_goal[:2])
    values.append(robot_radius)
    values.append(robot_heading)
    values.append(float(len(agents)))

    agent_positions: list[float] = []
    agent_velocities: list[float] = []
    for agent in agents:
        pos = np.asarray(agent.get("position", [0.0, 0.0]), dtype=float).reshape(-1)
        vel = np.asarray(agent.get("velocity", [0.0, 0.0]), dtype=float).reshape(-1)
        radius = float(np.asarray(agent.get("radius", [0.0]), dtype=float).reshape(-1)[0])
        values.extend(float(x) for x in pos[:2])
        values.extend(float(x) for x in vel[:2])
        values.append(radius)
        agent_positions.extend(float(x) for x in pos[:2])
        agent_velocities.extend(float(x) for x in vel[:2])

    components: dict[str, np.ndarray | float] = {
        "robot_position": robot_pos[:2],
        "robot_velocity": robot_vel[:2],
        "goal": robot_goal[:2],
        "agent_positions": np.asarray(agent_positions, dtype=float),
        "agent_velocities": np.asarray(agent_velocities, dtype=float),
        "agent_count": float(len(agents)),
    }
    return np.asarray(values, dtype=float), components


def _compare_policy_inputs(
    diff_obs: dict[str, Any],
    holo_obs: dict[str, Any],
    *,
    algo: str,
) -> tuple[float, dict[str, float]]:
    if str(algo).strip().lower() == "orca":
        diff_vec, diff_components = _flatten_orca_input(diff_obs)
        holo_vec, holo_components = _flatten_orca_input(holo_obs)
    else:
        diff_vec, diff_components = _flatten_ppo_input(diff_obs)
        holo_vec, holo_components = _flatten_ppo_input(holo_obs)
    diff_vec, holo_vec = _aligned_vectors(diff_vec, holo_vec)
    delta = diff_vec - holo_vec
    summary = {
        "policy_input_l2": float(np.linalg.norm(delta)),
        "robot_position_l2": float(
            np.linalg.norm(
                np.asarray(diff_components["robot_position"], dtype=float)
                - np.asarray(holo_components["robot_position"], dtype=float)
            )
        ),
        "robot_velocity_l2": float(
            np.linalg.norm(
                np.asarray(diff_components["robot_velocity"], dtype=float)
                - np.asarray(holo_components["robot_velocity"], dtype=float)
            )
        ),
        "goal_l2": float(
            np.linalg.norm(
                np.asarray(diff_components["goal"], dtype=float)
                - np.asarray(holo_components["goal"], dtype=float)
            )
        ),
        "pedestrians_position_l2": float(
            np.linalg.norm(
                np.asarray(diff_components.get("agent_positions", []), dtype=float)
                - np.asarray(holo_components.get("agent_positions", []), dtype=float)
            )
            if np.asarray(diff_components.get("agent_positions", [])).size
            and np.asarray(holo_components.get("agent_positions", [])).size
            else 0.0
        ),
        "pedestrians_velocity_l2": float(
            np.linalg.norm(
                np.asarray(diff_components.get("agent_velocities", []), dtype=float)
                - np.asarray(holo_components.get("agent_velocities", []), dtype=float)
            )
            if np.asarray(diff_components.get("agent_velocities", [])).size
            and np.asarray(holo_components.get("agent_velocities", [])).size
            else 0.0
        ),
        "agent_count_diff": round(
            float(diff_components["agent_count"]) - float(holo_components["agent_count"])
        ),
        "robot_heading_abs": _raw_heading_abs(diff_obs, holo_obs),
    }
    return float(np.linalg.norm(delta)), summary


def _compare_ppo_inputs(
    diff_obs: dict[str, Any],
    holo_obs: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Backward-compatible PPO-specific helper used by existing tests."""
    return _compare_policy_inputs(diff_obs, holo_obs, algo="ppo")


def _raw_heading_abs(diff_obs: dict[str, Any], holo_obs: dict[str, Any]) -> float:
    def _heading(obs: dict[str, Any]) -> float:
        robot = obs.get("robot", {}) if isinstance(obs.get("robot"), dict) else {}
        return float(np.asarray(robot.get("heading", [0.0]), dtype=float).reshape(-1)[0])

    delta = _heading(diff_obs) - _heading(holo_obs)
    wrapped = (delta + math.pi) % (2.0 * math.pi) - math.pi
    return abs(wrapped)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algo",
        type=str,
        default=DEFAULT_ALGO,
        help="Planner family to compare, such as ppo or orca.",
    )
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=DEFAULT_SCENARIO_FILE,
        help="Scenario manifest used to choose the benchmark scenario.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default=None,
        help="Scenario name/id from the manifest. Defaults to the first scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed used for both rollouts.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of paired rollout steps to record.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Absolute tolerance used to mark the first input divergence.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "output/analysis/holonomic_rollout_diff/<algo>_<scenario>_<seed>."
        ),
    )
    return parser


def _build_env_and_policy(
    *,
    scenario: dict[str, Any],
    scenario_path: Path,
    kinematics: str,
    algo: str,
    config_path: Path | None,
) -> tuple[Any, Any, dict[str, Any], Any]:
    base_config = _build_env_config(scenario, scenario_path=scenario_path)
    config = _override_robot_config(base_config, kinematics=kinematics)
    env = make_robot_env(config=config, seed=0, debug=False)
    algo_config = _load_yaml(config_path) if config_path is not None else {}
    policy, meta = _build_policy(
        algo,
        algo_config,
        robot_kinematics=kinematics,
    )
    return env, policy, meta, config


def run_pairwise_rollout(
    *,
    algo: str,
    scenario_file: Path,
    scenario_id: str | None,
    seed: int,
    max_steps: int,
    tolerance: float,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a paired rollout comparison and persist the markdown summary."""
    scenarios = load_scenarios(scenario_file, base_dir=scenario_file)
    scenario = dict(select_scenario(scenarios, scenario_id))
    scenario_name = _scenario_label(scenario)
    algo_key = str(algo).strip().lower()

    diff_env, diff_policy, diff_meta, diff_config = _build_env_and_policy(
        scenario=scenario,
        scenario_path=scenario_file,
        kinematics="differential_drive",
        algo=algo_key,
        config_path=DEFAULT_DIFF_CONFIGS.get(algo_key),
    )
    holo_env, holo_policy, holo_meta, holo_config = _build_env_and_policy(
        scenario=scenario,
        scenario_path=scenario_file,
        kinematics="holonomic",
        algo=algo_key,
        config_path=DEFAULT_HOLO_CONFIGS.get(algo_key),
    )

    diff_obs, _ = diff_env.reset(seed=seed)
    holo_obs, _ = holo_env.reset(seed=seed)

    rows: list[StepComparison] = []
    first_divergence_step: int | None = None
    first_divergence_payload: dict[str, Any] | None = None
    max_policy_input_l2 = 0.0
    max_heading_abs = 0.0

    for step in range(max_steps + 1):
        policy_input_l2, field_summary = _compare_policy_inputs(diff_obs, holo_obs, algo=algo_key)
        heading_abs = _raw_heading_abs(diff_obs, holo_obs)
        max_policy_input_l2 = max(max_policy_input_l2, policy_input_l2)
        max_heading_abs = max(max_heading_abs, heading_abs)

        if first_divergence_step is None and (
            policy_input_l2 > tolerance or heading_abs > tolerance
        ):
            first_divergence_step = step
            first_divergence_payload = {
                "step": step,
                "policy_input_l2": policy_input_l2,
                "heading_abs": heading_abs,
                "field_summary": field_summary,
            }

        rows.append(
            StepComparison(
                step=step,
                diff_done=False,
                holo_done=False,
                policy_input_l2=policy_input_l2,
                robot_position_l2=field_summary["robot_position_l2"],
                robot_velocity_l2=field_summary["robot_velocity_l2"],
                robot_heading_abs=heading_abs,
                goal_l2=field_summary["goal_l2"],
                pedestrians_position_l2=field_summary["pedestrians_position_l2"],
                pedestrians_velocity_l2=field_summary["pedestrians_velocity_l2"],
                agent_count_diff=field_summary["agent_count_diff"],
                diff_policy_status=str((diff_meta or {}).get("status", "unknown")),
                holo_policy_status=str((holo_meta or {}).get("status", "unknown")),
            )
        )

        if step >= max_steps:
            break

        diff_command = diff_policy(diff_obs)
        holo_command = holo_policy(holo_obs)
        diff_action = _policy_command_to_env_action(
            env=diff_env,
            config=diff_config,
            command=diff_command,
        )
        holo_action = _policy_command_to_env_action(
            env=holo_env,
            config=holo_config,
            command=holo_command,
        )

        diff_obs, _, diff_terminated, diff_truncated, _ = diff_env.step(diff_action)
        holo_obs, _, holo_terminated, holo_truncated, _ = holo_env.step(holo_action)

        rows[-1] = StepComparison(
            **{
                **asdict(rows[-1]),
                "diff_done": bool(diff_terminated or diff_truncated),
                "holo_done": bool(holo_terminated or holo_truncated),
            },
        )

        if diff_terminated or diff_truncated or holo_terminated or holo_truncated:
            break

    diff_env.close()
    holo_env.close()

    payload = {
        "scenario_file": str(scenario_file),
        "scenario_id": scenario_name,
        "algo": algo_key,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "tolerance": float(tolerance),
        "diff_policy_status": str((diff_meta or {}).get("status", "unknown")),
        "holo_policy_status": str((holo_meta or {}).get("status", "unknown")),
        "diff_robot_config": _robot_config_summary(
            _override_robot_config(
                _build_env_config(scenario, scenario_path=scenario_file),
                kinematics="differential_drive",
            )
        ),
        "holo_robot_config": _robot_config_summary(
            _override_robot_config(
                _build_env_config(scenario, scenario_path=scenario_file), kinematics="holonomic"
            )
        ),
        "first_divergence_step": first_divergence_step,
        "first_divergence": first_divergence_payload,
        "max_policy_input_l2": max_policy_input_l2,
        "max_ppo_input_l2": max_policy_input_l2,
        "max_raw_heading_abs": max_heading_abs,
        "rows": [asdict(row) for row in rows],
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "rollout_diff.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "rollout_diff.md").write_text(_render_markdown(payload), encoding="utf-8")
        _write_artifacts(payload, output_dir)

    return payload


def _write_artifacts(payload: dict[str, Any], output_dir: Path) -> None:
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return
    if plt is None:
        return
    steps = np.asarray([row["step"] for row in rows], dtype=float)
    policy_input_l2 = np.asarray([row["policy_input_l2"] for row in rows], dtype=float)
    robot_position_l2 = np.asarray([row["robot_position_l2"] for row in rows], dtype=float)
    robot_velocity_l2 = np.asarray([row["robot_velocity_l2"] for row in rows], dtype=float)
    heading_abs = np.asarray([row["robot_heading_abs"] for row in rows], dtype=float)
    ped_position_l2 = np.asarray([row["pedestrians_position_l2"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160, constrained_layout=True)
    ax.plot(steps, policy_input_l2, label="policy input L2", linewidth=2.0)
    ax.plot(steps, robot_position_l2, label="robot position delta", linewidth=1.5)
    ax.plot(steps, robot_velocity_l2, label="robot velocity delta", linewidth=1.5)
    ax.plot(steps, ped_position_l2, label="pedestrian position delta", linewidth=1.5)
    ax.plot(steps, heading_abs, label="raw heading delta", linewidth=1.5)
    ax.set_title("Holonomic vs differential rollout divergence")
    ax.set_xlabel("step")
    ax.set_ylabel("absolute difference")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.savefig(output_dir / "rollout_diff_plot.png", bbox_inches="tight")
    plt.close(fig)

    diff_path = np.asarray(
        [
            (
                row["step"],
                row["robot_position_l2"],
                row["robot_velocity_l2"],
                row["robot_heading_abs"],
                row["policy_input_l2"],
            )
            for row in rows
        ],
        dtype=float,
    )
    np.savetxt(
        output_dir / "rollout_diff_series.csv",
        diff_path,
        delimiter=",",
        header="step,robot_position_l2,robot_velocity_l2,robot_heading_abs,policy_input_l2",
        comments="",
    )


def _render_markdown(payload: dict[str, Any]) -> str:
    rows = payload.get("rows", [])
    lines = [
        "# Holonomic vs Differential Rollout Diff",
        "",
        f"- scenario: `{payload['scenario_id']}`",
        f"- seed: `{payload['seed']}`",
        f"- max steps: `{payload['max_steps']}`",
        f"- divergence tolerance: `{payload['tolerance']}`",
        f"- first divergence step: `{payload['first_divergence_step']}`",
        f"- max policy input L2: `{payload['max_policy_input_l2']:.8f}`",
        f"- max raw heading delta: `{payload['max_raw_heading_abs']:.8f}`",
        "",
        "## Robot Configs",
        "",
        f"- differential: `{payload['diff_robot_config']}`",
        f"- holonomic: `{payload['holo_robot_config']}`",
        "",
        "## Policy Status",
        "",
        f"- differential PPO status: `{payload['diff_policy_status']}`",
        f"- holonomic PPO status: `{payload['holo_policy_status']}`",
        "",
    ]
    if payload.get("first_divergence"):
        div = payload["first_divergence"]
        lines.extend(
            [
                "## First Divergence",
                "",
                f"- step: `{div['step']}`",
                f"- policy input L2: `{div['policy_input_l2']:.8f}`",
                f"- raw heading delta: `{div['heading_abs']:.8f}`",
                f"- robot position delta: `{div['field_summary']['robot_position_l2']:.8f}`",
                f"- robot velocity delta: `{div['field_summary']['robot_velocity_l2']:.8f}`",
                f"- pedestrian position delta: `{div['field_summary']['pedestrians_position_l2']:.8f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Step Summary",
            "",
            "| step | policy input L2 | robot pos | robot vel | heading | ped pos | ped vel | agent count diff | diff done | holo done |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows[:20]:
        lines.append(
            "| "
            f"{row['step']} | {row['policy_input_l2']:.8f} | {row['robot_position_l2']:.8f} | "
            f"{row['robot_velocity_l2']:.8f} | {row['robot_heading_abs']:.8f} | "
            f"{row['pedestrians_position_l2']:.8f} | {row['pedestrians_velocity_l2']:.8f} | "
            f"{row['agent_count_diff']} | {row['diff_done']} | {row['holo_done']} |"
        )
    if len(rows) > 20:
        lines.append("")
        lines.append(f"_Only the first 20 rows are shown; total recorded rows: {len(rows)}._")
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entry point."""
    args = _build_parser().parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        scenario_slug = str(args.scenario_id or "first_scenario").replace("/", "_")
        algo_slug = str(args.algo).strip().lower().replace("/", "_")
        output_dir = Path("output/analysis/holonomic_rollout_diff") / (
            f"{algo_slug}_{scenario_slug}_{args.seed}"
        )
    payload = run_pairwise_rollout(
        algo=args.algo,
        scenario_file=args.scenario_file.resolve(),
        scenario_id=args.scenario_id,
        seed=int(args.seed),
        max_steps=int(args.max_steps),
        tolerance=float(args.tolerance),
        output_dir=output_dir.resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
