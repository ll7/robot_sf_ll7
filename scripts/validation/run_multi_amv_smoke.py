#!/usr/bin/env python3
"""Run a minimal multi-AMV benchmark smoke and emit inter-robot metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.multi_amv import inter_robot_metrics, multi_amv_settings_from_scenario
from robot_sf.gym_env.environment_factory import make_multi_robot_env
from robot_sf.gym_env.unified_config import MultiRobotConfig
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios


def _multi_robot_config_from_scenario(
    scenario: dict[str, Any], scenario_path: Path
) -> MultiRobotConfig:
    """Build a ``MultiRobotConfig`` from an existing benchmark scenario."""
    base = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    settings = multi_amv_settings_from_scenario(scenario)
    config_values = {
        field.name: getattr(base, field.name)
        for field in fields(MultiRobotConfig)
        if field.name != "num_robots" and hasattr(base, field.name)
    }
    return MultiRobotConfig(**config_values, num_robots=settings.num_robots)


def _goal_actions(env: Any) -> np.ndarray:
    """Return simple unicycle goal-seeking actions for every robot in the env."""
    actions: list[list[float]] = []
    for sim in env.simulators:
        for robot, goal in zip(sim.robots, sim.goal_pos, strict=False):
            pos = np.asarray(robot.pose[0], dtype=float)
            heading = float(robot.pose[1])
            goal_xy = np.asarray(goal, dtype=float)
            delta = goal_xy - pos
            distance = float(np.linalg.norm(delta))
            target_heading = float(np.arctan2(delta[1], delta[0])) if distance > 1e-9 else heading
            heading_error = ((target_heading - heading + np.pi) % (2.0 * np.pi)) - np.pi
            linear = min(0.6, distance)
            angular = float(np.clip(2.0 * heading_error, -1.0, 1.0))
            actions.append([linear, angular])
    return np.asarray(actions, dtype=np.float32)


def _robot_positions(env: Any) -> np.ndarray:
    """Collect current robot positions from all simulators."""
    positions: list[tuple[float, float]] = []
    for sim in env.simulators:
        positions.extend(tuple(map(float, pos)) for pos in sim.robot_pos)
    return np.asarray(positions, dtype=float)


def run_smoke(*, scenario_path: Path, horizon: int) -> dict[str, Any]:
    """Run the first scenario in ``scenario_path`` and return a metrics record."""
    scenario = dict(load_scenarios(scenario_path)[0])
    settings = multi_amv_settings_from_scenario(scenario)
    config = _multi_robot_config_from_scenario(scenario, scenario_path)
    config.sim_config.sim_time_in_secs = max(
        config.sim_config.time_per_step_in_secs,
        horizon * config.sim_config.time_per_step_in_secs,
    )
    env = make_multi_robot_env(num_robots=settings.num_robots, config=config, debug=False)
    positions = []
    try:
        env.reset(seed=0)
        positions.append(_robot_positions(env))
        for _ in range(horizon):
            _obs, _reward, terminated, truncated, _info = env.step(_goal_actions(env))
            positions.append(_robot_positions(env))
            if terminated or truncated:
                break
    finally:
        env.close()
    robot_positions = np.stack(positions, axis=0)
    metrics = inter_robot_metrics(
        robot_positions,
        dt=float(config.sim_config.time_per_step_in_secs),
        settings=settings,
    )
    return {
        "scenario_id": scenario.get("name") or scenario.get("scenario_id") or scenario.get("id"),
        "horizon": horizon,
        "steps_recorded": int(robot_positions.shape[0]),
        "multi_amv": {
            "num_robots": settings.num_robots,
            "near_miss_distance_m": settings.near_miss_distance_m,
            "collision_distance_m": settings.collision_distance_m,
            "deadlock_speed_mps": settings.deadlock_speed_mps,
            "deadlock_window_steps": settings.deadlock_window_steps,
        },
        "metrics": {"inter_robot": metrics},
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    """Run the multi-AMV smoke CLI."""
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1")
    record = run_smoke(scenario_path=args.scenario, horizon=args.horizon)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
