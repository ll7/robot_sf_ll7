"""Observation bridge helpers for map-runner policy adapters."""

from __future__ import annotations

from typing import Any

import numpy as np


def _default_if_none(value: Any, default: Any) -> Any:
    """Return ``default`` when a parsed observation field is explicitly null."""
    return default if value is None else value


def normalize_xy_rows(values: Any) -> np.ndarray:
    """Normalize scalar/list/ndarray payloads to an ``(N, 2)`` float array.

    Returns:
        np.ndarray: ``(N, 2)`` array, or ``(0, 2)`` when input is empty/malformed.
    """
    if values is None:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            return np.zeros((0, 2), dtype=float)
        return arr.reshape(-1, 2)
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            return arr
        if arr.shape[1] > 2:
            return arr[:, :2]
        return np.pad(arr, ((0, 0), (0, 2 - arr.shape[1])), constant_values=0.0)
    return np.zeros((0, 2), dtype=float)


def extract_ppo_pedestrians(
    pedestrians: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract count-aware pedestrian positions, velocities, and shared radius.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: Pedestrian positions, velocities, and radius.
    """
    ped_pos = normalize_xy_rows(_default_if_none(pedestrians.get("positions"), []))
    ped_count_source = _default_if_none(pedestrians.get("count"), [ped_pos.shape[0]])
    ped_count_arr = np.asarray(ped_count_source, dtype=float).reshape(-1)
    ped_count = int(ped_count_arr[0]) if ped_count_arr.size else int(ped_pos.shape[0])
    ped_count = max(0, min(ped_count, int(ped_pos.shape[0])))
    ped_pos = ped_pos[:ped_count]

    ped_vel = normalize_xy_rows(_default_if_none(pedestrians.get("velocities"), []))
    if ped_vel.shape[0] < ped_count:
        ped_vel = np.pad(
            ped_vel,
            ((0, ped_count - ped_vel.shape[0]), (0, 0)),
            constant_values=0.0,
        )
    ped_vel = ped_vel[:ped_count]

    ped_radius_source = _default_if_none(pedestrians.get("radius"), [0.35])
    ped_radius_raw = np.asarray(ped_radius_source, dtype=float).reshape(-1)
    ped_radius = float(ped_radius_raw[0]) if ped_radius_raw.size else 0.35
    return ped_pos, ped_vel, ped_radius


def extract_ppo_dt(obs: dict[str, Any]) -> float:
    """Resolve PPO dt from structured sim metadata first, then fallback fields.

    Returns:
        float: Timestep for PPO planner observations.
    """
    sim_info = obs.get("sim")
    if isinstance(sim_info, dict) and "timestep" in sim_info:
        dt_source = sim_info.get("timestep")
    else:
        dt_source = obs.get("dt", 0.1)
    dt_raw = np.asarray(0.1 if dt_source is None else dt_source, dtype=float).reshape(-1)
    return float(dt_raw[0]) if dt_raw.size else 0.1


def obs_to_ppo_format(obs: dict[str, Any]) -> dict[str, Any]:
    """Convert map-runner observations into the PPO baseline observation contract.

    Returns:
        Mapping compatible with ``robot_sf.baselines.ppo.PPOPlanner.step``.
    """
    robot = obs.get("robot", {}) if isinstance(obs.get("robot"), dict) else {}
    goal = obs.get("goal", {}) if isinstance(obs.get("goal"), dict) else {}
    pedestrians = obs.get("pedestrians", {}) if isinstance(obs.get("pedestrians"), dict) else {}

    robot_pos = np.asarray(
        _default_if_none(robot.get("position"), [0.0, 0.0]), dtype=float
    ).reshape(-1)
    robot_vel = np.asarray(
        _default_if_none(robot.get("velocity"), [0.0, 0.0]), dtype=float
    ).reshape(-1)
    if robot_vel.size < 2:
        speed = float(
            np.asarray(_default_if_none(robot.get("speed"), [0.0]), dtype=float).reshape(-1)[0]
        )
        heading = float(
            np.asarray(_default_if_none(robot.get("heading"), [0.0]), dtype=float).reshape(-1)[0]
        )
        robot_vel = np.array([speed * np.cos(heading), speed * np.sin(heading)], dtype=float)
    robot_goal = np.asarray(_default_if_none(goal.get("current"), [0.0, 0.0]), dtype=float).reshape(
        -1
    )
    robot_heading = float(
        np.asarray(_default_if_none(robot.get("heading"), [0.0]), dtype=float).reshape(-1)[0]
    )
    robot_radius = float(
        np.asarray(_default_if_none(robot.get("radius"), [0.3]), dtype=float).reshape(-1)[0]
    )

    ped_pos, ped_vel, ped_radius = extract_ppo_pedestrians(pedestrians)

    agents = []
    for idx in range(ped_pos.shape[0]):
        vel = ped_vel[idx] if idx < ped_vel.shape[0] else np.zeros(2, dtype=float)
        agents.append(
            {
                "position": [float(ped_pos[idx, 0]), float(ped_pos[idx, 1])],
                "velocity": [float(vel[0]), float(vel[1])],
                "radius": ped_radius,
            }
        )

    dt = extract_ppo_dt(obs)
    return {
        "dt": dt,
        "robot": {
            "position": [float(robot_pos[0]), float(robot_pos[1])]
            if robot_pos.size >= 2
            else [0.0, 0.0],
            "velocity": [float(robot_vel[0]), float(robot_vel[1])]
            if robot_vel.size >= 2
            else [0.0, 0.0],
            "goal": [float(robot_goal[0]), float(robot_goal[1])]
            if robot_goal.size >= 2
            else [0.0, 0.0],
            "heading": robot_heading,
            "radius": robot_radius,
        },
        "agents": agents,
        "obstacles": [],
    }


def obs_to_external_mpc_format(obs: dict[str, Any]) -> dict[str, Any]:
    """Convert map-runner observations into the external MPC wrapper contract.

    Returns:
        dict[str, Any]: Structured observation with obstacles preserved when present.

    The external MPC wrappers use the same robot/human fields as the PPO bridge,
    but they may also reason about obstacle payloads when the upstream contract
    exposes them, so preserve the raw obstacle list when available.
    """
    payload = obs_to_ppo_format(obs)
    obstacles = obs.get("obstacles")
    if isinstance(obstacles, list):
        payload["obstacles"] = obstacles
    return payload
