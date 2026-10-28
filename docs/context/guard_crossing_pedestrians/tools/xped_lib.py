"""Shared helpers for the guard / crossing-pedestrian diagnostic (run from repo root).

Pedestrian measurement contract (frozen in contract.yaml):
- one fixed observation window of WINDOW_STEPS simulator steps from reset (dt 0.1 s),
  identical for robot and no-robot runs;
- a passage is a crossing of the restriction gate segment in the designed direction
  between consecutive frames; jumps longer than TELEPORT_M (route respawns) are ignored.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PACKET = Path("docs/context/guard_crossing_pedestrians")
SCENARIO_FILE = PACKET / "inputs/scenarios.yaml"
HORIZON_SCHEDULE = Path("configs/benchmarks/horizon_schedules/release_0_0_8_authored_v1.yaml")
HORIZON_SHA256 = "032bbf8ad354ea9394492659aa50188e2bc40078930975ef30cd3ba77208675e"
DT = 0.1
WINDOW_STEPS = 500  # 50 s = the authored budget of all four cells
TELEPORT_M = 1.0
PARK_FAR = (-1000.0, -1000.0)

# Gate segments: (axis, coordinate, lo, hi, direction sign).
GATES = {
    # Bottleneck corridor x 17.79..22.10, y 12.08..28.08; designed pedestrian travel +y.
    "classic_bottleneck_medium": ("y", 20.08, 17.79, 22.10, +1),
    "classic_bottleneck_high": ("y", 20.08, 17.79, 22.10, +1),
    # Doorway wall x 21..23 with opening y 12.24..15.54; designed pedestrian travel +x.
    "classic_doorway_medium": ("x", 22.0, 12.24, 15.54, +1),
    "classic_doorway_high": ("x", 22.0, 12.24, 15.54, +1),
}


def repo_scenario_path() -> Path:
    """Fake scenario path at repo root, as the camera-ready runner uses."""
    return Path.cwd().resolve() / "scoped_scenarios.json"


def load_scenarios(names: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    from robot_sf.benchmark.camera_ready._config import (  # noqa: PLC0415
        _apply_scenario_horizon_schedule,
    )

    data = yaml.safe_load(SCENARIO_FILE.read_text())["scenarios"]
    by_name = {s["name"]: s for s in data}
    chosen = []
    for n in names:
        s = deepcopy(by_name[n])
        s["seeds"] = list(seeds)
        chosen.append(s)
    # The schedule is keyed by base cell; bind it under the base name, then rename.
    renamed = []
    for s in chosen:
        t = deepcopy(s)
        t["name"] = s["metadata"]["xped_base_cell"]
        renamed.append(t)
    bound = _apply_scenario_horizon_schedule(
        renamed,
        schedule_path=HORIZON_SCHEDULE,
        expected_sha256=HORIZON_SHA256,
        protocol_version="0.0.8",
    )
    for s, b in zip(chosen, bound, strict=True):
        b["name"] = s["name"]
        assert b["simulation_config"]["max_episode_steps"] == WINDOW_STEPS, b["name"]
    return bound


def base_cell(scenario_name: str) -> str:
    return scenario_name.split("__", 1)[0]


def sim_config_digest(env: Any) -> str:
    cfg = env.simulator.config
    payload = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def park_robot(env: Any, *, remove: bool) -> None:
    """Stop the robot in place, or remove it from the scene (far outside every map)."""
    robot = env.simulator.robots[0]
    pose = robot.pose
    robot.reset_state((PARK_FAR, 0.0) if remove else (tuple(map(float, pose[0])), float(pose[1])))


def step_peds_only(env: Any) -> None:
    """Advance the simulator one step with a zero acceleration command."""
    env.simulator.step_once([(0.0, 0.0)])


def ped_xy(env: Any) -> np.ndarray:
    return np.asarray(env.simulator.ped_pos, dtype=np.float64).reshape(-1, 2).copy()


def robot_xy(env: Any) -> np.ndarray:
    return np.asarray(env.simulator.robots[0].pos, dtype=np.float64).copy()


def gate_crossings(cell: str, traj: np.ndarray) -> list[list[int]]:
    """Return per-pedestrian lists of frame indices k where k-1 -> k crosses the gate."""
    axis, coord, lo, hi, sign = GATES[cell]
    a, b = (1, 0) if axis == "y" else (0, 1)
    out: list[list[int]] = [[] for _ in range(traj.shape[1])]
    for k in range(1, traj.shape[0]):
        p0, p1 = traj[k - 1], traj[k]
        step = np.linalg.norm(p1 - p0, axis=1)
        s0 = (p0[:, a] - coord) * sign
        s1 = (p1[:, a] - coord) * sign
        hit = (s0 < 0) & (s1 >= 0) & (step <= TELEPORT_M)
        for i in np.flatnonzero(hit):
            t = s0[i] / (s0[i] - s1[i])
            lateral = p0[i, b] + t * (p1[i, b] - p0[i, b])
            if lo <= lateral <= hi:
                out[i].append(k)
    return out


def run_no_robot(scenario: dict[str, Any], seed: int) -> dict[str, Any]:
    """Reset exactly as the benchmark does, remove the robot, step pedestrians only."""
    from robot_sf.benchmark.map_runner.map_runner_env import build_env_config  # noqa: PLC0415
    from robot_sf.benchmark.map_runner.map_runner_identity import (  # noqa: PLC0415
        _scenario_with_episode_seed_defaults,
    )
    from robot_sf.gym_env.environment_factory import make_robot_env  # noqa: PLC0415

    # Same seed-derived scenario defaults (route_spawn_seed) as the benchmark runner.
    episode_scenario = _scenario_with_episode_seed_defaults(scenario, seed=int(seed))
    config = build_env_config(episode_scenario, scenario_path=repo_scenario_path())
    config.sim_config.time_per_step_in_secs = DT
    env = make_robot_env(config=config, seed=int(seed), debug=False)
    try:
        env.reset(seed=int(seed))
        frames = [ped_xy(env)]
        robot0 = robot_xy(env)
        park_robot(env, remove=True)
        for _ in range(WINDOW_STEPS):
            step_peds_only(env)
            frames.append(ped_xy(env))
        digest = sim_config_digest(env)
    finally:
        env.close()
    return {
        "ped_xy": np.stack(frames).astype(np.float32),
        "robot_xy": np.repeat(np.asarray(PARK_FAR)[None], WINDOW_STEPS + 1, 0).astype(np.float32),
        "robot_start_xy": robot0,
        "sim_config_sha256": digest,
    }
