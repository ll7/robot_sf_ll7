"""Deterministic closed-loop harness for hybrid-rule planner identity tests (issue #9726).

The harness runs a planner against synthetic pedestrian encounters with the
benchmark's geometry (1.0 m robot radius, 0.4 m pedestrian radius, 0.1 s step)
and a drive that clips speed changes to 1.0 m/s^2, like the default
``DifferentialDriveSettings``. It needs no simulator, so golden command
sequences recorded on one commit can be compared bit-for-bit on another.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.policy_search_manifest import resolve_candidate_manifest_runtime
from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)

ROBOT_RADIUS = 1.0
PED_RADIUS = 0.4
DT = 0.1
DRIVE_ACCEL = 1.0
DRIVE_DECEL = 1.0
DRIVE_MAX_SPEED = 2.0

# (name, robot start, heading, goal, pedestrian positions, pedestrian velocities)
ENCOUNTERS: dict[str, dict[str, Any]] = {
    "head_on": {
        "robot": (0.0, 0.0),
        "heading": 0.0,
        "goal": (14.0, 0.0),
        "peds": [(9.0, 0.1)],
        "vels": [(-0.65, 0.0)],
    },
    "crossing": {
        "robot": (0.0, 0.0),
        "heading": 0.0,
        "goal": (14.0, 0.0),
        "peds": [(6.0, -4.0), (7.5, 4.5)],
        "vels": [(0.0, 0.7), (0.0, -0.6)],
    },
    "doorway_like": {
        "robot": (0.0, 0.0),
        "heading": 0.0,
        "goal": (12.0, 0.0),
        "peds": [(5.0, 0.3), (6.5, -1.2), (8.0, 1.0)],
        "vels": [(-0.6, -0.05), (-0.5, 0.2), (0.0, 0.0)],
    },
}


def repo_root() -> Path:
    """Return the repository root that contains this test tree."""
    return Path(__file__).resolve().parents[2]


def load_planner_config(config_path: str, scenario_name: str) -> dict[str, Any]:
    """Resolve an algo or policy-search candidate config like the map runner does.

    Returns:
        dict[str, Any]: Flattened planner config for ``scenario_name``.
    """
    root = repo_root()

    def _load(path: object) -> dict[str, Any]:
        if not path:
            return {}
        with (root / str(path)).open(encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    manifest = _load(config_path)
    _algo, effective = resolve_candidate_manifest_runtime(
        default_algo="hybrid_rule_local_planner",
        manifest=manifest,
        scenario={"name": scenario_name},
        load_config=_load,
    )
    return effective


def _observation(
    robot: np.ndarray,
    heading: float,
    speed: float,
    goal: np.ndarray,
    peds: np.ndarray,
    vels: np.ndarray,
) -> dict[str, Any]:
    # SocNav observations carry world-frame positions but robot-ego-frame
    # pedestrian velocities; the planner rotates them back to the world frame.
    cos_h, sin_h = float(np.cos(heading)), float(np.sin(heading))
    world = np.asarray(vels, dtype=float).reshape(-1, 2)
    ego = np.column_stack(
        (cos_h * world[:, 0] + sin_h * world[:, 1], -sin_h * world[:, 0] + cos_h * world[:, 1])
    )
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "radius": np.asarray([ROBOT_RADIUS], dtype=float),
        },
        "goal": {"current": np.asarray(goal, dtype=float), "next": np.asarray(goal, dtype=float)},
        "pedestrians": {
            "positions": np.asarray(peds, dtype=float),
            "velocities": ego,
            "count": np.asarray([len(peds)], dtype=float),
            "radius": PED_RADIUS,
        },
        "sim": {"timestep": DT},
    }


def run_closed_loop(
    config: dict[str, Any],
    encounter: str,
    *,
    steps: int = 60,
) -> dict[str, Any]:
    """Run one planner through an encounter with a speed-limited drive.

    Returns:
        dict[str, Any]: Per-step commands and sources, minimum surface clearance
        with the robot's speed at that moment, and whether contact occurred.
    """
    spec = ENCOUNTERS[encounter]
    planner = HybridRuleLocalPlannerAdapter(build_hybrid_rule_local_planner_config(config))
    planner.reset(seed=0)
    robot = np.asarray(spec["robot"], dtype=float)
    heading = float(spec["heading"])
    goal = np.asarray(spec["goal"], dtype=float)
    peds = np.asarray(spec["peds"], dtype=float)
    vels = np.asarray(spec["vels"], dtype=float)
    speed = 0.0
    commands: list[list[float]] = []
    sources: list[str] = []
    min_clearance = float("inf")
    speed_at_min_clearance = 0.0
    contact_speeds: list[float] = []
    for _ in range(steps):
        obs = _observation(robot, heading, speed, goal, peds, vels)
        linear, angular = planner.plan(obs)
        decision = planner.last_decision() or {}
        commands.append([float(linear), float(angular)])
        sources.append(str(decision.get("selected_source")))
        delta = float(np.clip(float(linear) - speed, -DRIVE_DECEL * DT, DRIVE_ACCEL * DT))
        speed = float(np.clip(speed + delta, 0.0, DRIVE_MAX_SPEED))
        heading = float(heading + float(np.clip(angular, -1.0, 1.0)) * DT)
        robot = robot + speed * DT * np.array([np.cos(heading), np.sin(heading)])
        peds = peds + vels * DT
        clearances = np.linalg.norm(peds - robot[None, :], axis=1) - ROBOT_RADIUS - PED_RADIUS
        step_min = float(np.min(clearances))
        if step_min < min_clearance:
            min_clearance = step_min
            speed_at_min_clearance = speed
        if step_min < 0.0:
            contact_speeds.append(speed)
    return {
        "commands": commands,
        "sources": sources,
        "min_clearance": min_clearance,
        "speed_at_min_clearance": speed_at_min_clearance,
        "moving_contact": any(value > 0.05 for value in contact_speeds),
        "final_position": [float(robot[0]), float(robot[1])],
    }
