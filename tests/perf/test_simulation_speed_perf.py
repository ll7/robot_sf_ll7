"""Performance regression guard for simulation step throughput.

This test samples a small simulator step loop and records the throughput
so performance regressions are visible during development.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

import pytest

from robot_sf.common.artifact_paths import get_artifact_category_path
from robot_sf.common.types import Line2D, Rect
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.sim.simulator import init_simulators

if TYPE_CHECKING:
    from pathlib import Path

_SOFT_STEPS_PER_SEC = float(os.environ.get("ROBOT_SF_SIM_STEPS_SOFT", "2.0"))
_HARD_STEPS_PER_SEC = float(os.environ.get("ROBOT_SF_SIM_STEPS_HARD", "0.5"))


def _make_minimal_map() -> MapDefinition:
    """Construct a minimal map definition for performance sampling."""
    width = 10.0
    height = 10.0

    spawn_zone: Rect = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
    goal_zone: Rect = ((8.0, 8.0), (9.0, 8.0), (8.0, 9.0))
    bounds: list[Line2D] = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.2, 1.2), (8.8, 8.8)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )

    return MapDefinition(
        width=width,
        height=height,
        obstacles=[],
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[spawn_zone],
        robot_goal_zones=[goal_zone],
        bounds=bounds,
        robot_routes=[route],
        ped_goal_zones=[goal_zone],
        ped_crowded_zones=[],
        ped_routes=[route],
        single_pedestrians=[],
    )


def _write_perf_snapshot(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def test_simulation_step_throughput():
    """Track simulator step throughput and flag severe regressions."""
    config = RobotSimulationConfig()
    map_def = _make_minimal_map()

    sims = init_simulators(
        config,
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )
    sim = sims[0]

    num_steps = 10
    actions = [(0.0, 0.0)]

    start = time.perf_counter()
    for _ in range(num_steps):
        sim.step_once(actions)
    elapsed = time.perf_counter() - start

    steps_per_sec = num_steps / elapsed if elapsed > 0 else 0.0
    ms_per_step = (elapsed / num_steps) * 1000 if num_steps > 0 else 0.0

    output_dir = get_artifact_category_path("benchmarks")
    ped_pos = getattr(sim, "ped_pos", None)
    ped_count = int(ped_pos.shape[0]) if hasattr(ped_pos, "shape") else 0
    _write_perf_snapshot(
        output_dir / "simulation_speed_smoke.json",
        {
            "steps": num_steps,
            "elapsed_sec": elapsed,
            "steps_per_sec": steps_per_sec,
            "ms_per_step": ms_per_step,
            "soft_steps_per_sec": _SOFT_STEPS_PER_SEC,
            "hard_steps_per_sec": _HARD_STEPS_PER_SEC,
            "peds_have_obstacle_forces": True,
            "ped_count": ped_count,
        },
    )

    if steps_per_sec < _HARD_STEPS_PER_SEC:
        pytest.fail(
            "Simulation throughput below hard threshold: "
            f"{steps_per_sec:.2f} steps/sec < {_HARD_STEPS_PER_SEC:.2f}"
        )

    enforce = os.environ.get("ROBOT_SF_PERF_ENFORCE", "0") == "1"
    if steps_per_sec < _SOFT_STEPS_PER_SEC:
        msg = (
            "Simulation throughput below soft threshold: "
            f"{steps_per_sec:.2f} steps/sec < {_SOFT_STEPS_PER_SEC:.2f}"
        )
        if enforce:
            pytest.fail(msg)
        pytest.skip(f"{msg}; set ROBOT_SF_PERF_ENFORCE=1 to enforce")
