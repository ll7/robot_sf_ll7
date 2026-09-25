"""Reset-time spawn regressions from benchmark-data release 0.0.7 (issue #9725).

Each cell collided for every planner in the release because of a spawn defect:
A (pedestrian placed on the robot at reset), B (route-end respawn onto the robot),
or C (robot start within its radius of a wall). The cells are rebuilt exactly as
the map runner builds them and must now start clear and survive zero-action steps.
"""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner.map_runner_identity import _scenario_with_episode_seed_defaults
from robot_sf.benchmark.spawn_preflight import DEFAULT_MATRIX, run_preflight
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.sim.spawn_validation import reset_spawn_clearance
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX = REPO_ROOT / DEFAULT_MATRIX

# (scenario, seed, zero-action steps to survive). Station cells also cover the
# route-end respawn window (steps 5-16 in the release rows).
CELLS = [
    ("francis2023_circular_crossing", 111, 1),
    ("classic_cross_trap_high", 111, 1),
    ("classic_head_on_corridor_low", 116, 1),
    ("classic_station_platform_medium", 115, 20),
    ("classic_station_platform_medium", 118, 20),
    ("classic_station_platform_medium", 112, 20),
    ("classic_station_platform_medium", 131, 20),
]


@pytest.fixture(scope="module")
def scenarios() -> dict[str, dict]:
    """Load the release scenario matrix once."""
    return {str(row["name"]): dict(row) for row in load_scenarios(MATRIX)}


def _collided(info: dict) -> bool:
    meta = info.get("meta", info)
    return bool(
        meta.get("is_pedestrian_collision")
        or meta.get("is_obstacle_collision")
        or meta.get("is_robot_collision")
    )


@pytest.mark.parametrize(("name", "seed", "steps"), CELLS)
def test_formerly_defective_cell_starts_clear(scenarios, name: str, seed: int, steps: int) -> None:
    """The reset is clear of pedestrians and walls, and zero-action steps do not collide."""
    scenario = _scenario_with_episode_seed_defaults(scenarios[name], seed=seed)
    env = make_robot_env(
        config=build_env_config(scenario, scenario_path=MATRIX), seed=seed, debug=False
    )
    try:
        env.reset(seed=seed)
        clearance = reset_spawn_clearance(env.simulator)
        assert clearance["overlap"] is False, clearance
        assert clearance["robot_obstacle_min_surface_clearance_m"] > 0.0
        zero = np.zeros(env.action_space.shape, dtype=np.float32)
        for step in range(steps):
            _obs, _reward, terminated, _truncated, info = env.step(zero)
            assert not _collided(info), f"{name} seed {seed} collided at step {step + 1}"
            if terminated:
                break
        events = [
            event
            for behavior in env.simulator.peds_behaviors
            for event in getattr(behavior, "respawn_overlap_events", [])
        ]
        assert events == []
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("ROBOT_SF_SPAWN_MATRIX_PREFLIGHT") != "1",
    reason="full release matrix reset preflight; set ROBOT_SF_SPAWN_MATRIX_PREFLIGHT=1",
)
def test_release_matrix_reset_preflight_has_no_overlap() -> None:
    """Every release scenario x seed 111-140 resets without a spawn overlap."""
    workers = int(os.environ.get("ROBOT_SF_SPAWN_MATRIX_WORKERS", "4"))
    report = run_preflight(
        Namespace(
            matrix=MATRIX,
            seeds="111-140",
            scenario=[],
            workers=workers,
            step_zero=True,
            dump_spawns=False,
        )
    )
    assert report["cell_count"] == report["scenario_count"] * 30
    assert report["overlap_count"] == 0, report["overlaps"]
    assert report["step1_collision_count"] == 0
