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
# The v1 matrix keeps the original station-platform map; the code-level fixes alone
# (respawn exclusion, reset relocation, padded robot start) must also hold there.
MATRIX_V1 = REPO_ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml"

# Dev-only probe: b8dccb9c9^ (9c3452fac) versus f6d5ce86, with a hard
# 1001 <= seed <= 1030 assertion before construction. Pre-fix has no v2 matrix;
# its v1 is the closest predecessor, including the original station SVG. Both
# v1 and the actual successor v2 were measured at head, without changing maps.
# Reset robot-ped surface clearance in metres; first pre-fix collision is step 1:
# scenario / seed             pre-v1  collision    head-v1  head-v2  steps
# circular_crossing / 1001     -0.494  pedestrian      0.100    0.100      1
# cross_trap_high / 1020       -0.810  pedestrian      0.100    0.100      1
# head_on_corridor_low / 1001  23.050  obstacle       22.363   22.363      1
# station_medium / 1006       -0.261  pedestrian      0.100    4.070     20
# station_medium / 1024        0.673  pedestrian      0.673    3.490     20
# station_medium / 1001        6.029  pedestrian      6.029    6.029     20
# station_medium / 1021        5.938  pedestrian      5.938    5.743     20
# Every head cell has overlap=False, positive wall clearance, no step collision
# and no respawn-overlap events. Station cells survive the full respawn window.
# (matrix, scenario, seed, zero-action steps to survive).
CELLS = [
    ("v2", "francis2023_circular_crossing", 1001, 1),
    ("v2", "classic_cross_trap_high", 1020, 1),
    ("v2", "classic_head_on_corridor_low", 1001, 1),
    ("v2", "classic_station_platform_medium", 1006, 20),
    ("v2", "classic_station_platform_medium", 1024, 20),
    ("v2", "classic_station_platform_medium", 1001, 20),
    ("v2", "classic_station_platform_medium", 1021, 20),
    ("v1", "classic_head_on_corridor_low", 1001, 1),
    ("v1", "classic_station_platform_medium", 1006, 20),
    ("v1", "classic_station_platform_medium", 1024, 20),
    ("v1", "classic_station_platform_medium", 1001, 20),
    ("v1", "classic_station_platform_medium", 1021, 20),
]


@pytest.fixture(scope="module")
def scenarios() -> dict[str, dict[str, dict]]:
    """Load both release scenario matrices once."""
    return {
        label: {str(row["name"]): dict(row) for row in load_scenarios(path)}
        for label, path in (("v2", MATRIX), ("v1", MATRIX_V1))
    }


def _collided(info: dict) -> bool:
    meta = info.get("meta", info)
    return bool(
        meta.get("is_pedestrian_collision")
        or meta.get("is_obstacle_collision")
        or meta.get("is_robot_collision")
    )


@pytest.mark.parametrize(("matrix", "name", "seed", "steps"), CELLS)
def test_formerly_defective_cell_starts_clear(
    scenarios, matrix: str, name: str, seed: int, steps: int
) -> None:
    """The reset is clear of pedestrians and walls, and zero-action steps do not collide."""
    assert 1001 <= seed <= 1030
    scenario = _scenario_with_episode_seed_defaults(scenarios[matrix][name], seed=seed)
    path = MATRIX if matrix == "v2" else MATRIX_V1
    env = make_robot_env(
        config=build_env_config(scenario, scenario_path=path), seed=seed, debug=False
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
    reason="full development matrix reset preflight; set ROBOT_SF_SPAWN_MATRIX_PREFLIGHT=1",
)
def test_development_matrix_reset_preflight_has_no_overlap() -> None:
    """Every release scenario x dev seeds 1001-1030 resets without a spawn overlap."""
    workers = min(2, int(os.environ.get("ROBOT_SF_SPAWN_MATRIX_WORKERS", "1")))
    report = run_preflight(
        Namespace(
            matrix=MATRIX,
            seeds="1001-1030",
            scenario=[],
            workers=workers,
            step_zero=True,
            dump_spawns=False,
        )
    )
    assert report["cell_count"] == report["scenario_count"] * 30
    assert report["overlap_count"] == 0, report["overlaps"]
    assert report["step1_collision_count"] == 0
