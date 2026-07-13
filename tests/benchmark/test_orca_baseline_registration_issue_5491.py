"""Regression tests for historical ``orca`` baseline registration (issue #5491).

The exact-repeat campaign for #5263 resolves cells with ``algo: orca`` and
routes execution through ``runner.run_episode(algo="orca")`` ->
``_load_baseline_planner``.  That baseline registry only knew the map-based
``ORCAPlannerAdapter`` (via ``map_runner``), so the executor crashed with
``Unknown algorithm 'orca'`` instead of running the cell.  These tests
prove the ``orca`` baseline is registered and that an ``algo: orca`` cell
executes (or fail-closes with an explicit error, never a silent
``Unknown algorithm`` crash).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.baselines import get_baseline, list_baselines
from robot_sf.baselines.interface import Observation
from robot_sf.baselines.orca import OrcaPlanner

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_orca_is_registered_in_baseline_registry():
    """The baseline registry must expose ``orca`` after the #5491 fix."""
    assert "orca" in list_baselines()
    cls = get_baseline("orca")
    assert cls is OrcaPlanner


def test_orca_loader_does_not_raise_unknown_algorithm():
    """``_load_baseline_planner`` resolves ``orca`` without the historical crash.

    Issue #5491 reproduced ``ValueError: Unknown algorithm 'orca'`` because the
    baseline registry lacked the ``orca`` entry.  Constructing the planner via
    the public registry must succeed.
    """
    planner_class = get_baseline("orca")
    planner = planner_class({}, seed=1)
    assert planner is not None


def _orca_observation(robot_position=(0.0, 0.0), goal=(9.7, 3.0)) -> Observation:
    """Build a canonical benchmark Observation for the ORCA planner."""
    return Observation(
        dt=0.1,
        robot={
            "position": list(robot_position),
            "velocity": [0.0, 0.0],
            "goal": list(goal),
            "radius": 0.3,
        },
        agents=[],
    )


def test_orca_planner_returns_world_velocity_action():
    """The ORCA baseline yields a world-frame ``{"vx", "vy"}`` action."""
    planner = OrcaPlanner({}, seed=1)
    action = planner.step(_orca_observation())
    assert set(action.keys()) == {"vx", "vy"}
    for key in ("vx", "vy"):
        assert isinstance(action[key], float)
    speed = float(np.hypot(action["vx"], action["vy"]))
    assert speed <= float(planner._config.max_linear_speed) + 1e-6


def test_orca_planner_metadata_reports_orca():
    """Planner metadata identifies the algorithm as ``orca`` with status ``ok``."""
    planner = OrcaPlanner({}, seed=3)
    meta = planner.get_metadata()
    assert meta["algorithm"] == "orca"
    assert meta["status"] == "ok"
    assert "config_hash" in meta


def test_orca_planner_configure_updates_adapter():
    """Runtime configuration updates reach both planner and adapter state."""
    planner = OrcaPlanner({"max_linear_speed": 1.0}, seed=3)
    planner.configure({"max_linear_speed": 0.25})
    assert planner._config.max_linear_speed == 0.25
    assert planner._adapter.config.max_linear_speed == 0.25


def test_orca_planner_adapts_canonical_mapping_and_empty_scalars():
    """Canonical mappings and malformed optional scalars remain executable."""
    planner = OrcaPlanner({}, seed=3)
    adapted = planner._to_adapter_observation(
        {
            "sim": {"timestep": np.array([])},
            "robot": {
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "heading": None,
                "radius": np.array([]),
                "goal": [9.7, 3.0],
            },
            "agents": [
                {
                    "position": [1.0, 0.0],
                    "velocity": [0.0, 0.0],
                    "radius": None,
                }
            ],
        }
    )
    assert adapted["sim"]["timestep"] == [0.1]
    assert adapted["robot"]["heading"]
    assert adapted["robot"]["radius"] == [0.3]
    assert adapted["pedestrians"]["radius"] == [0.3]
    action = planner.step(
        {
            "dt": 0.1,
            "robot": {
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "goal": [9.7, 3.0],
                "radius": 0.3,
            },
            "agents": [],
        }
    )
    assert set(action) == {"vx", "vy"}


def test_orca_planner_reset_is_seeded_and_deterministic():
    """Resetting with a seed reproduces the initial action."""
    planner = OrcaPlanner({}, seed=5)
    obs = _orca_observation()
    first = planner.step(obs)["vx"]
    planner.reset(seed=5)
    assert planner.step(obs)["vx"] == first


def test_orca_planner_adapts_scalar_robot_state():
    """A minimal Observation without heading/speed still drives the rvo2 solver."""
    planner = OrcaPlanner({}, seed=1)
    obs = Observation(
        dt=0.1,
        robot={
            "position": [1.0, 0.3],
            "velocity": [1.0, 0.0],
            "goal": [9.7, 3.0],
            "radius": 0.3,
        },
        agents=[],
    )
    action = planner.step(obs)
    assert set(action.keys()) == {"vx", "vy"}


def test_run_episode_with_orca_algo_does_not_crash():
    """The exact-repeat executor path (``run_episode(algo="orca")``) runs the cell.

    This is the headless CPU contract from issue #5491 acceptance: execute
    over an ``algo: orca`` cell either runs or fail-closes, never raising
    the historical ``Unknown algorithm 'orca'`` ValueError.
    """
    from robot_sf.benchmark.runner import run_episode

    scenario_params: dict[str, Any] = {
        "map_file": str(REPO_ROOT / "maps/svg_maps/atomic_empty_frame_test.svg"),
        "num_peds": 2,
        "robot_radius": 0.3,
        "ped_radius": 0.35,
        "seed": 7,
    }
    record = run_episode(
        scenario_params,
        7,
        algo="orca",
        horizon=20,
        dt=0.1,
        record_forces=False,
    )
    assert "outcome" in record
    assert "metrics" in record
    planner_meta = record.get("algorithm_metadata", {})
    assert planner_meta.get("algorithm") == "orca"
    assert planner_meta.get("status") == "ok"
    assert planner_meta.get("policy_step_timeout", {}).get("fallback_actions") == 0
