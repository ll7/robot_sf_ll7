"""Planner-selected navigation target is exposed to the step trace (observational only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robot_sf.benchmark.map_runner.map_runner_episode import _planner_target_xy_from_stats
from robot_sf.planner.predictive_mppi import PredictiveMPPIAdapter, build_predictive_mppi_config
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, RiskDWAPlannerConfig


def _obs(robot: tuple[float, float], current: tuple[float, float], nxt: tuple[float, float]):
    """Build a minimal structured observation with distinct goal.current and goal.next."""
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([0.0], dtype=float),
            "speed": np.asarray([0.0], dtype=float),
            "angular_velocity": np.asarray([0.0], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(current, dtype=float),
            "next": np.asarray(nxt, dtype=float),
        },
        "pedestrians": {
            "positions": np.zeros((0, 2), dtype=float),
            "velocities": np.zeros((0, 2), dtype=float),
            "count": np.asarray([0.0], dtype=float),
            "radius": 0.25,
        },
    }


def _historical_selector(robot: np.ndarray, current: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Reproduce the pre-fix selector (#9883): prefer goal.next unless it coincides with the robot."""
    return nxt if np.linalg.norm(nxt - robot) > 1e-6 else current


def test_historical_selector_yields_goal_next_for_sentinel_next() -> None:
    """Robot (5,5), current (8,5), next (0,0): the historical selector returns goal.next."""
    robot, current, nxt = np.array([5.0, 5.0]), np.array([8.0, 5.0]), np.array([0.0, 0.0])
    assert np.array_equal(_historical_selector(robot, current, nxt), nxt)


def test_risk_dwa_records_selector_output() -> None:
    """RiskDWA diagnostics carry exactly the target its selector chose."""
    planner = RiskDWAPlannerAdapter(RiskDWAPlannerConfig(goal_tolerance=0.3))
    assert planner.diagnostics()["planner_target_xy"] is None
    obs = _obs((5.0, 5.0), (5.1, 5.0), (5.1, 5.0))
    selected = planner._extract_robot_goal_ped(obs)[2]
    planner.plan(obs)
    assert planner.diagnostics()["planner_target_xy"] == [float(selected[0]), float(selected[1])]
    assert planner.last_target_xy == (float(selected[0]), float(selected[1]))


def test_risk_dwa_target_matches_selector_for_sentinel_next() -> None:
    """The recorded target equals the live selector output on the #9883 geometry."""
    planner = RiskDWAPlannerAdapter(RiskDWAPlannerConfig())
    obs = _obs((5.0, 5.0), (8.0, 5.0), (0.0, 0.0))
    selected = planner._extract_robot_goal_ped(obs)[2]
    planner.plan(obs)
    assert planner.diagnostics()["planner_target_xy"] == [float(selected[0]), float(selected[1])]


def test_predictive_mppi_records_selector_output() -> None:
    """Predictive MPPI diagnostics carry exactly the target its selector chose."""
    planner = PredictiveMPPIAdapter(
        build_predictive_mppi_config({"goal_tolerance": 0.3}), allow_fallback=True
    )
    assert planner.diagnostics()["planner_target_xy"] is None
    obs = _obs((5.0, 5.0), (5.1, 5.0), (5.1, 5.0))
    selected = planner._extract_state(obs)[3]
    assert planner.plan(obs) == (0.0, 0.0)  # within goal tolerance: no predictor needed
    assert planner.diagnostics()["planner_target_xy"] == [float(selected[0]), float(selected[1])]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"planner_target_xy": [1.0, 2.0]}, [1.0, 2.0]),
        ({"last_decision": {"target_goal": {"kind": "next", "x": 3.0, "y": 4.0}}}, [3.0, 4.0]),
        ({"planner_target_xy": None}, None),
        ({"planner_type": "X"}, None),
        ({"planner_target_xy": [float("nan"), 1.0]}, None),
        ({"planner_target_xy": [1.0]}, None),
        (None, None),
    ],
)
def test_planner_target_extraction(payload: Any, expected: Any) -> None:
    """The step builder reads the target or reports null when the arm exposes none."""
    assert _planner_target_xy_from_stats(payload) == expected


def test_step_trace_field_present_for_risk_dwa_and_null_for_goal() -> None:
    """A real episode writes planner_target_xy per step: a pair for risk_dwa, null for goal."""
    from pathlib import Path

    import yaml

    from robot_sf.benchmark.map_runner.map_runner import _run_map_episode

    path = Path("configs/scenarios/canary_corridor.yaml")
    scenario = yaml.safe_load(path.read_text())["scenarios"][0]
    scenario["map_file"] = str(Path("maps/svg_maps/atomic_corridor_test.svg").resolve())
    found: dict[str, list[Any]] = {}
    for algo in ("risk_dwa", "goal"):
        row = _run_map_episode(
            dict(scenario),
            1001,
            horizon=4,
            dt=0.1,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo=algo,
            scenario_path=path,
            record_simulation_step_trace=True,
        )
        steps = row["algorithm_metadata"]["simulation_step_trace"]["steps"]
        assert steps
        assert all("planner_target_xy" in s["planner"] for s in steps)
        found[algo] = [s["planner"]["planner_target_xy"] for s in steps]
    assert all(t is not None and len(t) == 2 for t in found["risk_dwa"])
    assert all(t is None for t in found["goal"])
