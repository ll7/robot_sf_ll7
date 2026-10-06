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


def test_historical_selector_yields_goal_next_for_sentinel_next() -> None:
    """The real configured episode publishes the historical sentinel target.

    Geometry is independently specified: robot (5,5), current (8,5), next (0,0).
    This PR observes the existing selector; it does not correct its origin targeting.
    """
    from copy import deepcopy
    from pathlib import Path

    from robot_sf.benchmark.map_runner.map_runner import _build_policy, _run_map_episode
    from robot_sf.training.scenario_loader import load_scenarios

    path = Path("configs/scenarios/canary_corridor.yaml")
    scenario = load_scenarios(path)[0]

    def configured_policy(algo, config, **kwargs):
        policy, metadata = _build_policy(algo, config, **kwargs)

        def sentinel_observation(observation):
            observed = deepcopy(observation)
            observed.update(_obs((5.0, 5.0), (8.0, 5.0), (0.0, 0.0)))
            return policy(observed)

        sentinel_observation._planner_stats = policy._planner_stats
        return sentinel_observation, metadata

    row = _run_map_episode(
        scenario,
        1001,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="risk_dwa",
        algo_config_path="configs/algos/risk_dwa_camera_ready.yaml",
        scenario_path=path,
        record_simulation_step_trace=True,
        policy_builder=configured_policy,
    )
    steps = row["algorithm_metadata"]["simulation_step_trace"]["steps"]
    assert len(steps) == 1
    assert steps[0]["planner"]["planner_target_xy"] == [0.0, 0.0], (
        "missing or incorrect live planner target telemetry"
    )


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

    from robot_sf.benchmark.map_runner.map_runner import _run_map_episode

    path = Path("configs/scenarios/canary_corridor.yaml")
    from robot_sf.training.scenario_loader import load_scenarios

    scenario = load_scenarios(path)[0]
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


def test_guard_stats_publish_live_fallback_target_and_preserve_checkpoint() -> None:
    """The stats hook samples the current fallback target without losing provenance."""
    from types import SimpleNamespace

    from robot_sf.benchmark.map_runner.map_runner import _attach_guard_decision_stats

    policy = SimpleNamespace(
        _planner_stats=lambda: {"checkpoint_provenance": {"load_status": "loaded"}}
    )
    recovery_stats = {
        "no_admissible_command": False,
        "no_admissible_command_count": 0,
        "recovery_command_count": 0,
    }
    adapter = SimpleNamespace(
        last_fallback_target_xy=(8.0, 5.0), diagnostics=lambda: dict(recovery_stats)
    )
    decision = {"decision_label": "fallback_safe"}
    _attach_guard_decision_stats(policy, {"shield_stats": {"last_decision": decision}}, adapter)

    first = policy._planner_stats()
    assert first["checkpoint_provenance"] == {"load_status": "loaded"}
    assert first["last_decision"] == {**decision, **recovery_stats}
    assert _planner_target_xy_from_stats(first) == [8.0, 5.0]

    adapter.last_fallback_target_xy = (9.0, 6.0)
    assert _planner_target_xy_from_stats(policy._planner_stats()) == [9.0, 6.0]
    adapter.last_fallback_target_xy = None
    assert _planner_target_xy_from_stats(policy._planner_stats()) is None


def test_uncertainty_fallback_target_is_recorded_then_cleared_on_safe_step() -> None:
    """Uncertainty fallback telemetry belongs only to the step that consulted it."""
    from tests.planner.test_guarded_ppo import _FallbackAdapter, _obs
    from tests.planner.test_guarded_ppo_uncertainty_fallback import _uncertainty_guard

    fallback = _FallbackAdapter((0.05, -0.3))
    fallback.last_target_xy = (2.0, 0.0)
    guard = _uncertainty_guard(mode="fallback", fallback_adapter=fallback)

    decision = guard.choose_command_decision(
        _obs(ped_positions=[(1.0, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.4, 0.0),
    )
    assert decision.decision_label == "uncertainty_fallback_configured"
    assert decision.filtered_action == (0.05, -0.3)
    assert fallback.plan_calls == 1
    assert guard.last_fallback_target_xy == (2.0, 0.0)

    safe = guard.choose_command_decision(_obs(), (0.4, 0.0))
    assert safe.filtered_action == (0.4, 0.0)
    assert not safe.override_applied
    assert fallback.plan_calls == 1
    assert guard.last_fallback_target_xy is None
