"""Contract tests for the experimental issue #5310 topology-parallel NMPC arm."""

from __future__ import annotations

from time import perf_counter, sleep

import numpy as np
import pytest

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.benchmark.policy_builders import build_registered_adapter_policy_spec
from robot_sf.planner.nmpc_social import NMPCSocialConfig, NMPCSocialPlannerAdapter, NMPCSolveResult
from robot_sf.planner.topology_parallel_nmpc import (
    TopologyParallelNMPCConfig,
    TopologyParallelNMPCPlannerAdapter,
    build_topology_parallel_nmpc_config,
)
from tests.planner.test_nmpc_social import _obs


def _planner(*, hysteresis_ticks: int = 2) -> TopologyParallelNMPCPlannerAdapter:
    """Build a fast CPU-sized topology-parallel planner for unit tests."""
    return TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            nmpc=NMPCSocialConfig(
                horizon_steps=14,
                solver_max_iterations=12,
                max_angular_speed=1.5,
                obstacle_clearance_weight=20.0,
                occupancy_cost_weight=10.0,
                obstacle_margin=0.8,
            ),
            initialization_turn_rate=1.5,
            hysteresis_ticks=hysteresis_ticks,
        )
    )


def _conflict_observation() -> dict:
    """Build a controlled obstacle wall with an explicit obstacle-channel declaration."""
    shape = (16, 16)
    resolution = 0.25
    grid = np.zeros((4, *shape), dtype=np.float32)
    for row in range(2, 14):
        for col in range(8, 11):
            grid[0, row, col] = 1.0
    observation = _obs(goal=(3.0, 0.0))
    observation["occupancy_grid"] = grid
    observation["occupancy_grid_meta_origin"] = np.asarray([-2.0, -2.0], dtype=float)
    observation["occupancy_grid_meta_resolution"] = np.asarray([resolution], dtype=float)
    observation["occupancy_grid_meta_size"] = np.asarray([4.0, 4.0], dtype=float)
    # The fourth channel is absent, so the shared NMPC obstacle lookup uses channel zero.
    observation["occupancy_grid_meta_channel_indices"] = np.asarray(
        [0.0, 1.0, 2.0, -1.0], dtype=float
    )
    return observation


def test_topology_parallel_config_and_registration_are_explicit_opt_in() -> None:
    """The issue arm has a canonical config, registry key, and experimental readiness label."""
    cfg = build_topology_parallel_nmpc_config(
        {
            "horizon_steps": 3,
            "max_hypotheses": 3,
            "hypothesis_labels": ["pass_right", "pass_left", "yield_straight"],
        }
    )
    assert cfg.hypothesis_labels == ("pass_left", "yield_straight", "pass_right")
    spec = build_registered_adapter_policy_spec("topology_parallel_nmpc", {})
    assert spec is not None
    assert spec.adapter_name == "TopologyParallelNMPCPlannerAdapter"
    readiness = get_algorithm_readiness("topology_parallel_nmpc")
    assert readiness is not None
    assert readiness.tier == "experimental"
    assert readiness.requires_explicit_opt_in is True
    metadata = enrich_algorithm_metadata(
        algo="topology_parallel_nmpc",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    assert metadata["planner_kinematics"]["adapter_name"] == "TopologyParallelNMPCPlannerAdapter"


def test_explicit_initialization_reuses_legacy_nmpc_objective_and_solver() -> None:
    """The new solve seam matches the legacy planner's command for the same seed."""
    observation = _obs(goal=(2.5, 0.0))
    config = NMPCSocialConfig(horizon_steps=3, solver_max_iterations=12)
    explicit = TopologyParallelNMPCPlannerAdapter(TopologyParallelNMPCConfig(nmpc=config))
    context, goal_heading_error, goal_distance = explicit._build_rollout_context(observation)
    initial_guess = explicit._initial_guess(
        goal_heading_error=goal_heading_error,
        current_speed=context.current_speed,
        goal_distance=goal_distance,
        preferred_turn=0.0,
        speed_cap=context.speed_cap,
    )
    result = explicit._solve_context(context, initial_guess)
    legacy = NMPCSocialPlannerAdapter(config)
    command = legacy.plan(observation)
    expected = legacy._command_from_solution(result.solution, context=context)
    assert result.feasible is True
    assert result.objective is not None
    assert result.solver_iterations is not None
    np.testing.assert_allclose(command, expected, rtol=1e-6, atol=1e-6)


def test_k1_default_initialization_matches_legacy_plan() -> None:
    """The explicit K=1 default mode reproduces legacy NMPC at the pinned tolerance."""
    observation = _obs(goal=(2.5, 0.0))
    config = NMPCSocialConfig(horizon_steps=3, solver_max_iterations=12)
    topology = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            nmpc=config,
            hypothesis_labels=("default",),
            max_hypotheses=1,
        )
    )
    legacy = NMPCSocialPlannerAdapter(config)
    topology_command = topology.plan(observation)
    legacy_command = legacy.plan(observation)
    np.testing.assert_allclose(topology_command, legacy_command, rtol=1e-6, atol=1e-6)
    row = topology.diagnostics()["topology_parallel_nmpc"]["hypotheses"][0]
    assert row["hypothesis"] == "default"
    assert row["topology_valid"] is True


def test_parallel_arm_optimizes_distinct_xy_t_rollouts_and_emits_trace() -> None:
    """A controlled local obstacle scene produces multiple optimized geometric rollouts."""
    planner = _planner()
    planner.plan(_conflict_observation())
    decision = planner.diagnostics()["topology_parallel_nmpc"]
    rows = decision["hypotheses"]
    assert [row["hypothesis"] for row in rows] == [
        "pass_left",
        "yield_straight",
        "pass_right",
    ]
    assert all(row["feasible"] for row in rows)
    assert len({row["initialization_signature"] for row in rows}) == 3
    assert sum(row["topology_valid"] for row in rows) >= 2
    signed_sides = {
        row["rollout_diagnostics"]["side_sign"] for row in rows if row["topology_valid"]
    }
    assert signed_sides == {-1, 1}
    assert all(
        row["rollout_diagnostics"]["max_pairwise_separation_m"]
        >= row["rollout_diagnostics"]["minimum_topology_separation_m"]
        for row in rows
        if row["topology_valid"]
    )
    assert all(row["solver_iterations"] is not None for row in rows)
    eligible = [row for row in rows if row["topology_valid"]]
    assert (
        decision["selected_hypothesis"]
        == min(eligible, key=lambda row: float(row["objective"]))["hypothesis"]
    )


def test_collapsed_optimized_rollouts_fail_closed(monkeypatch) -> None:
    """Hash-only differences cannot make a collapsed topology set selectable."""
    planner = _planner()

    def collapsed_solve(_context, initial_guess, *, deadline=None):
        del deadline
        return NMPCSolveResult(
            feasible=True,
            solution=np.asarray(initial_guess, dtype=float),
            objective=1.0,
            solver_status="mocked",
            solver_iterations=1,
            runtime_s=0.0,
            rollout_states=np.asarray(
                [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=float
            ),
        )

    monkeypatch.setattr(planner, "_solve_context", collapsed_solve)
    command = planner.plan(_obs(goal=(3.0, 0.0)))
    decision = planner.diagnostics()["topology_parallel_nmpc"]
    assert command == (0.0, 0.0)
    assert decision["selected_hypothesis"] is None
    assert decision["switch_reason"] == "no_materially_distinct_hypotheses"


def test_hysteresis_holds_one_tick_and_releases_on_infeasibility(monkeypatch) -> None:
    """A transient alternative is held, while an infeasible commitment is released immediately."""
    planner = _planner(hysteresis_ticks=2)
    current_tick = 0
    preferred = ["pass_left", "pass_right", "pass_right", "pass_left"]

    def fake_solve(_context, initial_guess, *, deadline=None):
        del deadline
        desired = preferred[current_tick]
        label = (
            "pass_left"
            if float(initial_guess[1]) > 0.1
            else "pass_right"
            if float(initial_guess[1]) < -0.1
            else "yield_straight"
        )
        feasible = not (current_tick == 3 and label == "pass_right")
        side = 1.0 if label == "pass_left" else -1.0 if label == "pass_right" else 0.0
        return NMPCSolveResult(
            feasible=feasible,
            solution=np.asarray(initial_guess, dtype=float),
            objective=0.0 if label == desired else 10.0,
            solver_status="mocked",
            solver_iterations=1,
            runtime_s=0.0,
            rollout_states=np.asarray(
                [[0.1, 0.35 * side, 0.0], [0.2, 0.45 * side, 0.0], [0.3, 0.35 * side, 0.0]],
                dtype=float,
            ),
        )

    monkeypatch.setattr(planner, "_solve_context", fake_solve)
    observation = _obs(goal=(3.0, 0.0))
    planner.plan(observation)
    assert planner.diagnostics()["topology_parallel_nmpc"]["selected_hypothesis"] == "pass_left"
    current_tick = 1
    planner.plan(observation)
    second = planner.diagnostics()["topology_parallel_nmpc"]
    assert second["selected_hypothesis"] == "pass_left"
    assert second["switch_reason"] == "hysteresis_hold"
    current_tick = 2
    planner.plan(observation)
    assert planner.diagnostics()["topology_parallel_nmpc"]["selected_hypothesis"] == "pass_right"
    current_tick = 3
    planner.plan(observation)
    fourth = planner.diagnostics()["topology_parallel_nmpc"]
    assert fourth["selected_hypothesis"] == "pass_left"
    assert fourth["switch_reason"] == "committed_infeasible"


@pytest.mark.parametrize(
    "invalid",
    [
        {"max_hypotheses": 1},
        {"hypothesis_labels": ["bogus"]},
        {"max_runtime_s": float("nan")},
        {"initialization_turn_rate": float("inf")},
        {"enabled": False},
        {"arm": "nmpc_social"},
        {"control_period_s": 0.6, "max_runtime_s": 0.5},
    ],
)
def test_invalid_experimental_config_fails_closed(invalid: dict) -> None:
    """Invalid arm, topology, and predeclared runtime settings never default silently."""
    with pytest.raises(ValueError):
        build_topology_parallel_nmpc_config(invalid)


def test_objective_ties_use_canonical_order_before_hysteresis() -> None:
    """Near-equal objectives select the stable canonical label, not solver noise."""
    planner = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(hysteresis_ticks=1, objective_tolerance=1e-5)
    )
    rows = [
        {"hypothesis": "pass_left", "feasible": True, "topology_valid": True, "objective": 1.0},
        {
            "hypothesis": "yield_straight",
            "feasible": True,
            "topology_valid": True,
            "objective": 1.0 + 2e-7,
        },
        {
            "hypothesis": "pass_right",
            "feasible": True,
            "topology_valid": True,
            "objective": 1.0 + 3e-7,
        },
    ]
    selected, reason = planner._select_hypothesis(rows)
    assert selected is rows[0]
    assert reason == "initial_selection"


def test_solver_exception_stops_without_using_unverified_command(monkeypatch) -> None:
    """Unexpected solver exceptions become explicit stop diagnostics."""
    planner = _planner()

    def raise_solver(_context, _initial_guess, *, deadline=None):
        del deadline
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(planner, "_solve_context", raise_solver)
    command = planner.plan(_obs(goal=(3.0, 0.0)))
    decision = planner.diagnostics()["topology_parallel_nmpc"]
    assert command == (0.0, 0.0)
    assert decision["switch_reason"] == "no_feasible_hypothesis"
    assert {row["solver_status"] for row in decision["hypotheses"]} == {"solver_exception"}


def test_runtime_overrun_stops_without_using_late_command(monkeypatch) -> None:
    """A solve that misses the predeclared deadline cannot emit its late solution."""
    planner = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            control_period_s=0.005,
            max_runtime_s=0.005,
        )
    )

    def slow_solver(_context, initial_guess, *, deadline=None):
        del deadline
        sleep(0.02)
        return NMPCSolveResult(
            feasible=True,
            solution=np.asarray(initial_guess, dtype=float),
            objective=0.0,
            solver_status="late",
            solver_iterations=1,
            runtime_s=0.02,
            rollout_states=np.zeros((3, 3), dtype=float),
        )

    monkeypatch.setattr(planner, "_solve_context", slow_solver)
    command = planner.plan(_obs(goal=(3.0, 0.0)))
    decision = planner.diagnostics()["topology_parallel_nmpc"]
    assert command == (0.0, 0.0)
    assert decision["switch_reason"] == "runtime_budget_exceeded"


def test_adversarial_nonfinite_simulator_state_stops_before_solver() -> None:
    """A corrupted simulator pose is classified before any topology candidate is solved."""
    observation = _obs(goal=(3.0, 0.0))
    observation["robot"]["position"][0] = np.nan
    planner = _planner()
    command = planner.plan(observation)
    decision = planner.diagnostics()["topology_parallel_nmpc"]
    assert command == (0.0, 0.0)
    assert decision["hypotheses"] == []
    assert decision["switch_reason"] == "invalid_simulator_state"


def test_canonical_conflict_smoke_meets_p95_control_period() -> None:
    """The real multi-solve controlled smoke stays within its predeclared p95 period."""
    planner = _planner()
    durations = []
    for _ in range(5):
        planner.reset()
        started = perf_counter()
        planner.plan(_conflict_observation())
        durations.append(perf_counter() - started)
    assert np.percentile(durations, 95) <= planner.topology_config.control_period_s
