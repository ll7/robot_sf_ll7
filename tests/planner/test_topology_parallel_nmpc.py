"""Contract tests for the experimental issue #5310 topology-parallel NMPC arm."""

from __future__ import annotations

import numpy as np

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
            nmpc=NMPCSocialConfig(horizon_steps=3, solver_max_iterations=12),
            hysteresis_ticks=hysteresis_ticks,
        )
    )


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


def test_parallel_arm_optimizes_distinct_xy_t_rollouts_and_emits_trace() -> None:
    """A controlled local obstacle scene produces multiple optimized geometric rollouts."""
    planner = _planner()
    planner.plan(_obs(goal=(3.0, 0.0), obstacle_cells=[(1, 2), (2, 2)]))
    decision = planner.diagnostics()["topology_parallel_nmpc"]
    rows = decision["hypotheses"]
    assert [row["hypothesis"] for row in rows] == [
        "pass_left",
        "yield_straight",
        "pass_right",
    ]
    assert all(row["feasible"] for row in rows)
    assert len({row["initialization_signature"] for row in rows}) == 3
    assert len({row["rollout_signature"] for row in rows}) >= 2
    assert all(row["solver_iterations"] is not None for row in rows)
    assert (
        decision["selected_hypothesis"]
        == min(rows, key=lambda row: float(row["objective"]))["hypothesis"]
    )


def test_hysteresis_holds_one_tick_and_releases_on_infeasibility(monkeypatch) -> None:
    """A transient alternative is held, while an infeasible commitment is released immediately."""
    planner = _planner(hysteresis_ticks=2)
    call_count = 0
    preferred = ["pass_left", "pass_right", "pass_right", "pass_left"]

    def fake_solve(_context, initial_guess):
        nonlocal call_count
        tick = call_count // 3
        desired = preferred[tick]
        label = (
            "pass_left"
            if float(initial_guess[1]) > 0.1
            else "pass_right"
            if float(initial_guess[1]) < -0.1
            else "yield_straight"
        )
        call_count += 1
        feasible = not (tick == 3 and label == "pass_right")
        return NMPCSolveResult(
            feasible=feasible,
            solution=np.asarray(initial_guess, dtype=float),
            objective=0.0 if label == desired else 10.0,
            solver_status="mocked",
            solver_iterations=1,
            runtime_s=0.0,
            rollout_states=np.zeros((3, 3), dtype=float),
        )

    monkeypatch.setattr(planner, "_solve_context", fake_solve)
    observation = _obs(goal=(3.0, 0.0))
    planner.plan(observation)
    assert planner.diagnostics()["topology_parallel_nmpc"]["selected_hypothesis"] == "pass_left"
    planner.plan(observation)
    second = planner.diagnostics()["topology_parallel_nmpc"]
    assert second["selected_hypothesis"] == "pass_left"
    assert second["switch_reason"] == "hysteresis_hold"
    planner.plan(observation)
    assert planner.diagnostics()["topology_parallel_nmpc"]["selected_hypothesis"] == "pass_right"
    planner.plan(observation)
    fourth = planner.diagnostics()["topology_parallel_nmpc"]
    assert fourth["selected_hypothesis"] == "pass_left"
    assert fourth["switch_reason"] == "committed_infeasible"
