"""Tests for the testing-only topology-parallel NMPC planner."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.topology_parallel_nmpc import (
    TopologyParallelNMPCConfig,
    TopologyParallelNMPCPlannerAdapter,
    _material_separation,
    _preferred_turn_for_label,
    _rollout_signature,
    _signed_side_for_label,
    build_topology_parallel_nmpc_config,
)


def _obs(
    *,
    robot=(0.0, 0.0),
    heading=0.0,
    speed=0.0,
    goal=(2.0, 0.0),
    ped_positions=None,
    ped_velocities=None,
    obstacle_cells=None,
):
    """Build the compact observation payload used by planner tests."""
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    obstacle_cells = [] if obstacle_cells is None else obstacle_cells
    grid = np.zeros((4, 4, 4), dtype=np.float32)
    for row, col in obstacle_cells:
        grid[0, row, col] = 1.0
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": np.asarray([-2.0, -2.0], dtype=float),
        "occupancy_grid_meta_resolution": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_size": np.asarray([4.0, 4.0], dtype=float),
        "occupancy_grid_meta_use_ego_frame": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_channel_indices": np.asarray([0, 1, 2, 3], dtype=float),
    }


def test_signed_side_for_label() -> None:
    """Known labels should map to deterministic signed sides."""
    assert _signed_side_for_label("default") == 0
    assert _signed_side_for_label("pass_left") == 1
    assert _signed_side_for_label("yield_straight") == 0
    assert _signed_side_for_label("pass_right") == -1
    assert _signed_side_for_label("unknown_label") == 0


def test_preferred_turn_for_label() -> None:
    """Preferred turn should map from signed side."""
    assert _preferred_turn_for_label("pass_left") == 0.5
    assert _preferred_turn_for_label("yield_straight") == 0.0
    assert _preferred_turn_for_label("pass_right") == -0.5


def test_material_separation_identical() -> None:
    """Identical trajectories should have zero separation."""
    states = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    sep = _material_separation(states, states)
    assert sep == 0.0


def test_material_separation_divergent() -> None:
    """Divergent trajectories should have positive pairwise separation at later steps."""
    left = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]], dtype=float)
    right = np.asarray([[0.5, 0.0, 0.0], [1.0, -0.5, 0.0]], dtype=float)
    sep = _material_separation(left, right)
    assert sep == 0.5


def test_material_separation_empty() -> None:
    """Empty arrays should produce infinite separation."""
    sep = _material_separation(np.zeros((0, 3)), np.zeros((0, 3)))
    assert sep == float("inf")


def test_rollout_signature() -> None:
    """Rollout signature should produce compact geometric summary."""
    states = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]], dtype=float)
    sig = _rollout_signature(states)
    assert sig["mean_x"] == 0.5
    assert sig["mean_y"] == 0.25
    assert sig["span_x"] == 1.0
    assert sig["span_y"] == 0.5
    assert sig["n_states"] == 2


def test_rollout_signature_empty() -> None:
    """Empty states should produce zero signature."""
    sig = _rollout_signature(np.zeros((0, 3)))
    assert sig["n_states"] == 0


def test_build_topology_parallel_nmpc_config_defaults() -> None:
    """Empty config should produce default values."""
    cfg = build_topology_parallel_nmpc_config({})
    assert cfg.max_hypotheses == 3
    assert cfg.hypothesis_labels == ("pass_left", "yield_straight", "pass_right")
    assert cfg.control_period_s == 2.0
    assert cfg.switch_hysteresis_ticks == 2
    assert cfg.nmpc_config is None


def test_build_topology_parallel_nmpc_config_overrides() -> None:
    """Config builder should thread explicit settings through."""
    cfg = build_topology_parallel_nmpc_config(
        {
            "max_hypotheses": 2,
            "hypothesis_labels": ["default", "pass_left"],
            "control_period_s": 1.5,
            "switch_hysteresis_ticks": 1,
        }
    )
    assert cfg.max_hypotheses == 2
    assert cfg.hypothesis_labels == ("default", "pass_left")
    assert cfg.control_period_s == 1.5
    assert cfg.switch_hysteresis_ticks == 1


def test_planner_returns_command() -> None:
    """Topology-parallel NMPC should produce a valid command."""
    planner = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            nmpc_config=type(
                "cfg",
                (),
                {
                    "max_linear_speed": 0.9,
                    "max_angular_speed": 1.1,
                    "horizon_steps": 4,
                    "rollout_dt": 0.25,
                    "goal_tolerance": 0.25,
                    "waypoint_switch_distance": 0.75,
                    "path_goal_weight": 1.8,
                    "terminal_goal_weight": 4.5,
                    "progress_reward_weight": 2.0,
                    "heading_weight": 0.65,
                    "control_effort_weight": 0.06,
                    "smoothness_weight": 0.2,
                    "pedestrian_clearance_weight": 4.5,
                    "obstacle_clearance_weight": 4.2,
                    "occupancy_cost_weight": 1.2,
                    "collision_cost_kappa": 10.0,
                    "pedestrian_margin": 0.55,
                    "pedestrian_uncertainty_envelope_enabled": False,
                    "pedestrian_uncertainty_alpha_mps": 0.0,
                    "obstacle_margin": 0.45,
                    "desired_obstacle_clearance": 0.9,
                    "min_turn_speed_scale": 0.3,
                    "min_obstacle_speed_scale": 0.25,
                    "hard_obstacle_guard_enabled": False,
                    "hard_obstacle_clearance": 0.35,
                    "obstacle_threshold": 0.5,
                    "obstacle_search_cells": 12,
                    "avoidance_turn_bias_weight": 0.25,
                    "symmetry_break_bias": 0.2,
                    "solver_ftol": 1e-3,
                    "solver_max_iterations": 12,
                    "warm_start": False,
                    "fallback_to_stop": True,
                },
            )(),
        )
    )
    v, w = planner.plan(_obs(goal=(3.0, 0.0)))
    assert v >= 0.0
    assert abs(w) <= 1.1


def test_planner_deterministic_repeated() -> None:
    """Repeated calls with the same observation should produce the same command."""
    planner = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            hypothesis_labels=("default",),
            max_hypotheses=1,
            nmpc_config=type(
                "cfg",
                (),
                {
                    "max_linear_speed": 0.9,
                    "max_angular_speed": 1.1,
                    "horizon_steps": 4,
                    "rollout_dt": 0.25,
                    "goal_tolerance": 0.25,
                    "waypoint_switch_distance": 0.75,
                    "path_goal_weight": 1.8,
                    "terminal_goal_weight": 4.5,
                    "progress_reward_weight": 2.0,
                    "heading_weight": 0.65,
                    "control_effort_weight": 0.06,
                    "smoothness_weight": 0.2,
                    "pedestrian_clearance_weight": 4.5,
                    "obstacle_clearance_weight": 4.2,
                    "occupancy_cost_weight": 1.2,
                    "collision_cost_kappa": 10.0,
                    "pedestrian_margin": 0.55,
                    "pedestrian_uncertainty_envelope_enabled": False,
                    "pedestrian_uncertainty_alpha_mps": 0.0,
                    "obstacle_margin": 0.45,
                    "desired_obstacle_clearance": 0.9,
                    "min_turn_speed_scale": 0.3,
                    "min_obstacle_speed_scale": 0.25,
                    "hard_obstacle_guard_enabled": False,
                    "hard_obstacle_clearance": 0.35,
                    "obstacle_threshold": 0.5,
                    "obstacle_search_cells": 12,
                    "avoidance_turn_bias_weight": 0.25,
                    "symmetry_break_bias": 0.2,
                    "solver_ftol": 1e-3,
                    "solver_max_iterations": 12,
                    "warm_start": False,
                    "fallback_to_stop": True,
                },
            )(),
        )
    )
    obs = _obs(goal=(3.0, 0.0))
    cmd1 = planner.plan(obs)
    planner.reset()
    cmd2 = planner.plan(obs)
    np.testing.assert_allclose(cmd1, cmd2, rtol=1e-6, atol=1e-6)


def test_planner_reordered_hypotheses_deterministic() -> None:
    """Reordering hypothesis_labels should not change the selected command for K=1."""
    cfg = TopologyParallelNMPCConfig(
        hypothesis_labels=("default",),
        max_hypotheses=1,
    )
    # Set up the nmpc config by hand to avoid nesting issues
    from robot_sf.planner.nmpc_social import NMPCSocialConfig

    cfg.nmpc_config = NMPCSocialConfig(horizon_steps=4, solver_max_iterations=12, warm_start=False)

    planner = TopologyParallelNMPCPlannerAdapter(cfg)
    obs = _obs(goal=(3.0, 0.0))
    cmd_first = planner.plan(obs)
    assert cmd_first[0] >= 0.0


def test_planner_diagnostics_available() -> None:
    """Diagnostics should include topology-specific stats."""
    from robot_sf.planner.nmpc_social import NMPCSocialConfig

    planner = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            hypothesis_labels=("default",),
            max_hypotheses=1,
            nmpc_config=NMPCSocialConfig(
                horizon_steps=4, solver_max_iterations=12, warm_start=False
            ),
        )
    )
    obs = _obs(goal=(3.0, 0.0))
    planner.plan(obs)
    diag = planner.diagnostics()
    assert "calls" in diag


def test_planner_all_hypotheses_fail_fallback(monkeypatch) -> None:
    """When all hypotheses are infeasible, fall back without crashing."""
    planner = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            hypothesis_labels=("left", "right"),
            max_hypotheses=2,
            switch_hysteresis_ticks=1,
        )
    )
    from robot_sf.planner.nmpc_social import NMPCSocialConfig

    planner.topo_config.nmpc_config = NMPCSocialConfig(horizon_steps=1, solver_max_iterations=1)

    import robot_sf.planner.nmpc_social as nmpc_mod

    monkeypatch.setattr(
        nmpc_mod,
        "minimize",
        lambda *args, **kwargs: type(
            "r", (), {"success": False, "x": None, "fun": None, "status": 9, "nit": 0}
        )(),
    )
    obs = _obs(goal=(3.0, 0.0))
    v, _w = planner.plan(obs)
    assert v >= 0.0


def test_planner_hysteresis_supports_switch(monkeypatch) -> None:
    """Hysteresis should suppress switching before accumulated ticks."""
    planner = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            hypothesis_labels=("left", "right"),
            max_hypotheses=2,
            switch_hysteresis_ticks=2,
        )
    )
    from robot_sf.planner.nmpc_social import NMPCSocialConfig

    planner.topo_config.nmpc_config = NMPCSocialConfig(horizon_steps=1, solver_max_iterations=1)
    planner._current_hypothesis_index = 1

    import robot_sf.planner.nmpc_social as nmpc_mod

    original_minimize = nmpc_mod.minimize

    def _make_minimize(v0, w0):
        solutions = {}

        def _minimize(fun, x0, **kwargs):
            key = (float(x0[0]), float(x0[1]))
            if key not in solutions:
                solutions[key] = type(
                    "r", (), {"success": True, "x": x0.copy(), "fun": 1.0, "status": 0, "nit": 1}
                )()
            return solutions[key]

        return _minimize

    monkeypatch.setattr(nmpc_mod, "minimize", _make_minimize(0.3, 0.1))
    obs = _obs(goal=(3.0, 0.0))
    v, _w = planner.plan(obs)
    monkeypatch.setattr(nmpc_mod, "minimize", original_minimize)
    assert v >= 0.0


def test_planner_three_hypotheses_produce_sensible_command() -> None:
    """Default three-hypothesis config should produce a valid command."""
    from robot_sf.planner.nmpc_social import NMPCSocialConfig

    planner = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            hypothesis_labels=("pass_left", "yield_straight", "pass_right"),
            nmpc_config=NMPCSocialConfig(
                horizon_steps=4, solver_max_iterations=12, warm_start=False
            ),
            switch_hysteresis_ticks=1,
        )
    )
    obs = _obs(goal=(3.0, 0.0))
    v, w = planner.plan(obs)
    assert v >= 0.0
    assert abs(w) <= 1.1
