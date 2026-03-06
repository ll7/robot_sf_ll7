"""Tests for portfolio planner adapters."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.hybrid_portfolio import (
    HybridPortfolioAdapter,
    HybridPortfolioConfig,
    build_hybrid_portfolio_build_config,
)
from robot_sf.planner.mppi_social import MPPISocialConfig, MPPISocialPlannerAdapter
from robot_sf.planner.risk_dwa import (
    RiskDWAPlannerAdapter,
    RiskDWAPlannerConfig,
    _safe_mean,
    _wrap_angle,
    build_risk_dwa_config,
)


def _obs(
    *, robot=(0.0, 0.0), heading=0.0, goal=(2.0, 0.0), ped_positions=None, ped_velocities=None
):
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([0.2], dtype=float),
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
            "radius": 0.25,
        },
    }


def test_risk_dwa_returns_goal_directed_command() -> None:
    """Risk-DWA should output bounded commands toward goal."""
    planner = RiskDWAPlannerAdapter(RiskDWAPlannerConfig())
    v, w = planner.plan(_obs())
    assert 0.0 <= v <= planner.config.max_linear_speed
    assert abs(w) <= planner.config.max_angular_speed


def test_risk_dwa_helper_functions_and_config_defaults() -> None:
    """Helper functions and config builder should cover empty/default branches."""
    assert _wrap_angle(4.0 * np.pi) == pytest.approx(0.0)
    assert _safe_mean(np.asarray([], dtype=float)) == 0.0
    assert _safe_mean(np.asarray([np.nan], dtype=float)) == 0.0

    cfg = build_risk_dwa_config(None)
    assert isinstance(cfg, RiskDWAPlannerConfig)
    cfg = build_risk_dwa_config({"linear_candidates": "bad", "angular_candidates": "bad"})
    assert cfg.linear_candidates == RiskDWAPlannerConfig.linear_candidates
    assert cfg.angular_candidates == RiskDWAPlannerConfig.angular_candidates


def test_risk_dwa_handles_malformed_pedestrians_and_density_scaling() -> None:
    """Malformed pedestrian payloads should be sanitized and density cap should reduce speed."""
    planner = RiskDWAPlannerAdapter(
        RiskDWAPlannerConfig(
            max_linear_speed=1.2,
            near_field_distance=2.0,
            density_norm_count=2.0,
            near_field_speed_cap=0.4,
        )
    )
    robot_pos, heading, goal, ped_pos, ped_vel = planner._extract_robot_goal_ped(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [2.0, 0.0], "next": [2.0, 0.0]},
            "pedestrians": {"positions": [1.0, 2.0, 3.0], "velocities": [0.0]},
        }
    )
    assert robot_pos.tolist() == [0.0, 0.0]
    assert heading == 0.0
    assert goal.tolist() == [2.0, 0.0]
    assert ped_pos.shape == (0, 2)
    assert ped_vel.shape == (0, 2)

    dense_scale = planner._crowd_density_scale(
        np.asarray([0.0, 0.0], dtype=float),
        np.asarray([[0.5, 0.0], [0.8, 0.0], [1.5, 0.0]], dtype=float),
    )
    assert (
        dense_scale <= planner.config.near_field_speed_cap / planner.config.max_linear_speed + 1e-9
    )


def test_risk_dwa_obstacle_and_ttc_branches(monkeypatch) -> None:
    """Obstacle clearance and TTC helper branches should cover payload edge cases."""
    planner = RiskDWAPlannerAdapter(RiskDWAPlannerConfig())
    point = np.asarray([0.0, 0.0], dtype=float)
    obs = _obs()

    monkeypatch.setattr(planner, "_extract_grid_payload", lambda observation: None)
    assert planner._min_obstacle_clearance(point, obs) == float("inf")

    grid = np.zeros((1, 4, 4), dtype=float)
    meta = {"resolution": [0.5]}
    monkeypatch.setattr(planner, "_extract_grid_payload", lambda observation: (grid, meta))
    monkeypatch.setattr(planner, "_preferred_channel", lambda meta: 5)
    assert planner._min_obstacle_clearance(point, obs) == float("inf")

    monkeypatch.setattr(planner, "_preferred_channel", lambda meta: 0)
    monkeypatch.setattr(planner, "_world_to_grid", lambda point, meta, grid_shape: None)
    assert planner._min_obstacle_clearance(point, obs) == 0.0

    monkeypatch.setattr(planner, "_world_to_grid", lambda point, meta, grid_shape: (1, 1))
    grid[1 if False else 0, 1, 1] = 1.0
    assert planner._min_obstacle_clearance(point, obs) == 0.0

    grid.fill(0.0)
    assert planner._min_obstacle_clearance(point, obs) == float("inf")

    grid[0, 1, 2] = 1.0
    assert planner._min_obstacle_clearance(point, obs) > 0.0

    assert planner._ttc_proxy(
        np.asarray([0.0, 0.0], dtype=float),
        (0.5, 0.0),
        np.zeros((0, 2), dtype=float),
        np.zeros((0, 2), dtype=float),
        0.0,
    ) == float("inf")


def test_risk_dwa_stops_at_goal() -> None:
    """Risk-DWA should stop when already within goal tolerance."""
    planner = RiskDWAPlannerAdapter(RiskDWAPlannerConfig(goal_tolerance=0.3))
    v, w = planner.plan(_obs(robot=(0.0, 0.0), goal=(0.1, 0.0)))
    assert v == 0.0
    assert w == 0.0


def test_risk_dwa_progress_escape_breaks_stall() -> None:
    """Risk-DWA should inject progress command when scoring stalls in open space."""
    cfg = RiskDWAPlannerConfig(
        linear_candidates=(0.0,),
        angular_candidates=(0.0,),
        progress_escape_enabled=True,
        progress_escape_speed=0.6,
        progress_escape_distance=1.0,
        safe_distance=0.2,
    )
    planner = RiskDWAPlannerAdapter(cfg)
    v, w = planner.plan(_obs(goal=(3.0, 0.0)))
    assert v >= 0.59
    assert abs(w) <= cfg.max_angular_speed


def test_risk_dwa_progress_escape_keeps_scored_best_command(monkeypatch) -> None:
    """Progress-escape should not override a better-scored rollout command."""
    cfg = RiskDWAPlannerConfig(
        linear_candidates=(0.0, 0.2),
        angular_candidates=(0.0,),
        progress_escape_enabled=True,
        progress_escape_speed=0.8,
        progress_escape_distance=1.0,
    )
    planner = RiskDWAPlannerAdapter(cfg)
    monkeypatch.setattr(
        planner,
        "_rollout_score",
        lambda **kwargs: 10.0 if kwargs["command"] == (0.2, 0.0) else -5.0,
    )
    v, w = planner.plan(_obs(goal=(3.0, 0.0)))
    assert (v, w) == (0.2, 0.0)


def test_mppi_is_deterministic_for_fixed_seed() -> None:
    """Two planners with same seed should produce identical action on same observation."""
    cfg = MPPISocialConfig(random_seed=7, sample_count=24, iterations=2, horizon_steps=5)
    p1 = MPPISocialPlannerAdapter(cfg)
    p2 = MPPISocialPlannerAdapter(cfg)
    o = _obs(ped_positions=[(0.6, 0.2), (0.8, -0.1)], ped_velocities=[(0.0, 0.0), (0.0, 0.0)])
    a1 = p1.plan(o)
    a2 = p2.plan(o)
    assert a1 == a2


def test_mppi_progress_escape_breaks_stall() -> None:
    """MPPI should inject progress command when first action is too conservative."""
    cfg = MPPISocialConfig(
        random_seed=3,
        sample_count=12,
        iterations=1,
        horizon_steps=4,
        progress_escape_enabled=True,
        progress_escape_speed=0.8,
        progress_escape_distance=1.0,
    )
    planner = MPPISocialPlannerAdapter(cfg)
    v, w = planner.plan(_obs(goal=(3.0, 0.0)))
    assert v >= 0.79
    assert abs(w) <= cfg.max_angular_speed


class _DummyHead:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def plan(self, observation: dict) -> tuple[float, float]:
        """Return constant command and count calls."""
        _ = observation
        self.calls += 1
        return (1.0 if self.name == "risk" else 0.5, 0.0)


def test_hybrid_switches_to_orca_on_emergency_clearance() -> None:
    """Hybrid should route to ORCA head when pedestrian is inside emergency distance."""
    risk = _DummyHead("risk")
    orca = _DummyHead("orca")
    pred = _DummyHead("pred")
    mppi = _DummyHead("mppi")
    hybrid = HybridPortfolioAdapter(
        hybrid_config=HybridPortfolioConfig(emergency_clearance=0.7, caution_clearance=1.0),
        risk_dwa=risk,
        orca=orca,
        prediction=pred,
        mppi=mppi,
    )

    obs = _obs(ped_positions=[(0.3, 0.0)], ped_velocities=[(0.0, 0.0)])
    cmd = hybrid.plan(obs)
    assert cmd[0] == 0.5
    assert orca.calls == 1


def test_hybrid_prefers_prediction_mppi_and_hysteresis() -> None:
    """Hybrid head choice should cover prediction, MPPI, and hysteresis hold behavior."""
    risk = _DummyHead("risk")
    orca = _DummyHead("orca")
    pred = _DummyHead("pred")
    mppi = _DummyHead("mppi")
    hybrid = HybridPortfolioAdapter(
        hybrid_config=HybridPortfolioConfig(
            emergency_clearance=0.4,
            caution_clearance=1.0,
            dense_ped_count=2,
            hysteresis_steps=2,
        ),
        risk_dwa=risk,
        orca=orca,
        prediction=pred,
        mppi=mppi,
    )
    assert hybrid.plan(_obs(ped_positions=[(0.8, 0.0), (0.9, 0.0)]))[0] == 0.5
    assert pred.calls == 1
    # Hysteresis should keep previous head instead of immediately switching.
    assert hybrid.plan(_obs())[0] == 0.5
    # Open scene with no pedestrians and enough clearance can use MPPI once hold is released.
    assert hybrid.plan(_obs())[0] in {0.5, 1.0}


def test_hybrid_far_pedestrians_do_not_trigger_dense_head_switch() -> None:
    """Pedestrians outside near-field distance should not count toward dense switching."""
    risk = _DummyHead("risk")
    orca = _DummyHead("orca")
    pred = _DummyHead("pred")
    mppi = _DummyHead("mppi")
    hybrid = HybridPortfolioAdapter(
        hybrid_config=HybridPortfolioConfig(
            emergency_clearance=0.4,
            caution_clearance=1.0,
            dense_ped_count=2,
            near_field_distance=1.0,
            hysteresis_steps=0,
        ),
        risk_dwa=risk,
        orca=orca,
        prediction=pred,
        mppi=mppi,
    )
    cmd = hybrid.plan(_obs(ped_positions=[(4.0, 0.0), (5.0, 0.0)]))
    assert cmd[0] == 0.5
    assert mppi.calls == 1


def test_hybrid_fallback_on_exception_and_config_defaults() -> None:
    """Hybrid should fallback to ORCA on exception or re-raise when disabled."""

    class _FailingHead(_DummyHead):
        def plan(self, observation: dict) -> tuple[float, float]:
            raise RuntimeError("boom")

    risk = _FailingHead("risk")
    orca = _DummyHead("orca")
    pred = _DummyHead("pred")
    hybrid = HybridPortfolioAdapter(
        hybrid_config=HybridPortfolioConfig(fallback_on_exception=True),
        risk_dwa=risk,
        orca=orca,
        prediction=pred,
        mppi=None,
    )
    assert hybrid.plan(_obs())[0] == 0.5
    assert orca.calls == 1

    hybrid_raise = HybridPortfolioAdapter(
        hybrid_config=HybridPortfolioConfig(fallback_on_exception=False),
        risk_dwa=risk,
        orca=orca,
        prediction=pred,
        mppi=None,
    )
    with pytest.raises(RuntimeError, match="boom"):
        hybrid_raise.plan(_obs())

    build = build_hybrid_portfolio_build_config(None)
    assert isinstance(build.hybrid, HybridPortfolioConfig)


def test_hybrid_builder_applies_full_subhead_fields() -> None:
    """Hybrid build config should preserve full risk/mpii sub-config fields."""
    build = build_hybrid_portfolio_build_config(
        {
            "risk_dwa": {"goal_progress_weight": 9.1, "progress_escape_speed": 0.9},
            "mppi_social": {"smoothness_weight": 0.77, "progress_escape_speed": 0.88},
        }
    )
    assert abs(build.risk_dwa.goal_progress_weight - 9.1) < 1e-9
    assert abs(build.risk_dwa.progress_escape_speed - 0.9) < 1e-9
    assert abs(build.mppi.smoothness_weight - 0.77) < 1e-9
    assert abs(build.mppi.progress_escape_speed - 0.88) < 1e-9
