"""Tests for portfolio planner adapters."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.hybrid_portfolio import (
    HybridPortfolioAdapter,
    HybridPortfolioConfig,
    build_hybrid_portfolio_build_config,
)
from robot_sf.planner.mppi_social import MPPISocialConfig, MPPISocialPlannerAdapter
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, RiskDWAPlannerConfig


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
