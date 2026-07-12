"""Tests for portfolio planner adapters."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.hybrid_orca_sampler import (
    HybridORCASamplerAdapter,
    HybridORCASamplerConfig,
    build_hybrid_orca_sampler_build_config,
)
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
    *,
    robot=(0.0, 0.0),
    heading=0.0,
    goal=(2.0, 0.0),
    ped_positions=None,
    ped_velocities=None,
):
    """Build a planner observation fixture with optional pedestrian state."""
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


def test_risk_dwa_reshapes_flattened_pedestrians_using_count() -> None:
    """Flattened compatibility payloads should be reshaped instead of dropped."""
    planner = RiskDWAPlannerAdapter(RiskDWAPlannerConfig())
    _robot_pos, _heading, _goal, ped_pos, ped_vel = planner._extract_robot_goal_ped(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [2.0, 0.0], "next": [2.0, 0.0]},
            "pedestrians": {
                "positions": [0.5, 0.0, 1.0, 0.0, 5.0, 5.0],
                "velocities": [0.1, 0.0, 0.2, 0.0, 0.3, 0.0],
                "count": [2.0],
            },
        }
    )
    assert ped_pos.shape == (2, 2)
    assert ped_vel.shape == (2, 2)
    np.testing.assert_allclose(ped_pos, np.asarray([[0.5, 0.0], [1.0, 0.0]], dtype=float))
    np.testing.assert_allclose(ped_vel, np.asarray([[0.1, 0.0], [0.2, 0.0]], dtype=float))


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


def test_mppi_caches_absent_grid_payload_and_accepts_full_elite_fraction(monkeypatch) -> None:
    """MPPI should cache a missing grid and keep full-elite configurations valid."""
    planner = MPPISocialPlannerAdapter(
        MPPISocialConfig(sample_count=8, elite_fraction=1.0, iterations=1, horizon_steps=2)
    )
    calls = 0

    def _extract_grid_payload(_observation):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(planner, "_extract_grid_payload", _extract_grid_payload)

    linear, angular = planner.plan(_obs())

    assert calls == 1
    assert 0.0 <= linear <= planner.config.max_linear_speed
    assert abs(angular) <= planner.config.max_angular_speed


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
    """Planner head test double that tracks plan and reset calls."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.reset_calls = 0

    def plan(self, observation: dict) -> tuple[float, float]:
        """Return constant command and count calls."""
        _ = observation
        self.calls += 1
        return (1.0 if self.name == "risk" else 0.5, 0.0)

    def reset(self) -> None:
        """Track reset propagation from the hybrid adapter."""
        self.reset_calls += 1


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
        """Planner head that raises on every plan call."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Raise a deterministic planner failure."""
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


def test_hybrid_portfolio_records_selection_diagnostics_and_reset() -> None:
    """Hybrid diagnostics should expose selected heads and clear on episode reset."""
    risk = _DummyHead("risk")
    orca = _DummyHead("orca")
    pred = _DummyHead("pred")
    mppi = _DummyHead("mppi")
    hybrid = HybridPortfolioAdapter(
        hybrid_config=HybridPortfolioConfig(
            emergency_clearance=0.4,
            caution_clearance=1.0,
            dense_ped_count=2,
            hysteresis_steps=0,
        ),
        risk_dwa=risk,
        orca=orca,
        prediction=pred,
        mppi=mppi,
    )

    hybrid.plan(_obs(ped_positions=[(0.8, 0.0), (0.9, 0.0)]))
    hybrid.plan(_obs(ped_positions=[(5.0, 0.0)]))

    diagnostics = hybrid.diagnostics()

    assert diagnostics["steps"] == 2
    assert diagnostics["selected_head_counts"] == {"prediction": 1, "mppi": 1}
    assert diagnostics["fallback_count"] == 0
    assert diagnostics["last_decision"]["desired_head"] == "mppi"
    assert diagnostics["last_decision"]["selected_head"] == "mppi"
    assert diagnostics["last_decision"]["fallback"] is False

    diagnostics["selected_head_counts"]["mppi"] = 99
    assert hybrid.diagnostics()["selected_head_counts"]["mppi"] == 1

    hybrid.reset()
    cleared = hybrid.diagnostics()
    assert cleared["steps"] == 0
    assert cleared["selected_head_counts"] == {}
    assert cleared["last_decision"] is None


def test_hybrid_portfolio_last_decision_returns_copy_and_resets() -> None:
    """Step-level tooling should be able to read the latest hybrid decision safely."""
    risk = _DummyHead("risk")
    orca = _DummyHead("orca")
    pred = _DummyHead("pred")
    mppi = _DummyHead("mppi")
    hybrid = HybridPortfolioAdapter(
        hybrid_config=HybridPortfolioConfig(hysteresis_steps=0),
        risk_dwa=risk,
        orca=orca,
        prediction=pred,
        mppi=mppi,
    )

    assert hybrid.last_decision() is None

    hybrid.plan(_obs(ped_positions=[(5.0, 0.0)]))
    last = hybrid.last_decision()

    assert last is not None
    assert last["selected_head"] == "mppi"
    last["selected_head"] = "mutated"
    assert hybrid.last_decision()["selected_head"] == "mppi"

    hybrid.reset()
    assert hybrid.last_decision() is None


def test_hybrid_portfolio_records_fallback_diagnostics() -> None:
    """Fallback diagnostics should explain degraded ORCA selection after head failure."""

    class _FailingHead(_DummyHead):
        """Planner head that always fails for fallback diagnostics."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Raise a deterministic planner failure."""
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

    assert hybrid.plan(_obs()) == (0.5, 0.0)
    diagnostics = hybrid.diagnostics()

    assert diagnostics["steps"] == 1
    assert diagnostics["fallback_count"] == 1
    assert diagnostics["selected_head_counts"] == {"orca": 1}
    assert diagnostics["last_decision"]["desired_head"] == "risk_dwa"
    assert diagnostics["last_decision"]["selected_head"] == "orca"
    assert diagnostics["last_decision"]["fallback"] is True
    assert diagnostics["last_decision"]["fallback_from"] == "risk_dwa"
    assert "boom" in diagnostics["last_decision"]["error"]


def test_hybrid_portfolio_preserves_fallback_diagnostics_if_orca_raises() -> None:
    """Fallback intent should still be recorded when the ORCA fallback raises."""

    class _FailingHead(_DummyHead):
        """Planner head that raises after recording the attempted call."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Record the call and raise a named failure."""
            _ = observation
            self.calls += 1
            raise RuntimeError(f"{self.name} boom")

    risk = _FailingHead("risk")
    orca = _FailingHead("orca")
    pred = _DummyHead("pred")
    hybrid = HybridPortfolioAdapter(
        hybrid_config=HybridPortfolioConfig(fallback_on_exception=True),
        risk_dwa=risk,
        orca=orca,
        prediction=pred,
        mppi=None,
    )

    with pytest.raises(RuntimeError, match="orca boom"):
        hybrid.plan(_obs())

    diagnostics = hybrid.diagnostics()

    assert risk.calls == 1
    assert orca.calls == 1
    assert diagnostics["steps"] == 1
    assert diagnostics["fallback_count"] == 1
    assert diagnostics["selected_head_counts"] == {"orca": 1}
    assert diagnostics["last_decision"]["desired_head"] == "risk_dwa"
    assert diagnostics["last_decision"]["selected_head"] == "orca"
    assert diagnostics["last_decision"]["fallback"] is True
    assert diagnostics["last_decision"]["fallback_from"] == "risk_dwa"
    assert diagnostics["last_decision"]["error"] == "risk boom"


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


def test_hybrid_reset_clears_hysteresis_and_resets_child_heads() -> None:
    """Episode reset should clear active-head stickiness and child planner state."""
    risk = _DummyHead("risk")
    orca = _DummyHead("orca")
    pred = _DummyHead("pred")
    mppi = _DummyHead("mppi")
    hybrid = HybridPortfolioAdapter(
        hybrid_config=HybridPortfolioConfig(hysteresis_steps=3),
        risk_dwa=risk,
        orca=orca,
        prediction=pred,
        mppi=mppi,
    )
    hybrid._active_head = "prediction"
    hybrid._hold_remaining = 2

    hybrid.reset()

    assert hybrid._active_head == "risk_dwa"
    assert hybrid._hold_remaining == 0
    assert risk.reset_calls == 1
    assert orca.reset_calls == 1
    assert pred.reset_calls == 1
    assert mppi.reset_calls == 1


def test_hybrid_orca_sampler_prefers_sampler_when_orca_progress_is_low(
    monkeypatch,
) -> None:
    """Hybrid ORCA sampler should choose MPPI when ORCA is safe but stalls."""

    class _PrimaryHead:
        """Primary ORCA head test double with low progress."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Return a low-progress primary command."""
            _ = observation
            return 0.1, 0.0

    class _SamplerHead:
        """Sampler head test double with higher progress."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Return a higher-progress sampler command."""
            _ = observation
            return 0.6, 0.0

    planner = HybridORCASamplerAdapter(
        config=HybridORCASamplerConfig(sampler_progress_margin=0.05, near_field_distance=2.0),
        orca_adapter=_PrimaryHead(),
        sampler_adapter=_SamplerHead(),
    )

    def _eval(_observation: dict, command: tuple[float, float]) -> dict[str, float | bool]:
        """Score low primary commands below the sampler progress threshold."""
        if command[0] < 0.2:
            return {"safe": True, "progress": 0.01, "min_ped_clear": 1.0}
        return {"safe": True, "progress": 0.25, "min_ped_clear": 1.0}

    monkeypatch.setattr(planner, "_evaluate_command", _eval)
    cmd = planner.plan(_obs(ped_positions=[(0.8, 0.0)], ped_velocities=[(0.0, 0.0)]))
    assert cmd == (0.6, 0.0)


def test_hybrid_orca_sampler_keeps_orca_when_scene_is_clear(monkeypatch) -> None:
    """Hybrid ORCA sampler should keep ORCA in open scenes with safe progress."""

    class _PrimaryHead:
        """Primary ORCA head test double for clear-scene checks."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Return a safe primary command."""
            _ = observation
            return 0.4, 0.1

    class _SamplerHead:
        """Sampler head test double for clear-scene checks."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Return a sampler command that should not be selected."""
            _ = observation
            return 0.7, 0.0

    planner = HybridORCASamplerAdapter(
        config=HybridORCASamplerConfig(sampler_progress_margin=0.05, near_field_distance=1.0),
        orca_adapter=_PrimaryHead(),
        sampler_adapter=_SamplerHead(),
    )
    monkeypatch.setattr(
        planner,
        "_evaluate_command",
        lambda _observation, _command: {
            "safe": True,
            "progress": 0.2,
            "min_ped_clear": 2.0,
        },
    )
    assert planner.plan(_obs()) == (0.4, 0.1)


def test_hybrid_orca_sampler_uses_sampler_in_clear_scene_when_orca_stalls(
    monkeypatch,
) -> None:
    """Hybrid ORCA sampler should still repair low-progress ORCA in open scenes."""

    class _PrimaryHead:
        """Primary ORCA head test double that stalls in open space."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Return a low-progress primary command."""
            _ = observation
            return 0.1, 0.0

    class _SamplerHead:
        """Sampler head test double for open-space repair."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Return a higher-progress sampler command."""
            _ = observation
            return 0.7, 0.0

    planner = HybridORCASamplerAdapter(
        config=HybridORCASamplerConfig(sampler_progress_margin=0.05, near_field_distance=1.0),
        orca_adapter=_PrimaryHead(),
        sampler_adapter=_SamplerHead(),
    )

    def _eval(_observation: dict, command: tuple[float, float]) -> dict[str, float | bool]:
        """Score primary commands as stalled and sampler commands as progressing."""
        if command[0] < 0.2:
            return {"safe": True, "progress": 0.01, "min_ped_clear": 3.0}
        return {"safe": True, "progress": 0.25, "min_ped_clear": 3.0}

    monkeypatch.setattr(planner, "_evaluate_command", _eval)

    assert planner.plan(_obs()) == (0.7, 0.0)


def test_hybrid_orca_sampler_records_diagnostics_and_reset(monkeypatch) -> None:
    """Hybrid ORCA sampler should expose last-step diagnostics and clear them on reset."""

    class _PrimaryHead:
        """Primary ORCA head test double with reset support."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Return a low-progress primary command."""
            _ = observation
            return 0.1, 0.0

        def reset(self) -> None:
            """Accept reset propagation from the hybrid wrapper."""
            return None

    class _SamplerHead:
        """Sampler head test double with reset support."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Return a higher-progress sampler command."""
            _ = observation
            return 0.7, 0.0

        def reset(self) -> None:
            """Accept reset propagation from the hybrid wrapper."""
            return None

    planner = HybridORCASamplerAdapter(
        config=HybridORCASamplerConfig(sampler_progress_margin=0.05, near_field_distance=1.0),
        orca_adapter=_PrimaryHead(),
        sampler_adapter=_SamplerHead(),
    )

    def _eval(_observation: dict, command: tuple[float, float]) -> dict[str, float | bool]:
        """Score diagnostic commands for sampler repair selection."""
        if command[0] < 0.2:
            return {"safe": True, "progress": 0.01, "min_ped_clear": 3.0}
        return {"safe": True, "progress": 0.25, "min_ped_clear": 3.0}

    monkeypatch.setattr(planner, "_evaluate_command", _eval)

    assert planner.plan(_obs()) == (0.7, 0.0)
    diagnostics = planner.diagnostics()

    assert diagnostics["decision_counts"]["sampler_progress_repair"] == 1
    assert diagnostics["selected_head_counts"]["sampler"] == 1
    assert diagnostics["last_decision"]["decision"] == "sampler_progress_repair"
    assert diagnostics["last_decision"]["primary_eval"]["progress"] == pytest.approx(0.01)
    assert planner.last_decision()["selected_head"] == "sampler"

    planner.reset()

    cleared = planner.diagnostics()
    assert cleared["decision_counts"] == {}
    assert cleared["selected_head_counts"] == {"orca": 0, "sampler": 0, "stop": 0}
    assert cleared["last_decision"] is None


def test_hybrid_orca_sampler_uses_sampler_after_route_goal_regression(
    monkeypatch,
) -> None:
    """Route-goal regression should disable the clear-scene ORCA fast path."""

    class _PrimaryHead:
        """Primary ORCA head test double for route-regression checks."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Return the primary command before route regression."""
            _ = observation
            return 0.4, 0.0

    class _SamplerHead:
        """Sampler head test double for route-regression repair."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Return the sampler command after route regression."""
            _ = observation
            return 0.7, 0.0

    planner = HybridORCASamplerAdapter(
        config=HybridORCASamplerConfig(
            goal_tolerance=0.1,
            sampler_progress_margin=0.05,
            near_field_distance=1.0,
            route_stall_cycles_before_sampler=1,
            route_goal_regression_tolerance=0.5,
        ),
        orca_adapter=_PrimaryHead(),
        sampler_adapter=_SamplerHead(),
    )

    def _eval(_observation: dict, command: tuple[float, float]) -> dict[str, float | bool]:
        """Score primary and sampler commands for route-regression repair."""
        if command[0] < 0.5:
            return {"safe": True, "progress": 0.2, "min_ped_clear": 3.0}
        return {"safe": True, "progress": 0.35, "min_ped_clear": 3.0}

    monkeypatch.setattr(planner, "_evaluate_command", _eval)

    assert planner.plan(_obs(goal=(0.4, 0.0))) == (0.4, 0.0)
    assert planner.plan(_obs(goal=(3.0, 0.0))) == (0.7, 0.0)

    last_decision = planner.last_decision()
    assert last_decision is not None
    assert last_decision["decision"] == "sampler_progress_repair"
    assert last_decision["route_state"]["route_regressed"] is True


def test_hybrid_orca_sampler_switches_to_sampler_after_route_stall_without_progress(
    monkeypatch,
) -> None:
    """Route-level stall should override clear-scene fast-path ORCA selection."""

    class _PrimaryHead:
        """Primary ORCA head test double for stalled route cases."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Return a modest command that keeps short-horizon progress above margin."""
            _ = observation
            return 0.2, 0.0

    class _SamplerHead:
        """Sampler head test double that should win once route stalls."""

        def plan(self, observation: dict) -> tuple[float, float]:
            """Return a fallback command with lower short-horizon local gain."""
            _ = observation
            return 0.5, 0.0

    planner = HybridORCASamplerAdapter(
        config=HybridORCASamplerConfig(
            sampler_progress_margin=0.05,
            near_field_distance=2.0,
            route_stall_cycles_before_sampler=1,
            route_progress_epsilon=0.05,
        ),
        orca_adapter=_PrimaryHead(),
        sampler_adapter=_SamplerHead(),
    )

    def _eval(_observation: dict, command: tuple[float, float]) -> dict[str, float | bool]:
        """Keep both heads safe while keeping ORCA local-progress slightly higher."""
        if command[0] < 0.3:
            return {"safe": True, "progress": 0.2, "min_ped_clear": 3.0}
        return {"safe": True, "progress": 0.1, "min_ped_clear": 3.0}

    monkeypatch.setattr(planner, "_evaluate_command", _eval)
    observation = _obs(goal=(3.0, 0.0))

    first = planner.plan(observation)
    assert first == (0.2, 0.0)

    second = planner.plan(observation)
    assert second == (0.5, 0.0)
    assert planner.last_decision()["decision"] == "sampler_progress_repair"
    assert planner.last_decision()["route_state"]["route_stalled"] is True


def test_hybrid_orca_sampler_builder_preserves_nested_configs() -> None:
    """Hybrid ORCA sampler builder should parse guard, ORCA, and MPPI knobs."""
    build = build_hybrid_orca_sampler_build_config(
        {
            "max_linear_speed": 1.05,
            "orca_obstacle_margin": 0.18,
            "hybrid_guard": {
                "progress_margin": 0.08,
                "hard_ped_clearance": 0.61,
                "route_goal_regression_tolerance": 0.8,
            },
            "mppi_social": {"sample_count": 12, "goal_progress_weight": 6.0},
        }
    )
    assert build.guard.sampler_progress_margin == pytest.approx(0.08)
    assert build.guard.hard_ped_clearance == pytest.approx(0.61)
    assert build.guard.route_goal_regression_tolerance == pytest.approx(0.8)
    assert build.socnav.orca_obstacle_margin == pytest.approx(0.18)
    assert build.mppi.sample_count == 12
    assert build.mppi.max_linear_speed == pytest.approx(1.05)
