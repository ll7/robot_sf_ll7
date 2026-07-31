"""Planner integration smoke for the proxemic costmap layer."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleCandidate,
    HybridRuleLocalPlannerAdapter,
    HybridRuleLocalPlannerConfig,
    build_hybrid_rule_local_planner_config,
)


def _obs(
    *,
    ped_positions: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
) -> dict:
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray([0.0, 0.0], dtype=float),
            "heading": np.asarray([0.0], dtype=float),
            "speed": np.asarray([0.0], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray([2.0, 0.0], dtype=float),
            "next": np.asarray([2.0, 0.0], dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
            "radius": 0.25,
        },
        "sim": {"timestep": 0.1},
    }


def test_hybrid_rule_continuous_static_clearance_skips_dynamic_check_on_violation(
    monkeypatch,
) -> None:
    """Continuous static acceptance preserves the legacy per-step dynamic skip.

    Regression guard for the ``_evaluate_candidate`` decomposition: when continuous
    static checking is active and a static-clearance violation is accepted (rather than
    rejected), the original monolithic loop issued ``continue`` and skipped that step's
    dynamic-collision check. The refactor must preserve that, so a pedestrian that is
    only in dynamic-collision range at such a step is not flagged, and the running
    ``min_dynamic_clearance`` stays at infinity for the skipped step.
    """
    cfg = HybridRuleLocalPlannerConfig(
        rollout_horizon=0.2,
        continuous_static_clearance_enabled=True,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    # A non-None context enables the continuous path; collision/clearance lookups below
    # are patched, so the concrete environment shape is irrelevant to this regression.
    planner._continuous_static_context = object()
    observation = _obs(ped_positions=[(0.05, 0.0)])
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "dynamic_window")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    # No continuous static *collision* (so clearance, not collision, is the violated gate).
    monkeypatch.setattr(planner, "_static_collision_rejection", lambda **_kwargs: None)
    # Static clearance below the required hard threshold -> violation accepted via
    # the continuous static path, which triggers the legacy per-step dynamic skip.
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 0.05)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
    )

    assert evaluation["accepted"] is True
    assert evaluation["continuous_static_checked"] is True
    assert evaluation["min_dynamic_clearance"] == pytest.approx(float("inf"))


def test_hybrid_rule_nonfinite_clearance_preserves_dynamic_collision_check(monkeypatch) -> None:
    """Non-finite clearance follows the legacy dynamic-collision path.

    The original evaluator only entered static-clearance policy when
    ``min_static_clearance <= required_static_clearance``. In particular, a NaN
    clearance did not activate the continuous-static ``continue`` and therefore
    still allowed the dynamic-collision check to reject the candidate.
    """
    cfg = HybridRuleLocalPlannerConfig(
        rollout_horizon=0.2,
        continuous_static_clearance_enabled=True,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    planner._continuous_static_context = object()
    observation = _obs(ped_positions=[(0.05, 0.0)])
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "dynamic_window")

    monkeypatch.setattr(planner, "_static_collision_rejection", lambda **_kwargs: None)
    monkeypatch.setattr(
        planner, "_min_obstacle_clearance", lambda _point, _observation: float("nan")
    )

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "dynamic_collision"


def test_hybrid_rule_records_proxemic_costmap_metadata_and_cost_term() -> None:
    """Opt-in proxemic layer contributes a soft score term and metadata hash."""
    cfg = HybridRuleLocalPlannerConfig(
        proxemic_costmap_enabled=True,
        proxemic_costmap_weight=2.0,
        proxemic_costmap_personal_radius=0.45,
        proxemic_costmap_social_radius=1.2,
        proxemic_costmap_social_weight=1.0,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(ped_positions=[(0.8, 0.0)], ped_velocities=[(0.0, 0.0)])
    state = planner._extract_state(observation)
    evaluation = planner._evaluate_candidate(
        candidate=HybridRuleCandidate(0.0, 0.0, "stop"),
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=0.8,
    )

    assert evaluation["accepted"] is True
    assert evaluation["terms"]["proxemic_cost"] > 0.0
    assert evaluation["proxemic_cost_summary"]["enabled"] is True

    planner.plan(observation)
    metadata = planner.diagnostics()["proxemic_costmap"]
    assert metadata["enabled"] is True
    assert metadata["status"] == "ok"
    assert metadata["config_hash"]
    assert metadata["soft_cost_only"] is True


def test_hybrid_rule_config_builder_accepts_nested_proxemic_costmap() -> None:
    """YAML-style nested layer config maps into planner fields."""
    cfg = build_hybrid_rule_local_planner_config(
        {
            "proxemic_costmap": {
                "enabled": True,
                "personal_radius": 0.4,
                "social_radius": 1.4,
                "personal_weight": 1.2,
                "social_weight": 0.4,
                "velocity_elongation_factor": 0.5,
                "max_cost": 4.0,
                "decay_function": "gaussian",
            },
            "proxemic_costmap_weight": 3.0,
        }
    )

    assert cfg.proxemic_costmap_enabled is True
    assert cfg.proxemic_costmap_personal_radius == 0.4
    assert cfg.proxemic_costmap_social_radius == 1.4
    assert cfg.proxemic_costmap_decay_function == "gaussian"
    assert cfg.proxemic_costmap_weight == 3.0


def test_nested_proxemic_config_validation_does_not_mutate_input() -> None:
    """Nested proxemic config reuses validation without caller side effects."""
    payload = {
        "proxemic_costmap": {
            "enabled": True,
            "personal_radius": 0.4,
            "social_radius": 1.4,
            "unknown_field": 1.0,
        },
        "proxemic_costmap_weight": 3.0,
    }

    with pytest.raises(ValueError, match="unknown proxemic costmap config fields"):
        build_hybrid_rule_local_planner_config(payload)

    assert "proxemic_costmap" in payload
    assert payload["proxemic_costmap"]["unknown_field"] == 1.0


def test_proxemic_costmap_config_reflects_runtime_config_updates() -> None:
    """Planner resolves proxemic config dynamically from current config fields."""
    cfg = HybridRuleLocalPlannerConfig(
        proxemic_costmap_enabled=False,
        proxemic_costmap_social_radius=1.2,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)

    disabled_hash = planner._proxemic_costmap_metadata()["config_hash"]

    planner.config.proxemic_costmap_enabled = True
    planner.config.proxemic_costmap_social_radius = 1.6

    metadata = planner._proxemic_costmap_metadata()
    assert metadata["enabled"] is True
    assert metadata["config_hash"] != disabled_hash
