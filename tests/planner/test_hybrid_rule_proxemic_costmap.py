"""Planner integration smoke for the proxemic costmap layer."""

from __future__ import annotations

import numpy as np

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
