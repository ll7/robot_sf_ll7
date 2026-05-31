"""Tests for the diagnostic actuation-aware hybrid-rule planner wrapper."""

from __future__ import annotations

import pytest

from robot_sf.planner.actuation_aware_hybrid_rule import (
    ActuationAwareHybridRuleAdapter,
    ActuationAwareHybridRuleConfig,
    build_actuation_aware_hybrid_rule_config,
)
from robot_sf.planner.hybrid_rule_local_planner import HybridRuleLocalPlannerConfig


class _FixedCommandPlanner:
    """Minimal planner stub that returns a fixed unicycle command."""

    def __init__(self, command: tuple[float, float]) -> None:
        self.command = command
        self.reset_calls = 0

    def reset(self, *, seed: int | None = None) -> None:
        self.reset_calls += 1

    def plan(self, observation: dict[str, object]) -> tuple[float, float]:
        return self.command

    def diagnostics(self) -> dict[str, object]:
        return {"wrapped": "ok"}


def test_actuation_aware_wrapper_projects_command_and_reports_diagnostics() -> None:
    """High acceleration and yaw-rate requests should be projected and auditable."""
    base = _FixedCommandPlanner((3.0, 4.0))
    planner = ActuationAwareHybridRuleAdapter(
        config=ActuationAwareHybridRuleConfig(
            base_planner=HybridRuleLocalPlannerConfig(),
            max_linear_accel_m_s2=2.0,
            max_angular_accel_rad_s2=4.0,
            max_yaw_rate_rad_s=1.2,
            projection_dt=0.1,
        ),
        base_planner=base,
    )

    command = planner.plan({"sim": {"timestep": 0.1}})

    assert command == pytest.approx((0.2, 0.4))
    diagnostics = planner.diagnostics()
    projection = diagnostics["actuation_projection"]
    assert diagnostics["diagnostic_only"] is True
    assert diagnostics["calibrated_hardware_evidence"] is False
    assert projection["status"] == "ok"
    assert projection["projection_fraction"] == pytest.approx(1.0)
    assert projection["linear_accel_limited_fraction"] == pytest.approx(1.0)
    assert projection["angular_accel_limited_fraction"] == pytest.approx(1.0)
    assert projection["yaw_rate_limited_fraction"] == pytest.approx(1.0)
    assert planner.last_decision()["wrapped_decision"] is None


def test_actuation_aware_config_keeps_synthetic_non_hardware_claim_boundary() -> None:
    """Config parsing should keep the AMV diagnostic boundary explicit."""
    config = build_actuation_aware_hybrid_rule_config(
        {
            "planner_variant": "actuation_aware_hybrid_rule_v0",
            "projection_profile_name": "amv-actuation-stress-v0",
            "claim_scope": "synthetic-only",
            "diagnostic_only": "true",
            "calibrated_hardware_evidence": "false",
        }
    )

    assert config.base_planner.planner_variant == "actuation_aware_hybrid_rule_v0"
    assert config.projection_profile_name == "amv-actuation-stress-v0"
    assert config.claim_scope == "synthetic-only"
    assert config.diagnostic_only is True
    assert config.calibrated_hardware_evidence is False


def test_actuation_aware_config_rejects_ambiguous_boolean_strings() -> None:
    """Claim-boundary flags should fail closed on ambiguous values."""
    with pytest.raises(ValueError, match="calibrated_hardware_evidence"):
        build_actuation_aware_hybrid_rule_config(
            {
                "calibrated_hardware_evidence": "maybe",
            }
        )
