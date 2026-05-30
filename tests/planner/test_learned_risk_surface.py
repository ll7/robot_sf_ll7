"""Tests for the local learned-risk surface planner contract."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.learned_risk_surface import (
    LocalRiskSurfaceSpec,
    RiskSurfacePlannerAdapter,
    RiskSurfaceUnavailable,
    attach_risk_surface_to_observation,
    deterministic_pedestrian_risk_surface,
)


def _observation_with_pedestrian() -> dict:
    """Return a tiny structured observation with one pedestrian in front."""
    return {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [0.0],
            "speed": [0.0],
            "radius": [0.3],
        },
        "goal": {"current": [2.0, 0.0], "next": [2.0, 0.0]},
        "pedestrians": {
            "positions": [[0.75, 0.0]],
            "velocities": [[0.0, 0.0]],
            "count": [1],
            "radius": [0.3],
        },
    }


def test_deterministic_surface_has_reviewable_contract_and_peak_near_pedestrian() -> None:
    """The fixture producer should emit a normalized ego-frame risk surface."""
    spec = LocalRiskSurfaceSpec(resolution=0.25, width=2.0, height=2.0, risk_threshold=0.6)

    surface = deterministic_pedestrian_risk_surface(_observation_with_pedestrian(), spec)
    diagnostics = surface.diagnostics()

    assert surface.values.shape == (8, 8)
    assert diagnostics["status"] == "available"
    assert diagnostics["frame"] == "ego"
    assert diagnostics["risk_cells_at_or_above_threshold"] > 0
    assert 0.0 <= diagnostics["mean_risk"] <= diagnostics["max_risk"] <= 1.0

    meta = surface.occupancy_meta()
    assert meta["use_ego_frame"] == [1.0]
    assert meta["channel_indices"] == [0, 1, -1, 2]
    assert meta["risk_surface"]["risk_threshold"] == 0.6


def test_surface_attaches_as_occupancy_payload_consumable_by_planners() -> None:
    """Risk surfaces should use the existing occupancy-grid observation contract."""
    spec = LocalRiskSurfaceSpec(resolution=0.5, width=3.0, height=3.0)
    surface = deterministic_pedestrian_risk_surface(_observation_with_pedestrian(), spec)

    enriched = attach_risk_surface_to_observation(_observation_with_pedestrian(), surface)

    assert enriched["occupancy_grid"].shape == (3, 6, 6)
    np.testing.assert_allclose(enriched["occupancy_grid"][0], surface.values)
    np.testing.assert_allclose(enriched["occupancy_grid"][2], surface.values)
    assert enriched["local_risk_surface_diagnostics"]["producer_id"] == ("deterministic_fixture_v0")


def test_risk_surface_planner_adapter_produces_bounded_command_and_diagnostics() -> None:
    """A wrapped occupancy-aware planner should consume the produced surface."""
    adapter = RiskSurfacePlannerAdapter(
        spec=LocalRiskSurfaceSpec(resolution=0.25, width=4.0, height=4.0)
    )

    linear, angular = adapter.plan(_observation_with_pedestrian())
    diagnostics = adapter.diagnostics()

    assert 0.0 <= linear <= 1.2
    assert abs(angular) <= np.pi
    assert diagnostics["status"] == "ok"
    assert diagnostics["execution_mode"] == "adapter"
    assert diagnostics["availability_status"] == "available"
    assert diagnostics["benchmark_strength"] is False
    assert diagnostics["surface"]["risk_cells_at_or_above_threshold"] > 0


def test_risk_surface_adapter_fails_closed_without_robot_state() -> None:
    """Missing required state should stop instead of falling back silently."""
    adapter = RiskSurfacePlannerAdapter()

    assert adapter.plan({"pedestrians": {"positions": [[1.0, 0.0]], "count": [1]}}) == (
        0.0,
        0.0,
    )
    diagnostics = adapter.diagnostics()
    assert diagnostics["status"] == "not_available"
    assert diagnostics["availability_status"] == "not_available"
    assert "robot state" in str(diagnostics["error"])


def test_risk_surface_contract_rejects_invalid_resolution() -> None:
    """Invalid geometry should be rejected before any planner run."""
    try:
        LocalRiskSurfaceSpec(resolution=0.0)
    except RiskSurfaceUnavailable as exc:
        assert "resolution" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected RiskSurfaceUnavailable")
