"""Unit tests for route-complete success handling in predictive diagnostics."""

from __future__ import annotations

from robot_sf.benchmark.termination_reason import route_complete_success


def test_route_complete_success_ignores_waypoint_only_flags() -> None:
    """Predictive diagnostics must only mark route-complete success as successful."""
    assert route_complete_success({"meta": {"is_route_complete": True}}) is True
    assert route_complete_success({"meta": {"is_route_complete": False}}) is False
    assert (
        route_complete_success({"success": True, "meta": {"is_waypoint_complete": True}}) is False
    )
    assert route_complete_success({}) is False
