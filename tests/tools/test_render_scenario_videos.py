"""Unit tests for route-complete success handling in render_scenario_videos."""

from __future__ import annotations

from robot_sf.benchmark.termination_reason import route_complete_success
from scripts.tools import render_scenario_videos


def test_route_complete_success_true_only_for_route_complete() -> None:
    """Helper should only return true when full route is complete."""
    assert route_complete_success({"meta": {"is_route_complete": True}}) is True
    assert route_complete_success({"meta": {"is_route_complete": False}}) is False
    assert (
        route_complete_success({"success": True, "meta": {"is_waypoint_complete": True}}) is False
    )
    assert route_complete_success({}) is False


def test_policy_choices_include_orca_variants() -> None:
    """Renderer CLI should expose the SocNav ORCA variants."""
    parser = render_scenario_videos._build_parser()
    policy_action = next(
        action for action in parser._actions if getattr(action, "dest", None) == "policy"
    )
    choices = set(policy_action.choices)
    for policy in (
        "socnav_orca_nonholonomic",
        "socnav_orca_dd",
        "socnav_orca_relaxed",
        "socnav_hrvo",
    ):
        assert policy in choices
