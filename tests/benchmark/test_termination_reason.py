"""Tests for canonical termination-reason helpers."""

from __future__ import annotations

from robot_sf.benchmark.termination_reason import (
    collision_event,
    outcome_contradictions,
    resolve_termination_reason,
    route_complete_success,
    status_from_termination_reason,
)


def test_resolve_termination_reason_all_paths() -> None:
    """Resolve helper should cover explicit flags and fallback behavior."""
    assert (
        resolve_termination_reason(
            terminated=False,
            truncated=False,
            success=False,
            collision=False,
            had_error=True,
        )
        == "error"
    )
    assert (
        resolve_termination_reason(
            terminated=True,
            truncated=False,
            success=True,
            collision=False,
        )
        == "success"
    )
    assert (
        resolve_termination_reason(
            terminated=True,
            truncated=False,
            success=False,
            collision=True,
        )
        == "collision"
    )
    assert (
        resolve_termination_reason(
            terminated=True,
            truncated=False,
            success=True,
            collision=True,
        )
        == "collision"
    )
    assert (
        resolve_termination_reason(
            terminated=True,
            truncated=False,
            success=False,
            collision=False,
        )
        == "terminated"
    )
    assert (
        resolve_termination_reason(
            terminated=False,
            truncated=True,
            success=False,
            collision=False,
        )
        == "truncated"
    )
    assert (
        resolve_termination_reason(
            terminated=False,
            truncated=False,
            success=False,
            collision=False,
            reached_max_steps=True,
        )
        == "max_steps"
    )
    assert (
        resolve_termination_reason(
            terminated=False,
            truncated=False,
            success=True,
            collision=False,
        )
        == "success"
    )
    assert (
        resolve_termination_reason(
            terminated=False,
            truncated=False,
            success=False,
            collision=True,
        )
        == "collision"
    )
    assert (
        resolve_termination_reason(
            terminated=False,
            truncated=False,
            success=True,
            collision=True,
        )
        == "collision"
    )
    assert (
        resolve_termination_reason(
            terminated=False,
            truncated=False,
            success=False,
            collision=False,
        )
        == "max_steps"
    )


def test_status_from_termination_reason_mapping() -> None:
    """Status mapper should preserve success/collision and collapse the rest to failure."""
    assert status_from_termination_reason("success") == "success"
    assert status_from_termination_reason("collision") == "collision"
    assert status_from_termination_reason("max_steps") == "failure"
    assert status_from_termination_reason("error") == "failure"


def test_route_complete_success_reads_only_route_complete_flag() -> None:
    """Route success helper should ignore waypoint-only payloads."""
    assert route_complete_success({"meta": {"is_route_complete": True}}) is True
    assert route_complete_success({"meta": {"is_route_complete": False}}) is False
    assert route_complete_success({"meta": {"is_waypoint_complete": True}}) is False


def test_collision_event_reads_info_and_meta_flags() -> None:
    """Collision helper should accept either top-level or metadata collision flags."""
    assert collision_event({"collision": True}) is True
    assert collision_event({"meta": {"is_obstacle_collision": True}}) is True
    assert collision_event({"meta": {"is_route_complete": True}}) is False


def test_outcome_contradictions_detect_success_mismatch() -> None:
    """Outcome integrity checks should flag success/outcome mismatches."""
    contradictions = outcome_contradictions(
        termination_reason="max_steps",
        outcome={"route_complete": False, "collision_event": False, "timeout_event": True},
        metrics={"success": 1.0, "collisions": 0.0},
    )
    assert contradictions
