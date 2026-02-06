"""Tests for canonical termination-reason helpers."""

from __future__ import annotations

from robot_sf.benchmark.termination_reason import (
    resolve_termination_reason,
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
