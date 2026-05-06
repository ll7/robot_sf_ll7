"""Tests for policy-search portfolio summaries."""

from __future__ import annotations

from pathlib import Path

from scripts.tools.summarize_policy_search_portfolio import _report_name_parts


def test_report_name_parts_accepts_leader_collision_h500_stage() -> None:
    """The h500 collision-slice stage should be parsed as a registered report stage."""
    parts = _report_name_parts(
        Path(
            "docs/context/policy_search/reports/"
            "2026-05-06_scenario_adaptive_orca_v1_leader_collision_slice_h500.md"
        )
    )

    assert parts is not None
    _, candidate, stage = parts
    assert candidate == "scenario_adaptive_orca_v1"
    assert stage == "leader_collision_slice_h500"
