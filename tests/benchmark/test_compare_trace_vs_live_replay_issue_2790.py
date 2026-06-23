"""Tests for the compare_trace_vs_live_replay_issue_2790 script."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.analysis.compare_trace_vs_live_replay_issue_2790 import (
    build_comparison_report,
    build_lineage_check,
    render_markdown_report,
)


@pytest.fixture
def dummy_trace_data() -> dict[str, Any]:
    """Provide dummy trace-derived summary data."""
    return {
        "schema_version": "observation_noise_envelope.v1",
        "issue": 2755,
        "fixture": {
            "scenario_id": "issue_2756_occluded_emergence",
            "seed": 111,
            "planner_id": "hybrid_rule_v0_minimal",
            "episode_id": "issue_2756_occluded_emergence_ep0000",
        },
        "conditions": [
            {
                "condition": cond,
                "classification": {
                    "label": "robustness_evidence" if cond == "delay_only" else "diagnostic_only",
                    "rationale": f"Dummy rationale for {cond}",
                },
            }
            for cond in (
                "noop",
                "low_noise",
                "medium_noise",
                "missed_detection_only",
                "occlusion_only",
                "delay_only",
                "combined",
            )
        ],
    }


@pytest.fixture
def dummy_live_data() -> dict[str, Any]:
    """Provide dummy live replay summary data."""
    return {
        "schema_version": "issue_2777_observation_noise_live_replay.v1",
        "issue": 2777,
        "fixture_contract": {
            "required_source_issue": 2756,
            "required_scenario": "issue_2756_occluded_emergence",
            "required_seed": 111,
            "scenario_matrix": "configs/scenarios/sets/example.yaml",
            "satisfied": True,
            "blocker": None,
        },
        "conditions": [
            {
                "name": cond,
                "command_summary": {
                    "sequence_changed": False,
                },
                "progress_delta": {
                    "net_goal_progress": {"changed": False},
                },
                "classification": {
                    "label": "policy_insensitive",
                    "rationale": f"Dummy rationale for {cond}",
                },
            }
            for cond in (
                "noop",
                "low_noise",
                "medium_noise",
                "missed_detection_only",
                "occlusion_only",
                "delay_only",
                "combined",
            )
        ],
    }


def test_build_comparison_report_false_positive(dummy_trace_data, dummy_live_data) -> None:
    """If trace predicts sensitivity but live replay has no changes, it is a false positive."""
    report = build_comparison_report(dummy_trace_data, dummy_live_data)

    assert report["schema_version"] == "compare_trace_vs_live_replay.v1"
    assert report["prefilter_trustworthy"] is False
    assert report["recommended_action"] == "demote_to_debugging_only"
    assert report["lineage_check"]["satisfied"] is True

    delay_row = next(r for r in report["comparisons"] if r["condition"] == "delay_only")
    assert delay_row["comparison_label"] == "false_positive"
    assert delay_row["comparison_category"] == "trace_only_effect"

    noop_row = next(r for r in report["comparisons"] if r["condition"] == "noop")
    assert noop_row["comparison_label"] == "confirmed"
    assert noop_row["comparison_category"] == "confirmed_no_effect"


def test_build_comparison_report_confirmed_sensitivity(dummy_trace_data, dummy_live_data) -> None:
    """If trace predicts sensitivity and live replay shows changes, it is confirmed."""
    # Modify live data so delay_only shows command sequence changed
    delay_live = next(c for c in dummy_live_data["conditions"] if c["name"] == "delay_only")
    delay_live["command_summary"]["sequence_changed"] = True

    report = build_comparison_report(dummy_trace_data, dummy_live_data)

    # Still false due to other default/noop mismatches? Wait, all other conditions match
    # (Trace: diagnostic_only / Live: no-effect, so they agree!).
    # So with delay_only matching, prefilter should be trustworthy (all_matched = True)
    assert report["prefilter_trustworthy"] is True
    assert report["recommended_action"] == "keep_as_cheap_prefilter"

    delay_row = next(r for r in report["comparisons"] if r["condition"] == "delay_only")
    assert delay_row["comparison_label"] == "confirmed"
    assert delay_row["comparison_category"] == "confirmed_sensitivity"


def test_lineage_check_fails_closed_on_seed_mismatch(dummy_trace_data, dummy_live_data) -> None:
    """Mismatched trace/live fixture lineage blocks prefilter trust."""
    dummy_live_data["fixture_contract"]["required_seed"] = 999

    lineage_check = build_lineage_check(dummy_trace_data, dummy_live_data)
    report = build_comparison_report(dummy_trace_data, dummy_live_data)

    assert lineage_check["satisfied"] is False
    assert lineage_check["checks"]["seed_matches"] is False
    assert report["prefilter_trustworthy"] is False
    assert report["recommended_action"] == "demote_to_debugging_only"
    assert "lineage check failed" in report["verdict"]


def test_render_markdown_report(dummy_trace_data, dummy_live_data) -> None:
    """The markdown renderer should output the formatted report."""
    report = build_comparison_report(dummy_trace_data, dummy_live_data)
    md_content = render_markdown_report(report)

    assert "# Issue #2790" in md_content
    assert "*Lineage Check Satisfied*: `True`" in md_content
    assert "| `delay_only` |" in md_content
    assert "**`false_positive`**" in md_content
