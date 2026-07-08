"""Tests for the issue #2910 publication-suite certification report."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.analysis.build_issue_2910_publication_suite_certification_report import (
    DEFAULT_RELEASE_CLAIM_MATRIX,
    DEFAULT_SCENARIO_CERTIFICATION_SUMMARY,
    InputPaths,
    build_report,
    main,
    render_markdown,
)

if TYPE_CHECKING:
    from pathlib import Path


def _default_inputs() -> InputPaths:
    """Return default tracked input paths."""

    return InputPaths(
        scenario_certification_summary=DEFAULT_SCENARIO_CERTIFICATION_SUMMARY,
        release_claim_matrix=DEFAULT_RELEASE_CLAIM_MATRIX,
    )


def _load_json(path: Path) -> dict:
    """Load a JSON object from ``path``."""

    return json.loads(path.read_text(encoding="utf-8"))


def test_report_maps_current_remaining_scenarios_and_keeps_gate_blocked() -> None:
    """Current tracked evidence lists the remaining #2910 scenario blockers."""

    report = build_report(
        _load_json(DEFAULT_SCENARIO_CERTIFICATION_SUMMARY),
        _load_json(DEFAULT_RELEASE_CLAIM_MATRIX),
        inputs=_default_inputs(),
    )

    assert report["schema_version"].endswith(".v1")
    assert report["status"] == "blocked"
    assert report["summary"]["benchmark_eligibility_counts"] == {
        "eligible": 37,
        "excluded": 2,
        "stress_only": 9,
    }
    assert [item["scenario_id"] for item in report["excluded_scenarios"]] == [
        "francis2023_exiting_elevator",
        "francis2023_narrow_doorway",
    ]
    assert len(report["stress_only_scenarios"]) == 9
    assert report["summary"]["release_artifact_rows_blocked_on_certification"] == 0
    assert {blocker["check"] for blocker in report["blockers"]} == {
        "excluded_scenarios",
        "stress_only_scenarios",
    }


def test_report_can_pass_for_synthetic_accepted_suite() -> None:
    """The report status is data-driven, not permanently blocked."""

    summary = {
        "scenario_count": 1,
        "benchmark_eligibility_counts": {
            "eligible": 1,
            "excluded": 0,
            "stress_only": 0,
        },
        "excluded_scenarios": [],
        "stress_only_scenarios": [],
    }
    matrix = {
        "rows": [
            {
                "section": "release_artifact",
                "row_id": "release_artifact:complete",
                "classification": "benchmark evidence",
                "scenario_certification": "scenario_cert.v1:accepted",
            }
        ]
    }

    report = build_report(summary, matrix, inputs=_default_inputs())

    assert report["status"] == "pass"
    assert report["blockers"] == []
    assert report["summary"]["release_artifact_rows_blocked_on_certification"] == 0


def test_report_policy_blocks_detail_count_mismatch() -> None:
    """Publication-suite policy cannot pass when detail lists disagree with counts."""
    summary = {
        "scenario_count": 3,
        "benchmark_eligibility_counts": {
            "eligible": 1,
            "excluded": 2,
            "stress_only": 0,
        },
        "excluded_scenarios": [{"scenario_id": "blocked_geometry"}],
        "stress_only_scenarios": [],
    }
    matrix = {
        "rows": [
            {
                "section": "release_artifact",
                "row_id": "release_artifact:complete",
                "classification": "benchmark evidence",
                "scenario_certification": "scenario_cert.v1:accepted_reviewed",
            }
        ]
    }
    policy = {
        "nominal_release_suite": {
            "certification_status": "scenario_cert.v1:accepted_reviewed",
            "excluded_scenarios": [
                {
                    "scenario_id": "blocked_geometry",
                    "action": "exclude_from_nominal_publication",
                }
            ],
        }
    }

    report = build_report(
        summary,
        matrix,
        inputs=_default_inputs(),
        publication_suite_policy=policy,
    )

    assert report["status"] == "blocked"
    assert {blocker["check"] for blocker in report["blockers"]} == {"excluded_scenarios"}


def test_render_markdown_lists_blockers_and_next_action() -> None:
    """Markdown report preserves the scenario lists and next empirical action."""

    report = build_report(
        _load_json(DEFAULT_SCENARIO_CERTIFICATION_SUMMARY),
        _load_json(DEFAULT_RELEASE_CLAIM_MATRIX),
        inputs=_default_inputs(),
    )
    markdown = render_markdown(report)

    assert "Status: `blocked`" in markdown
    assert "`francis2023_narrow_doorway`" in markdown
    assert "`classic_cross_trap_low`" in markdown
    assert "Next Empirical Action" in markdown


def test_cli_writes_json_and_markdown(tmp_path) -> None:
    """CLI writes both report artifacts for review."""

    assert main(["--output-dir", tmp_path.as_posix()]) == 0

    payload = _load_json(tmp_path / "report.json")
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert payload["status"] == "blocked_pending_rebase"
    assert payload["summary"]["blocker_count"] == 0
    assert payload["summary"]["publication_suite_policy_status"] == "applied"
    assert "Publication Suite Certification Report" in markdown
