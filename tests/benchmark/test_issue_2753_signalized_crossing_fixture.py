"""Tests for the issue #2753 signalized crossing metrics report generator.

Validates that the generator script produces correct evidence artifacts:
- All expected files are written
- All four canonical row types appear
- Excluded rows have denominator 0 and compliance_eligible false
- Observable rows are the only eligible rows
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools.generate_signalized_crossing_metrics_report import generate

if TYPE_CHECKING:
    from pathlib import Path

_EXPECTED_FILES = {"summary.json", "report.md", "README.md"}
_EXPECTED_ROW_TYPES = {
    "red_required_stop",
    "green_proceed",
    "unavailable_no_claim",
    "proxy_only_denominator_excluded",
}


@pytest.fixture()
def evidence_dir(tmp_path: Path) -> Path:
    """Generate evidence artifacts into a temporary directory."""
    generate(tmp_path)
    return tmp_path


def test_generator_writes_all_expected_files(evidence_dir: Path) -> None:
    """summary.json, report.md, and README.md must all be created."""
    written = {f.name for f in evidence_dir.iterdir()}
    assert _EXPECTED_FILES == written


def test_summary_json_is_valid(evidence_dir: Path) -> None:
    """summary.json must parse and contain required keys."""
    summary = json.loads((evidence_dir / "summary.json").read_text())
    assert summary["issue"] == 2753
    assert "row_types_present" in summary
    assert "eligible_rows" in summary
    assert "excluded_rows" in summary
    assert summary["claim_boundary"]  # non-empty


def test_all_four_row_types_present(evidence_dir: Path) -> None:
    """All four canonical row types must appear in the summary."""
    summary = json.loads((evidence_dir / "summary.json").read_text())
    assert set(summary["row_types_present"]) == _EXPECTED_ROW_TYPES


def test_total_rows_count(evidence_dir: Path) -> None:
    """Exactly four fixture rows must be generated."""
    summary = json.loads((evidence_dir / "summary.json").read_text())
    assert summary["total_rows"] == 4
    assert summary["observable_count"] == 2
    assert summary["excluded_count"] == 2


def test_excluded_rows_denominator_zero(evidence_dir: Path) -> None:
    """Excluded rows must have signal_metrics_denominator == 0."""
    summary = json.loads((evidence_dir / "summary.json").read_text())
    for row in summary["excluded_rows"]:
        assert row["signal_metrics_denominator"] == 0, (
            f"{row['episode_id']} has nonzero denominator"
        )
    assert summary["excluded_denominator_zero"] is True


def test_excluded_rows_not_compliance_eligible(evidence_dir: Path) -> None:
    """Excluded rows must not be compliance eligible."""
    summary = json.loads((evidence_dir / "summary.json").read_text())
    assert summary["excluded_not_compliance_eligible"] is True
    for row in summary["excluded_rows"]:
        assert row["planner_observable"] is False
        assert row["benchmark_evidence"] is False


def test_observable_rows_are_only_eligible(evidence_dir: Path) -> None:
    """Only planner_observable + benchmark_evidence rows should be eligible."""
    summary = json.loads((evidence_dir / "summary.json").read_text())
    for row in summary["eligible_rows"]:
        assert row["planner_observable"] is True
        assert row["benchmark_evidence"] is True
        assert row["signal_metrics_denominator"] == 1
    eligible_ids = {r["episode_id"] for r in summary["eligible_rows"]}
    assert "fixture_red_required_stop" in eligible_ids
    assert "fixture_green_proceed" in eligible_ids


def test_report_md_contains_table(evidence_dir: Path) -> None:
    """report.md must contain a Markdown table with the expected episode IDs."""
    content = (evidence_dir / "report.md").read_text()
    assert "| episode_id" in content
    assert "|---|" in content
    for eid in [
        "fixture_red_required_stop",
        "fixture_green_proceed",
        "fixture_unavailable_no_claim",
        "fixture_proxy_only_denominator_excluded",
    ]:
        assert eid in content


def test_readme_md_mentions_issue(evidence_dir: Path) -> None:
    """README.md must reference issue #2753."""
    content = (evidence_dir / "README.md").read_text()
    assert "2753" in content
    assert "fixture" in content.lower()
