"""Tests for scripts/dev/branch_coverage_report.py."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.dev.branch_coverage_report import (
    _branch_pct,
    load_coverage,
    overall_summary,
    per_package_table,
    threshold_proposal,
    worst_evidence_critical,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def sample_coverage(tmp_path: Path) -> Path:
    """Create a minimal coverage.json for testing."""
    data = {
        "totals": {
            "covered_lines": 100,
            "missing_lines": 200,
            "covered_branches": 30,
            "missing_branches": 70,
            "percent_covered": 33.33,
        },
        "files": {
            "robot_sf/sim/core.py": {
                "summary": {
                    "covered_lines": 50,
                    "missing_lines": 10,
                    "covered_branches": 20,
                    "missing_branches": 5,
                }
            },
            "robot_sf/benchmark/map_runner.py": {
                "summary": {
                    "covered_lines": 10,
                    "missing_lines": 90,
                    "covered_branches": 0,
                    "missing_branches": 40,
                }
            },
            "robot_sf/planner/socnav.py": {
                "summary": {
                    "covered_lines": 5,
                    "missing_lines": 95,
                    "covered_branches": 0,
                    "missing_branches": 60,
                }
            },
            "robot_sf/render/draw.py": {
                "summary": {
                    "covered_lines": 35,
                    "missing_lines": 5,
                    "covered_branches": 10,
                    "missing_branches": 0,
                }
            },
        },
    }
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(data))
    return path


def test_branch_pct_with_branches() -> None:
    """Branch pct returns correct percentage when branches exist."""
    assert _branch_pct({"covered_branches": 10, "missing_branches": 10}) == 50.0


def test_branch_pct_no_branches() -> None:
    """Branch pct returns 100 when no branches exist."""
    assert _branch_pct({"covered_branches": 0, "missing_branches": 0}) == 100.0


def test_load_coverage(sample_coverage: Path) -> None:
    """Load coverage returns dict with expected top-level keys."""
    data = load_coverage(sample_coverage)
    assert "totals" in data
    assert "files" in data


def test_overall_summary(sample_coverage: Path) -> None:
    """Overall summary contains expected percentages."""
    data = load_coverage(sample_coverage)
    text = overall_summary(data)
    assert "30.0%" in text
    assert "33.3%" in text


def test_per_package_table(sample_coverage: Path) -> None:
    """Per-package table contains expected package names."""
    data = load_coverage(sample_coverage)
    table = per_package_table(data)
    assert "robot_sf/sim" in table
    assert "robot_sf/benchmark" in table
    assert "robot_sf/planner" in table


def test_worst_evidence_critical(sample_coverage: Path) -> None:
    """Worst evidence-critical table lists lowest-coverage modules first."""
    data = load_coverage(sample_coverage)
    table = worst_evidence_critical(data, n=2)
    assert "socnav.py" in table
    assert "map_runner.py" in table


def test_threshold_proposal(sample_coverage: Path) -> None:
    """Threshold proposal contains all five phases."""
    data = load_coverage(sample_coverage)
    text = threshold_proposal(data)
    assert "Phase 1" in text
    assert "Phase 5" in text
    assert "462" not in text  # should use actual count from data


def test_report_script_runs(sample_coverage: Path, capsys: pytest.CaptureFixture) -> None:
    """Main entrypoint produces expected report sections."""
    import sys
    from unittest.mock import patch

    from scripts.dev.branch_coverage_report import main

    with patch.object(sys, "argv", ["branch_coverage_report.py", "--json", str(sample_coverage)]):
        main()
    out = capsys.readouterr().out
    assert "Branch-Coverage Threshold Analysis Report" in out
    assert "Per-Package" in out
    assert "Worst-Covered" in out
