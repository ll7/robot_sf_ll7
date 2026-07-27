"""Branch-coverage tests for ``robot_sf.coverage_tools.report_formatter``.

Locks terminal, JSON, and Markdown formatting for coverage/gap/trend/baseline
reports, the unknown-report-type fallback, empty/default payloads, gap top-10
truncation, baseline positive/zero/negative status direction, changed-file
truncation, coverage-decrease warnings, and the JSON ``default=str``
serialization path.

Assertions target stable semantic fields, headings, and numeric formatting
rather than incidental whitespace, so the tests survive benign layout changes.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

import pytest

from robot_sf.coverage_tools.report_formatter import (
    format_json_report,
    format_markdown_report,
    format_terminal_report,
)

# ---------------------------------------------------------------------------
# In-memory payload builders (no coverage data files are read or written).
# ---------------------------------------------------------------------------


def _gaps(count: int) -> list[dict[str, Any]]:
    """Build ``count`` synthetic gap entries for truncation tests."""
    return [
        {
            "file": f"mod/file_{i}.py",
            "coverage_percent": 50.0,
            "uncovered_lines": i,
            "priority_score": float(i),
        }
        for i in range(1, count + 1)
    ]


def _changed_files(count: int) -> list[dict[str, Any]]:
    """Build ``count`` synthetic changed-file entries for truncation tests."""
    return [
        {"file": f"mod/changed_{i}.py", "current": 60.0, "baseline": 80.0, "delta": -20.0}
        for i in range(1, count + 1)
    ]


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------


def test_terminal_coverage_report_formats_totals(sample_coverage_data):
    """Terminal coverage report renders totals with two-decimal percentage."""
    out = format_terminal_report(sample_coverage_data, "coverage")
    assert "Coverage Summary" in out
    assert "Total Lines: 18" in out
    assert "Covered Lines: 12" in out
    assert "Coverage: 66.67%" in out


def test_terminal_default_report_type_is_coverage(sample_coverage_data):
    """Omitting ``report_type`` defaults to the coverage formatter."""
    out = format_terminal_report(sample_coverage_data)
    assert "Coverage Summary" in out
    assert "Coverage: 66.67%" in out


def test_terminal_coverage_empty_payload_uses_defaults():
    """An empty payload falls back to zero totals without error."""
    out = format_terminal_report({}, "coverage")
    assert "Total Lines: 0" in out
    assert "Covered Lines: 0" in out
    assert "Coverage: 0.00%" in out


def test_markdown_coverage_report_formats_totals(sample_coverage_data):
    """Markdown coverage report renders a summary with two-decimal percentage."""
    out = format_markdown_report(sample_coverage_data, "coverage")
    assert "# Coverage Summary" in out
    assert "**Total Lines**: 18" in out
    assert "**Covered Lines**: 12" in out
    assert "**Coverage**: 66.67%" in out


def test_markdown_default_report_type_is_coverage(sample_coverage_data):
    """Omitting ``report_type`` defaults to the Markdown coverage formatter."""
    out = format_markdown_report(sample_coverage_data)
    assert "# Coverage Summary" in out
    assert "**Coverage**: 66.67%" in out


def test_markdown_coverage_empty_payload_uses_defaults():
    """An empty Markdown coverage payload falls back to zero totals without error."""
    out = format_markdown_report({}, "coverage")
    assert "**Total Lines**: 0" in out
    assert "**Covered Lines**: 0" in out
    assert "**Coverage**: 0.00%" in out


# ---------------------------------------------------------------------------
# Gap report
# ---------------------------------------------------------------------------


def test_terminal_gap_report_lists_gaps(sample_gap_data):
    """Terminal gap report enumerates each gap with one-decimal coverage."""
    out = format_terminal_report(sample_gap_data, "gap")
    assert "Top Coverage Gaps (2 found)" in out
    assert "1. robot_sf/gym_env/environment.py" in out
    assert "2. robot_sf/sim/simulator.py" in out
    assert "Coverage: 70.0% | Uncovered: 3 lines" in out
    assert "Coverage: 62.5% | Uncovered: 3 lines" in out
    assert "Priority Score: 4.5" in out


def test_markdown_gap_report_builds_table(sample_gap_data):
    """Markdown gap report renders a table with stable headers and one-decimal fields."""
    out = format_markdown_report(sample_gap_data, "gap")
    assert "# Coverage Gaps (2 found)" in out
    assert "| # | File | Coverage | Uncovered Lines | Priority |" in out
    assert "| 1 | `robot_sf/gym_env/environment.py` | 70.0% | 3 | 4.5 |" in out
    assert "| 2 | `robot_sf/sim/simulator.py` | 62.5% | 3 | 4.5 |" in out


def test_terminal_gap_report_empty_payload():
    """An empty gap payload reports zero gaps and lists none."""
    out = format_terminal_report({}, "gap")
    assert "Top Coverage Gaps (0 found)" in out


def test_markdown_gap_report_empty_payload():
    """An empty gap payload still renders the table header with zero rows."""
    out = format_markdown_report({}, "gap")
    assert "# Coverage Gaps (0 found)" in out
    assert "| # | File | Coverage | Uncovered Lines | Priority |" in out


def test_terminal_gap_report_truncates_to_top_10():
    """Gap terminal report keeps only the first 10 gaps regardless of total count."""
    data = {"gaps": _gaps(12)}
    out = format_terminal_report(data, "gap")
    assert "Top Coverage Gaps (12 found)" in out
    assert "mod/file_10.py" in out
    assert "mod/file_11.py" not in out
    assert "mod/file_12.py" not in out


def test_markdown_gap_report_truncates_to_top_10():
    """Gap Markdown report keeps only the first 10 table rows."""
    data = {"gaps": _gaps(12)}
    out = format_markdown_report(data, "gap")
    assert "Coverage Gaps (12 found)" in out
    assert "mod/file_10.py" in out
    assert "mod/file_11.py" not in out
    assert "mod/file_12.py" not in out


# ---------------------------------------------------------------------------
# Trend report
# ---------------------------------------------------------------------------


def test_terminal_trend_report(sample_trend_data):
    """Terminal trend report renders direction and signed weekly rate."""
    out = format_terminal_report(sample_trend_data, "trend")
    assert "Coverage Trend Analysis" in out
    assert "Current Coverage: 66.67%" in out
    assert "Oldest Coverage: 60.00%" in out
    assert "Trend Direction: improving" in out
    assert "Rate: +0.50% per week" in out


def test_markdown_trend_report(sample_trend_data):
    """Markdown trend report renders direction and signed weekly rate."""
    out = format_markdown_report(sample_trend_data, "trend")
    assert "# Coverage Trend Analysis" in out
    assert "**Current Coverage**: 66.67%" in out
    assert "**Oldest Coverage**: 60.00%" in out
    assert "**Trend Direction**: improving" in out
    assert "**Rate**: +0.50% per week" in out


def test_terminal_trend_empty_payload():
    """An empty trend payload uses defaults: zero coverage and unknown direction."""
    out = format_terminal_report({}, "trend")
    assert "Current Coverage: 0.00%" in out
    assert "Oldest Coverage: 0.00%" in out
    assert "Trend Direction: unknown" in out
    assert "Rate: +0.00% per week" in out


def test_markdown_trend_empty_payload_uses_defaults():
    """An empty Markdown trend payload uses zero values and unknown direction."""
    out = format_markdown_report({}, "trend")
    assert "**Current Coverage**: 0.00%" in out
    assert "**Oldest Coverage**: 0.00%" in out
    assert "**Trend Direction**: unknown" in out
    assert "**Rate**: +0.00% per week" in out


# ---------------------------------------------------------------------------
# Baseline report
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta", "terminal_status", "markdown_status", "expect_warning"),
    [
        pytest.param(-3.33, "DECREASED", "🔻 DECREASED", True, id="negative"),
        pytest.param(3.33, "INCREASED", "🔺 INCREASED", False, id="positive"),
        pytest.param(0.0, "UNCHANGED", "➡️ UNCHANGED", False, id="zero"),
    ],
)
def test_baseline_status_direction(delta, terminal_status, markdown_status, expect_warning):
    """Baseline status direction covers decrease/increase/unchanged in both formats."""
    data = {"current_coverage": 66.67, "baseline_coverage": 70.0, "delta": delta}
    signed = f"{delta:+.2f}"

    term = format_terminal_report(data, "baseline")
    md = format_markdown_report(data, "baseline")

    assert f"Change: {signed}% ({terminal_status})" in term
    assert f"**Change**: {signed}% ({markdown_status})" in md

    if expect_warning:
        assert "WARNING: Coverage has decreased!" in term
        assert "## ⚠️ Warning: Coverage Decreased" in md
    else:
        assert "WARNING" not in term
        assert "Warning: Coverage Decreased" not in md


def test_terminal_baseline_decreased_with_changed_files(sample_baseline_data):
    """A decrease emits the warning and lists the affected file deltas."""
    out = format_terminal_report(sample_baseline_data, "baseline")
    assert "Coverage Baseline Comparison" in out
    assert "Baseline Coverage: 70.00%" in out
    assert "Current Coverage: 66.67%" in out
    assert "Change: -3.33% (DECREASED)" in out
    assert "WARNING: Coverage has decreased!" in out
    assert "Affected files:" in out
    assert "robot_sf/gym_env/environment.py" in out
    assert "-5.00%" in out


def test_markdown_baseline_decreased_with_changed_files(sample_baseline_data):
    """A decrease emits the Markdown warning section and affected-file deltas."""
    out = format_markdown_report(sample_baseline_data, "baseline")
    assert "# Coverage Baseline Comparison" in out
    assert "**Baseline Coverage**: 70.00%" in out
    assert "**Current Coverage**: 66.67%" in out
    assert "**Change**: -3.33% (🔻 DECREASED)" in out
    assert "## ⚠️ Warning: Coverage Decreased" in out
    assert "### Affected Files" in out
    assert "`robot_sf/gym_env/environment.py`" in out
    assert "-5.00%" in out


def test_terminal_baseline_decreased_without_changed_files():
    """A decrease with no changed files warns but omits the affected-files list."""
    data = {"current_coverage": 60.0, "baseline_coverage": 80.0, "delta": -20.0}
    out = format_terminal_report(data, "baseline")
    assert "Change: -20.00% (DECREASED)" in out
    assert "WARNING: Coverage has decreased!" in out
    assert "Affected files:" not in out


def test_markdown_baseline_decreased_without_changed_files():
    """A Markdown decrease with no changed files warns but omits affected files."""
    data = {"current_coverage": 60.0, "baseline_coverage": 80.0, "delta": -20.0}
    out = format_markdown_report(data, "baseline")
    assert "**Change**: -20.00% (🔻 DECREASED)" in out
    assert "## ⚠️ Warning: Coverage Decreased" in out
    assert "### Affected Files" not in out


def test_terminal_baseline_truncates_changed_files_to_top_5():
    """Terminal baseline report keeps only the first five changed files."""
    data = {
        "current_coverage": 60.0,
        "baseline_coverage": 80.0,
        "delta": -20.0,
        "changed_files": _changed_files(7),
    }
    out = format_terminal_report(data, "baseline")
    assert "mod/changed_5.py" in out
    assert "mod/changed_6.py" not in out
    assert "mod/changed_7.py" not in out


def test_markdown_baseline_truncates_changed_files_to_top_5():
    """Markdown baseline report keeps only the first five changed files."""
    data = {
        "current_coverage": 60.0,
        "baseline_coverage": 80.0,
        "delta": -20.0,
        "changed_files": _changed_files(7),
    }
    out = format_markdown_report(data, "baseline")
    assert "mod/changed_5.py" in out
    assert "mod/changed_6.py" not in out
    assert "mod/changed_7.py" not in out


def test_terminal_baseline_empty_payload_is_unchanged():
    """An empty baseline payload defaults to zero coverage and unchanged status."""
    out = format_terminal_report({}, "baseline")
    assert "Baseline Coverage: 0.00%" in out
    assert "Current Coverage: 0.00%" in out
    assert "Change: +0.00% (UNCHANGED)" in out
    assert "WARNING" not in out


def test_markdown_baseline_empty_payload_is_unchanged():
    """An empty Markdown baseline payload defaults to zero values and no warning."""
    out = format_markdown_report({}, "baseline")
    assert "**Baseline Coverage**: 0.00%" in out
    assert "**Current Coverage**: 0.00%" in out
    assert "**Change**: +0.00% (➡️ UNCHANGED)" in out
    assert "Warning: Coverage Decreased" not in out


# ---------------------------------------------------------------------------
# Unknown report type
# ---------------------------------------------------------------------------


def test_terminal_unknown_report_type():
    """Unknown report types fall back to a stable terminal message."""
    assert format_terminal_report({}, "bogus") == "Unknown report type: bogus"


def test_markdown_unknown_report_type():
    """Unknown report types fall back to a stable Markdown message."""
    assert format_markdown_report({}, "bogus") == "Unknown report type: bogus"


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


def test_json_report_round_trips(sample_coverage_data):
    """JSON output parses back into the original structure with native numerics."""
    out = format_json_report(sample_coverage_data, "coverage")
    parsed = json.loads(out)
    assert parsed["totals"]["percent_covered"] == 66.67
    assert parsed["totals"]["num_statements"] == 18
    assert parsed["totals"]["covered_lines"] == 12


def test_json_report_serializes_non_native_values_via_default_str():
    """Non-JSON-native values are rendered through the documented default=str path."""
    moment = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    data = {"snapshot_at": moment, "totals": {"percent_covered": 50.0}}

    # Without default=str the datetime is not JSON serializable, proving the
    # formatter's default=str argument is the load-bearing serialization path.
    with pytest.raises(TypeError):
        json.dumps(data)

    parsed = json.loads(format_json_report(data))
    assert parsed["snapshot_at"] == str(moment)
    # Native numerics remain typed, not stringified.
    assert parsed["totals"]["percent_covered"] == 50.0
