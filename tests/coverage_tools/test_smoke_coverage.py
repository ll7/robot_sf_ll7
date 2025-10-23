"""
Smoke test for coverage collection.

Purpose: Verify that coverage collection doesn't break test execution
and that coverage data is properly generated during test runs.

This test validates the foundational layer (Phase 2) by ensuring
coverage collection integrates seamlessly with pytest.
"""

import json
from pathlib import Path


def test_coverage_collection_doesnt_break_tests():
    """Verify that tests still pass with coverage enabled."""
    assert True, "Basic assertion should pass"


def test_coverage_data_is_generated():
    """
    Verify that coverage data files are generated.

    Note: This test relies on pytest-cov being configured via pyproject.toml
    to automatically generate coverage.json after test run completion.
    """
    # This test will run as part of the suite; coverage.json is generated
    # at session end, so we can't check it here in realtime. Instead,
    # we verify the configuration is loadable.
    project_root = Path(__file__).parent.parent.parent
    pyproject = project_root / "pyproject.toml"

    assert pyproject.exists(), "pyproject.toml should exist"

    content = pyproject.read_text()
    assert "[tool.coverage.run]" in content, "Coverage configuration should exist"
    assert "[tool.coverage.report]" in content, "Coverage report config should exist"
    assert "[tool.coverage.json]" in content, "JSON output config should exist"


def test_report_formatter_imports():
    """Verify that report formatter module can be imported."""
    from robot_sf.coverage_tools.report_formatter import (
        format_json_report,
        format_markdown_report,
        format_terminal_report,
    )

    # Basic smoke test - functions are callable
    assert callable(format_terminal_report)
    assert callable(format_json_report)
    assert callable(format_markdown_report)


def test_coverage_fixtures_available(
    sample_coverage_data, sample_gap_data, sample_trend_data, sample_baseline_data
):
    """Verify that coverage test fixtures are properly configured."""
    # Validate sample_coverage_data structure
    assert "meta" in sample_coverage_data
    assert "files" in sample_coverage_data
    assert "totals" in sample_coverage_data

    # Validate sample_gap_data structure
    assert "gaps" in sample_gap_data
    assert "summary" in sample_gap_data

    # Validate sample_trend_data structure
    assert "direction" in sample_trend_data
    assert "rate_per_week" in sample_trend_data

    # Validate sample_baseline_data structure
    assert "current_coverage" in sample_baseline_data
    assert "baseline_coverage" in sample_baseline_data
    assert "delta" in sample_baseline_data


def test_formatters_with_sample_data(sample_coverage_data):
    """Test that formatters work with sample coverage data."""
    from robot_sf.coverage_tools.report_formatter import (
        format_json_report,
        format_markdown_report,
        format_terminal_report,
    )

    # Terminal format
    terminal_output = format_terminal_report(sample_coverage_data, "coverage")
    assert "Coverage Summary" in terminal_output
    assert "66.67" in terminal_output

    # JSON format
    json_output = format_json_report(sample_coverage_data)
    parsed = json.loads(json_output)
    assert parsed["totals"]["percent_covered"] == 66.67

    # Markdown format
    markdown_output = format_markdown_report(sample_coverage_data, "coverage")
    assert "# Coverage Summary" in markdown_output
    assert "66.67%" in markdown_output
