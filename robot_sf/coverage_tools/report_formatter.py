"""
Report formatting utilities for coverage analysis.

Provides multiple output formats for coverage reports:
- Terminal: Human-readable colored output for console
- JSON: Machine-readable structured data
- Markdown: Documentation-friendly tables and summaries

All formatters are pure functions with no I/O side effects,
following the library-first principle (Constitution XI).
"""

import json
from typing import Any


def format_terminal_report(data: dict[str, Any], report_type: str = "coverage") -> str:
    """
    Format coverage data for terminal display with basic structure.

    Args:
        data: Coverage data dictionary
        report_type: Type of report ('coverage', 'gap', 'trend', 'baseline')

    Returns:
        Formatted string for terminal output

    Side Effects:
        None (pure function)
    """
    if report_type == "coverage":
        return _format_coverage_terminal(data)
    if report_type == "gap":
        return _format_gap_terminal(data)
    if report_type == "trend":
        return _format_trend_terminal(data)
    if report_type == "baseline":
        return _format_baseline_terminal(data)
    return f"Unknown report type: {report_type}"


def format_json_report(data: dict[str, Any], _report_type: str = "coverage") -> str:
    """
    Format coverage data as JSON string.

    Args:
        data: Coverage data dictionary
        _report_type: Type of report (reserved for future use)

    Returns:
        JSON-formatted string

    Side Effects:
        None (pure function)
    """
    return json.dumps(data, indent=2, default=str)


def format_markdown_report(data: dict[str, Any], report_type: str = "coverage") -> str:
    """
    Format coverage data as Markdown tables and text.

    Args:
        data: Coverage data dictionary
        report_type: Type of report ('coverage', 'gap', 'trend', 'baseline')

    Returns:
        Markdown-formatted string

    Side Effects:
        None (pure function)
    """
    if report_type == "coverage":
        return _format_coverage_markdown(data)
    if report_type == "gap":
        return _format_gap_markdown(data)
    if report_type == "trend":
        return _format_trend_markdown(data)
    if report_type == "baseline":
        return _format_baseline_markdown(data)
    return f"Unknown report type: {report_type}"


# Internal helper functions


def _format_coverage_terminal(data: dict[str, Any]) -> str:
    """Format basic coverage summary for terminal."""
    total = data.get("totals", {})
    covered = total.get("covered_lines", 0)
    total_lines = total.get("num_statements", 0)
    percent = total.get("percent_covered", 0.0)

    lines = [
        "=" * 60,
        "Coverage Summary",
        "=" * 60,
        f"Total Lines: {total_lines}",
        f"Covered Lines: {covered}",
        f"Coverage: {percent:.2f}%",
        "=" * 60,
    ]
    return "\n".join(lines)


def _format_gap_terminal(data: dict[str, Any]) -> str:
    """Format gap analysis for terminal."""
    gaps = data.get("gaps", [])
    lines = [
        "=" * 60,
        f"Top Coverage Gaps ({len(gaps)} found)",
        "=" * 60,
    ]

    for i, gap in enumerate(gaps[:10], 1):
        file_path = gap.get("file", "unknown")
        coverage_pct = gap.get("coverage_percent", 0.0)
        uncovered = gap.get("uncovered_lines", 0)
        priority = gap.get("priority_score", 0.0)

        lines.append(f"\n{i}. {file_path}")
        lines.append(f"   Coverage: {coverage_pct:.1f}% | Uncovered: {uncovered} lines")
        lines.append(f"   Priority Score: {priority:.1f}")

    lines.append("=" * 60)
    return "\n".join(lines)


def _format_trend_terminal(data: dict[str, Any]) -> str:
    """Format trend report for terminal."""
    direction = data.get("direction", "unknown")
    rate = data.get("rate_per_week", 0.0)
    current = data.get("current_coverage", 0.0)
    oldest = data.get("oldest_coverage", 0.0)

    lines = [
        "=" * 60,
        "Coverage Trend Analysis",
        "=" * 60,
        f"Current Coverage: {current:.2f}%",
        f"Oldest Coverage: {oldest:.2f}%",
        f"Trend Direction: {direction}",
        f"Rate: {rate:+.2f}% per week",
        "=" * 60,
    ]
    return "\n".join(lines)


def _format_baseline_terminal(data: dict[str, Any]) -> str:
    """Format baseline comparison for terminal."""
    current = data.get("current_coverage", 0.0)
    baseline = data.get("baseline_coverage", 0.0)
    delta = data.get("delta", 0.0)
    status = "DECREASED" if delta < 0 else "INCREASED" if delta > 0 else "UNCHANGED"

    lines = [
        "=" * 60,
        "Coverage Baseline Comparison",
        "=" * 60,
        f"Baseline Coverage: {baseline:.2f}%",
        f"Current Coverage: {current:.2f}%",
        f"Change: {delta:+.2f}% ({status})",
        "=" * 60,
    ]

    if delta < 0:
        lines.append("\n⚠️  WARNING: Coverage has decreased!")
        changed_files = data.get("changed_files", [])
        if changed_files:
            lines.append("\nAffected files:")
            for file_info in changed_files[:5]:
                fname = file_info.get("file", "unknown")
                file_delta = file_info.get("delta", 0.0)
                lines.append(f"  - {fname}: {file_delta:+.2f}%")

    return "\n".join(lines)


def _format_coverage_markdown(data: dict[str, Any]) -> str:
    """Format coverage summary as Markdown."""
    total = data.get("totals", {})
    covered = total.get("covered_lines", 0)
    total_lines = total.get("num_statements", 0)
    percent = total.get("percent_covered", 0.0)

    lines = [
        "# Coverage Summary",
        "",
        f"- **Total Lines**: {total_lines}",
        f"- **Covered Lines**: {covered}",
        f"- **Coverage**: {percent:.2f}%",
    ]
    return "\n".join(lines)


def _format_gap_markdown(data: dict[str, Any]) -> str:
    """Format gap analysis as Markdown table."""
    gaps = data.get("gaps", [])
    lines = [
        f"# Coverage Gaps ({len(gaps)} found)",
        "",
        "| # | File | Coverage | Uncovered Lines | Priority |",
        "|---|------|----------|-----------------|----------|",
    ]

    for i, gap in enumerate(gaps[:10], 1):
        file_path = gap.get("file", "unknown")
        coverage_pct = gap.get("coverage_percent", 0.0)
        uncovered = gap.get("uncovered_lines", 0)
        priority = gap.get("priority_score", 0.0)

        lines.append(
            f"| {i} | `{file_path}` | {coverage_pct:.1f}% | {uncovered} | {priority:.1f} |"
        )

    return "\n".join(lines)


def _format_trend_markdown(data: dict[str, Any]) -> str:
    """Format trend report as Markdown."""
    direction = data.get("direction", "unknown")
    rate = data.get("rate_per_week", 0.0)
    current = data.get("current_coverage", 0.0)
    oldest = data.get("oldest_coverage", 0.0)

    lines = [
        "# Coverage Trend Analysis",
        "",
        f"- **Current Coverage**: {current:.2f}%",
        f"- **Oldest Coverage**: {oldest:.2f}%",
        f"- **Trend Direction**: {direction}",
        f"- **Rate**: {rate:+.2f}% per week",
    ]
    return "\n".join(lines)


def _format_baseline_markdown(data: dict[str, Any]) -> str:
    """Format baseline comparison as Markdown."""
    current = data.get("current_coverage", 0.0)
    baseline = data.get("baseline_coverage", 0.0)
    delta = data.get("delta", 0.0)
    status = "🔻 DECREASED" if delta < 0 else "🔺 INCREASED" if delta > 0 else "➡️ UNCHANGED"

    lines = [
        "# Coverage Baseline Comparison",
        "",
        f"- **Baseline Coverage**: {baseline:.2f}%",
        f"- **Current Coverage**: {current:.2f}%",
        f"- **Change**: {delta:+.2f}% ({status})",
    ]

    if delta < 0:
        lines.append("")
        lines.append("## ⚠️ Warning: Coverage Decreased")
        changed_files = data.get("changed_files", [])
        if changed_files:
            lines.append("")
            lines.append("### Affected Files")
            lines.append("")
            for file_info in changed_files[:5]:
                fname = file_info.get("file", "unknown")
                file_delta = file_info.get("delta", 0.0)
                lines.append(f"- `{fname}`: {file_delta:+.2f}%")

    return "\n".join(lines)
