"""
Coverage analysis and quality tracking tools.

This package provides utilities for:
- Coverage gap analysis and prioritization
- Historical coverage trend tracking
- CI/CD baseline comparison and warnings
- Multiple report formats (terminal, JSON, markdown)

All modules follow the library-first principle with no direct I/O;
orchestration is handled by CLI scripts in scripts/coverage/.
"""

from robot_sf.coverage_tools.report_formatter import (
    format_json_report,
    format_markdown_report,
    format_terminal_report,
)

__all__ = [
    "format_json_report",
    "format_markdown_report",
    "format_terminal_report",
]
