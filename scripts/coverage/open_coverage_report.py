#!/usr/bin/env python3
"""Cross-platform browser launcher for coverage reports.

Purpose: Opens the coverage HTML report in the default browser,
working reliably on macOS, Linux, and Windows.

Usage:
    python scripts/coverage/open_coverage_report.py [--path htmlcov/index.html]
"""

import argparse
import sys
import webbrowser
from pathlib import Path

from robot_sf.common.artifact_paths import get_artifact_category_path


def open_coverage_report(report_path: Path) -> int:
    """
    Open coverage report in default browser.

    Args:
        report_path: Path to HTML coverage report

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if not report_path.exists():
        print(f"❌ Coverage report not found: {report_path}", file=sys.stderr)
        print("\nRun 'uv run pytest tests' first to generate coverage data", file=sys.stderr)
        return 1

    # Convert to absolute path for reliable browser opening
    abs_path = report_path.resolve()

    # Use file:// URL for cross-platform compatibility
    url = abs_path.as_uri()

    try:
        # Open in default browser (new=2 means new tab if possible)
        success = webbrowser.open(url, new=2)

        if success:
            print(f"✅ Coverage report opened: {abs_path}")
            return 0
        else:
            print("⚠️  Failed to open browser automatically", file=sys.stderr)
            print(f"Please open manually: {abs_path}", file=sys.stderr)
            return 1

    except Exception as e:
        print(f"❌ Error opening browser: {e}", file=sys.stderr)
        print(f"Please open manually: {abs_path}", file=sys.stderr)
        return 1


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Open coverage HTML report in browser")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help=(
            "Path to coverage report (default: output/coverage/htmlcov/index.html via artifact helper)"
        ),
    )

    args = parser.parse_args()
    default_path = get_artifact_category_path("coverage") / "htmlcov/index.html"
    target = args.path or default_path
    return open_coverage_report(target)


if __name__ == "__main__":
    sys.exit(main())
