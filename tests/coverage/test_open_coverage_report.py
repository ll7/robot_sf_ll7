"""Tests for scripts/coverage/open_coverage_report.py."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from scripts.coverage.open_coverage_report import main, open_coverage_report

if TYPE_CHECKING:
    from pathlib import Path


def test_missing_report_prints_truthful_remediation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing coverage report explains how to generate it with the canonical command."""
    missing = tmp_path / "htmlcov" / "index.html"
    exit_code = open_coverage_report(missing)

    assert exit_code == 1
    err = capsys.readouterr().err
    assert f"Coverage report not found: {missing}" in err
    assert "ROBOT_SF_PYTEST_COVERAGE=1 scripts/dev/run_tests_parallel.sh --lane all" in err


def test_successful_browser_open(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """When webbrowser.open returns True, exit 0 and report success."""
    report = tmp_path / "index.html"
    report.write_text("<html><body>Coverage</body></html>", encoding="utf-8")

    with patch("webbrowser.open", return_value=True) as mock_open:
        exit_code = open_coverage_report(report)

    assert exit_code == 0
    mock_open.assert_called_once_with(report.resolve().as_uri(), new=2)
    out = capsys.readouterr().out
    assert f"Coverage report opened: {report.resolve()}" in out


def test_browser_open_returns_false(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """When webbrowser.open returns False, exit 1 and instruct manual opening."""
    report = tmp_path / "index.html"
    report.write_text("<html><body>Coverage</body></html>", encoding="utf-8")

    with patch("webbrowser.open", return_value=False) as mock_open:
        exit_code = open_coverage_report(report)

    assert exit_code == 1
    mock_open.assert_called_once_with(report.resolve().as_uri(), new=2)
    err = capsys.readouterr().err
    assert "Failed to open browser automatically" in err
    assert f"Please open manually: {report.resolve()}" in err


def test_browser_open_raises_exception(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """When webbrowser.open raises an exception, catch it and instruct manual opening."""
    report = tmp_path / "index.html"
    report.write_text("<html><body>Coverage</body></html>", encoding="utf-8")

    with patch("webbrowser.open", side_effect=RuntimeError("Browser launch failed")):
        exit_code = open_coverage_report(report)

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Error opening browser: Browser launch failed" in err
    assert f"Please open manually: {report.resolve()}" in err


def test_main_cli_with_explicit_path(tmp_path: Path) -> None:
    """main() forwards --path to open_coverage_report."""
    report = tmp_path / "custom_cov" / "index.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("<html></html>", encoding="utf-8")

    with (
        patch.object(sys, "argv", ["open_coverage_report.py", "--path", str(report)]),
        patch("webbrowser.open", return_value=True) as mock_open,
    ):
        exit_code = main()

    assert exit_code == 0
    mock_open.assert_called_once_with(report.resolve().as_uri(), new=2)


def test_main_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """main() displays help text including default path description."""
    with (
        patch.object(sys, "argv", ["open_coverage_report.py", "--help"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "Open coverage HTML report in browser" in out
    assert "output/coverage/htmlcov/index.html" in out
