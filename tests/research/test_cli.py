"""Smoke test for CLI invocation."""

import subprocess
import sys

import pytest


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory."""
    output_path = tmp_path / "cli_test"
    output_path.mkdir(exist_ok=True)
    return output_path


def test_generate_report_cli_help():
    """Test CLI help message."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.research.generate_report", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "--experiment-name" in result.stdout
    assert "--output" in result.stdout
    assert "--threshold" in result.stdout


def test_generate_report_cli_missing_args():
    """Test CLI with missing required arguments."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.research.generate_report"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Should fail with missing arguments
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "error" in result.stderr.lower()


def test_validate_report_cli_help():
    """Test validation CLI help message."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.tools.validate_report", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "--report-dir" in result.stdout


def test_validate_report_cli_missing_dir():
    """Test validation CLI with missing directory."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.tools.validate_report"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Should fail with missing arguments
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "error" in result.stderr.lower()


def test_validate_report_cli_nonexistent_dir():
    """Test validation CLI with nonexistent directory."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.tools.validate_report",
            "--report-dir",
            "/nonexistent/path",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # Should fail with directory not found
    assert result.returncode != 0
