"""Tiny sanity check for the CI step timer helper."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_ci_step_timer_shell_syntax():
    """Validate that the CI step timer helper passes bash syntax checks."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    assert script.exists(), "ci_step_timer.sh helper is missing"
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_ci_step_timer_help_flag():
    """--help prints usage and exits 0 without running any command."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "ci_step_timer step_start" not in result.stdout
    assert "ci_step_timer step_end" not in result.stdout
    assert "::group::" not in result.stdout


def test_ci_step_timer_h_flag():
    """-h prints usage and exits 0 without running any command."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script), "-h"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "ci_step_timer step_start" not in result.stdout
    assert "ci_step_timer step_end" not in result.stdout
    assert "::group::" not in result.stdout


def test_ci_step_timer_requires_label_and_command():
    """Ensure the helper exits with usage info when arguments are missing."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_ci_step_timer_propagates_failure():
    """Verify that a failing command is reflected in the reported status."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script), "failing-check", "false"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "failing-check" in result.stdout
    assert "::endgroup::" in result.stdout


def test_ci_step_timer_reports_success_duration():
    """Check that a successful step is reported with zero exit status and duration."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script), "echo-test", "echo", "hello"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "echo-test" in result.stdout
    assert "hello" in result.stdout
    assert 'ci_step_timer step_end label="echo-test" status=0 duration_seconds=' in result.stdout
