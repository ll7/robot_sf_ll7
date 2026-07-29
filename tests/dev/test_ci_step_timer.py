"""Tiny sanity check for the CI step timer helper."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_HAS_TIMEOUT = shutil.which("timeout") is not None


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


@pytest.mark.skipif(not _HAS_TIMEOUT, reason="GNU timeout(1) is required for timeout tests")
def test_ci_step_timer_timeout_does_not_affect_fast_command() -> None:
    """A small timeout should not change the outcome of a command that finishes quickly."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    env = os.environ.copy()
    env["CI_STEP_TIMEOUT_SECONDS"] = "5"
    result = subprocess.run(
        ["bash", str(script), "fast-timeout", "echo", "ok"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    assert "ok" in result.stdout
    assert 'ci_step_timer step_end label="fast-timeout" status=0 duration_seconds=' in result.stdout
    assert "::endgroup::" in result.stdout


@pytest.mark.skipif(not _HAS_TIMEOUT, reason="GNU timeout(1) is required for timeout tests")
def test_ci_step_timer_timeout_kills_long_command() -> None:
    """A short timeout must kill a long command and still report the step end."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    env = os.environ.copy()
    env["CI_STEP_TIMEOUT_SECONDS"] = "0.1"
    result = subprocess.run(
        ["bash", str(script), "slow-timeout", "sleep", "10"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 124
    assert "slow-timeout" in result.stdout
    assert "ci_step_timer step_end" in result.stdout
    assert "::notice" in result.stdout
    assert "::endgroup::" in result.stdout


def test_ci_step_timer_timeout_requires_gnu_timeout(tmp_path: Path) -> None:
    """If CI_STEP_TIMEOUT_SECONDS is set but timeout(1) is missing, fail clearly."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    bash_path = shutil.which("bash")
    date_path = shutil.which("date")
    true_path = shutil.which("true")
    assert bash_path and date_path and true_path, "required system binaries missing"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    os.symlink(date_path, fake_bin / "date")
    os.symlink(true_path, fake_bin / "true")

    env = os.environ.copy()
    env["CI_STEP_TIMEOUT_SECONDS"] = "1"
    env["PATH"] = str(fake_bin)
    result = subprocess.run(
        [bash_path, str(script), "missing-timeout", true_path],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 127
    assert "GNU timeout" in result.stderr
    assert "ci_step_timer step_end" in result.stdout
    assert "::endgroup::" in result.stdout
