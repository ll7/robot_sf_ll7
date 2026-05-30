"""Tiny sanity check for the CI step timer helper."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_ci_step_timer_shell_syntax():
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    assert script.exists(), "ci_step_timer.sh helper is missing"
    assert subprocess.run(["bash", "-n", str(script)], check=True).returncode == 0


def test_ci_step_timer_requires_label_and_command():
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_ci_step_timer_propagates_failure():
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script), "failing-check", "false"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "failing-check" in result.stdout
    assert "::endgroup::" in result.stdout


def test_ci_step_timer_reports_success_duration():
    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_step_timer.sh"
    result = subprocess.run(
        ["bash", str(script), "echo-test", "echo", "hello"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "echo-test" in result.stdout
    assert "hello" in result.stdout
    assert "ci_step_timer step_end label=echo-test status=0 duration_seconds=" in result.stdout
