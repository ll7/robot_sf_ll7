"""Run shell-only INT regressions against the actual unsubmitted pilot script."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("trainer_mode", ["cooperative", "ignore_term"])
def test_shell_only_int_terminates_trainer_and_reaches_custody(trainer_mode):
    """Ignored INT and even ignored TERM must not strand the EXIT custody path."""
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "tests/training/shell/ftplan_pilot_int.sh"),
            str(ROOT / "SLURM/ftplan/released_ppo_pilot.slurm"),
            trainer_mode,
        ],
        capture_output=True,
        text=True,
        timeout=35,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
