"""Tests for the dependency-only linked-worktree preflight."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "dev" / "check_worktree_optional_deps.py"
VALIDATOR = REPO_ROOT / "scripts" / "dev" / "validate_worktree_optional_deps.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _validate(
    report: dict[str, object], observed_exit_code: int
) -> subprocess.CompletedProcess[str]:
    """Run the report validator against one captured probe payload."""
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--observed-exit-code", str(observed_exit_code)],
        cwd=REPO_ROOT,
        input=json.dumps(report),
        capture_output=True,
        text=True,
        check=False,
    )


def test_custom_profile_distinguishes_missing_optional_imports() -> None:
    """Missing imports get a setup-specific status and exit code."""
    result = _run(
        "--module",
        "json",
        "--module",
        "robot_sf_ll7_module_that_does_not_exist",
        "--json",
    )

    assert result.returncode == 2, result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "robot_sf.worktree_optional_deps.v1"
    assert report["status"] == "missing_optional"
    assert report["project_imports_performed"] is False
    assert report["missing_optional"] == ["robot_sf_ll7_module_that_does_not_exist"]
    assert (
        "environment/setup evidence"
        in _run("--module", "robot_sf_ll7_module_that_does_not_exist").stdout
    )


def test_available_custom_profile_is_ready() -> None:
    """A dependency-only probe succeeds without importing project code."""
    result = _run("--module", "json", "--json")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "ready"
    assert report["missing_optional"] == []
    assert report["project_imports_performed"] is False


def test_all_extras_profile_reports_current_environment() -> None:
    """The bootstrapped development environment satisfies the documented profile."""
    result = _run("--profile", "all-extras", "--json")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["profile"] == "all-extras"
    assert report["checked_count"] >= 30
    assert report["status"] == "ready"


def test_named_training_profile_matches_training_extra() -> None:
    """A named bootstrap extra can use the corresponding focused profile."""
    result = _run("--profile", "training", "--json")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["profile"] == "training"
    assert {check["module"] for check in report["checks"]} == {
        "stable_baselines3",
        "torch",
        "sklearn",
        "optuna",
        "tensorboard",
        "wandb",
        "optuna_dashboard",
    }


@pytest.mark.parametrize(
    ("status", "exit_code", "missing_optional", "check_failures", "expected"),
    [
        ("ready", 0, [], [], 0),
        ("missing_optional", 2, ["pandas"], [], 2),
        ("check_failed", 1, [], ["pandas"], 1),
    ],
)
def test_validator_accepts_matching_status_and_exit_contract(
    status: str,
    exit_code: int,
    missing_optional: list[str],
    check_failures: list[str],
    expected: int,
) -> None:
    """Only a matching report/exit pair reaches its semantic exit code."""
    result = _validate(
        {
            "schema": "robot_sf.worktree_optional_deps.v1",
            "profile": "all-extras",
            "status": status,
            "exit_code": exit_code,
            "missing_optional": missing_optional,
            "check_failures": check_failures,
            "project_imports_performed": False,
        },
        exit_code,
    )
    assert result.returncode == expected, result.stderr


@pytest.mark.parametrize(
    ("report", "observed_exit_code"),
    [
        (
            {
                "schema": "robot_sf.worktree_optional_deps.v1",
                "profile": "all-extras",
                "status": "ready",
                "exit_code": 0,
                "missing_optional": [],
                "check_failures": [],
                "project_imports_performed": False,
            },
            1,
        ),
        ({"status": "ready"}, 0),
        (
            {
                "schema": "robot_sf.worktree_optional_deps.v1",
                "profile": "all-extras",
                "status": "ready",
                "exit_code": 7,
                "missing_optional": [],
                "check_failures": [],
                "project_imports_performed": False,
            },
            7,
        ),
    ],
    ids=["status-exit-disagreement", "malformed-fields", "unknown-exit"],
)
def test_validator_rejects_invalid_status_exit_contract(
    report: dict[str, object], observed_exit_code: int
) -> None:
    """Malformed, disagreeing, and unknown probe outcomes fail as tool errors."""
    result = _validate(report, observed_exit_code)
    assert result.returncode == 1
    assert '"status": "invalid"' in result.stderr
