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


def test_check_entry_points_matching_current_environment() -> None:
    """Issue #9591: declared entry points match the installed package metadata."""
    result = _run("--profile", "core", "--check-entry-points", "--json")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "ready"
    assert "entry_points" in report
    ep = report["entry_points"]
    assert ep["status"] == "ready"
    assert ep["installed"] is True
    assert ep["missing"] == []
    assert ep["mismatched"] == []
    assert len(ep["matching"]) >= 2
    matching_names = {item["name"] for item in ep["matching"]}
    assert {"srev29-example-analyzer", "srev29-example-renderer"} <= matching_names


def test_check_entry_points_missing_declared_fails(tmp_path: Path) -> None:
    """Issue #9591: missing declared entry point fails with exit code 2 and remedy."""
    fake_pyproject = tmp_path / "pyproject.toml"
    fake_pyproject.write_text(
        """
[project]
name = "robot_sf"

[project.entry-points."robot_sf.scenario_review"]
srev29-example-analyzer = "examples.scenario_review.components.episode_analyzer:DESCRIPTOR"
srev29-example-renderer = "examples.scenario_review.components.telemetry_renderer:DESCRIPTOR"
srev29-missing-plugin = "robot_sf.plugins:MISSING"
""",
        encoding="utf-8",
    )

    result = _run(
        "--profile",
        "core",
        "--check-entry-points",
        "--pyproject",
        str(fake_pyproject),
        "--json",
    )

    assert result.returncode == 2, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "missing_entry_points"
    assert report["exit_code"] == 2
    ep = report["entry_points"]
    assert ep["status"] == "missing_entry_points"
    assert len(ep["missing"]) == 1
    assert ep["missing"][0]["name"] == "srev29-missing-plugin"
    assert ep["missing"][0]["expected_value"] == "robot_sf.plugins:MISSING"

    human_result = _run(
        "--profile",
        "core",
        "--check-entry-points",
        "--pyproject",
        str(fake_pyproject),
    )
    assert human_result.returncode == 2
    assert "Missing declared entry points:" in human_result.stdout
    assert "srev29-missing-plugin" in human_result.stdout
    assert "Remedy: install this checkout into the active environment" in human_result.stdout


def test_check_entry_points_mismatched_declared_fails(tmp_path: Path) -> None:
    """Issue #9591: mismatched declared entry point fails with exit code 2 and remedy."""
    fake_pyproject = tmp_path / "pyproject.toml"
    fake_pyproject.write_text(
        """
[project]
name = "robot_sf"

[project.entry-points."robot_sf.scenario_review"]
srev29-example-analyzer = "examples.scenario_review.components.episode_analyzer:WRONG_TARGET"
srev29-example-renderer = "examples.scenario_review.components.telemetry_renderer:DESCRIPTOR"
""",
        encoding="utf-8",
    )

    result = _run(
        "--profile",
        "core",
        "--check-entry-points",
        "--pyproject",
        str(fake_pyproject),
        "--json",
    )

    assert result.returncode == 2, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "mismatched_entry_points"
    assert report["exit_code"] == 2
    ep = report["entry_points"]
    assert ep["status"] == "mismatched_entry_points"
    assert len(ep["mismatched"]) == 1
    assert ep["mismatched"][0]["name"] == "srev29-example-analyzer"
    assert ep["mismatched"][0]["expected_value"] == (
        "examples.scenario_review.components.episode_analyzer:WRONG_TARGET"
    )
    assert ep["mismatched"][0]["actual_value"] == (
        "examples.scenario_review.components.episode_analyzer:DESCRIPTOR"
    )

    human_result = _run(
        "--profile",
        "core",
        "--check-entry-points",
        "--pyproject",
        str(fake_pyproject),
    )
    assert human_result.returncode == 2
    assert "Mismatched declared entry points:" in human_result.stdout
    assert "srev29-example-analyzer" in human_result.stdout
    assert "Remedy: install this checkout into the active environment" in human_result.stdout


def test_check_entry_points_missing_distribution() -> None:
    """Issue #9591: uninstalled project package reports missing entry points."""
    py_code = """
import json
from scripts.dev.check_worktree_optional_deps import check_declared_entry_points

report = check_declared_entry_points("pyproject.toml", distribution_name="nonexistent-robot-sf")
print(json.dumps(report))
"""
    result = subprocess.run(
        [sys.executable, "-c", py_code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "missing_entry_points"
    assert report["installed"] is False
    assert report["exit_code"] == 2
    assert "not installed in the active environment" in report["error"]


def test_check_entry_points_missing_pyproject(tmp_path: Path) -> None:
    """Issue #9591: missing pyproject path fails closed with check_failed."""
    missing_file = tmp_path / "does_not_exist.toml"
    result = _run("--check-entry-points", "--pyproject", str(missing_file), "--json")

    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "check_failed"
    assert report["exit_code"] == 1
    assert "entry_points" in report
    assert report["entry_points"]["status"] == "check_failed"


def test_validator_accepts_entry_point_reports() -> None:
    """Issue #9591: validator recognizes entry point statuses and exits."""
    for status in ("missing_entry_points", "mismatched_entry_points"):
        payload = {
            "schema": "robot_sf.worktree_optional_deps.v1",
            "profile": "all-extras",
            "status": status,
            "exit_code": 2,
            "missing_optional": [],
            "check_failures": [],
            "entry_points": {
                "status": status,
                "distribution": "robot-sf",
                "declared_count": 2,
            },
            "project_imports_performed": False,
        }
        result = _validate(payload, 2)
        assert result.returncode == 2, result.stderr


def test_review_registry_discovery_requires_installed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #9591: review registry discovery relies on installed distribution metadata."""
    import importlib.metadata

    from robot_sf.analysis_workbench.review_registry import discover_components

    # With actual installed metadata, discovery finds both components
    components, _rows = discover_components()
    assert {"srev29-example-analyzer", "srev29-example-renderer"} <= set(components)

    # When metadata is missing (stale environment where entry points were not installed),
    # discovery returns empty, reproducing why preflight verification is critical.
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kwargs: ())
    empty_components, _empty_rows = discover_components()
    assert not {"srev29-example-analyzer", "srev29-example-renderer"}.intersection(empty_components)
