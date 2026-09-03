"""Tests for deterministic changed-module fast-lane routing diagnostics."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.dev.check_fast_lane_routing import (
    FastLanePolicy,
    _changed_modules,
    audit_changed_modules,
    load_fast_lane_policy,
)


def _policy(*fast_files: str) -> FastLanePolicy:
    return FastLanePolicy(
        fast_files=frozenset(fast_files),
        fast_path_fragments=(),
        fast_file_prefixes=(),
        slow_file_overrides=frozenset(),
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_changed_modules_excludes_changes_unique_to_advanced_base(tmp_path: Path) -> None:
    """A stale ordinary-CAS head must audit only changes introduced on its branch."""

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Fast Lane Test")
    _git(tmp_path, "config", "user.email", "fast-lane@example.invalid")
    source_root = tmp_path / "robot_sf"
    source_root.mkdir()
    (source_root / "shared.py").write_text("VALUE = 'common'\n", encoding="utf-8")
    _git(tmp_path, "add", "robot_sf/shared.py")
    _git(tmp_path, "commit", "-m", "common base")
    common_sha = _git(tmp_path, "rev-parse", "HEAD")

    _git(tmp_path, "switch", "-c", "feature")
    (source_root / "feature_only.py").write_text("FEATURE = True\n", encoding="utf-8")
    _git(tmp_path, "add", "robot_sf/feature_only.py")
    _git(tmp_path, "commit", "-m", "feature change")
    feature_sha = _git(tmp_path, "rev-parse", "HEAD")

    _git(tmp_path, "switch", "-c", "advanced-base", common_sha)
    (source_root / "shared.py").write_text("VALUE = 'base only'\n", encoding="utf-8")
    _git(tmp_path, "add", "robot_sf/shared.py")
    _git(tmp_path, "commit", "-m", "base-only change")
    advanced_base_sha = _git(tmp_path, "rev-parse", "HEAD")

    assert _changed_modules(tmp_path, advanced_base_sha, feature_sha) == [
        "robot_sf/feature_only.py"
    ]


def test_adversarial_harness_and_atlas_contracts_report_missing_registration() -> None:
    """The two observed research fixtures produce actionable findings when unregistered."""

    test_contents = {
        "tests/adversarial/test_search_harness.py": '"""Search harness contract tests."""\n',
        "tests/benchmark/test_mechanism_boundary_atlas.py": '"""Atlas schema tests."""\n',
    }

    observations = audit_changed_modules(
        (
            "robot_sf/adversarial/search_harness.py",
            "robot_sf/benchmark/mechanism_boundary_atlas.py",
        ),
        test_contents,
        _policy(),
    )

    assert [(item.source_module, item.policy_state) for item in observations] == [
        ("robot_sf/adversarial/search_harness.py", "missing-fast-registration"),
        ("robot_sf/benchmark/mechanism_boundary_atlas.py", "missing-fast-registration"),
    ]
    assert all(
        "tests/conftest.py:_FAST_FILES" in item.suggested_registration for item in observations
    )


def test_registered_contracts_are_not_reported() -> None:
    """Existing fast-file registrations suppress the diagnostic without changing thresholds."""

    test_contents = {
        "tests/adversarial/test_search_harness.py": '"""Search harness contract tests."""\n',
        "tests/benchmark/test_mechanism_boundary_atlas.py": '"""Atlas schema tests."""\n',
    }

    observations = audit_changed_modules(
        (
            "robot_sf/adversarial/search_harness.py",
            "robot_sf/benchmark/mechanism_boundary_atlas.py",
        ),
        test_contents,
        _policy("test_search_harness.py", "test_mechanism_boundary_atlas.py"),
    )

    assert all(item.policy_state == "registered-fast" for item in observations)
    assert all(not item.needs_attention for item in observations)


def test_release_checkpoint_producer_tests_are_registered_in_fast_lane() -> None:
    """Release-smoke producer coverage must reach the hosted changed-line combiner."""
    policy = load_fast_lane_policy(Path("tests/conftest.py").read_text(encoding="utf-8"))

    assert {
        "test_checkpoint_provenance_issue_4970.py",
        "test_post_execution_release_doctor.py",
        "test_predictive_mppi_planner.py",
        "test_runtime_smoke_admission.py",
    } <= policy.fast_files


def test_simulation_and_campaign_tests_remain_slow() -> None:
    """Simulation-heavy tests are reported as intentional slow coverage, not waived."""

    observations = audit_changed_modules(
        ("robot_sf/scenarios/campaign_runner.py",),
        {
            "tests/scenarios/test_campaign_runner.py": (
                "import pytest\n"
                "@pytest.mark.slow\n"
                "def test_runs_episode():\n"
                "    simulator.run_episode()\n"
            )
        },
        _policy(),
    )

    assert len(observations) == 1
    assert observations[0].classification == "slow-simulation-or-campaign"
    assert observations[0].policy_state == "slow-by-policy"
    assert not observations[0].needs_attention


def test_ambiguous_nearby_test_requires_explicit_classification() -> None:
    """Unknown nearby tests fail closed instead of silently losing changed coverage."""

    observations = audit_changed_modules(
        ("robot_sf/nav/route_helpers.py",),
        {"tests/nav/test_route_helpers.py": "def test_route_helper():\n    assert True\n"},
        _policy(),
    )

    assert len(observations) == 1
    assert observations[0].policy_state == "needs-classification"
    assert observations[0].needs_attention
