"""Tests for deterministic changed-module fast-lane routing diagnostics."""

from __future__ import annotations

from scripts.dev.check_fast_lane_routing import (
    FastLanePolicy,
    audit_changed_modules,
)


def _policy(*fast_files: str) -> FastLanePolicy:
    return FastLanePolicy(
        fast_files=frozenset(fast_files),
        fast_path_fragments=(),
        fast_file_prefixes=(),
        slow_file_overrides=frozenset(),
    )


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
