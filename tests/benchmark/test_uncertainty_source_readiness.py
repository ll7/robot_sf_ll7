"""Tests for issue #3557 uncertainty-source readiness inventory."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import robot_sf.benchmark.uncertainty_source_readiness as readiness
from robot_sf.benchmark.uncertainty_source_readiness import (
    UNCERTAINTY_SOURCE_READINESS_SCHEMA,
    UncertaintySourceRunSpec,
    build_uncertainty_source_readiness_inventory,
    classify_uncertainty_source_readiness,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "validation"
    / "check_uncertainty_source_readiness_issue_3557.py"
)
_SPEC = importlib.util.spec_from_file_location("_issue_3557_readiness", _SCRIPT_PATH)
assert _SPEC is not None
_SCRIPT = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_SCRIPT)


def test_default_inventory_reports_readiness_without_running_benchmarks() -> None:
    """Default inventory labels current hooks without executing episode runs."""

    report = build_uncertainty_source_readiness_inventory()

    assert report["schema_version"] == UNCERTAINTY_SOURCE_READINESS_SCHEMA
    assert report["issue"] == 3557
    assert report["claim_boundary"] == "readiness_inventory_only"
    assert report["not_benchmark_evidence"] is True
    assert report["runs_executed"] is False
    assert report["surrogate_semantics_changed"] is False
    assert report["all_sources_ready"] is False

    by_source = {row["source"]: row for row in report["sources"]}
    assert set(by_source) == {
        "existence_degraded",
        "visibility_limited",
        "covariance_inflated",
        "class_probability",
        "tracking_noise",
    }

    assert by_source["existence_degraded"]["status"] == "ready"
    assert by_source["existence_degraded"]["missing"] == []
    assert by_source["visibility_limited"]["missing"] == ["scenario_hook"]
    assert by_source["covariance_inflated"]["missing"] == ["scenario_hook"]
    assert by_source["class_probability"]["missing"] == ["scenario_hook"]
    assert by_source["tracking_noise"]["missing"] == ["condition_builder", "scenario_hook"]


def test_supported_source_with_all_surrogate_outputs_is_ready() -> None:
    """A source with known builder, hook, and output fields is runnable-ready."""

    row = classify_uncertainty_source_readiness(
        UncertaintySourceRunSpec(
            source="synthetic",
            condition_builder="_condition_existence_degraded",
            scenario_hook="build_belief_for_mode",
        )
    )

    assert row["status"] == "ready"
    assert row["condition_builder_present"] is True
    assert row["scenario_hook_present"] is True
    assert row["expected_surrogate_outputs_present"] is True


def test_unknown_condition_builder_fails_closed() -> None:
    """A misspelled or not-yet-written builder blocks the source."""

    row = classify_uncertainty_source_readiness(
        UncertaintySourceRunSpec(
            source="tracking_noise",
            condition_builder="_condition_tracking_noise",
            scenario_hook="build_belief_for_mode",
        )
    )

    assert row["status"] == "blocked"
    assert row["missing"] == ["condition_builder"]
    assert row["condition_builder_present"] is False


def test_missing_scenario_hook_fails_closed() -> None:
    """Single-step condition builders alone are not episode-run readiness."""

    row = classify_uncertainty_source_readiness(
        UncertaintySourceRunSpec(
            source="visibility_limited",
            condition_builder="_condition_visibility_limited",
            scenario_hook=None,
        )
    )

    assert row["status"] == "blocked"
    assert row["missing"] == ["scenario_hook"]
    assert row["scenario_hook_present"] is False


def test_missing_expected_surrogate_output_fails_closed() -> None:
    """Future run specs must expose all fields consumed by the decision layer."""

    row = classify_uncertainty_source_readiness(
        UncertaintySourceRunSpec(
            source="existence_degraded",
            condition_builder="_condition_existence_degraded",
            scenario_hook="build_belief_for_mode",
            expected_surrogate_outputs=(
                "source",
                "retained_unsafe_commit_rate",
                "dropped_unsafe_commit_rate",
                "n_episodes",
            ),
        )
    )

    assert row["status"] == "blocked"
    assert row["missing"] == ["expected_surrogate_outputs"]
    assert row["missing_expected_surrogate_outputs"] == ["min_separation_delta_m"]


def test_inventory_requires_at_least_one_source() -> None:
    """An empty inventory would hide readiness gaps."""

    with pytest.raises(ValueError, match="at least one uncertainty source"):
        build_uncertainty_source_readiness_inventory([])


def test_inventory_reuses_discovered_symbols_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inventory discovery should happen once per report, not once per source."""

    calls = {"builders": 0, "hooks": 0}

    def discover_builders() -> frozenset[str]:
        calls["builders"] += 1
        return frozenset(
            {
                "_condition_existence_degraded",
                "_condition_visibility_limited",
                "_condition_covariance_inflated",
                "_condition_class_probability",
            }
        )

    def discover_hooks() -> frozenset[str]:
        calls["hooks"] += 1
        return frozenset({"build_belief_for_mode"})

    monkeypatch.setattr(readiness, "_discover_condition_builders", discover_builders)
    monkeypatch.setattr(readiness, "_discover_scenario_hooks", discover_hooks)

    report = build_uncertainty_source_readiness_inventory()

    assert report["ready_sources"] == ["existence_degraded"]
    assert calls == {"builders": 1, "hooks": 1}


def test_owner_import_failures_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Broken owner modules should make sources blocked, not crash the inventory CLI."""

    def broken_import_module(module_name: str) -> object:
        raise RuntimeError(f"{module_name} unavailable")

    monkeypatch.setattr(readiness.importlib, "import_module", broken_import_module)

    report = build_uncertainty_source_readiness_inventory(
        [
            UncertaintySourceRunSpec(
                source="synthetic",
                condition_builder="_condition_existence_degraded",
                scenario_hook="build_belief_for_mode",
            )
        ]
    )

    assert report["all_sources_ready"] is False
    assert report["blocked_sources"] == ["synthetic"]
    assert report["sources"][0]["missing"] == ["condition_builder", "scenario_hook"]


def test_cli_reports_inventory_and_can_fail_on_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI prints JSON by default and can fail closed when blocked sources remain."""

    assert _SCRIPT.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == UNCERTAINTY_SOURCE_READINESS_SCHEMA
    assert "tracking_noise" in report["blocked_sources"]

    assert _SCRIPT.main(["--fail-on-blocked"]) == 1
