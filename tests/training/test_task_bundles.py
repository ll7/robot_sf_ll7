"""Tests for versioned task-bundle scenario packages."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.training.scenario_loader import load_scenarios
from robot_sf.training.task_bundles import (
    TASK_BUNDLE_SCHEMA_VERSION,
    describe_task_bundle_source,
    load_task_bundle,
    load_task_bundle_scenarios,
)


def _write_yaml(path: Path, payload: object) -> None:
    """Write a YAML fixture with stable key ordering."""
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_scenario_file(path: Path, names: list[str]) -> None:
    map_path = path.parent / "map.svg"
    map_path.write_text("<svg></svg>", encoding="utf-8")
    _write_yaml(
        path,
        {
            "scenarios": [
                {
                    "name": name,
                    "map_file": "map.svg",
                    "simulation_config": {"max_episode_steps": 50},
                    "metadata": {"archetype": "crossing", "density": "low"},
                }
                for name in names
            ]
        },
    )


def test_load_task_bundle_scenarios_expands_deterministically(tmp_path: Path) -> None:
    """A bundle should expand scenario files in explicit YAML order."""
    scenario_a = tmp_path / "scenario_a.yaml"
    scenario_b = tmp_path / "scenario_b.yaml"
    _write_scenario_file(scenario_a, ["alpha"])
    _write_scenario_file(scenario_b, ["beta"])
    bundle_path = tmp_path / "bundle.yaml"
    _write_yaml(
        bundle_path,
        {
            "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
            "name": "demo-bundle",
            "scenario_files": ["scenario_a.yaml", "scenario_b.yaml"],
        },
    )

    first = load_task_bundle_scenarios(bundle_path)
    second = load_task_bundle_scenarios(bundle_path)

    assert [scenario["name"] for scenario in first] == ["alpha", "beta"]
    assert first == second


def test_load_task_bundle_scenarios_applies_selection_order(tmp_path: Path) -> None:
    """Bundle-level selection should provide a compact deterministic subset."""
    scenario_file = tmp_path / "scenarios.yaml"
    _write_scenario_file(scenario_file, ["alpha", "beta", "gamma"])
    bundle_path = tmp_path / "bundle.yaml"
    _write_yaml(
        bundle_path,
        {
            "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
            "name": "demo-bundle",
            "scenario_files": ["scenarios.yaml"],
            "select_scenarios": ["gamma", "alpha"],
        },
    )

    scenarios = load_task_bundle_scenarios(bundle_path)

    assert [scenario["name"] for scenario in scenarios] == ["gamma", "alpha"]


def test_load_scenarios_accepts_named_registry_bundle() -> None:
    """The default registry should resolve bundle:<name> through load_scenarios."""
    scenarios = load_scenarios("bundle:sanity-smoke-v1")

    assert [scenario["name"] for scenario in scenarios] == [
        "planner_sanity_simple",
        "empty_map_8_directions_east",
        "goal_behind_robot",
        "single_ped_crossing_orthogonal",
    ]


def test_describe_task_bundle_source_reports_inputs() -> None:
    """CLI source reports should expose the bundle registry expansion."""
    source = describe_task_bundle_source("bundle:sanity-smoke-v1")

    assert source["format"] == "task_bundle"
    assert source["bundle_name"] == "sanity-smoke-v1"
    assert source["schema_version"] == TASK_BUNDLE_SCHEMA_VERSION
    assert len(source["scenario_files"]) == 1


def test_load_task_bundle_rejects_unsupported_schema(tmp_path: Path) -> None:
    """Malformed bundle schema versions should fail closed."""
    bundle_path = tmp_path / "bundle.yaml"
    _write_yaml(
        bundle_path,
        {
            "schema_version": "robot_sf.task_bundle.v0",
            "name": "demo-bundle",
            "scenario_files": [],
        },
    )

    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_task_bundle(bundle_path)


def test_load_task_bundle_rejects_local_output_inputs(tmp_path: Path) -> None:
    """Bundles should not make worktree-local output files durable by reference."""
    output_scenario = Path("output") / "tmp_bundle_scenario.yaml"
    bundle_path = tmp_path / "bundle.yaml"
    _write_yaml(
        bundle_path,
        {
            "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
            "name": "demo-bundle",
            "scenario_files": [str(Path.cwd() / output_scenario)],
        },
    )

    with pytest.raises(ValueError, match="local output path"):
        load_task_bundle(bundle_path)
