"""Tests for versioned task-bundle scenario packages."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.training.scenario_loader import load_scenarios
from robot_sf.training.task_bundles import (
    TASK_BUNDLE_SCHEMA_VERSION,
    describe_task_bundle_source,
    is_task_bundle_reference,
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


def test_load_task_bundle_scenarios_expands_nested_bundles(tmp_path: Path) -> None:
    """Bundles may reuse smaller bundles without changing deterministic order."""
    scenario_file = tmp_path / "scenarios.yaml"
    _write_scenario_file(scenario_file, ["alpha", "beta"])
    inner_bundle = tmp_path / "inner.yaml"
    outer_bundle = tmp_path / "outer.yaml"
    _write_yaml(
        inner_bundle,
        {
            "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
            "name": "inner-bundle",
            "scenario_files": ["scenarios.yaml"],
            "select_scenarios": ["beta"],
        },
    )
    _write_yaml(
        outer_bundle,
        {
            "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
            "name": "outer-bundle",
            "scenario_files": ["inner.yaml"],
        },
    )

    scenarios = load_task_bundle_scenarios(outer_bundle)

    assert [scenario["name"] for scenario in scenarios] == ["beta"]


def test_load_task_bundle_scenarios_rejects_bundle_cycles(tmp_path: Path) -> None:
    """Recursive bundle references should fail closed with a useful cycle error."""
    bundle_a = tmp_path / "bundle_a.yaml"
    bundle_b = tmp_path / "bundle_b.yaml"
    _write_yaml(
        bundle_a,
        {
            "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
            "name": "bundle-a",
            "scenario_files": ["bundle_b.yaml"],
        },
    )
    _write_yaml(
        bundle_b,
        {
            "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
            "name": "bundle-b",
            "scenario_files": ["bundle_a.yaml"],
        },
    )

    with pytest.raises(ValueError, match="Task bundle include cycle detected"):
        load_task_bundle_scenarios(bundle_a)


def test_load_task_bundle_selection_ignores_plain_id_field(tmp_path: Path) -> None:
    """Bundle selection should match the scenario-loader name/scenario_id contract."""
    map_path = tmp_path / "map.svg"
    map_path.write_text("<svg></svg>", encoding="utf-8")
    scenario_file = tmp_path / "scenarios.yaml"
    _write_yaml(
        scenario_file,
        {
            "scenarios": [
                {
                    "name": "alpha",
                    "id": "legacy-id",
                    "map_file": "map.svg",
                    "simulation_config": {"max_episode_steps": 50},
                }
            ]
        },
    )
    bundle_path = tmp_path / "bundle.yaml"
    _write_yaml(
        bundle_path,
        {
            "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
            "name": "demo-bundle",
            "scenario_files": ["scenarios.yaml"],
            "select_scenarios": ["legacy-id"],
        },
    )

    with pytest.raises(ValueError, match="Unknown select_scenarios entry 'legacy-id'"):
        load_task_bundle_scenarios(bundle_path)


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


def test_is_task_bundle_reference_uses_yaml_header_guard(tmp_path: Path) -> None:
    """Task-bundle detection should avoid parsing unrelated YAML documents."""
    scenario_file = tmp_path / "scenarios.yaml"
    _write_scenario_file(scenario_file, ["alpha"])
    bundle_path = tmp_path / "bundle.yaml"
    _write_yaml(
        bundle_path,
        {
            "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
            "name": "demo-bundle",
            "scenario_files": ["scenarios.yaml"],
        },
    )

    assert not is_task_bundle_reference(scenario_file)
    assert is_task_bundle_reference(bundle_path)


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
