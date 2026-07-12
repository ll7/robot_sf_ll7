"""Tests for include-aware scenario matrix loading in the benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from robot_sf.benchmark.runner import load_scenario_matrix

if TYPE_CHECKING:
    from pathlib import Path


def _write_yaml(path: Path, payload: object) -> None:
    """Write a YAML fixture with stable key ordering."""
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_load_scenario_matrix_expands_includes(tmp_path: Path) -> None:
    """Verify manifest include expansion so CLI list/validate honors new layout."""
    map_path = tmp_path / "map.svg"
    map_path.write_text("<svg></svg>", encoding="utf-8")

    include_path = tmp_path / "include.yaml"
    _write_yaml(
        include_path,
        {
            "scenarios": [
                {
                    "name": "sc_a",
                    "map_file": "map.svg",
                    "simulation_config": {"max_episode_steps": 100},
                    "metadata": {"archetype": "crossing", "density": "low"},
                }
            ]
        },
    )

    manifest_path = tmp_path / "manifest.yaml"
    _write_yaml(manifest_path, {"includes": ["include.yaml"]})

    scenarios = load_scenario_matrix(manifest_path)

    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "sc_a"


def test_load_scenario_matrix_yaml_stream_legacy(tmp_path: Path) -> None:
    """Ensure multi-doc YAML streams still load without include expansion regressions."""
    map_path = tmp_path / "map.svg"
    map_path.write_text("<svg></svg>", encoding="utf-8")

    stream_path = tmp_path / "stream.yaml"
    stream_path.write_text(
        """name: sc_a\nmap_file: map.svg\nsimulation_config:\n  max_episode_steps: 50\nmetadata:\n  archetype: crossing\n  density: low\n---\nname: sc_b\nmap_file: map.svg\nsimulation_config:\n  max_episode_steps: 60\nmetadata:\n  archetype: overtaking\n  density: medium\n""",
        encoding="utf-8",
    )

    scenarios = load_scenario_matrix(stream_path)

    assert [sc["name"] for sc in scenarios] == ["sc_a", "sc_b"]


def test_load_scenario_matrix_accepts_task_bundle_reference() -> None:
    """Benchmark matrix loading should consume named task bundles."""
    scenarios = load_scenario_matrix("bundle:sanity-smoke-v1")

    assert [scenario["name"] for scenario in scenarios] == [
        "planner_sanity_simple",
        "empty_map_8_directions_east",
        "goal_behind_robot",
        "single_ped_crossing_orthogonal",
    ]


# Abstract benchmark scenarios (density/flow/obstacle form) carry neither a
# ``name``/``scenario_id`` nor a ``map_file``/``map_id``. They must load from a
# single-document top-level list the same way they load from a multi-document
# stream, without routing through the map-oriented manifest validator (#5429).
_ABSTRACT_SCENARIOS = [
    {
        "id": "batch-uni-low-open",
        "density": "low",
        "flow": "uni",
        "obstacle": "open",
        "groups": 0.0,
        "speed_var": "low",
        "goal_topology": "point",
        "robot_context": "embedded",
        "repeats": 2,
    },
    {
        "id": "batch-uni-high-open",
        "density": "high",
        "flow": "uni",
        "obstacle": "open",
        "groups": 0.0,
        "speed_var": "low",
        "goal_topology": "point",
        "robot_context": "embedded",
        "repeats": 2,
    },
]


def _capture_loader_logs() -> tuple[list[str], int]:
    """Attach a loguru sink capturing scenario-loader messages.

    Returns:
        Tuple of (captured message list, sink handler id) for later removal.
    """
    from loguru import logger

    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        filter="robot_sf.training.scenario_loader",
        level="WARNING",
    )
    return messages, handler_id


def test_load_scenario_matrix_single_doc_list_returns_abstract_scenarios(tmp_path: Path) -> None:
    """A single-document list of abstract scenarios loads verbatim (#5429)."""
    from loguru import logger

    matrix_path = tmp_path / "abstract_matrix.yaml"
    _write_yaml(matrix_path, _ABSTRACT_SCENARIOS)

    messages, handler_id = _capture_loader_logs()
    try:
        scenarios = load_scenario_matrix(matrix_path)
    finally:
        logger.remove(handler_id)

    assert scenarios == _ABSTRACT_SCENARIOS
    # No entry gained a name/map field, and the map-oriented validator was not
    # invoked, so its misleading "missing name"/"no map_file" warnings are absent.
    assert all("name" not in sc and "map_file" not in sc for sc in scenarios)
    assert not any("missing a name" in m or "no map_file" in m for m in messages)


def test_load_scenario_matrix_single_doc_list_matches_stream(tmp_path: Path) -> None:
    """Single-document list and multi-document stream yield identical scenarios (#5429)."""
    list_path = tmp_path / "list.yaml"
    list_path.write_text(yaml.safe_dump(_ABSTRACT_SCENARIOS), encoding="utf-8")

    stream_path = tmp_path / "stream.yaml"
    stream_path.write_text(yaml.safe_dump_all(_ABSTRACT_SCENARIOS), encoding="utf-8")

    assert load_scenario_matrix(list_path) == load_scenario_matrix(stream_path)


def test_load_scenario_matrix_empty_single_doc_list_fails_closed(tmp_path: Path) -> None:
    """An empty single-document list raises instead of a silent zero-job run (#5429)."""
    import pytest

    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty scenario list"):
        load_scenario_matrix(empty_path)


def test_load_scenario_matrix_single_doc_map_list_preserves_manifest_validation(
    tmp_path: Path,
) -> None:
    """Top-level map manifests still resolve and validate through the shared loader."""
    import pytest

    matrix_path = tmp_path / "map_matrix.yaml"
    matrix_path.write_text(
        "- name: named_map\n  map_id: definitely-not-a-real-map\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown map_id"):
        load_scenario_matrix(matrix_path)
