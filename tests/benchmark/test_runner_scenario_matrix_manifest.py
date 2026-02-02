"""Tests for include-aware scenario matrix loading in the benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from robot_sf.benchmark.runner import load_scenario_matrix

if TYPE_CHECKING:
    from pathlib import Path


def _write_yaml(path: Path, payload: object) -> None:
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
