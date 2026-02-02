"""Scenario loader include and validation coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.training.scenario_loader import load_scenarios

if TYPE_CHECKING:
    from pathlib import Path


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_scenarios_with_includes(tmp_path: Path) -> None:
    """Scenario manifests can include multiple files in order."""
    include_a = tmp_path / "a.yaml"
    include_b = tmp_path / "b.yaml"
    manifest = tmp_path / "manifest.yaml"

    _write_yaml(
        include_a,
        """
scenarios:
  - name: scenario_a
    map_file: maps/a.svg
""",
    )
    _write_yaml(
        include_b,
        """
- name: scenario_b
  map_file: maps/b.svg
""",
    )
    _write_yaml(
        manifest,
        """
includes:
  - a.yaml
  - b.yaml
scenarios:
  - name: scenario_local
    map_file: maps/local.svg
""",
    )

    scenarios = load_scenarios(manifest)
    names = [scenario.get("name") for scenario in scenarios]
    assert names == ["scenario_a", "scenario_b", "scenario_local"]


def test_load_scenarios_detects_include_cycles(tmp_path: Path) -> None:
    """Include cycles are rejected with a clear error."""
    file_a = tmp_path / "a.yaml"
    file_b = tmp_path / "b.yaml"

    _write_yaml(
        file_a,
        """
includes:
  - b.yaml
""",
    )
    _write_yaml(
        file_b,
        """
includes:
  - a.yaml
scenarios:
  - name: scenario_b
    map_file: maps/b.svg
""",
    )

    with pytest.raises(ValueError, match="cycle"):
        load_scenarios(file_a)


def test_load_scenarios_validates_seed_types(tmp_path: Path) -> None:
    """Seed lists must contain integers."""
    scenario_file = tmp_path / "seeds.yaml"
    _write_yaml(
        scenario_file,
        """
scenarios:
  - name: bad_seeds
    map_file: maps/bad.svg
    seeds: [1, "2"]
""",
    )

    with pytest.raises(ValueError, match="seeds must contain integers"):
        load_scenarios(scenario_file)


def test_map_search_paths_rebases_map_paths(tmp_path: Path) -> None:
    """map_search_paths should resolve map_file and rebase to the manifest root."""
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    map_path = maps_dir / "demo.svg"
    map_path.write_text("<svg></svg>", encoding="utf-8")

    scenario_file = tmp_path / "scenario.yaml"
    _write_yaml(
        scenario_file,
        """
scenarios:
  - name: map_search
    map_file: demo.svg
""",
    )

    manifest = tmp_path / "manifest.yaml"
    _write_yaml(
        manifest,
        """
includes:
  - scenario.yaml
map_search_paths:
  - maps
""",
    )

    scenarios = load_scenarios(manifest, base_dir=manifest)
    assert scenarios[0]["map_file"] == "maps/demo.svg"


def test_map_search_paths_use_base_dir_and_ignore_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure base_dir directories anchor map resolution instead of CWD."""
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    maps_dir = base_dir / "maps"
    maps_dir.mkdir()
    (maps_dir / "map.svg").write_text("<svg></svg>", encoding="utf-8")

    scenario_file = base_dir / "scenario.yaml"
    _write_yaml(
        scenario_file,
        """
scenarios:
  - name: map_search
    map_file: map.svg
""",
    )

    manifest = base_dir / "manifest.yaml"
    _write_yaml(
        manifest,
        """
includes:
  - scenario.yaml
map_search_paths:
  - maps
""",
    )

    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    (cwd_dir / "map.svg").write_text("<svg></svg>", encoding="utf-8")
    monkeypatch.chdir(cwd_dir)

    scenarios = load_scenarios(manifest, base_dir=base_dir)
    assert scenarios[0]["map_file"] == "maps/map.svg"
