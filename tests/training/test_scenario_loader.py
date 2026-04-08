"""Scenario loader include and validation coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.training import scenario_loader
from robot_sf.training.scenario_loader import load_scenarios


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


def test_load_scenarios_select_scenarios_preserves_explicit_order(tmp_path: Path) -> None:
    """Scenario selection should keep an explicit, deterministic subset order."""
    source = tmp_path / "source.yaml"
    manifest = tmp_path / "manifest.yaml"

    _write_yaml(
        source,
        """
scenarios:
  - name: scenario_a
    map_file: maps/a.svg
  - name: scenario_b
    map_file: maps/b.svg
  - name: scenario_c
    map_file: maps/c.svg
""",
    )
    _write_yaml(
        manifest,
        """
includes:
  - source.yaml
select_scenarios:
  - scenario_c
  - scenario_a
""",
    )

    scenarios = load_scenarios(manifest)
    names = [scenario.get("name") for scenario in scenarios]
    assert names == ["scenario_c", "scenario_a"]


def test_load_scenarios_select_scenarios_rejects_duplicate_names(tmp_path: Path) -> None:
    """Selecting by name should fail closed when the expanded manifest is ambiguous."""
    source_a = tmp_path / "a.yaml"
    source_b = tmp_path / "b.yaml"
    manifest = tmp_path / "manifest.yaml"

    _write_yaml(
        source_a,
        """
scenarios:
  - name: scenario_a
    map_file: maps/a.svg
""",
    )
    _write_yaml(
        source_b,
        """
scenarios:
  - name: scenario_a
    map_file: maps/b.svg
""",
    )
    _write_yaml(
        manifest,
        """
includes:
  - a.yaml
  - b.yaml
select_scenarios:
  - scenario_a
""",
    )

    with pytest.raises(ValueError, match="Duplicate scenario name"):
        load_scenarios(manifest)


def test_load_scenarios_select_scenarios_rejects_unknown_names(tmp_path: Path) -> None:
    """Selecting a missing scenario should raise instead of silently skipping it."""
    source = tmp_path / "source.yaml"
    manifest = tmp_path / "manifest.yaml"

    _write_yaml(
        source,
        """
scenarios:
  - name: scenario_a
    map_file: maps/a.svg
""",
    )
    _write_yaml(
        manifest,
        """
includes:
  - source.yaml
select_scenarios:
  - scenario_b
""",
    )

    with pytest.raises(ValueError, match="Unknown select_scenarios entry"):
        load_scenarios(manifest)


def test_classic_interactions_uses_canonical_cross_trap_names() -> None:
    """The default classic suite should use issue-594 canonical cross-trap IDs."""
    scenarios = load_scenarios(Path("configs/scenarios/classic_interactions.yaml"))
    names = {scenario["name"] for scenario in scenarios}

    assert {
        "classic_cross_trap_low",
        "classic_cross_trap_medium",
        "classic_cross_trap_high",
    }.issubset(names)
    assert "classic_crossing_low" not in names


def test_legacy_classic_crossing_manifest_still_loads_aliases() -> None:
    """Legacy crossing IDs should keep working through the deprecated alias manifest."""
    scenarios = load_scenarios(Path("configs/scenarios/sets/classic_crossing_subset.yaml"))
    names = [scenario["name"] for scenario in scenarios]

    assert names == ["classic_crossing_low", "classic_crossing_high"]
    assert scenarios[0]["metadata"]["deprecated_alias_for"] == "classic_cross_trap_low"


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


def test_map_id_resolves_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Map ids resolve via the registry to keep map_id-based scenarios portable."""
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    (maps_dir / "demo.svg").write_text("<svg></svg>", encoding="utf-8")

    registry_path = tmp_path / "registry.yaml"
    _write_yaml(
        registry_path,
        """
version: 1
maps:
  - map_id: demo_map
    path: maps/demo.svg
""",
    )

    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    manifest = scenarios_dir / "manifest.yaml"
    _write_yaml(
        manifest,
        """
scenarios:
  - name: demo
    map_id: demo_map
""",
    )

    monkeypatch.setenv("ROBOT_SF_MAP_REGISTRY", str(registry_path))
    scenario_loader._load_map_registry.cache_clear()

    scenarios = load_scenarios(manifest, base_dir=manifest)
    scenario_loader._load_map_registry.cache_clear()
    assert scenarios[0]["map_file"] == "../maps/demo.svg"
