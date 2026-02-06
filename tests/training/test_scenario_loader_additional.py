"""Additional scenario loader branch coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.training import scenario_loader

if TYPE_CHECKING:
    from pathlib import Path


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_resolve_includes_rejects_directories(tmp_path: Path) -> None:
    """Reject directory includes to prevent ambiguous scenario discovery."""
    include_dir = tmp_path / "includes"
    include_dir.mkdir()
    source = tmp_path / "manifest.yaml"
    with pytest.raises(ValueError, match="directory"):
        scenario_loader._resolve_includes(
            {"includes": [str(include_dir)]},
            source=source,
        )


def test_map_search_paths_skip_missing_entries(tmp_path: Path) -> None:
    """Ignore missing map_search_paths entries instead of crashing."""
    root = tmp_path / "manifest.yaml"
    paths = scenario_loader._resolve_map_search_paths(
        {"map_search_paths": ["missing"]},
        root=root,
    )
    assert paths == []


def test_load_map_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Duplicate map_id entries should raise to avoid ambiguity."""
    registry = tmp_path / "registry.yaml"
    _write_yaml(
        registry,
        """
maps:
  - map_id: dup
    path: maps/a.svg
  - map_id: dup
    path: maps/b.svg
""",
    )
    scenario_loader._load_map_registry.cache_clear()
    with pytest.raises(ValueError, match="Duplicate map_id"):
        scenario_loader._load_map_registry(registry)
    scenario_loader._load_map_registry.cache_clear()


def test_resolve_map_id_requires_registry(tmp_path: Path) -> None:
    """Reject map_id references when registry is empty."""
    source = tmp_path / "scenarios.yaml"
    with pytest.raises(ValueError, match="registry is empty"):
        scenario_loader._resolve_map_id("demo", map_registry={}, source=source)


def test_rebase_scenario_paths_keeps_unresolved(tmp_path: Path) -> None:
    """Preserve scenario entries when map files cannot be resolved."""
    scenario = {"name": "missing_map", "map_file": "missing.svg"}
    result = scenario_loader._rebase_scenario_paths(
        scenario,
        source=tmp_path / "manifest.yaml",
        root=tmp_path,
        map_search_paths=[],
        map_registry={},
    )
    assert result == scenario


def test_load_map_definition_unsupported_extension(tmp_path: Path) -> None:
    """Skip unsupported map formats gracefully."""
    bad_map = tmp_path / "map.txt"
    bad_map.write_text("not a map", encoding="utf-8")
    scenario_loader._load_map_definition.cache_clear()
    assert scenario_loader._load_map_definition(str(bad_map)) is None


def test_iter_registry_mapping_skips_version_and_bad_entries() -> None:
    """Registry mapping iterators should skip metadata and invalid paths."""
    entries = {"version": 1, "demo": "maps/demo.svg", "bad": 123}
    assert list(scenario_loader._iter_registry_mapping(entries)) == [("demo", "maps/demo.svg")]


def test_iter_registry_list_handles_invalid_entries() -> None:
    """Registry list iterators should ignore malformed records."""
    entries = [
        "bad",
        {"map_id": "demo", "path": "maps/demo.svg"},
        {"map_id": 1, "path": "maps/other.svg"},
        {"id": "alt", "map_file": "maps/alt.svg"},
    ]
    assert list(scenario_loader._iter_registry_list(entries)) == [
        ("demo", "maps/demo.svg"),
        ("alt", "maps/alt.svg"),
    ]


def test_iter_map_registry_entries_rejects_invalid_format(tmp_path: Path) -> None:
    """Invalid registry schemas should raise a helpful error."""
    registry = {"maps": "bad"}
    with pytest.raises(ValueError, match="invalid format"):
        list(
            scenario_loader._iter_map_registry_entries(
                registry,
                registry_path=tmp_path / "registry.yaml",
            )
        )


def test_register_map_entry_resolves_relative_paths_and_duplicates(tmp_path: Path) -> None:
    """Registry entries should resolve relative paths and reject duplicates."""
    registry: dict[str, Path] = {}
    registry_path = tmp_path / "registry.yaml"

    with pytest.raises(ValueError, match="empty map_id"):
        scenario_loader._register_map_entry(
            registry,
            map_id=" ",
            map_path="maps/demo.svg",
            registry_path=registry_path,
        )

    scenario_loader._register_map_entry(
        registry,
        map_id="demo",
        map_path="maps/demo.svg",
        registry_path=registry_path,
    )
    assert registry["demo"] == (registry_path.parent / "maps/demo.svg").resolve()

    with pytest.raises(ValueError, match="Duplicate map_id"):
        scenario_loader._register_map_entry(
            registry,
            map_id="demo",
            map_path="maps/other.svg",
            registry_path=registry_path,
        )


def test_select_scenario_matches_case_insensitive() -> None:
    """Scenario selection should be case-insensitive and raise on missing ids."""
    scenarios = [{"name": "Demo"}, {"scenario_id": "Second"}]
    assert scenario_loader.select_scenario(scenarios, "demo") == scenarios[0]
    with pytest.raises(ValueError, match="not found"):
        scenario_loader.select_scenario(scenarios, "missing")
