"""Scenario loader include and validation coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.training import scenario_loader
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    resolve_map_definition,
)


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


def test_nested_manifests_resolve_map_search_paths_from_each_manifest(
    tmp_path: Path,
) -> None:
    """Nested include manifests should keep their own map_search_paths roots."""
    maps_dir = tmp_path / "maps" / "svg_maps"
    maps_dir.mkdir(parents=True)
    (maps_dir / "demo.svg").write_text("<svg></svg>", encoding="utf-8")

    scenarios_dir = tmp_path / "configs" / "scenarios"
    archetypes_dir = scenarios_dir / "archetypes"
    sets_dir = scenarios_dir / "sets"
    archetypes_dir.mkdir(parents=True)
    sets_dir.mkdir(parents=True)

    archetype_file = archetypes_dir / "demo.yaml"
    _write_yaml(
        archetype_file,
        """
scenarios:
  - name: nested_demo
    map_file: demo.svg
""",
    )

    child_manifest = scenarios_dir / "classic.yaml"
    _write_yaml(
        child_manifest,
        """
includes:
  - archetypes/demo.yaml
map_search_paths:
  - ../../maps/svg_maps
""",
    )

    top_manifest = sets_dir / "combo.yaml"
    _write_yaml(
        top_manifest,
        """
includes:
  - ../classic.yaml
""",
    )

    scenarios = load_scenarios(top_manifest)
    assert scenarios[0]["map_file"] == "../../../maps/svg_maps/demo.svg"


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


# ---------------------------------------------------------------------------
# Regression tests for issue #830: empty map-pool crash in long-run PPO jobs
# ---------------------------------------------------------------------------


def test_build_robot_config_raises_early_on_missing_map_file(tmp_path: Path) -> None:
    """build_robot_config_from_scenario must raise a clear ValueError when the
    map_file in the scenario does not exist, rather than silently falling back
    to the default map pool (which causes a confusing 'Map pool is empty!' error
    much later during scenario reset — the original issue #830 failure mode).
    """
    manifest = tmp_path / "manifest.yaml"
    _write_yaml(
        manifest,
        """
scenarios:
  - name: missing_map_scenario
    map_file: does_not_exist.svg
""",
    )

    scenarios = load_scenarios(manifest)
    assert len(scenarios) == 1

    with pytest.raises(ValueError, match="does_not_exist.svg"):
        build_robot_config_from_scenario(scenarios[0], scenario_path=manifest)


def test_build_robot_config_raises_early_on_unresolvable_relative_map_file(
    tmp_path: Path,
) -> None:
    """Relative map_file paths that cannot be resolved should also fail fast."""
    manifest = tmp_path / "manifest.yaml"
    _write_yaml(
        manifest,
        """
scenarios:
  - name: unresolvable_scenario
    map_file: subdir/nonexistent.svg
""",
    )

    scenarios = load_scenarios(manifest)
    with pytest.raises(ValueError, match="nonexistent.svg"):
        build_robot_config_from_scenario(scenarios[0], scenario_path=manifest)


def test_build_robot_config_with_no_map_file_does_not_raise(tmp_path: Path) -> None:
    """Scenarios without any map_file should not raise; they rely on the default pool."""
    manifest = tmp_path / "manifest.yaml"
    _write_yaml(
        manifest,
        """
scenarios:
  - name: no_map_scenario
""",
    )

    scenarios = load_scenarios(manifest)
    # Should not raise — no map_file means the caller uses the default map pool.
    cfg = build_robot_config_from_scenario(scenarios[0], scenario_path=manifest)
    assert cfg is not None


def test_classic_interactions_francis2023_manifest_resolves_all_maps() -> None:
    """All scenarios in the combined issue-791 manifest must resolve to real map files.

    This is the direct regression guard for issue #830: if any map_file in the
    classic_interactions_francis2023 scenario chain is broken, this test catches
    it before a long SLURM allocation is consumed.
    """
    manifest = Path("configs/scenarios/classic_interactions_francis2023.yaml")
    if not manifest.exists():
        pytest.skip("Combined manifest not found; run from repo root.")

    scenarios = load_scenarios(manifest)
    assert scenarios, "Manifest loaded no scenarios"

    missing: list[str] = []
    for scenario in scenarios:
        map_file = scenario.get("map_file")
        if not map_file:
            continue
        result = resolve_map_definition(map_file, scenario_path=manifest)
        if result is None:
            missing.append(f"{scenario.get('name')}: {map_file}")

    assert not missing, (
        f"The following scenarios in classic_interactions_francis2023.yaml could not "
        f"resolve their map_file from '{manifest}':\n" + "\n".join(missing)
    )
