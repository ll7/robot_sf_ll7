"""Scenario loader include and validation coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256_file
from robot_sf.training import scenario_loader
from robot_sf.training.scenario_loader import (
    _coerce_positive_float,
    build_robot_config_from_scenario,
    load_scenarios,
    resolve_map_definition,
)


def _write_yaml(path: Path, content: str) -> None:
    """Write a YAML fixture file."""
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


def test_build_robot_config_parses_observation_visibility_settings(tmp_path: Path) -> None:
    """Scenario configs should expose opt-in planner observation visibility limits."""
    config = build_robot_config_from_scenario(
        {
            "name": "visibility_case",
            "observation_visibility": {
                "fov_degrees": 120,
                "max_range_m": 8.5,
                "static_occlusion": True,
                "dynamic_occlusion": True,
            },
        },
        scenario_path=tmp_path / "scenario.yaml",
    )

    assert config.observation_visibility.enabled is True
    assert config.observation_visibility.fov_degrees == pytest.approx(120.0)
    assert config.observation_visibility.max_range_m == pytest.approx(8.5)
    assert config.observation_visibility.static_occlusion is True
    assert config.observation_visibility.dynamic_occlusion is True


def test_build_robot_config_rejects_invalid_observation_visibility(tmp_path: Path) -> None:
    """Visibility config validation should fail closed for unsupported sensor settings."""
    with pytest.raises(ValueError, match="fov_degrees must be <= 360"):
        build_robot_config_from_scenario(
            {
                "name": "visibility_case",
                "observation_visibility": {
                    "fov_degrees": 361,
                },
            },
            scenario_path=tmp_path / "scenario.yaml",
        )

    def test_coerce_positive_float_prefixes_robot_config_fields() -> None:
        """Positive float validation should match sibling robot_config error prefixes."""
        with pytest.raises(ValueError, match=r"robot_config\.radius must be > 0\."):
            _coerce_positive_float(0, field_name="radius")


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


def test_load_scenarios_applies_named_overrides_after_includes(tmp_path: Path) -> None:
    """Named overrides should tune one expanded scenario without touching siblings."""
    source = tmp_path / "source.yaml"
    manifest = tmp_path / "manifest.yaml"

    _write_yaml(
        source,
        """
scenarios:
  - name: scenario_a
    map_file: maps/a.svg
    simulation_config:
      max_episode_steps: 100
      ped_density: 0.1
  - name: scenario_b
    map_file: maps/b.svg
    simulation_config:
      max_episode_steps: 100
      ped_density: 0.2
""",
    )
    _write_yaml(
        manifest,
        """
includes:
  - source.yaml
scenario_overrides_by_name:
  scenario_b:
    simulation_config:
      max_episode_steps: 240
""",
    )

    scenarios = load_scenarios(manifest)
    by_name = {scenario["name"]: scenario for scenario in scenarios}

    assert by_name["scenario_a"]["simulation_config"] == {
        "max_episode_steps": 100,
        "ped_density": 0.1,
    }
    assert by_name["scenario_b"]["simulation_config"] == {
        "max_episode_steps": 240,
        "ped_density": 0.2,
    }


def test_build_robot_config_applies_archetype_simulation_overrides(tmp_path: Path) -> None:
    """Scenario YAML can carry pedestrian archetype composition into runtime config."""
    scenario = {
        "name": "archetype-runtime-smoke",
        "simulation_config": {
            "ped_density": 0.08,
            "archetype_composition": {"cautious": 0.5, "hurried": 0.5},
            "archetype_speed_factors": {"cautious": 0.7, "hurried": 1.4},
            "archetype_seed": 3206,
        },
    }

    config = build_robot_config_from_scenario(scenario, scenario_path=tmp_path / "scenario.yaml")

    assert config.sim_config.archetype_composition == {"cautious": 0.5, "hurried": 0.5}
    assert config.sim_config.archetype_speed_factors == {"cautious": 0.7, "hurried": 1.4}
    assert config.sim_config.archetype_seed == 3206


def test_build_robot_config_applies_pedestrian_uncertainty_envelope_fields(
    tmp_path: Path,
) -> None:
    """Scenario YAML can thread pedestrian uncertainty-envelope fields to runtime config."""
    scenario = {
        "name": "uncertainty-envelope-runtime-smoke",
        "simulation_config": {
            "pedestrian_uncertainty_envelope_enabled": True,
            "pedestrian_uncertainty_alpha_mps": 0.1,
        },
    }

    config = build_robot_config_from_scenario(scenario, scenario_path=tmp_path / "scenario.yaml")

    assert config.sim_config.pedestrian_uncertainty_envelope_enabled is True
    assert config.sim_config.pedestrian_uncertainty_alpha_mps == 0.1


def test_build_robot_config_applies_action_latency_overrides(tmp_path: Path) -> None:
    """Scenario configuration exposes both discrete and millisecond action delay forms."""
    step_config = build_robot_config_from_scenario(
        {
            "name": "action-latency-step-runtime-smoke",
            "simulation_config": {"action_latency_steps": 2},
        },
        scenario_path=tmp_path / "scenario.yaml",
    )
    ms_config = build_robot_config_from_scenario(
        {
            "name": "action-latency-ms-runtime-smoke",
            "simulation_config": {"action_latency_ms": 250.0},
        },
        scenario_path=tmp_path / "scenario.yaml",
    )

    assert step_config.sim_config.resolved_action_latency_steps == 2
    assert ms_config.sim_config.resolved_action_latency_steps == 3


def test_build_robot_config_rejects_ambiguous_action_latency_overrides(tmp_path: Path) -> None:
    """A scenario cannot choose both step and millisecond delay representations."""
    scenario = {
        "name": "ambiguous-action-latency",
        "simulation_config": {"action_latency_steps": 1, "action_latency_ms": 100.0},
    }

    with pytest.raises(ValueError, match="cannot both be configured"):
        build_robot_config_from_scenario(scenario, scenario_path=tmp_path / "scenario.yaml")


def test_build_robot_config_rejects_negative_uncertainty_alpha(tmp_path: Path) -> None:
    """Negative scenario alpha fails closed at config-construction time."""
    scenario = {
        "name": "uncertainty-envelope-negative-alpha",
        "simulation_config": {"pedestrian_uncertainty_alpha_mps": -0.1},
    }

    with pytest.raises(ValueError, match="pedestrian_uncertainty_alpha_mps"):
        build_robot_config_from_scenario(scenario, scenario_path=tmp_path / "scenario.yaml")


def test_build_robot_config_alpha_zero_preserves_default_config(tmp_path: Path) -> None:
    """Explicit alpha zero is equivalent to the default deterministic envelope setting."""
    scenario = {
        "name": "uncertainty-envelope-alpha-zero",
        "simulation_config": {"pedestrian_uncertainty_alpha_mps": 0.0},
    }

    config = build_robot_config_from_scenario(scenario, scenario_path=tmp_path / "scenario.yaml")

    assert config.sim_config.pedestrian_uncertainty_envelope_enabled is False
    assert config.sim_config.pedestrian_uncertainty_alpha_mps == 0.0


def test_load_scenarios_allows_duplicate_names_outside_named_overrides(
    tmp_path: Path,
) -> None:
    """Unchanged duplicate names may pass through when only another scenario is overridden."""
    source = tmp_path / "source.yaml"
    manifest = tmp_path / "manifest.yaml"

    _write_yaml(
        source,
        """
scenarios:
  - name: repeated
    map_file: maps/a.svg
  - name: scenario_b
    map_file: maps/b.svg
    simulation_config:
      max_episode_steps: 100
  - name: repeated
    map_file: maps/c.svg
""",
    )
    _write_yaml(
        manifest,
        """
includes:
  - source.yaml
scenario_overrides_by_name:
  scenario_b:
    simulation_config:
      max_episode_steps: 240
""",
    )

    scenarios = load_scenarios(manifest)

    assert [scenario["name"] for scenario in scenarios] == [
        "repeated",
        "scenario_b",
        "repeated",
    ]
    assert scenarios[1]["simulation_config"]["max_episode_steps"] == 240


def test_load_scenarios_rejects_case_duplicate_named_overrides(tmp_path: Path) -> None:
    """Per-scenario override targets should be unambiguous after case normalization."""
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
scenario_overrides_by_name:
  scenario_a:
    simulation_config:
      max_episode_steps: 240
  Scenario_A:
    simulation_config:
      max_episode_steps: 480
""",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate case-insensitive scenario_overrides_by_name",
    ):
        load_scenarios(manifest)


def test_load_scenarios_rejects_unknown_named_override(tmp_path: Path) -> None:
    """Per-scenario overrides must fail closed when a target name is absent."""
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
scenario_overrides_by_name:
  scenario_missing:
    simulation_config:
      max_episode_steps: 240
""",
    )

    with pytest.raises(ValueError, match="Unknown scenario_overrides_by_name"):
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


def test_load_scenarios_preserves_platform_semantics_regions(tmp_path: Path) -> None:
    """Platform semantics should be explicit YAML metadata, not implicit map geometry."""
    scenario_file = tmp_path / "platform_semantics.yaml"
    _write_yaml(
        scenario_file,
        """
scenarios:
  - name: platform_semantics_ok
    map_file: maps/platform.svg
    platform_semantics:
      status: metadata_only
      regions:
        - id: platform_edge
          kind: hazard
          shape: polygon
          points:
            - [0.0, 0.0]
            - [2.0, 0.0]
            - [2.0, 0.5]
        - id: train_door_keep_clear
          kind: keep_clear
          shape: bbox
          bounds: [4.0, 1.0, 6.0, 2.0]
""",
    )

    scenarios = load_scenarios(scenario_file)

    semantics = scenarios[0]["platform_semantics"]
    assert semantics["status"] == "metadata_only"
    assert [region["kind"] for region in semantics["regions"]] == ["hazard", "keep_clear"]


def test_load_scenarios_rejects_invalid_platform_semantics(tmp_path: Path) -> None:
    """Invalid platform semantics should fail closed instead of being silently ignored."""
    scenario_file = tmp_path / "bad_platform_semantics.yaml"
    _write_yaml(
        scenario_file,
        """
scenarios:
  - name: platform_semantics_bad
    map_file: maps/platform.svg
    platform_semantics:
      status: metadata_only
      regions:
        - id: platform_edge
          kind: escalator
          shape: polygon
          points:
            - [0.0, 0.0]
            - [2.0, 0.0]
            - [2.0, 0.5]
""",
    )

    with pytest.raises(ValueError, match="platform_semantics.regions\\[0\\].kind"):
        load_scenarios(scenario_file)


def test_build_robot_config_rejects_required_platform_semantic_consumers(tmp_path: Path) -> None:
    """Scenarios requiring platform semantic consumers should fail before benchmark use."""
    scenario_file = tmp_path / "required_platform_semantics.yaml"
    _write_yaml(
        scenario_file,
        """
scenarios:
  - name: platform_semantics_requires_consumers
    map_file: maps/platform.svg
    platform_semantics:
      status: require_consumers
      regions:
        - id: platform_edge
          kind: hazard
          shape: polygon
          points:
            - [0.0, 0.0]
            - [2.0, 0.0]
            - [2.0, 0.5]
""",
    )
    scenario = load_scenarios(scenario_file)[0]

    with pytest.raises(NotImplementedError, match="platform_semantics consumers"):
        build_robot_config_from_scenario(scenario, scenario_path=scenario_file)


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


def test_map_id_enforces_robot_runtime_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario map_id resolution should fail closed on missing runtime capability."""
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    map_path = maps_dir / "demo.svg"
    map_path.write_text("<svg></svg>", encoding="utf-8")

    registry_path = tmp_path / "registry.yaml"
    _write_yaml(
        registry_path,
        f"""
version: 2
schema: robot_sf.map_catalog.v2
parser_version: parser-capability-metadata.v1
maps:
  - map_id: demo_map
    path: maps/demo.svg
    source_sha256: {_sha256_file(map_path)}
    role: obstacle_only
    profile: obstacle_only
    capabilities:
      robot_runtime: false
      pedestrian_runtime: false
      route_only: false
      obstacle_source: true
      benchmark_candidate: false
    limitations:
      - no_robot_routes
    validation:
      status: unchecked
""",
    )
    manifest = tmp_path / "manifest.yaml"
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

    with pytest.raises(ValueError, match="requested profile 'robot_runtime'.*no_robot_routes"):
        load_scenarios(manifest, base_dir=manifest)
    scenario_loader._load_map_registry.cache_clear()


def test_map_id_allows_route_only_for_requested_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route-only maps should pass only when callers request route-only capability."""
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    map_path = maps_dir / "route_only.svg"
    map_path.write_text("<svg></svg>", encoding="utf-8")

    registry_path = tmp_path / "registry.yaml"
    _write_yaml(
        registry_path,
        f"""
version: 2
schema: robot_sf.map_catalog.v2
parser_version: parser-capability-metadata.v1
maps:
  - map_id: route_only
    path: maps/route_only.svg
    source_sha256: {_sha256_file(map_path)}
    role: route_only
    profile: route_only
    capabilities:
      robot_runtime: false
      pedestrian_runtime: false
      route_only: true
      obstacle_source: false
      benchmark_candidate: false
    limitations:
      - route_derived_zones
    validation:
      status: unchecked
""",
    )
    manifest = tmp_path / "manifest.yaml"
    _write_yaml(
        manifest,
        """
scenarios:
  - name: demo
    map_id: route_only
    required_map_profile: route_only
""",
    )

    monkeypatch.setenv("ROBOT_SF_MAP_REGISTRY", str(registry_path))
    scenario_loader._load_map_registry.cache_clear()

    scenarios = load_scenarios(manifest, base_dir=manifest)

    scenario_loader._load_map_registry.cache_clear()
    assert scenarios[0]["map_file"] == "maps/route_only.svg"


def test_map_id_allows_benchmark_candidate_for_requested_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark callers should be able to request benchmark_candidate rows."""
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    map_path = maps_dir / "benchmark.svg"
    map_path.write_text("<svg></svg>", encoding="utf-8")

    registry_path = tmp_path / "registry.yaml"
    _write_yaml(
        registry_path,
        f"""
version: 2
schema: robot_sf.map_catalog.v2
parser_version: parser-capability-metadata.v1
maps:
  - map_id: benchmark_map
    path: maps/benchmark.svg
    source_sha256: {_sha256_file(map_path)}
    role: benchmark_candidate
    profile: benchmark_candidate
    capabilities:
      robot_runtime: true
      pedestrian_runtime: true
      route_only: false
      obstacle_source: true
      benchmark_candidate: true
    limitations: []
    validation:
      status: reviewed
""",
    )
    manifest = tmp_path / "manifest.yaml"
    _write_yaml(
        manifest,
        """
scenarios:
  - name: benchmark
    map_id: benchmark_map
    required_map_profile: benchmark_candidate
""",
    )

    monkeypatch.setenv("ROBOT_SF_MAP_REGISTRY", str(registry_path))
    scenario_loader._load_map_registry.cache_clear()

    scenarios = load_scenarios(manifest, base_dir=manifest)

    scenario_loader._load_map_registry.cache_clear()
    assert scenarios[0]["map_file"] == "maps/benchmark.svg"


def test_map_id_rejects_stale_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catalog source hashes should fail closed when SVG files drift."""
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    (maps_dir / "demo.svg").write_text("<svg></svg>", encoding="utf-8")

    registry_path = tmp_path / "registry.yaml"
    _write_yaml(
        registry_path,
        """
version: 2
schema: robot_sf.map_catalog.v2
parser_version: parser-capability-metadata.v1
maps:
  - map_id: demo_map
    path: maps/demo.svg
    source_sha256: stale
    role: robot_runtime
    profile: robot_runtime
    capabilities:
      robot_runtime: true
      pedestrian_runtime: false
      route_only: false
      obstacle_source: false
      benchmark_candidate: false
    limitations: []
    validation:
      status: unchecked
""",
    )
    manifest = tmp_path / "manifest.yaml"
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

    with pytest.raises(ValueError, match="source_sha256 is stale"):
        load_scenarios(manifest, base_dir=manifest)
    scenario_loader._load_map_registry.cache_clear()


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
