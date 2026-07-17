"""End-to-end acceptance test for issue #5829: a geometry-less single-interaction
scenario (no pedestrian routes and no crowded zones) with a forced ``population_size``
must build a valid harness manifest at build time AND realize the exact declared
population at simulation time.

Before option (a) (PR #5837) this raised
``ValueError: force_population_size requires a pedestrian route or crowded zone for the
remaining N background pedestrians`` only after hours of valid crowd-scenario compute.
With option (a), the forced remainder is scatter-synthesized into free walkable space, so
the cell builds a valid manifest row (per build-time realizability, PR #5831) and the
declared count is realized at runtime. This test locks both halves of that contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from robot_sf.benchmark.heterogeneous_population_ablation import (
    build_mean_matched_harness_manifest,
)
from robot_sf.nav.navigation import get_prepared_obstacles
from robot_sf.nav.svg_map_parser import SvgMapConverter
from robot_sf.ped_npc.ped_archetypes import assign_archetype_labels
from robot_sf.ped_npc.ped_population import PedSpawnConfig, populate_simulation

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = (
    _REPO_ROOT / "configs" / "benchmarks" / "issue_5829_geometryless_forced_population_smoke.yaml"
)
_FRANCIS_SVG = _REPO_ROOT / "maps" / "svg_maps" / "francis2023" / "francis2023_blind_corner.svg"
_FORCED_SIZE = 12


def test_geometryless_forced_population_builds_valid_manifest_row():
    """Build-time gate accepts the now-synthesizable cell and emits a valid row.

    Acceptance (issue #5829): a manifest containing a single-interaction scenario +
    forced population builds with a valid spawn plan (option a) and never fails at
    simulation hour N. The manifest build must return a ``ready`` status with a
    realized population_size of 12 and no blockers.
    """
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = build_mean_matched_harness_manifest(
        config,
        config_path=str(_CONFIG_PATH),
    )

    # The cell must build successfully: no build-time blocker and no "not realizable"
    # rejection. ``pending_runtime_capture`` is the expected pre-run state because the
    # pedestrian control trace is only captured during an actual campaign run.
    assert isinstance(manifest, dict)
    assert manifest["status"] == "pending_runtime_capture"
    assert manifest["blockers"] == []
    assert manifest["row_count"] == 2

    scenario_rows = manifest["scenario_rows"]
    assert len(scenario_rows) == 1
    row = scenario_rows[0]
    assert row["scenario_id"] == "francis2023_blind_corner_forced_pop_12"
    assert row["population_size"] == _FORCED_SIZE
    # The heterogeneous arm must allocate the full forced count, not drop to the
    # single-interaction map's native 1 pedestrian.
    het_counts = row["heterogeneous_counts"]
    assert sum(het_counts.values()) == _FORCED_SIZE


def test_geometryless_forced_population_realizes_declared_count_at_runtime():
    """The produced scenario actually populates the exact forced count at runtime.

    Drives the geometry-less francis map through ``populate_simulation`` with the
    forced population contract the manifest encodes, asserting 12 peds are realized
    without the historical startup ``ValueError``.
    """
    map_def = SvgMapConverter(str(_FRANCIS_SVG)).get_map_definition()
    assert not map_def.ped_routes
    assert not map_def.ped_crowded_zones

    spawn_config = PedSpawnConfig(
        peds_per_area_m2=0.0,
        max_group_members=3,
        initial_speed=0.5,
        force_population_size=_FORCED_SIZE,
        route_spawn_seed=219,
        archetype_composition={"cautious": 0.25, "standard": 0.5, "hurried": 0.25},
        archetype_speed_factors={"cautious": 0.7, "standard": 1.0, "hurried": 1.4},
        archetype_seed=3574,
    )
    ped_states, _groups, _behaviors = populate_simulation(
        tau=0.5,
        spawn_config=spawn_config,
        ped_routes=map_def.ped_routes,
        ped_crowded_zones=map_def.ped_crowded_zones,
        obstacle_polygons=get_prepared_obstacles(map_def),
        single_pedestrians=map_def.single_pedestrians,
        map_bounds=map_def.get_map_bounds(),
        reserved_zones=[*map_def.robot_spawn_zones, *map_def.robot_goal_zones],
        ped_radius=0.4,
        reserved_zone_radius=0.3,
    )

    assert ped_states.num_peds == _FORCED_SIZE
    speeds = np.linalg.norm(ped_states.ped_velocities, axis=1)
    labels = assign_archetype_labels(
        _FORCED_SIZE,
        spawn_config.archetype_composition,
        seed=spawn_config.archetype_seed,
    )
    expected_speeds = np.asarray(
        [0.5 * spawn_config.archetype_speed_factors[label] for label in labels],
    )
    assert np.allclose(speeds, expected_speeds)
