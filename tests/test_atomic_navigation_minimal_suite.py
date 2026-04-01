"""Validation tests for the issue-596 atomic navigation scenario suite."""

from __future__ import annotations

from math import dist
from pathlib import Path

import pytest

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.ped_npc.ped_zone import sample_zone
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

ROOT = Path(__file__).resolve().parent.parent
FULL_MANIFEST = ROOT / "configs" / "scenarios" / "sets" / "atomic_navigation_minimal_full_v1.yaml"
SUBSET_MANIFEST = ROOT / "configs" / "scenarios" / "sets" / "verified_simple_subset_v1.yaml"
VALIDATION_MANIFEST = (
    ROOT / "configs" / "scenarios" / "sets" / "atomic_navigation_validation_fixtures_v1.yaml"
)

REQUIRED_METADATA_FIELDS = {
    "purpose",
    "expected_behavior",
    "expected_pass_criteria",
    "failure_modes",
    "primary_capability",
    "target_failure_mode",
    "determinism",
    "plausibility",
}
EXPECTED_FULL_SCENARIOS = {
    "empty_map_8_directions_east",
    "empty_map_8_directions_northeast",
    "empty_map_8_directions_north",
    "empty_map_8_directions_northwest",
    "empty_map_8_directions_west",
    "empty_map_8_directions_southwest",
    "empty_map_8_directions_south",
    "empty_map_8_directions_southeast",
    "goal_behind_robot",
    "small_angle_precision",
    "single_obstacle_circle",
    "single_obstacle_rectangle",
    "line_wall_detour",
    "narrow_passage",
    "corner_90_turn",
    "u_trap_local_minimum",
    "corridor_following",
    "single_ped_crossing_orthogonal",
    "head_on_interaction",
    "overtaking_interaction",
    "start_near_obstacle",
    "goal_very_close",
    "symmetry_ambiguous_choice",
}
EXPECTED_SUBSET_SCENARIOS = {
    "empty_map_8_directions_east",
    "empty_map_8_directions_north",
    "empty_map_8_directions_west",
    "goal_behind_robot",
    "single_obstacle_circle",
    "line_wall_detour",
    "narrow_passage",
    "single_ped_crossing_orthogonal",
    "head_on_interaction",
    "overtaking_interaction",
}
ZERO_DENSITY_ADVISORY = "zero_baseline_route_spawn"
ATOMIC_MAP_FILENAMES = [
    "atomic_corner_90_test.svg",
    "atomic_corridor_test.svg",
    "atomic_empty_frame_test.svg",
    "atomic_goal_close_test.svg",
    "atomic_goal_inside_obstacle_invalid.svg",
    "atomic_line_wall_obstacle.svg",
    "atomic_narrow_passage_test.svg",
    "atomic_single_circle_obstacle.svg",
    "atomic_single_rectangle_obstacle.svg",
    "atomic_start_near_obstacle_test.svg",
    "atomic_symmetry_split_test.svg",
    "atomic_u_trap_test.svg",
]


def _load(path: Path) -> list[dict]:
    return list(load_scenarios(path, base_dir=path))


def test_atomic_navigation_full_manifest_loads_with_expected_scenarios() -> None:
    """The full atomic suite should load all expected runnable scenario entries."""
    scenarios = _load(FULL_MANIFEST)

    assert len(scenarios) == len(EXPECTED_FULL_SCENARIOS)
    assert {scenario["name"] for scenario in scenarios} == EXPECTED_FULL_SCENARIOS


def test_verified_simple_subset_loads_with_expected_scenarios() -> None:
    """The verified-simple subset should stay a strict named subset of the full suite."""
    scenarios = _load(SUBSET_MANIFEST)

    assert len(scenarios) == len(EXPECTED_SUBSET_SCENARIOS)
    assert {scenario["name"] for scenario in scenarios} == EXPECTED_SUBSET_SCENARIOS


def test_atomic_navigation_scenarios_include_metadata_contract_and_resolved_maps() -> None:
    """Each scenario should retain its metadata contract and resolve to an existing map file."""
    scenarios = _load(FULL_MANIFEST)

    for scenario in scenarios:
        metadata = scenario["metadata"]
        missing = REQUIRED_METADATA_FIELDS - metadata.keys()
        assert not missing, f"{scenario['name']} missing metadata fields: {sorted(missing)}"
        assert isinstance(metadata["failure_modes"], list) and metadata["failure_modes"], (
            f"{scenario['name']} should list concrete failure modes"
        )
        assert (
            isinstance(metadata["expected_pass_criteria"], str)
            and metadata["expected_pass_criteria"].strip()
        ), f"{scenario['name']} should define pass criteria"
        assert (
            isinstance(metadata["target_failure_mode"], str)
            and metadata["target_failure_mode"].strip()
        ), f"{scenario['name']} should define a normalized target failure mode"
        map_file = (FULL_MANIFEST.parent / scenario["map_file"]).resolve()
        assert map_file.exists(), f"Map file missing for {scenario['name']}: {map_file}"


def test_verified_simple_subset_scenarios_include_metadata_contract() -> None:
    """The verified-simple subset should preserve the same metadata contract as the full suite."""
    scenarios = _load(SUBSET_MANIFEST)

    for scenario in scenarios:
        metadata = scenario["metadata"]
        missing = REQUIRED_METADATA_FIELDS - metadata.keys()
        assert not missing, f"{scenario['name']} missing metadata fields: {sorted(missing)}"


def test_atomic_issue_596_maps_pass_repo_verifier(tmp_path: Path) -> None:
    """The new atomic SVG maps should be accepted by the repository map verifier."""
    import subprocess
    import sys

    for map_name in ATOMIC_MAP_FILENAMES:
        output_path = tmp_path / f"{map_name}.json"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/validation/verify_maps.py",
                "--scope",
                map_name,
                "--mode",
                "ci",
                "--output",
                str(output_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert output_path.exists(), f"Verifier did not emit a manifest for {map_name}"


@pytest.mark.parametrize(
    "scenario_name",
    ["line_wall_detour", "narrow_passage", "symmetry_ambiguous_choice"],
)
def test_atomic_topology_scenarios_rebase_initial_handoff_target(scenario_name: str) -> None:
    """Topology-heavy scenarios should not start with an immediately satisfiable handoff target."""
    scenarios = {scenario["name"]: scenario for scenario in _load(FULL_MANIFEST)}
    scenario = scenarios[scenario_name]

    config = build_robot_config_from_scenario(scenario, scenario_path=FULL_MANIFEST)
    map_def = next(iter(config.map_pool.map_defs.values()))
    env = make_robot_env(config=config, seed=0, debug=False)
    try:
        env.reset(seed=0)
        nav = env.simulator.robot_navs[0]
        robot_pos = env.simulator.robots[0].pose[0]
        original_first_waypoint = map_def.robot_routes[0].waypoints[0]

        assert dist(original_first_waypoint, robot_pos) <= nav.proximity_threshold
        assert dist(nav.current_waypoint, robot_pos) > nav.proximity_threshold
        assert not nav.reached_waypoint
    finally:
        env.close()


def test_zero_density_scenarios_set_density_advisory() -> None:
    """Zero-density scenarios should declare the advisory used by the matrix validation rules."""
    scenarios = _load(FULL_MANIFEST)

    for scenario in scenarios:
        density = scenario["simulation_config"].get("ped_density")
        if density == 0.0:
            assert scenario["metadata"].get("density_advisory") == ZERO_DENSITY_ADVISORY


def test_empty_direction_family_expands_all_compass_variants() -> None:
    """The empty-map directional family should cover all eight compass directions explicitly."""
    scenarios = _load(FULL_MANIFEST)

    family = [
        scenario
        for scenario in scenarios
        if scenario["metadata"].get("variation_family") == "goal_direction"
    ]
    assert len(family) == 8
    assert {scenario["metadata"]["variation_value"] for scenario in family} == {
        "east",
        "northeast",
        "north",
        "northwest",
        "west",
        "southwest",
        "south",
        "southeast",
    }


def test_dynamic_scenarios_remain_single_pedestrian_cases() -> None:
    """Dynamic scenarios should remain sparse single-pedestrian interaction cases."""
    scenarios = _load(FULL_MANIFEST)
    dynamic = {
        scenario["name"]: scenario
        for scenario in scenarios
        if scenario["metadata"]["primary_capability"] == "dynamic_interaction"
    }

    assert set(dynamic) == {
        "single_ped_crossing_orthogonal",
        "head_on_interaction",
        "overtaking_interaction",
    }
    for scenario in dynamic.values():
        assert scenario["simulation_config"]["ped_density"] == 0.0
        assert len(scenario.get("single_pedestrians", [])) == 1


def test_goal_inside_obstacle_fixture_fails_closed_when_sampling_goal() -> None:
    """The invalid goal fixture should fail when the system tries to sample a usable goal."""
    scenarios = _load(VALIDATION_MANIFEST)
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["name"] == "goal_inside_obstacle_invalid"

    config = build_robot_config_from_scenario(scenario, scenario_path=VALIDATION_MANIFEST)
    _map_name, map_def = next(iter(config.map_pool.map_defs.items()))
    obstacle_polygons = [obstacle.vertices for obstacle in map_def.obstacles]

    with pytest.raises(RuntimeError, match="Failed to sample"):
        sample_zone(map_def.robot_goal_zones[0], 1, obstacle_polygons=obstacle_polygons)
