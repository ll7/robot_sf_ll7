"""Static witnesses for release endpoint safety; no environment is constructed."""

from pathlib import Path

from shapely.geometry import LineString

from robot_sf.evidence.writers import write_json
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from scripts.validation.check_scenario_archetype_geometry import _rect_polygon

MATRIX = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")


def test_overtaking_lane_cannot_intersect_full_robot_spawn_rectangle():
    scenario = next(
        row for row in load_scenarios(MATRIX) if row["name"] == "francis2023_pedestrian_overtaking"
    )
    config = build_robot_config_from_scenario(scenario, scenario_path=MATRIX.resolve())
    definition = next(iter(config.map_pool.map_defs.values()))
    pedestrian = definition.single_pedestrians[0]
    lane = LineString([pedestrian.start, pedestrian.goal])
    assert all(
        _rect_polygon(zone).distance(lane) > config.sim_config.ped_radius
        for zone in definition.robot_spawn_zones
    ), "pedestrian overtaking lane intersects full robot spawn rectangle"


def test_overtaking_retains_a_faster_pedestrian_behind_the_full_robot_spawn():
    scenario = next(
        row for row in load_scenarios(MATRIX) if row["name"] == "francis2023_pedestrian_overtaking"
    )
    config = build_robot_config_from_scenario(scenario, scenario_path=MATRIX.resolve())
    definition = next(iter(config.map_pool.map_defs.values()))
    pedestrian = definition.single_pedestrians[0]
    walker_speed = scenario["single_pedestrians"][0]["speed_m_s"]
    assert pedestrian.start[0] < min(
        _rect_polygon(zone).bounds[0] for zone in definition.robot_spawn_zones
    ), "overtaking pedestrian must start behind every sampled robot"
    assert config.robot_config.max_linear_speed < walker_speed, (
        "the robot must be slower than the pedestrian being tested as an overtaker"
    )
    assert config.sim_config.sim_time_in_secs > (34 / config.robot_config.max_linear_speed + 5), (
        "the slower robot needs enough time to finish the unchanged route"
    )


def test_active_crowd_zones_cannot_intersect_robot_destination_rectangles():
    for scenario in load_scenarios(MATRIX):
        if scenario["name"] not in {
            "classic_station_platform_medium",
            "francis2023_robot_crowding",
        }:
            continue
        config = build_robot_config_from_scenario(scenario, scenario_path=MATRIX.resolve())
        definition = next(iter(config.map_pool.map_defs.values()))
        for robot_zone in definition.robot_goal_zones:
            for crowd_zone in definition.ped_spawn_zones + definition.ped_crowded_zones:
                assert (
                    _rect_polygon(robot_zone).distance(_rect_polygon(crowd_zone))
                    > config.sim_config.ped_radius
                ), f"{scenario['name']}: active crowd spawn intersects robot goal"


def test_release_zone_audit_has_all_51_scenarios_and_102_endpoint_rectangles():
    from scripts.validation.check_scenario_archetype_geometry import inspect_release_zones

    rows = inspect_release_zones()
    assert len(rows) == 102
    assert len({row["scenario"] for row in rows}) == 51
    assert {row["ped_radius_m"] for row in rows} == {0.4}


def test_checked_in_dispositions_are_exact_and_do_not_waive_overtaking():
    from scripts.validation.check_scenario_archetype_geometry import (
        enforce_release_zone_waivers,
        inspect_release_zones,
    )

    rows = inspect_release_zones()
    assert not any(
        row["intersections"]
        for row in rows
        if row["scenario"] == "francis2023_pedestrian_overtaking"
    )
    enforce_release_zone_waivers(
        rows, Path("configs/scenarios/release_0_0_8_endpoint_dispositions.yaml")
    )


def test_original_overtaking_map_is_refused_without_a_disposition(tmp_path):
    import pytest

    from scripts.validation.check_scenario_archetype_geometry import (
        enforce_release_zone_waivers,
        inspect_release_zones,
    )
    from scripts.validation.scenario_validation_waivers import WaiverValidationError

    manifest = tmp_path / "original.yaml"
    write_json(
        manifest,
        {
            "scenarios": [
                {
                    "name": "francis2023_pedestrian_overtaking",
                    "map_file": str(
                        Path(
                            "maps/successor_svg_maps/issue_9762_francis2023_ped_overtaking_goal_zone_entry_v2.svg"
                        ).resolve()
                    ),
                    "simulation_config": {"ped_density": 0.0},
                }
            ]
        },
    )
    rows = inspect_release_zones([manifest])
    assert rows[0]["bounds"] == [3.0, 4.0, 5.0, 6.0]
    assert rows[0]["intersections"][0]["actor"] == "h1"
    waivers = tmp_path / "waivers.yaml"
    write_json(waivers, {"schema": "scenario_validation_waivers.v1", "release_zones": []})
    with pytest.raises(WaiverValidationError, match="missing release zone overlap"):
        enforce_release_zone_waivers(rows, waivers)


def test_radius_only_intersection_and_trajectory_override_are_detected(tmp_path):

    from scripts.validation.check_scenario_archetype_geometry import inspect_release_zones

    manifest = tmp_path / "radius.yaml"
    write_json(
        manifest,
        {
            "scenarios": [
                {
                    "name": "radius_probe",
                    "map_file": str(
                        Path(
                            "maps/successor_svg_maps/issue_10063_francis2023_ped_overtaking_safe_spawn_v1.svg"
                        ).resolve()
                    ),
                    "simulation_config": {"ped_density": 0.0},
                    "single_pedestrians": [
                        {"id": "h1", "goal": None, "trajectory": [[2, 4.8], [8, 4.8]]}
                    ],
                }
            ]
        },
    )
    rows = inspect_release_zones([manifest])
    hit = rows[0]["intersections"][0]
    assert hit["distance_m"] > 0
    assert hit["distance_m"] < rows[0]["ped_radius_m"]
    assert (2.0, 4.8) in hit["evidence"]["points"]


def test_geometry_change_invalidates_intended_overlap_disposition(tmp_path, monkeypatch):
    import pytest
    import yaml

    from scripts.validation.check_scenario_archetype_geometry import (
        enforce_release_zone_waivers,
        inspect_release_zones,
    )
    from scripts.validation.scenario_validation_waivers import WaiverValidationError

    rows = inspect_release_zones()
    doc = yaml.safe_load(
        Path("configs/scenarios/release_0_0_8_endpoint_dispositions.yaml").read_text()
    )
    doc["release_zones"][0]["geometry_sha256"] = "0" * 64
    waivers = tmp_path / "waivers.yaml"
    write_json(waivers, doc)
    with pytest.raises(WaiverValidationError, match="changed"):
        enforce_release_zone_waivers(rows, waivers)

    # Density zero is not dormant when an exact population override is supplied.
    # Resolve a real scenario through the normal loader; don't forge the fingerprint.
    from robot_sf.training import scenario_loader

    original_loader = scenario_loader.load_scenarios

    def force_crowd_population(path):
        scenarios = original_loader(path)
        for scenario in scenarios:
            if scenario["name"] == "classic_bottleneck_low":
                scenario["simulation_config"]["population_size"] = 3
        return scenarios

    monkeypatch.setattr(scenario_loader, "load_scenarios", force_crowd_population)
    forced_rows = inspect_release_zones()
    with pytest.raises(WaiverValidationError, match="changed"):
        enforce_release_zone_waivers(
            forced_rows, Path("configs/scenarios/release_0_0_8_endpoint_dispositions.yaml")
        )


def test_station_route_starts_in_moved_zone_and_crowding_keeps_24_pedestrians():
    from math import ceil

    for scenario in load_scenarios(MATRIX):
        if scenario["name"] not in {
            "classic_station_platform_medium",
            "francis2023_robot_crowding",
        }:
            continue
        config = build_robot_config_from_scenario(scenario, scenario_path=MATRIX.resolve())
        definition = next(iter(config.map_pool.map_defs.values()))
        if scenario["name"] == "francis2023_robot_crowding":
            assert (
                ceil(
                    sum(_rect_polygon(z).area for z in definition.ped_crowded_zones)
                    * config.sim_config.peds_per_area_m2
                )
                == 24
            )
        else:
            from shapely.geometry import Point

            assert _rect_polygon(definition.ped_spawn_zones[1]).covers(
                Point(definition.ped_routes[1].waypoints[0])
            )


def test_release_zone_cli_enforces_dispositions(capsys, tmp_path):

    from scripts.validation.check_scenario_archetype_geometry import main

    assert (
        main(
            [
                "--release-zones",
                "--waiver-file",
                "configs/scenarios/release_0_0_8_endpoint_dispositions.yaml",
            ]
        )
        == 0
    )
    assert "classic_bottleneck_low" in capsys.readouterr().out
    assert main(["--release-zones"]) == 2
    assert "requires --waiver-file" in capsys.readouterr().err
    invalid = tmp_path / "invalid.yaml"
    write_json(invalid, {"schema": "scenario_validation_waivers.v1", "release_zones": []})
    assert main(["--release-zones", "--waiver-file", str(invalid)]) == 2
    assert "missing release zone overlap" in capsys.readouterr().err


def test_invalid_disposition_and_missing_map_fail_closed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from robot_sf.training import scenario_loader
    from scripts.validation.check_scenario_archetype_geometry import (
        enforce_release_zone_waivers,
        inspect_release_zones,
    )
    from scripts.validation.scenario_validation_waivers import WaiverValidationError

    invalid = tmp_path / "invalid.yaml"
    write_json(
        invalid,
        {
            "schema": "scenario_validation_waivers.v1",
            "release_zones": [{"rationale": "test", "decision_ref": "#10063"}],
        },
    )
    with pytest.raises(WaiverValidationError, match="requires exact identity"):
        enforce_release_zone_waivers([], invalid)
    monkeypatch.setattr(
        scenario_loader,
        "build_robot_config_from_scenario",
        lambda *a, **k: SimpleNamespace(map_pool=None),
    )
    with pytest.raises(ValueError, match="Missing map"):
        inspect_release_zones([MATRIX])


def test_stationary_single_pedestrian_is_a_radius_expanded_point():
    from types import SimpleNamespace

    from scripts.validation.check_scenario_archetype_geometry import _release_actors

    definition = SimpleNamespace(
        single_pedestrians=[
            SimpleNamespace(id="stationary", start=(4, 5), goal=None, trajectory=None, role="wait")
        ],
        ped_spawn_zones=[],
        ped_crowded_zones=[],
    )
    actors = _release_actors(definition, 0)
    assert actors[0][2].geom_type == "Point"
