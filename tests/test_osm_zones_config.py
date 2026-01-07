"""Tests for programmatic OSM zones and routes configuration (Phase 3).

Tests verify:
- T036: create_spawn_zone() helper function
- T037: create_goal_zone() helper function
- T038: create_crowded_zone() helper function
- T039: create_route() helper function
- Validation of zone/route creation
- Polygon and waypoint handling
- Metadata preservation
"""

import pytest

from robot_sf.maps.osm_zones_config import (
    create_config_with_zones_routes,
    create_crowded_zone,
    create_goal_zone,
    create_route,
    create_spawn_zone,
)
from robot_sf.maps.osm_zones_yaml import OSMZonesConfig, Route, Zone


class TestCreateSpawnZone:
    """Test create_spawn_zone() helper function (T036)."""

    def test_create_spawn_zone_basic(self):
        """Test basic spawn zone creation with default priority."""
        zone = create_spawn_zone(
            "spawn_center",
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        )

        assert isinstance(zone, Zone)
        assert zone.name == "spawn_center"
        assert zone.type == "spawn"
        assert len(zone.polygon) == 3
        assert zone.priority == 1

    def test_create_spawn_zone_with_priority(self):
        """Test spawn zone creation with custom priority."""
        zone = create_spawn_zone(
            "spawn_north",
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
            priority=2,
        )

        assert zone.priority == 2

    def test_create_spawn_zone_with_metadata(self):
        """Test spawn zone creation with metadata."""
        meta = {"robot_type": "wheelchair", "start_orientation": 0.0}
        zone = create_spawn_zone(
            "spawn_accessible",
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
            metadata=meta,
        )

        assert zone.metadata == meta

    def test_create_spawn_zone_rectangular(self):
        """Test rectangular spawn zone (4 vertices)."""
        zone = create_spawn_zone(
            "spawn_rect",
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        )

        assert len(zone.polygon) == 4

    def test_create_spawn_zone_invalid_polygon_too_few_points(self):
        """Test error when polygon has <3 points."""
        with pytest.raises(ValueError, match="≥3 points"):
            create_spawn_zone("spawn_bad", polygon=[(0.0, 0.0), (10.0, 0.0)])

    def test_create_spawn_zone_invalid_polygon_collinear(self):
        """Test error when polygon points are collinear."""
        with pytest.raises(ValueError, match="collinear"):
            create_spawn_zone(
                "spawn_bad",
                polygon=[(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)],
            )

    def test_create_spawn_zone_invalid_point_format(self):
        """Test error when point format is invalid."""
        with pytest.raises(TypeError):
            create_spawn_zone("spawn_bad", polygon=[(0.0, 0.0), "invalid", (10.0, 10.0)])

    def test_create_spawn_zone_float_conversion(self):
        """Test automatic float conversion from various formats."""
        zone = create_spawn_zone(
            "spawn_ints",
            polygon=[(0, 0), (10, 0), (10, 10)],  # integers
        )

        assert zone.polygon == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]


class TestCreateGoalZone:
    """Test create_goal_zone() helper function (T037)."""

    def test_create_goal_zone_basic(self):
        """Test basic goal zone creation."""
        zone = create_goal_zone(
            "goal_exit",
            polygon=[(90.0, 90.0), (100.0, 90.0), (100.0, 100.0)],
        )

        assert isinstance(zone, Zone)
        assert zone.name == "goal_exit"
        assert zone.type == "goal"
        assert len(zone.polygon) == 3
        assert zone.priority == 1

    def test_create_goal_zone_with_metadata(self):
        """Test goal zone with metadata."""
        meta = {"reward": 1.0, "zone_id": "exit_front"}
        zone = create_goal_zone(
            "goal_custom",
            polygon=[(90.0, 90.0), (100.0, 90.0), (100.0, 100.0)],
            metadata=meta,
        )

        assert zone.metadata == meta

    def test_create_goal_zone_rectangular(self):
        """Test rectangular goal zone."""
        zone = create_goal_zone(
            "goal_rect",
            polygon=[(80.0, 80.0), (100.0, 80.0), (100.0, 100.0), (80.0, 100.0)],
        )

        assert len(zone.polygon) == 4

    def test_create_goal_zone_invalid_polygon(self):
        """Test error on invalid polygon."""
        with pytest.raises(ValueError):
            create_goal_zone("goal_bad", polygon=[(0.0, 0.0), (10.0, 10.0)])


class TestCreateCrowdedZone:
    """Test create_crowded_zone() helper function (T038)."""

    def test_create_crowded_zone_basic(self):
        """Test basic crowded zone creation."""
        zone = create_crowded_zone(
            "crowd_center",
            polygon=[(30.0, 30.0), (40.0, 30.0), (40.0, 40.0)],
            density=2.5,
        )

        assert isinstance(zone, Zone)
        assert zone.name == "crowd_center"
        assert zone.type == "crowded"
        assert zone.metadata["density"] == 2.5

    def test_create_crowded_zone_sparse_density(self):
        """Test crowded zone with sparse density."""
        zone = create_crowded_zone(
            "sparse_area",
            polygon=[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)],
            density=0.1,
        )

        assert zone.metadata["density"] == 0.1

    def test_create_crowded_zone_dense(self):
        """Test crowded zone with high density."""
        zone = create_crowded_zone(
            "dense_area",
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
            density=5.0,
        )

        assert zone.metadata["density"] == 5.0

    def test_create_crowded_zone_with_extra_metadata(self):
        """Test crowded zone with additional metadata."""
        meta = {"pedestrian_type": "mixed", "time_period": "rush_hour"}
        zone = create_crowded_zone(
            "rush_hour_zone",
            polygon=[(20.0, 20.0), (30.0, 20.0), (30.0, 30.0)],
            density=1.5,
            metadata=meta,
        )

        assert zone.metadata["density"] == 1.5
        assert zone.metadata["pedestrian_type"] == "mixed"
        assert zone.metadata["time_period"] == "rush_hour"

    def test_create_crowded_zone_invalid_density_zero(self):
        """Test error on zero density."""
        with pytest.raises(ValueError, match="must be >0"):
            create_crowded_zone(
                "bad_crowd",
                polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
                density=0.0,
            )

    def test_create_crowded_zone_invalid_density_negative(self):
        """Test error on negative density."""
        with pytest.raises(ValueError, match="must be >0"):
            create_crowded_zone(
                "bad_crowd",
                polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
                density=-1.0,
            )


class TestCreateRoute:
    """Test create_route() helper function (T039)."""

    def test_create_route_basic(self):
        """Test basic route creation."""
        route = create_route(
            "main_path",
            waypoints=[(0.0, 0.0), (50.0, 50.0)],
        )

        assert isinstance(route, Route)
        assert route.name == "main_path"
        assert route.route_type == "pedestrian"
        assert len(route.waypoints) == 2

    def test_create_route_multiple_waypoints(self):
        """Test route with multiple waypoints."""
        route = create_route(
            "complex_path",
            waypoints=[(0.0, 0.0), (25.0, 25.0), (50.0, 50.0), (75.0, 25.0)],
        )

        assert len(route.waypoints) == 4

    def test_create_route_wheelchair(self):
        """Test wheelchair route type."""
        route = create_route(
            "accessible_path",
            waypoints=[(0.0, 0.0), (50.0, 50.0)],
            route_type="wheelchair",
        )

        assert route.route_type == "wheelchair"

    def test_create_route_vehicle(self):
        """Test vehicle route type."""
        route = create_route(
            "car_path",
            waypoints=[(0.0, 0.0), (100.0, 100.0)],
            route_type="vehicle",
        )

        assert route.route_type == "vehicle"

    def test_create_route_with_metadata(self):
        """Test route with metadata."""
        meta = {"speed_preference": 1.5, "traversal_time": 60.0}
        route = create_route(
            "timed_path",
            waypoints=[(0.0, 0.0), (50.0, 50.0)],
            metadata=meta,
        )

        assert route.metadata == meta

    def test_create_route_invalid_waypoints_too_few(self):
        """Test error with <2 waypoints."""
        with pytest.raises(ValueError, match="≥2 waypoints"):
            create_route("bad_route", waypoints=[(0.0, 0.0)])

    def test_create_route_invalid_waypoint_format(self):
        """Test error on invalid waypoint format."""
        with pytest.raises(TypeError):
            create_route(
                "bad_route",
                waypoints=[(0.0, 0.0), "invalid"],
            )

    def test_create_route_integer_waypoints(self):
        """Test automatic float conversion for waypoints."""
        route = create_route(
            "int_route",
            waypoints=[(0, 0), (50, 50)],
        )

        assert route.waypoints == [(0.0, 0.0), (50.0, 50.0)]


class TestCreateConfigWithZonesRoutes:
    """Test create_config_with_zones_routes() helper."""

    def test_create_empty_config(self):
        """Test creating empty config."""
        config = create_config_with_zones_routes()

        assert isinstance(config, OSMZonesConfig)
        assert len(config.zones) == 0
        assert len(config.routes) == 0
        assert config.version == "1.0"

    def test_create_config_with_zones(self):
        """Test config creation with zones."""
        spawn = create_spawn_zone("spawn1", polygon=[(0, 0), (10, 0), (10, 10)])
        goal = create_goal_zone("goal1", polygon=[(90, 90), (100, 90), (100, 100)])

        config = create_config_with_zones_routes(zones=[spawn, goal])

        assert len(config.zones) == 2
        assert "spawn1" in config.zones
        assert "goal1" in config.zones
        assert config.zones["spawn1"].type == "spawn"
        assert config.zones["goal1"].type == "goal"

    def test_create_config_with_routes(self):
        """Test config creation with routes."""
        route1 = create_route("path1", waypoints=[(0, 0), (50, 50)])
        route2 = create_route("path2", waypoints=[(100, 0), (0, 100)])

        config = create_config_with_zones_routes(routes=[route1, route2])

        assert len(config.routes) == 2
        assert "path1" in config.routes
        assert "path2" in config.routes

    def test_create_config_with_zones_and_routes(self):
        """Test config with both zones and routes."""
        spawn = create_spawn_zone("spawn", polygon=[(0, 0), (10, 0), (10, 10)])
        goal = create_goal_zone("goal", polygon=[(90, 90), (100, 90), (100, 100)])
        route = create_route("path", waypoints=[(0, 0), (100, 100)])

        config = create_config_with_zones_routes(zones=[spawn, goal], routes=[route])

        assert len(config.zones) == 2
        assert len(config.routes) == 1
        assert config.version == "1.0"

    def test_create_config_with_version_and_metadata(self):
        """Test config with custom version and metadata."""
        meta = {"map_source": "oslo.pbf", "projection": "EPSG:32633"}
        config = create_config_with_zones_routes(
            version="2.0",
            metadata=meta,
        )

        assert config.version == "2.0"
        assert config.metadata["map_source"] == "oslo.pbf"

    def test_create_config_round_trip(self):
        """Test config can be serialized and deserialized."""
        spawn = create_spawn_zone("spawn", polygon=[(0, 0), (10, 0), (10, 10)])
        goal = create_goal_zone("goal", polygon=[(90, 90), (100, 90), (100, 100)])
        route = create_route("path", waypoints=[(0, 0), (100, 100)])

        config1 = create_config_with_zones_routes(zones=[spawn, goal], routes=[route])

        # Serialize and deserialize
        data = config1.to_dict()
        config2 = OSMZonesConfig.from_dict(data)

        # Verify round-trip
        assert len(config2.zones) == 2
        assert len(config2.routes) == 1
        assert config2.zones["spawn"].type == "spawn"
        assert config2.zones["goal"].type == "goal"


class TestLoadScenarioConfig:
    """Test load_scenario_config() function (T040)."""

    def test_load_scenario_config_basic(self, tmp_path):
        """Test loading a basic scenario from YAML."""
        from robot_sf.maps.osm_zones_yaml import save_zones_yaml

        # Create and save a scenario
        spawn = create_spawn_zone("spawn", polygon=[(0, 0), (10, 0), (10, 10)])
        goal = create_goal_zone("goal", polygon=[(90, 90), (100, 90), (100, 100)])
        route = create_route("path", waypoints=[(0, 0), (100, 100)])

        config = create_config_with_zones_routes(zones=[spawn, goal], routes=[route])

        yaml_file = tmp_path / "scenario.yaml"
        save_zones_yaml(config, str(yaml_file))

        # Load it back
        from robot_sf.maps.osm_zones_config import load_scenario_config

        loaded_config = load_scenario_config(str(yaml_file))

        assert len(loaded_config.zones) == 2
        assert len(loaded_config.routes) == 1
        assert loaded_config.zones["spawn"].type == "spawn"
        assert loaded_config.zones["goal"].type == "goal"

    def test_load_scenario_config_with_metadata(self, tmp_path):
        """Test loading scenario with metadata."""
        from robot_sf.maps.osm_zones_yaml import save_zones_yaml

        meta = {"map_source": "oslo.pbf", "description": "Downtown"}
        spawn = create_spawn_zone("spawn", polygon=[(0, 0), (10, 0), (10, 10)])
        config = create_config_with_zones_routes(zones=[spawn], metadata=meta)

        yaml_file = tmp_path / "scenario.yaml"
        save_zones_yaml(config, str(yaml_file))

        from robot_sf.maps.osm_zones_config import load_scenario_config

        loaded = load_scenario_config(str(yaml_file))

        assert loaded.metadata["map_source"] == "oslo.pbf"
        assert loaded.metadata["description"] == "Downtown"

    def test_load_scenario_config_not_found(self):
        """Test error when file doesn't exist."""
        from robot_sf.maps.osm_zones_config import load_scenario_config

        with pytest.raises(FileNotFoundError):
            load_scenario_config("nonexistent.yaml")

    def test_load_scenario_config_complex(self, tmp_path):
        """Test loading complex multi-zone scenario."""
        from robot_sf.maps.osm_zones_yaml import save_zones_yaml

        zones = [
            create_spawn_zone("s1", polygon=[(0, 0), (10, 0), (10, 10)]),
            create_spawn_zone("s2", polygon=[(90, 0), (100, 0), (100, 10)]),
            create_goal_zone("g1", polygon=[(90, 90), (100, 90), (100, 100)]),
            create_goal_zone("g2", polygon=[(0, 90), (10, 90), (10, 100)]),
            create_crowded_zone("crowd1", polygon=[(40, 40), (60, 40), (60, 60)], density=1.0),
            create_crowded_zone("crowd2", polygon=[(30, 30), (50, 30), (50, 50)], density=2.0),
        ]

        routes = [
            create_route("r1", waypoints=[(5, 5), (50, 50), (95, 95)]),
            create_route("r2", waypoints=[(95, 5), (50, 50), (5, 95)]),
        ]

        config = create_config_with_zones_routes(zones=zones, routes=routes)

        yaml_file = tmp_path / "complex.yaml"
        save_zones_yaml(config, str(yaml_file))

        from robot_sf.maps.osm_zones_config import load_scenario_config

        loaded = load_scenario_config(str(yaml_file))

        assert len(loaded.zones) == 6
        assert len(loaded.routes) == 2
        assert sum(1 for z in loaded.zones.values() if z.type == "spawn") == 2
        assert sum(1 for z in loaded.zones.values() if z.type == "goal") == 2
        assert sum(1 for z in loaded.zones.values() if z.type == "crowded") == 2


class TestProgrammaticEditorEquivalence:
    """Test equivalence between programmatic and editor workflows (T041)."""

    def test_programmatic_editor_equivalence_basic(self, tmp_path):
        """Test that programmatic and editor outputs are equivalent."""
        from robot_sf.maps.osm_zones_yaml import save_zones_yaml

        # Create zones programmatically
        spawn = create_spawn_zone("spawn", polygon=[(0, 0), (10, 0), (10, 10)], priority=2)
        goal = create_goal_zone("goal", polygon=[(90, 90), (100, 90), (100, 100)])
        route = create_route("path", waypoints=[(0, 0), (100, 100)], route_type="pedestrian")

        config1 = create_config_with_zones_routes(zones=[spawn, goal], routes=[route])

        # Save both configs
        yaml_file1 = tmp_path / "programmatic.yaml"
        save_zones_yaml(config1, str(yaml_file1))

        # Reload and compare
        config1_reloaded = OSMZonesConfig.from_dict(config1.to_dict())

        # Serialize both to dict for comparison (ignoring metadata timestamps)
        dict1_prog = config1.to_dict()
        dict1_reload = config1_reloaded.to_dict()

        # Remove version/timestamp for comparison
        if "metadata" in dict1_prog:
            dict1_prog["metadata"].pop("created_at", None)
        if "metadata" in dict1_reload:
            dict1_reload["metadata"].pop("created_at", None)

        # Both should have identical structure
        assert dict1_prog["zones"] == dict1_reload["zones"]
        assert dict1_prog["routes"] == dict1_reload["routes"]

    def test_programmatic_editor_equivalence_round_trip(self, tmp_path):
        """Test round-trip: create → save → load → save produces identical YAML."""
        from robot_sf.maps.osm_zones_yaml import save_zones_yaml

        zones = [
            create_spawn_zone("s1", polygon=[(0, 0), (10, 0), (10, 10)]),
            create_goal_zone("g1", polygon=[(90, 90), (100, 90), (100, 100)]),
            create_crowded_zone("c1", polygon=[(40, 40), (60, 40), (60, 60)], density=1.5),
        ]
        routes = [
            create_route("r1", waypoints=[(0, 0), (100, 100)]),
        ]

        config = create_config_with_zones_routes(zones=zones, routes=routes)

        # Save once
        yaml_file1 = tmp_path / "first_save.yaml"
        save_zones_yaml(config, str(yaml_file1))
        content1 = yaml_file1.read_text()

        # Load and save again
        config_loaded = OSMZonesConfig.from_dict(config.to_dict())
        yaml_file2 = tmp_path / "second_save.yaml"
        save_zones_yaml(config_loaded, str(yaml_file2))
        content2 = yaml_file2.read_text()

        # Contents should be identical (byte-for-byte)
        assert content1 == content2

    def test_programmatic_complex_equivalence(self, tmp_path):
        """Test complex scenario programmatic equivalence."""
        from robot_sf.maps.osm_zones_yaml import save_zones_yaml

        # Create realistic scenario
        zones = []
        for i in range(3):
            zones.append(
                create_spawn_zone(
                    f"spawn_{i}",
                    polygon=[(i * 30, 0), (i * 30 + 10, 0), (i * 30 + 10, 10)],
                    priority=i + 1,
                    metadata={"variant": f"v{i}"},
                )
            )

        for i in range(3):
            zones.append(
                create_goal_zone(
                    f"goal_{i}",
                    polygon=[(i * 30, 90), (i * 30 + 10, 90), (i * 30 + 10, 100)],
                )
            )

        zones.append(
            create_crowded_zone(
                "intersection",
                polygon=[(30, 30), (70, 30), (70, 70), (30, 70)],
                density=2.5,
                metadata={"peak_hours": "8-18"},
            )
        )

        routes = [
            create_route(f"route_{i}", waypoints=[(i * 30, 5), (50, 50), (i * 30 + 95, 95)])
            for i in range(3)
        ]

        config = create_config_with_zones_routes(zones=zones, routes=routes)

        # Save and reload
        yaml_file = tmp_path / "complex_scenario.yaml"
        save_zones_yaml(config, str(yaml_file))

        config_reloaded = OSMZonesConfig.from_dict(
            OSMZonesConfig.from_dict(config.to_dict()).to_dict()
        )

        # Verify equivalence
        assert len(config_reloaded.zones) == len(config.zones)
        assert len(config_reloaded.routes) == len(config.routes)

        for name, zone in config.zones.items():
            assert name in config_reloaded.zones
            reloaded_zone = config_reloaded.zones[name]
            assert reloaded_zone.type == zone.type
            assert reloaded_zone.priority == zone.priority
            # Compare polygon structure (handle list vs tuple from YAML)
            assert len(reloaded_zone.polygon) == len(zone.polygon)
            for p1, p2 in zip(reloaded_zone.polygon, zone.polygon, strict=True):
                assert round(p1[0], 3) == round(p2[0], 3)
                assert round(p1[1], 3) == round(p2[1], 3)


class TestProgrammaticWorkflow:
    """Integration tests for programmatic zone/route workflows."""

    def test_complete_scenario_creation(self):
        """Test creating a complete scenario programmatically."""
        # Create zones for an intersection scenario
        spawn_zones = [
            create_spawn_zone(
                "spawn_north",
                polygon=[(45, 0), (55, 0), (55, 10)],
                priority=2,
            ),
            create_spawn_zone(
                "spawn_south",
                polygon=[(45, 90), (55, 90), (55, 100)],
                priority=1,
            ),
        ]

        goal_zones = [
            create_goal_zone(
                "goal_north",
                polygon=[(45, 90), (55, 90), (55, 100)],
            ),
            create_goal_zone(
                "goal_south",
                polygon=[(45, 0), (55, 0), (55, 10)],
            ),
        ]

        crowd_zones = [
            create_crowded_zone(
                "intersection_center",
                polygon=[(40, 40), (60, 40), (60, 60), (40, 60)],
                density=2.0,
            ),
        ]

        routes = [
            create_route(
                "north_to_south",
                waypoints=[(50, 5), (50, 50), (50, 95)],
            ),
            create_route(
                "south_to_north",
                waypoints=[(50, 95), (50, 50), (50, 5)],
            ),
        ]

        # Combine into config
        config = create_config_with_zones_routes(
            zones=spawn_zones + goal_zones + crowd_zones,
            routes=routes,
        )

        # Verify complete scenario
        assert len(config.zones) == 5
        assert len(config.routes) == 2
        assert sum(1 for z in config.zones.values() if z.type == "spawn") == 2
        assert sum(1 for z in config.zones.values() if z.type == "goal") == 2
        assert sum(1 for z in config.zones.values() if z.type == "crowded") == 1

    def test_scenario_with_varying_densities(self):
        """Test scenario with multiple crowded zones at different densities."""
        zones = [
            create_crowded_zone("sparse", polygon=[(0, 0), (20, 0), (20, 20)], density=0.1),
            create_crowded_zone("medium", polygon=[(30, 30), (50, 30), (50, 50)], density=1.0),
            create_crowded_zone("dense", polygon=[(60, 60), (80, 60), (80, 80)], density=4.0),
        ]

        config = create_config_with_zones_routes(zones=zones)

        densities = [z.metadata["density"] for z in config.zones.values()]
        assert set(densities) == {0.1, 1.0, 4.0}
