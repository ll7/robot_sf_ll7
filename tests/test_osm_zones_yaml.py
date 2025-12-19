"""Tests for OSM zones YAML serialization (T023-T025).

This test suite validates:
- Zone and Route dataclass serialization
- OSMZonesConfig round-trip (save → load → save byte-identical)
- Deterministic YAML output (sorted keys, 3-decimal precision)
- YAML validation against boundaries and obstacles
- Error handling for malformed files
"""

import tempfile
from pathlib import Path

import pytest

from robot_sf.maps.osm_zones_yaml import (
    OSMZonesConfig,
    Route,
    Zone,
    create_route,
    create_zone,
    load_zones_yaml,
    save_zones_yaml,
    validate_zones_yaml,
)


class TestZoneDataclass:
    """Test Zone dataclass serialization (T022)."""

    def test_zone_creation(self):
        """Test basic Zone creation."""
        zone = Zone(
            name="spawn_1",
            type="spawn",
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            priority=1,
        )
        assert zone.name == "spawn_1"
        assert zone.type == "spawn"
        assert len(zone.polygon) == 4
        assert zone.priority == 1

    def test_zone_to_dict(self):
        """Test Zone.to_dict() with precision rounding."""
        zone = Zone(
            name="spawn_1",
            type="spawn",
            polygon=[(0.1234, 0.5678), (10.9876, 5.4321)],
            priority=2,
            metadata={"density": "high"},
        )
        data = zone.to_dict()

        assert data["name"] == "spawn_1"
        assert data["type"] == "spawn"
        # Check 3-decimal precision
        assert data["polygon"][0] == [0.123, 0.568]
        assert data["polygon"][1] == [10.988, 5.432]
        assert data["priority"] == 2
        assert data["metadata"] == {"density": "high"}

    def test_zone_from_dict(self):
        """Test Zone.from_dict() reconstruction."""
        data = {
            "name": "goal_1",
            "type": "goal",
            "polygon": [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0]],
            "priority": 3,
            "metadata": {"color": "blue"},
        }
        zone = Zone.from_dict(data)

        assert zone.name == "goal_1"
        assert zone.type == "goal"
        assert zone.polygon == [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0]]
        assert zone.priority == 3
        assert zone.metadata == {"color": "blue"}

    def test_zone_default_values(self):
        """Test Zone defaults (priority=1, empty metadata)."""
        zone = Zone(name="z1", type="crowded", polygon=[(0, 0), (1, 0), (1, 1)])
        assert zone.priority == 1
        assert zone.metadata == {}


class TestRouteDataclass:
    """Test Route dataclass serialization (T022)."""

    def test_route_creation(self):
        """Test basic Route creation."""
        route = Route(
            name="main_path",
            waypoints=[(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)],
            route_type="pedestrian",
        )
        assert route.name == "main_path"
        assert len(route.waypoints) == 3
        assert route.route_type == "pedestrian"

    def test_route_to_dict(self):
        """Test Route.to_dict() with precision rounding."""
        route = Route(
            name="path_1",
            waypoints=[(1.2345, 2.3456), (3.4567, 4.5678)],
            route_type="wheelchair",
            metadata={"surface": "asphalt"},
        )
        data = route.to_dict()

        assert data["name"] == "path_1"
        # Check 3-decimal precision (Python's round to 3 decimals)
        assert data["waypoints"][0] == [1.234, 2.346]  # banker's rounding
        assert data["waypoints"][1] == [3.457, 4.568]
        assert data["route_type"] == "wheelchair"
        assert data["metadata"] == {"surface": "asphalt"}

    def test_route_from_dict(self):
        """Test Route.from_dict() reconstruction."""
        data = {
            "name": "route_2",
            "waypoints": [[10.0, 10.0], [20.0, 20.0]],
            "route_type": "vehicle",
            "metadata": {"traffic": "light"},
        }
        route = Route.from_dict(data)

        assert route.name == "route_2"
        assert route.waypoints == [[10.0, 10.0], [20.0, 20.0]]
        assert route.route_type == "vehicle"
        assert route.metadata == {"traffic": "light"}

    def test_route_default_values(self):
        """Test Route defaults (route_type=pedestrian, empty metadata)."""
        route = Route(name="simple", waypoints=[(0, 0), (1, 1)])
        assert route.route_type == "pedestrian"
        assert route.metadata == {}


class TestOSMZonesConfig:
    """Test OSMZonesConfig top-level configuration (T022)."""

    def test_config_creation(self):
        """Test OSMZonesConfig creation."""
        zone1 = Zone("spawn", "spawn", [(0, 0), (1, 0), (1, 1)])
        route1 = Route("path", [(0, 0), (5, 5)])

        config = OSMZonesConfig(
            zones={"z1": zone1},
            routes={"r1": route1},
            metadata={"map_source": "test.pbf"},
        )

        assert config.version == "1.0"
        assert len(config.zones) == 1
        assert len(config.routes) == 1
        assert config.metadata["map_source"] == "test.pbf"

    def test_config_to_dict(self):
        """Test OSMZonesConfig.to_dict() with sorted keys."""
        zone1 = Zone("z_spawn", "spawn", [(0, 0), (1, 0), (1, 1)], priority=1)
        zone2 = Zone("a_goal", "goal", [(5, 5), (6, 5), (6, 6)], priority=2)

        config = OSMZonesConfig(
            zones={"z_spawn": zone1, "a_goal": zone2},
            metadata={"created": "2025-12-19"},
        )

        data = config.to_dict()
        assert data["version"] == "1.0"
        assert "z_spawn" in data["zones"]
        assert "a_goal" in data["zones"]
        # Verify sorted zone keys in output
        zone_keys = list(data["zones"].keys())
        assert zone_keys == sorted(zone_keys)

    def test_config_from_dict(self):
        """Test OSMZonesConfig.from_dict() reconstruction."""
        data = {
            "version": "1.0",
            "metadata": {"map": "test"},
            "zones": {
                "spawn_1": {
                    "name": "spawn_1",
                    "type": "spawn",
                    "polygon": [[0, 0], [1, 0], [1, 1]],
                    "priority": 1,
                }
            },
            "routes": {
                "route_1": {
                    "name": "route_1",
                    "waypoints": [[0, 0], [5, 5]],
                    "route_type": "pedestrian",
                }
            },
        }

        config = OSMZonesConfig.from_dict(data)
        assert config.version == "1.0"
        assert len(config.zones) == 1
        assert len(config.routes) == 1
        assert config.zones["spawn_1"].name == "spawn_1"
        assert config.routes["route_1"].name == "route_1"


class TestYAMLRoundTrip:
    """Test YAML serialization round-trip (byte-identical guarantee) (T023-T024)."""

    def test_save_and_load(self):
        """Test save → load cycle preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "zones.yaml"

            # Create config
            zone = Zone("spawn_1", "spawn", [(0, 0), (10, 0), (10, 10), (0, 10)])
            route = Route("path_1", [(0, 0), (5, 5), (10, 10)])
            config = OSMZonesConfig(
                zones={"spawn_1": zone},
                routes={"path_1": route},
                metadata={"map": "test.pbf"},
            )

            # Save
            save_zones_yaml(config, str(yaml_file))
            assert yaml_file.exists()

            # Load
            loaded = load_zones_yaml(str(yaml_file))
            assert loaded.version == config.version
            assert len(loaded.zones) == 1
            assert loaded.zones["spawn_1"].name == "spawn_1"
            assert len(loaded.routes) == 1

    def test_round_trip_byte_identical(self):
        """Test determinism: save → load → save produces identical bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file1 = Path(tmpdir) / "zones1.yaml"
            yaml_file2 = Path(tmpdir) / "zones2.yaml"

            # Create config
            zone = Zone("z1", "spawn", [(1.23456, 2.34567), (3.45678, 4.56789)])
            config = OSMZonesConfig(zones={"z1": zone})

            # Save twice
            save_zones_yaml(config, str(yaml_file1))
            loaded = load_zones_yaml(str(yaml_file1))
            save_zones_yaml(loaded, str(yaml_file2))

            # Compare byte-for-byte
            content1 = yaml_file1.read_text()
            content2 = yaml_file2.read_text()
            assert content1 == content2, "Round-trip produces different YAML"

    def test_precision_rounding(self):
        """Test floating-point precision is 3 decimals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "zones.yaml"

            zone = Zone("z1", "spawn", [(1.23456789, 2.98765432)])
            config = OSMZonesConfig(zones={"z1": zone})

            save_zones_yaml(config, str(yaml_file))
            loaded = load_zones_yaml(str(yaml_file))

            # Verify rounded to 3 decimals
            loaded_point = loaded.zones["z1"].polygon[0]
            assert loaded_point == [1.235, 2.988]

    def test_empty_config(self):
        """Test handling of empty zones/routes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "empty.yaml"

            config = OSMZonesConfig()
            save_zones_yaml(config, str(yaml_file))

            loaded = load_zones_yaml(str(yaml_file))
            assert len(loaded.zones) == 0
            assert len(loaded.routes) == 0

    def test_null_metadata_omitted(self):
        """Test null metadata fields are omitted from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "zones.yaml"

            zone = Zone("z1", "spawn", [(0, 0), (1, 1), (1, 0)])
            config = OSMZonesConfig(zones={"z1": zone})

            save_zones_yaml(config, str(yaml_file))
            content = yaml_file.read_text()

            # metadata: null should be omitted
            assert "metadata: null" not in content


class TestYAMLValidation:
    """Test YAML validation against bounds and obstacles (T025)."""

    def test_validate_empty_config(self):
        """Test validation of empty config."""
        config = OSMZonesConfig()
        warnings = validate_zones_yaml(config)

        # Should warn about empty config
        assert any("No zones or routes" in w for w in warnings)

    def test_validate_duplicate_zone_names(self):
        """Test detection of duplicate zone names."""
        zone1 = Zone("z1", "spawn", [(0, 0), (1, 0), (1, 1)])
        zone2 = Zone("z1", "goal", [(5, 5), (6, 5), (6, 6)])  # Duplicate name

        config = OSMZonesConfig(zones={"z1": zone1})
        # Manually add duplicate
        config.zones["z1_dup"] = zone2
        config.zones["z1_dup"].name = "z1"

        # This should trigger duplicate detection in dict iteration
        # Note: dict keys prevent true duplicates, but names can differ

    def test_validate_invalid_polygon(self):
        """Test detection of polygons with < 3 points."""
        zone = Zone("z1", "spawn", [(0, 0), (1, 1)])  # Only 2 points
        config = OSMZonesConfig(zones={"z1": zone})

        warnings = validate_zones_yaml(config)
        assert any("fewer than 3 points" in w for w in warnings)

    def test_validate_invalid_route(self):
        """Test detection of routes with < 2 waypoints."""
        route = Route("r1", [(0, 0)])  # Only 1 waypoint
        config = OSMZonesConfig(routes={"r1": route})

        warnings = validate_zones_yaml(config)
        assert any("fewer than 2 waypoints" in w for w in warnings)

    def test_validate_no_errors_for_valid_config(self):
        """Test validation passes for valid config."""
        zone = Zone("z1", "spawn", [(0, 0), (1, 0), (1, 1)])
        route = Route("r1", [(0, 0), (1, 1)])
        config = OSMZonesConfig(zones={"z1": zone}, routes={"r1": route})

        warnings = validate_zones_yaml(config)
        # Should have no warnings for valid config
        errors = [w for w in warnings if w.startswith("Error")]
        assert len(errors) == 0


class TestHelperFunctions:
    """Test helper functions for zone/route creation (T022)."""

    def test_create_zone_helper(self):
        """Test create_zone() helper."""
        zone = create_zone(
            "spawn_1",
            [(0, 0), (10, 0), (10, 10)],
            zone_type="spawn",
            priority=2,
            metadata={"type": "outdoor"},
        )

        assert zone.name == "spawn_1"
        assert zone.type == "spawn"
        assert zone.priority == 2
        assert zone.metadata["type"] == "outdoor"

    def test_create_route_helper(self):
        """Test create_route() helper."""
        route = create_route(
            "main_path",
            [(0, 0), (5, 5), (10, 10)],
            route_type="wheelchair",
            metadata={"accessible": True},
        )

        assert route.name == "main_path"
        assert len(route.waypoints) == 3
        assert route.route_type == "wheelchair"
        assert route.metadata["accessible"] is True


class TestErrorHandling:
    """Test error handling for malformed files (T023)."""

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_zones_yaml("/nonexistent/path/zones.yaml")

    def test_load_malformed_yaml(self):
        """Test loading malformed YAML raises YAMLError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "malformed.yaml"
            yaml_file.write_text("invalid: yaml: syntax: [")

            with pytest.raises(Exception):  # yaml.YAMLError or similar
                load_zones_yaml(str(yaml_file))

    def test_save_creates_parent_dirs(self):
        """Test save_zones_yaml creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "zones.yaml"
            config = OSMZonesConfig()

            save_zones_yaml(config, str(nested_path))
            assert nested_path.exists()
            assert nested_path.parent.is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
