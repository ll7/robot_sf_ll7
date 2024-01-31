"""
test map_loader.py
"""
import json
import pytest

from pysocialforce.map_loader import load_map
from pysocialforce.map_config import MapDefinition

def test_load_map():
    # Test with a valid map file
    map_file = 'tests/test_maps/map_regular.json'
    map_definition = load_map(map_file)
    assert isinstance(map_definition, MapDefinition)
    assert len(map_definition.obstacles) > 0
    assert len(map_definition.routes) > 0
    assert len(map_definition.crowded_zones) > 0

    # Test with a map file that has no obstacles
    map_file = 'tests/test_maps/map_no_obstacles.json'
    map_definition = load_map(map_file)
    assert len(map_definition.obstacles) == 0

    # Test with a map file that has no routes
    map_file = 'tests/test_maps/map_no_routes.json'
    map_definition = load_map(map_file)
    assert len(map_definition.routes) == 0

    # Test with a map file that has no crowded zones
    map_file = 'tests/test_maps/map_no_crowded_zone.json'
    map_definition = load_map(map_file)
    assert len(map_definition.crowded_zones) == 0

def test_load_map_with_invalid_file():
    # Test with a non-existent file
    with pytest.raises(FileNotFoundError):
        load_map('non_existent_file.json')

    # Test with a file that is not a valid JSON
    with pytest.raises(json.JSONDecodeError):
        load_map('tests/test_maps/invalid_json_file.json')
