"""Integration tests for map registry with environment creation."""

import pytest


def test_map_pool_loads_from_canonical_location():
    """Test that MapDefinitionPool loads from maps/metadata/ by default."""
    from robot_sf.nav.map_config import MapDefinitionPool
    
    pool = MapDefinitionPool()
    assert len(pool.map_defs) > 0, "MapDefinitionPool should load at least one map"
    assert "uni_campus_big" in pool.map_defs, "Should load uni_campus_big.json"


def test_environment_creation_with_default_map_pool():
    """Test that environments can be created with the default map pool."""
    from robot_sf.gym_env.environment_factory import make_robot_env
    
    # Create environment with default config (should use MapDefinitionPool default)
    env = make_robot_env(debug=True)
    
    # Verify environment was created successfully
    assert env is not None
    
    # Reset to ensure map loading works
    obs, info = env.reset()
    assert obs is not None
    assert info is not None
    
    env.close()


def test_registry_and_map_pool_consistency():
    """Test that registry finds SVG files and map pool finds JSON metadata."""
    from robot_sf.maps import registry
    from robot_sf.nav.map_config import MapDefinitionPool
    
    # Get IDs from registry (SVG files)
    svg_ids = set(registry.list_ids())
    
    # Get IDs from map pool (JSON metadata)
    pool = MapDefinitionPool()
    json_ids = set(pool.map_defs.keys())
    
    # Registry should find many SVG files
    assert len(svg_ids) > 0, "Registry should find SVG maps"
    
    # Map pool should find JSON metadata
    assert len(json_ids) > 0, "Map pool should find JSON metadata"
    
    # Log the difference for debugging
    print(f"SVG maps found by registry: {len(svg_ids)}")
    print(f"JSON metadata found by pool: {len(json_ids)}")
    print(f"Maps with both SVG and JSON: {len(svg_ids & json_ids)}")
