"""Tests for map registry module."""

import pytest

from robot_sf.maps import registry


def test_build_registry():
    """Test that registry builds successfully."""
    reg = registry.build_registry()
    assert isinstance(reg, dict)
    assert len(reg) > 0  # Should find maps in maps/svg_maps/


def test_list_ids():
    """Test listing map IDs."""
    ids = registry.list_ids()
    assert isinstance(ids, list)
    assert len(ids) > 0
    # Should be sorted
    assert ids == sorted(ids)


def test_get_existing_map():
    """Test getting an existing map."""
    ids = registry.list_ids()
    if not ids:
        pytest.skip("No maps found in registry")
    
    # Get first map
    map_id = ids[0]
    svg_path, json_path = registry.get(map_id)
    
    assert svg_path.exists()
    assert svg_path.suffix == ".svg"
    # JSON path may be None if no metadata exists
    if json_path:
        assert json_path.exists()
        assert json_path.suffix == ".json"


def test_validate_map_id_valid():
    """Test validate_map_id with valid ID."""
    ids = registry.list_ids()
    if not ids:
        pytest.skip("No maps found in registry")
    
    # Should not raise for valid ID
    registry.validate_map_id(ids[0])


def test_validate_map_id_invalid():
    """Test validate_map_id with invalid ID."""
    with pytest.raises(ValueError) as exc_info:
        registry.validate_map_id("definitely_not_a_real_map_id_12345")
    
    error_msg = str(exc_info.value)
    assert "not found in registry" in error_msg
    assert "Available map IDs" in error_msg


def test_get_invalid_map():
    """Test getting a non-existent map."""
    with pytest.raises(ValueError) as exc_info:
        registry.get("definitely_not_a_real_map_id_12345")
    
    error_msg = str(exc_info.value)
    assert "not found in registry" in error_msg


def test_cache_clearing():
    """Test that cache can be cleared and rebuilt."""
    # Build registry
    ids1 = registry.list_ids()
    
    # Clear cache
    registry.clear_cache()
    
    # Rebuild
    ids2 = registry.list_ids()
    
    # Should get same results
    assert ids1 == ids2


def test_registry_consistency():
    """Test that registry returns consistent results."""
    ids1 = registry.list_ids()
    ids2 = registry.list_ids()
    
    assert ids1 == ids2
