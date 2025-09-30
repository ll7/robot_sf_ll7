"""Test the get_nested_value utility function."""

from robot_sf.benchmark.utils import get_nested_value


def test_get_nested_value_basic():
    """Test basic path traversal."""
    data = {
        "metrics": {
            "collision_rate": 0.1,
            "success_rate": 0.9,
        },
        "scenario_params": {"algo": "social_force"},
    }

    assert get_nested_value(data, "metrics.collision_rate") == 0.1
    assert get_nested_value(data, "scenario_params.algo") == "social_force"


def test_get_nested_value_deep_nesting():
    """Test deeply nested path traversal."""
    data = {"level1": {"level2": {"level3": {"value": 42}}}}

    assert get_nested_value(data, "level1.level2.level3.value") == 42


def test_get_nested_value_missing_paths():
    """Test missing paths return default."""
    data = {"existing": {"key": "value"}}

    assert get_nested_value(data, "missing.path") is None
    assert get_nested_value(data, "missing.path", "fallback") == "fallback"
    assert get_nested_value(data, "existing.missing", 0) == 0


def test_get_nested_value_partial_paths():
    """Test partial paths that don't exist."""
    data = {"metrics": {"collision_rate": 0.1}}

    assert get_nested_value(data, "metrics.collision_rate.deeper") is None


def test_get_nested_value_edge_cases():
    """Test edge cases."""
    data = {"key": "value"}

    # Test empty path
    assert get_nested_value(data, "") is None

    # Test single key
    assert get_nested_value(data, "key") == "value"

    # Test non-existent single key
    assert get_nested_value(data, "missing") is None


def test_get_nested_value_non_dict_intermediate():
    """Test when intermediate values are not dictionaries."""
    data = {"metrics": "not_a_dict"}

    assert get_nested_value(data, "metrics.collision_rate") is None
