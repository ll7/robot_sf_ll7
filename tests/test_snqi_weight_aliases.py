"""Tests for SNQI weight alias handling.

Covers the backward-compatible alias 'w_near_misses' -> 'w_near'.
"""

from robot_sf.benchmark.snqi.compute import WEIGHT_NAMES
from robot_sf.benchmark.snqi.weights_validation import validate_weights_mapping


def _base_weights():
    return {
        "w_success": 1.0,
        "w_time": 1.0,
        "w_collisions": 1.0,
        "w_near": 1.0,
        "w_comfort": 1.0,
        "w_force_exceed": 1.0,
        "w_jerk": 1.0,
    }


def test_alias_w_near_misses_alone_maps_to_w_near():
    raw = _base_weights()
    # Remove canonical and provide alias
    del raw["w_near"]
    raw["w_near_misses"] = 2.5

    validated = validate_weights_mapping(raw)
    assert set(validated.keys()) == set(WEIGHT_NAMES)
    assert validated["w_near"] == 2.5


def test_alias_ignored_when_canonical_present():
    raw = _base_weights()
    raw["w_near"] = 3.0
    raw["w_near_misses"] = 2.0  # Should be ignored since canonical present

    validated = validate_weights_mapping(raw)
    assert validated["w_near"] == 3.0
