"""Tests for SNQI schema validation utilities."""

from __future__ import annotations

import copy
import math

import pytest

from robot_sf.benchmark.snqi.schema import (
    EXPECTED_SCHEMA_VERSION,
    assert_all_finite,
    validate_snqi,
)


def _base_metadata():
    """Base metadata.

    Returns:
        Any: Auto-generated placeholder description.
    """
    return {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "generated_at": "2025-01-01T00:00:00+00:00",
        "git_commit": "abc123",
        "seed": 123,
        "provenance": {"episodes_file": "episodes.jsonl"},
    }


def test_validate_optimization_success():
    """Test validate optimization success.

    Returns:
        Any: Auto-generated placeholder description.
    """
    obj = {
        "recommended": {"weights": {"w_success": 1.0, "w_time": 2.0}},
        "_metadata": _base_metadata(),
    }
    validate_snqi(obj, "optimization")  # Should not raise


def test_validate_recompute_success():
    """Test validate recompute success.

    Returns:
        Any: Auto-generated placeholder description.
    """
    obj = {"recommended_weights": {"w_success": 1.0}, "_metadata": _base_metadata()}
    validate_snqi(obj, "recompute")


def test_validate_sensitivity_success():
    """Test validate sensitivity success.

    Returns:
        Any: Auto-generated placeholder description.
    """
    obj = {"weight_sweep": {"dummy": 1}, "_metadata": _base_metadata()}
    validate_snqi(obj, "sensitivity")


@pytest.mark.parametrize(
    "kind,patch",
    [
        ("optimization", lambda o: o.pop("recommended")),
        ("recompute", lambda o: o.pop("recommended_weights")),
        ("sensitivity", lambda o: o.pop("weight_sweep")),
    ],
)
def test_missing_key_errors(kind, patch):
    """Test missing key errors.

    Args:
        kind: Auto-generated placeholder description.
        patch: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    base = {
        "optimization": {"recommended": {"weights": {"w_success": 1.0}}},
        "recompute": {"recommended_weights": {"w_success": 1.0}},
        "sensitivity": {"weight_sweep": {"dummy": 1}},
    }[kind]
    base["_metadata"] = _base_metadata()
    obj = copy.deepcopy(base)
    patch(obj)
    with pytest.raises(ValueError):
        validate_snqi(obj, kind)


def test_non_finite_detection():
    """Test non finite detection.

    Returns:
        Any: Auto-generated placeholder description.
    """
    obj = {"recommended": {"weights": {"w_success": math.nan}}, "_metadata": _base_metadata()}
    with pytest.raises(ValueError):
        validate_snqi(obj, "optimization", check_finite=True)


def test_assert_all_finite_list_nested():
    """Test assert all finite list nested.

    Returns:
        Any: Auto-generated placeholder description.
    """
    good = {"a": [1.0, 2.0, {"b": 3.0}]}
    assert_all_finite(good)  # Should not raise
    bad = {"a": [1.0, float("inf")]}
    with pytest.raises(ValueError):
        assert_all_finite(bad)
