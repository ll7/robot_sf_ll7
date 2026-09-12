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
    """Return a minimal valid _metadata block for SNQI payload fixtures."""
    return {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "generated_at": "2025-01-01T00:00:00+00:00",
        "git_commit": "abc123",
        "seed": 123,
        "provenance": {"episodes_file": "episodes.jsonl"},
    }


def test_validate_optimization_success():
    """Verify validate_snqi accepts an optimization payload with recommended weights."""
    obj = {
        "recommended": {"weights": {"w_success": 1.0, "w_time": 2.0}},
        "_metadata": _base_metadata(),
    }
    validate_snqi(obj, "optimization")  # Should not raise


def test_validate_recompute_success():
    """Verify validate_snqi accepts a recompute payload with recommended_weights."""
    obj = {"recommended_weights": {"w_success": 1.0}, "_metadata": _base_metadata()}
    validate_snqi(obj, "recompute")


def test_validate_sensitivity_success():
    """Verify validate_snqi accepts a sensitivity payload with a weight_sweep block."""
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
    """Verify removing a kind's required key makes validate_snqi raise ValueError.

    Args:
        kind: SNQI output kind under test (optimization, recompute, or sensitivity).
        patch: Callable that removes the required key from a valid payload.
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
    """Verify validate_snqi rejects a NaN weight when check_finite is enabled."""
    obj = {"recommended": {"weights": {"w_success": math.nan}}, "_metadata": _base_metadata()}
    with pytest.raises(ValueError):
        validate_snqi(obj, "optimization", check_finite=True)


def test_assert_all_finite_list_nested():
    """Verify assert_all_finite accepts nested finite values but rejects infinity."""
    good = {"a": [1.0, 2.0, {"b": 3.0}]}
    assert_all_finite(good)  # Should not raise
    bad = {"a": [1.0, float("inf")]}
    with pytest.raises(ValueError):
        assert_all_finite(bad)
