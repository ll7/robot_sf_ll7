"""Tests for performance_visuals.schema.json validity."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = Path("specs/127-enhance-benchmark-visual/contracts/performance_visuals.schema.json")


def _schema():
    """Load the performance visual schema from disk."""
    return json.loads(SCHEMA_PATH.read_text())


def _validate(inst):
    """Validate an instance against the performance visual schema.

    Args:
        inst: Candidate performance visual payload.
    """
    jsonschema.validate(instance=inst, schema=_schema())


def test_valid_minimal():
    """Minimal performance visual metrics should satisfy the schema."""
    _validate({"plots_time_s": 0.5, "plots_over_budget": False, "video_over_budget": False})


def test_missing_required_field():
    """Performance visual metrics should require plot timing."""
    with pytest.raises(jsonschema.ValidationError):
        _validate({"plots_over_budget": False, "video_over_budget": False})
