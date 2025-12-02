"""Tests for performance_visuals.schema.json validity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCHEMA_PATH = Path("specs/127-enhance-benchmark-visual/contracts/performance_visuals.schema.json")
jsonschema_spec = importlib.util.find_spec("jsonschema")
pytestmark = pytest.mark.skipif(jsonschema_spec is None, reason="jsonschema not installed")
if jsonschema_spec:  # type: ignore
    import jsonschema  # type: ignore


def _schema():
    """Schema.

    Returns:
        Any: Auto-generated placeholder description.
    """
    return json.loads(SCHEMA_PATH.read_text())


def _validate(inst):
    """Validate.

    Args:
        inst: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    jsonschema.validate(instance=inst, schema=_schema())  # type: ignore


def test_valid_minimal():
    """Test valid minimal.

    Returns:
        Any: Auto-generated placeholder description.
    """
    _validate({"plots_time_s": 0.5, "plots_over_budget": False, "video_over_budget": False})


def test_missing_required_field():
    """Test missing required field.

    Returns:
        Any: Auto-generated placeholder description.
    """
    import pytest as _pytest

    with _pytest.raises(Exception):
        _validate({"plots_over_budget": False, "video_over_budget": False})
