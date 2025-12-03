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
    """TODO docstring. Document this function."""
    return json.loads(SCHEMA_PATH.read_text())


def _validate(inst):
    """TODO docstring. Document this function.

    Args:
        inst: TODO docstring.
    """
    jsonschema.validate(instance=inst, schema=_schema())  # type: ignore


def test_valid_minimal():
    """TODO docstring. Document this function."""
    _validate({"plots_time_s": 0.5, "plots_over_budget": False, "video_over_budget": False})


def test_missing_required_field():
    """TODO docstring. Document this function."""
    import pytest as _pytest

    with _pytest.raises(Exception):
        _validate({"plots_over_budget": False, "video_over_budget": False})
