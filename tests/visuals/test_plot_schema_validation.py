"""Tests for plot_artifacts.schema.json validity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCHEMA_PATH = Path("specs/127-enhance-benchmark-visual/contracts/plot_artifacts.schema.json")
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


def test_valid_empty():
    """Test valid empty.

    Returns:
        Any: Auto-generated placeholder description.
    """
    _validate({"artifacts": []})


def test_success_entry():
    """Test success entry.

    Returns:
        Any: Auto-generated placeholder description.
    """
    _validate({"artifacts": [{"name": "plot1", "status": "success", "filename": "plot1.pdf"}]})


def test_success_missing_filename():
    """Test success missing filename.

    Returns:
        Any: Auto-generated placeholder description.
    """
    with pytest.raises(Exception):
        _validate({"artifacts": [{"name": "p1", "status": "success"}]})
