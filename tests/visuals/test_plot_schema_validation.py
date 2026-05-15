"""Tests for plot_artifacts.schema.json validity."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = Path("specs/127-enhance-benchmark-visual/contracts/plot_artifacts.schema.json")


def _schema():
    """Load the plot artifact schema from disk."""
    return json.loads(SCHEMA_PATH.read_text())


def _validate(inst):
    """Validate an instance against the plot artifact schema.

    Args:
        inst: Candidate plot artifact payload.
    """
    jsonschema.validate(instance=inst, schema=_schema())


def test_valid_empty():
    """Empty artifact lists should satisfy the plot schema."""
    _validate({"artifacts": []})


def test_success_entry():
    """Successful plot entries should validate when filename is present."""
    _validate({"artifacts": [{"name": "plot1", "status": "success", "filename": "plot1.pdf"}]})


def test_success_missing_filename():
    """Successful plot entries should require a filename."""
    with pytest.raises(jsonschema.ValidationError):
        _validate({"artifacts": [{"name": "p1", "status": "success"}]})
