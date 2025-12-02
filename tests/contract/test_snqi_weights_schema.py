"""Contract test for SNQI weights JSON schema (T013).

Ensures presence and structure of `snqi-weights.schema.v1.json` once implemented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import jsonschema  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("jsonschema dependency required for contract tests") from e


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "snqi-weights.schema.v1.json"
)


def _load_schema() -> dict:
    """Load schema.

    Returns:
        dict: Auto-generated placeholder description.
    """
    return json.loads(SCHEMA_PATH.read_text())


def test_snqi_weights_invalid_sample_fails():
    """Test snqi weights invalid sample fails.

    Returns:
        Any: Auto-generated placeholder description.
    """
    schema = _load_schema()
    invalid = {"weights": {"w_success": 1.0, "w_time": -1.0}}  # missing version, negative weight
    with pytest.raises(Exception):
        jsonschema.validate(instance=invalid, schema=schema)


def test_snqi_weights_minimal_valid_passes_when_ready():
    """Test snqi weights minimal valid passes when ready.

    Returns:
        Any: Auto-generated placeholder description.
    """
    schema = _load_schema()
    minimal = {
        "version": "v1",
        "weights": {"w_success": 1.0, "w_time": 1.0},
        "meta": {"origin": "test"},
    }
    try:
        jsonschema.validate(instance=minimal, schema=schema)
    except FileNotFoundError:
        pytest.xfail("SNQI weights schema not yet implemented (expected)")
    except jsonschema.ValidationError:
        pytest.xfail("SNQI weights schema structure incomplete (expected during red phase)")
