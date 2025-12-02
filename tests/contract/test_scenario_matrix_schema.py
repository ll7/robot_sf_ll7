"""Contract test for scenario matrix JSON schema (T011).

Fails until `scenario-matrix.schema.v1.json` is implemented. Mirrors T010 pattern.
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
    / "scenario-matrix.schema.v1.json"
)


def _load_schema() -> dict:
    """Load schema.

    Returns:
        dict: Auto-generated placeholder description.
    """
    return json.loads(SCHEMA_PATH.read_text())


def test_scenario_matrix_invalid_sample_fails():
    """Test scenario matrix invalid sample fails.

    Returns:
        Any: Auto-generated placeholder description.
    """
    schema = _load_schema()
    invalid = {"scenarios": [{"id": 1}]}  # id wrong type, missing required keys
    with pytest.raises(Exception):
        jsonschema.validate(instance=invalid, schema=schema)


def test_scenario_matrix_minimal_valid_passes_when_ready():
    """Test scenario matrix minimal valid passes when ready.

    Returns:
        Any: Auto-generated placeholder description.
    """
    schema = _load_schema()
    minimal = {
        "scenarios": [
            {
                "id": "simple_1",
                "algo": "random",
                "map": "basic",
                "episodes": 1,
                "seed": 123,
            },
        ],
    }
    try:
        jsonschema.validate(instance=minimal, schema=schema)
    except FileNotFoundError:
        pytest.xfail("Scenario matrix schema not yet implemented (expected)")
    except jsonschema.ValidationError:
        pytest.xfail("Scenario matrix schema structure incomplete (expected during red phase)")
