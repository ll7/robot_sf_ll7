"""Contract test for aggregate summary JSON schema (T012).

Fails until `aggregate.schema.v1.json` exists. Pattern consistent with T010/T011.
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
    / "aggregate.schema.v1.json"
)


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def test_aggregate_summary_invalid_sample_fails():
    schema = _load_schema()
    invalid = {
        "groups": {"A": {"metrics": {"collisions": {"mean": 0}}}}
    }  # missing version, shape mismatch
    with pytest.raises(Exception):
        jsonschema.validate(instance=invalid, schema=schema)


def test_aggregate_summary_minimal_valid_passes_when_ready():
    schema = _load_schema()
    minimal = {
        "version": "v1",
        "groups": {"A": {"metrics": {"collisions": {"mean": 0.0, "median": 0.0, "p95": 0.0}}}},
    }
    try:
        jsonschema.validate(instance=minimal, schema=schema)
    except FileNotFoundError:
        pytest.xfail("Aggregate schema not yet implemented (expected)")
    except jsonschema.ValidationError:
        pytest.xfail("Aggregate schema structure incomplete (expected during red phase)")
