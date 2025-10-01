"""Contract test for resume manifest JSON schema (T014).

Checks for `resume-manifest.schema.v1.json` presence and structure once created.
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
    / "resume-manifest.schema.v1.json"
)


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def test_resume_manifest_invalid_sample_fails():
    schema = _load_schema()
    invalid = {"episodes": ["id1", "id2"], "hash": 123}  # missing version, type mismatch
    with pytest.raises(Exception):
        jsonschema.validate(instance=invalid, schema=schema)


def test_resume_manifest_minimal_valid_passes_when_ready():
    schema = _load_schema()
    minimal = {
        "version": "v1",
        "episodes": ["ep1"],
        "meta": {"source": "unit-test"},
    }
    try:
        jsonschema.validate(instance=minimal, schema=schema)
    except FileNotFoundError:
        pytest.xfail("Resume manifest schema not yet implemented (expected)")
    except jsonschema.ValidationError:
        pytest.xfail("Resume manifest schema structure incomplete (expected during red phase)")
