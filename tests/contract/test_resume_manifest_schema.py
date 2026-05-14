"""Contract tests for the resume manifest JSON schema (T014)."""

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
    """Load the canonical resume manifest schema from the repository."""
    return json.loads(SCHEMA_PATH.read_text())


def test_resume_manifest_invalid_sample_fails():
    """A malformed resume manifest should fail the v1 schema."""
    schema = _load_schema()
    invalid = {"episodes": ["id1", "id2"], "hash": 123}  # missing version, type mismatch
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=schema)


def test_resume_manifest_minimal_valid_passes_when_ready():
    """A minimal resume manifest should validate against the canonical schema."""
    schema = _load_schema()
    minimal = {
        "version": "v1",
        "episodes": ["ep1"],
        "meta": {"source": "unit-test"},
    }
    jsonschema.validate(instance=minimal, schema=schema)
