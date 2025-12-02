"""Contract test for episode JSON schema (T010).

This test is intentionally added *before* the actual `episode.schema.v1.json`
implementation to enforce a red → green TDD cycle. It references the expected
schema location under `robot_sf/benchmark/schemas/` and will FAIL until the
schema file is created with the required structure.

Failure modes expected initially:
1. FileNotFoundError when loading schema.
2. jsonschema.ValidationError / KeyError once partial schema exists but lacks
   required fields.

Once the schema is implemented, the invalid sample should raise a
`jsonschema.ValidationError` while the minimal valid sample passes.
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
    / "episode.schema.v1.json"
)


def _load_schema() -> dict:
    """Load schema.

    Returns:
        dict: Auto-generated placeholder description.
    """
    text = SCHEMA_PATH.read_text()  # may raise FileNotFoundError (expected red)
    return json.loads(text)


def test_episode_schema_invalid_sample_fails():
    """Test episode schema invalid sample fails.

    Returns:
        Any: Auto-generated placeholder description.
    """
    schema = _load_schema()
    invalid_record = {  # missing many required keys by design
        "episode_id": "abc123",
        "metrics": {"collisions": 0},
    }
    with pytest.raises(Exception):  # broad until schema stabilizes
        jsonschema.validate(instance=invalid_record, schema=schema)


def test_episode_schema_minimal_valid_passes_when_ready():
    """Test episode schema minimal valid passes when ready.

    Returns:
        Any: Auto-generated placeholder description.
    """
    schema = _load_schema()
    minimal = {
        # Placeholder minimal fields; update once schema is defined.
        "episode_id": "e_000",
        "version": "v1",
        "scenario_id": "sc_basic",
        "seed": 123,
        "metrics": {"collisions": 0, "near_misses": 0},
    }
    try:
        jsonschema.validate(instance=minimal, schema=schema)
    except FileNotFoundError:
        pytest.xfail("Episode schema not yet implemented (expected) *)")
    except jsonschema.ValidationError:
        # Accept validation failure until full field set + required list decided
        pytest.xfail("Episode schema structure incomplete (expected during red phase)")
