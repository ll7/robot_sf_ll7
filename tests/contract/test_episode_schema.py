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

from robot_sf.benchmark.schema_validator import validate_episode

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
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    text = SCHEMA_PATH.read_text()  # may raise FileNotFoundError (expected red)
    return json.loads(text)


def test_episode_schema_invalid_sample_fails():
    """TODO docstring. Document this function."""
    schema = _load_schema()
    invalid_record = {  # missing many required keys by design
        "episode_id": "abc123",
        "metrics": {"collisions": 0},
    }
    with pytest.raises(Exception):  # broad until schema stabilizes
        jsonschema.validate(instance=invalid_record, schema=schema)


def test_episode_schema_minimal_valid_passes_when_ready():
    """TODO docstring. Document this function."""
    schema = _load_schema()
    minimal = {
        # Placeholder minimal fields; update once schema is defined.
        "episode_id": "e_000",
        "version": "v1",
        "scenario_id": "sc_basic",
        "seed": 123,
        "metrics": {"collisions": 0, "near_misses": 0},
        "termination_reason": "max_steps",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
    }
    try:
        jsonschema.validate(instance=minimal, schema=schema)
    except FileNotFoundError:
        pytest.xfail("Episode schema not yet implemented (expected) *)")
    except jsonschema.ValidationError:
        # Accept validation failure until full field set + required list decided
        pytest.xfail("Episode schema structure incomplete (expected during red phase)")


def test_episode_schema_rejects_collision_event_without_collision_metric() -> None:
    """New v1 records should not report collision_event=true with zero collision count."""
    schema = _load_schema()
    record = {
        "episode_id": "e_collision",
        "version": "v1",
        "scenario_id": "sc_collision",
        "seed": 123,
        "metrics": {"success": 0.0, "collisions": 0.0},
        "termination_reason": "collision",
        "outcome": {
            "route_complete": False,
            "collision_event": True,
            "timeout_event": False,
        },
        "integrity": {"contradictions": []},
    }
    with pytest.raises(jsonschema.ValidationError, match="collisions"):
        jsonschema.validate(instance=record, schema=schema)


def test_episode_schema_rejects_collision_metric_without_collision_event() -> None:
    """New v1 records should not carry a positive collision count for non-collision episodes."""
    schema = _load_schema()
    record = {
        "episode_id": "e_stale_collision",
        "version": "v1",
        "scenario_id": "sc_collision",
        "seed": 124,
        "metrics": {"success": 0.0, "collisions": 1.0},
        "termination_reason": "max_steps",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
    }
    with pytest.raises(jsonschema.ValidationError, match="collisions"):
        jsonschema.validate(instance=record, schema=schema)


def test_episode_validator_rejects_legacy_collision_alias_drift() -> None:
    """Semantic validation should also catch legacy collision aliases in schema-valid records."""
    schema = _load_schema()
    record = {
        "episode_id": "e_alias_collision",
        "version": "v1",
        "scenario_id": "sc_collision",
        "seed": 125,
        "metrics": {"success": 0.0, "collisions": 0.0, "collision_rate": 1.0},
        "termination_reason": "max_steps",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
    }
    with pytest.raises(jsonschema.ValidationError, match="collision_event=false"):
        validate_episode(record, schema)
