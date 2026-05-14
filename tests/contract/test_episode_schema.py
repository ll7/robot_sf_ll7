"""Contract tests for the episode JSON schema (T010)."""

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
    """Load the canonical episode schema from the repository."""
    text = SCHEMA_PATH.read_text()
    return json.loads(text)


def test_episode_schema_invalid_sample_fails():
    """A partial episode record should fail the v1 schema."""
    schema = _load_schema()
    invalid_record = {  # missing many required keys by design
        "episode_id": "abc123",
        "metrics": {"collisions": 0},
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_record, schema=schema)


def test_episode_schema_minimal_valid_passes_when_ready():
    """A minimal v1 episode record should validate against the canonical schema."""
    schema = _load_schema()
    minimal = {
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
    jsonschema.validate(instance=minimal, schema=schema)


def test_episode_schema_validates_pedestrian_impact_block() -> None:
    """The pedestrian-impact metric block should be schema-backed when present."""
    schema = _load_schema()
    record = {
        "episode_id": "e_ped_impact",
        "version": "v1",
        "scenario_id": "sc_ped_impact",
        "seed": 123,
        "metrics": {
            "collisions": 0,
            "near_misses": 0,
            "pedestrian_impact": {
                "schema_version": "pedestrian-impact.v1",
                "parameters": {"near_radius_m": 2.0, "window_steps": 1},
                "units": {
                    "accel": "m/s^2",
                    "turn_rate": "rad/s",
                    "near_radius": "m",
                    "sample_counts": "samples",
                    "sample_fraction": "fraction",
                },
                "sample_counts": {
                    "pedestrians": 1,
                    "near_samples": 4,
                    "far_samples": 5,
                    "near_sample_frac": 4.0 / 9.0,
                },
                "canonical_reductions": {
                    "accel_delta_mean": 0.75,
                    "accel_delta_median": 0.70,
                    "accel_delta_valid_pedestrians": 1,
                    "turn_rate_delta_mean": 0.20,
                    "turn_rate_delta_median": 0.18,
                    "turn_rate_delta_valid_pedestrians": 1,
                },
            },
        },
        "termination_reason": "max_steps",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
    }

    jsonschema.validate(instance=record, schema=schema)
    record["metrics"]["pedestrian_impact"]["schema_version"] = "wrong"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=record, schema=schema)

    record["metrics"]["pedestrian_impact"]["schema_version"] = "pedestrian-impact.v1"
    record["metrics"]["pedestrian_impact"]["sample_counts"]["pedestrians"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=record, schema=schema)


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
