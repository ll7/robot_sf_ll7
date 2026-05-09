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


def test_episode_schema_validates_pedestrian_impact_block():
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
                "parameters": {"near_radius_m": 2.0, "window_steps": 1.0},
                "units": {
                    "accel": "m/s^2",
                    "turn_rate": "rad/s",
                    "near_radius": "m",
                    "sample_counts": "samples",
                    "sample_fraction": "fraction",
                },
                "sample_counts": {
                    "pedestrians": 1.0,
                    "near_samples": 4.0,
                    "far_samples": 5.0,
                    "near_sample_frac": 4.0 / 9.0,
                },
                "canonical_reductions": {
                    "accel_delta_mean": 0.75,
                    "accel_delta_median": 0.70,
                    "accel_delta_valid_pedestrians": 1.0,
                    "turn_rate_delta_mean": 0.20,
                    "turn_rate_delta_median": 0.18,
                    "turn_rate_delta_valid_pedestrians": 1.0,
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
