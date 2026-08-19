"""Tests for robot_sf.benchmark.map_runner.map_runner_jsonl — JSONL record writing."""

from __future__ import annotations

import io
import json

import pytest
from jsonschema import ValidationError

from robot_sf.benchmark.map_runner.map_runner_jsonl import write_validated_to_handle


def _valid_record() -> dict:
    """Return a minimal valid episode record for JSONL writing."""
    return {
        "version": "v1",
        "episode_id": "ep-1",
        "scenario_id": "sc-1",
        "seed": 42,
        "algo": "goal",
        "git_hash": "abc123",
        "metrics": {
            "success": 1.0,
            "collisions": 0.0,
            "near_misses": 0.0,
            "min_clearance": 1.5,
        },
        "termination_reason": "success",
        "outcome": {
            "route_complete": True,
            "collision_event": False,
            "timeout_event": False,
        },
    }


def _permissive_schema() -> dict:
    """Return a permissive JSON schema that accepts any object."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": True,
    }


class TestWriteValidatedToHandle:
    """Tests for write_validated_to_handle JSONL writing."""

    def test_writes_valid_json_line(self) -> None:
        """A valid record must be written as a single JSON line."""
        handle = io.StringIO()
        write_validated_to_handle(handle, _permissive_schema(), _valid_record())
        output = handle.getvalue()
        lines = output.strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["episode_id"] == "ep-1"

    def test_output_is_sorted_keys(self) -> None:
        """Output JSON must use sorted keys for determinism."""
        handle = io.StringIO()
        write_validated_to_handle(handle, _permissive_schema(), _valid_record())
        line = handle.getvalue().strip()
        parsed = json.loads(line)
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_appends_to_existing_content(self) -> None:
        """Multiple writes must append separate lines."""
        handle = io.StringIO()
        write_validated_to_handle(handle, _permissive_schema(), _valid_record())
        record2 = _valid_record()
        record2["episode_id"] = "ep-2"
        write_validated_to_handle(handle, _permissive_schema(), record2)
        lines = handle.getvalue().strip().split("\n")
        assert len(lines) == 2

    def test_collision_success_contradiction_rejected(self) -> None:
        """A record with collision_event=True but success=1.0 must be rejected."""
        record = _valid_record()
        record["metrics"]["success"] = 1.0
        record["metrics"]["collisions"] = 1.0
        record["outcome"]["collision_event"] = True
        record["termination_reason"] = "collision"
        handle = io.StringIO()
        with pytest.raises(ValueError, match="integrity"):
            write_validated_to_handle(handle, _permissive_schema(), record)

    def test_schema_validation_enforced(self) -> None:
        """A record violating the schema must be rejected."""
        strict_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["nonexistent_field"],
            "additionalProperties": True,
        }
        handle = io.StringIO()
        with pytest.raises(ValidationError):
            write_validated_to_handle(handle, strict_schema, _valid_record())
