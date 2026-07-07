#!/usr/bin/env python3
"""Tests for convert_failure_record_to_scenario.py."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import yaml

from scripts.tools.convert_failure_record_to_scenario import (
    FAILURE_RECORD_SCHEMA_VERSION,
    _build_scenario_payload,
    _generate_assumptions,
    _generate_invalidity_warnings,
    _generate_pedestrians,
    _validate_failure_record,
    convert_failure_record,
)

VALID_EVENT_RECORD = {
    "schema_version": FAILURE_RECORD_SCHEMA_VERSION,
    "failure_record": {
        "id": "test-event-001",
        "source": "Test source",
        "date": "2025-01-01",
        "environment": "event",
        "actors": [
            {"type": "pedestrian", "count": 10, "description": "Dense crowd"},
            {"type": "temporary_obstacle", "count": 1, "description": "Barrier"},
        ],
        "triggering_condition": "Sudden crowd surge",
        "failure_mode": "blocked_path",
        "contextual_factors": ["dense_crowd", "temporary_obstacle"],
        "required_manual_review": True,
        "claim_boundary": "scenario hypothesis only; not evidence",
    },
}

VALID_SIDEWALK_RECORD = {
    "schema_version": FAILURE_RECORD_SCHEMA_VERSION,
    "failure_record": {
        "id": "test-sidewalk-001",
        "source": "Test source",
        "date": "2025-01-01",
        "environment": "sidewalk",
        "actors": [
            {"type": "pedestrian", "count": 4, "description": "Passing pedestrians"},
        ],
        "triggering_condition": "Blocked sidewalk",
        "failure_mode": "blocked_path",
        "contextual_factors": ["temporary_obstacle"],
        "required_manual_review": True,
        "claim_boundary": "scenario hypothesis only; not evidence",
    },
}


class TestValidation:
    """Tests for failure record validation."""

    def test_valid_event_record(self):
        """Valid event-style record passes validation."""
        errors = _validate_failure_record(VALID_EVENT_RECORD)
        assert errors == []

    def test_valid_sidewalk_record(self):
        """Valid AMMV sidewalk record passes validation."""
        errors = _validate_failure_record(VALID_SIDEWALK_RECORD)
        assert errors == []

    def test_missing_schema_version(self):
        """Missing schema_version fails with clear error."""
        record = {"failure_record": VALID_EVENT_RECORD["failure_record"]}
        errors = _validate_failure_record(record)
        assert any("schema_version" in err for err in errors)

    def test_wrong_schema_version(self):
        """Wrong schema_version fails with clear error."""
        record = copy.deepcopy(VALID_EVENT_RECORD)
        record["schema_version"] = "wrong-version"
        errors = _validate_failure_record(record)
        assert any("Unsupported schema_version" in err for err in errors)

    def test_missing_failure_record(self):
        """Missing failure_record field fails."""
        record = {"schema_version": FAILURE_RECORD_SCHEMA_VERSION}
        errors = _validate_failure_record(record)
        assert any("failure_record" in err for err in errors)

    def test_unknown_environment(self):
        """Unknown environment fails with clear error."""
        record = copy.deepcopy(VALID_EVENT_RECORD)
        record["failure_record"]["environment"] = "unknown_env"
        errors = _validate_failure_record(record)
        assert any("Unknown environment" in err for err in errors)

    def test_unknown_failure_mode(self):
        """Unknown failure_mode fails with clear error."""
        record = copy.deepcopy(VALID_EVENT_RECORD)
        record["failure_record"]["failure_mode"] = "unknown_mode"
        errors = _validate_failure_record(record)
        assert any("Unknown failure_mode" in err for err in errors)

    def test_manual_review_must_be_true(self):
        """required_manual_review must be exactly True."""
        record = copy.deepcopy(VALID_EVENT_RECORD)
        record["failure_record"]["required_manual_review"] = False
        errors = _validate_failure_record(record)
        assert any("required_manual_review must be true" in err for err in errors)

    def test_claim_boundary_must_mention_evidence(self):
        """claim_boundary must state 'not evidence'."""
        record = copy.deepcopy(VALID_EVENT_RECORD)
        record["failure_record"]["claim_boundary"] = "some other boundary"
        errors = _validate_failure_record(record)
        assert any("not evidence" in err for err in errors)

    def test_actors_must_be_list(self):
        """actors field must be a list."""
        record = copy.deepcopy(VALID_EVENT_RECORD)
        record["failure_record"]["actors"] = "not a list"
        errors = _validate_failure_record(record)
        assert any("actors must be a list" in err for err in errors)

    def test_actor_must_have_type_and_count(self):
        """Each actor must have type and count fields."""
        record = copy.deepcopy(VALID_EVENT_RECORD)
        record["failure_record"]["actors"] = [{"type": "pedestrian"}]
        errors = _validate_failure_record(record)
        assert any("missing required fields" in err for err in errors)


class TestAssumptions:
    """Tests for assumption generation."""

    def test_basic_assumptions(self):
        """Basic assumptions include environment and failure mode mapping."""
        assumptions = _generate_assumptions(VALID_EVENT_RECORD["failure_record"])
        assert any("Environment" in a and "event" in a for a in assumptions)
        assert any("Failure mode" in a and "blocked_path" in a for a in assumptions)

    def test_actor_count_assumption(self):
        """High actor count generates truncation assumption."""
        record = copy.deepcopy(VALID_EVENT_RECORD["failure_record"])
        record["actors"] = [{"type": "pedestrian", "count": 20}]
        assumptions = _generate_assumptions(record)
        assert any("truncated" in a for a in assumptions)

    def test_unmapped_factor_assumption(self):
        """Unmapped contextual factors generate assumptions."""
        record = copy.deepcopy(VALID_EVENT_RECORD["failure_record"])
        record["contextual_factors"] = ["unknown_factor"]
        assumptions = _generate_assumptions(record)
        assert any("unknown_factor" in a for a in assumptions)


class TestInvalidityWarnings:
    """Tests for invalidity warning generation."""

    def test_known_factor_warnings(self):
        """Known contextual factors generate specific warnings."""
        warnings = _generate_invalidity_warnings(VALID_EVENT_RECORD["failure_record"])
        assert any("High pedestrian density" in w for w in warnings)
        assert any("Obstacle representation" in w for w in warnings)

    def test_unknown_factor_warning(self):
        """Unknown factors generate unmapped warning."""
        record = dict(VALID_EVENT_RECORD["failure_record"])
        record["contextual_factors"] = ["unknown_factor"]
        warnings = _generate_invalidity_warnings(record)
        assert any("Unmapped contextual factor" in w for w in warnings)

    def test_collision_warning(self):
        """Collision failure mode generates safety warning."""
        record = dict(VALID_EVENT_RECORD["failure_record"])
        record["failure_mode"] = "collision"
        warnings = _generate_invalidity_warnings(record)
        assert any("safety review" in w.lower() for w in warnings)


class TestPedestrianGeneration:
    """Tests for pedestrian list generation."""

    def test_generates_pedestrians_from_actors(self):
        """Pedestrians are generated from pedestrian actors."""
        pedestrians = _generate_pedestrians(VALID_EVENT_RECORD["failure_record"])
        assert len(pedestrians) == 4

    def test_respects_max_count(self):
        """Pedestrian count is capped at max_count."""
        record = copy.deepcopy(VALID_EVENT_RECORD["failure_record"])
        record["actors"] = [{"type": "pedestrian", "count": 100}]
        pedestrians = _generate_pedestrians(record)
        assert len(pedestrians) == 4

    def test_pedestrian_ids_sequential(self):
        """Pedestrian IDs are sequential (h1, h2, ...)."""
        pedestrians = _generate_pedestrians(VALID_EVENT_RECORD["failure_record"])
        assert pedestrians[0]["id"] == "h1"
        assert pedestrians[1]["id"] == "h2"

    def test_non_pedestrian_actors_ignored(self):
        """Non-pedestrian actors are not converted to pedestrians."""
        record = {
            "actors": [
                {"type": "vehicle", "count": 5},
                {"type": "obstacle", "count": 2},
            ]
        }
        pedestrians = _generate_pedestrians(record)
        assert pedestrians == []


class TestScenarioPayload:
    """Tests for complete scenario payload generation."""

    def test_event_style_scenario(self):
        """Event-style record generates event/disruption scenario."""
        payload = _build_scenario_payload(VALID_EVENT_RECORD["failure_record"])
        assert "scenarios" in payload
        scenario = payload["scenarios"][0]
        assert "classic_crossing" in scenario["map_file"]
        assert "failure_record_test-event-001" in scenario["name"]

    def test_sidewalk_scenario(self):
        """Sidewalk record generates sidewalk scenario."""
        payload = _build_scenario_payload(VALID_SIDEWALK_RECORD["failure_record"])
        scenario = payload["scenarios"][0]
        assert "classic_doorway" in scenario["map_file"]

    def test_metadata_required_manual_review(self):
        """Output always has required_manual_review: true."""
        payload = _build_scenario_payload(VALID_EVENT_RECORD["failure_record"])
        metadata = payload["scenarios"][0]["metadata"]
        assert metadata["required_manual_review"] is True

    def test_metadata_claim_boundary(self):
        """Output includes claim boundary statement."""
        payload = _build_scenario_payload(VALID_EVENT_RECORD["failure_record"])
        metadata = payload["scenarios"][0]["metadata"]
        assert "not executed evidence" in metadata["claim_boundary"]

    def test_metadata_generated_assumptions(self):
        """Output includes generated_assumptions list."""
        payload = _build_scenario_payload(VALID_EVENT_RECORD["failure_record"])
        metadata = payload["scenarios"][0]["metadata"]
        assert isinstance(metadata["generated_assumptions"], list)
        assert len(metadata["generated_assumptions"]) > 0

    def test_metadata_invalidity_warnings(self):
        """Output includes invalidity_warnings list."""
        payload = _build_scenario_payload(VALID_EVENT_RECORD["failure_record"])
        metadata = payload["scenarios"][0]["metadata"]
        assert isinstance(metadata["invalidity_warnings"], list)

    def test_metadata_expected_failure_modes(self):
        """Output includes expected_failure_modes list."""
        payload = _build_scenario_payload(VALID_EVENT_RECORD["failure_record"])
        metadata = payload["scenarios"][0]["metadata"]
        assert isinstance(metadata["expected_failure_modes"], list)
        assert "stuck" in metadata["expected_failure_modes"]

    def test_metadata_authoring_draft(self):
        """Output includes authoring metadata with draft status."""
        payload = _build_scenario_payload(VALID_EVENT_RECORD["failure_record"])
        metadata = payload["scenarios"][0]["metadata"]
        assert metadata["authoring"]["status"] == "draft"
        assert metadata["authoring"]["benchmark_evidence"] is False

    def test_seeds_included(self):
        """Output includes default seeds."""
        payload = _build_scenario_payload(VALID_EVENT_RECORD["failure_record"])
        scenario = payload["scenarios"][0]
        assert scenario["seeds"] == [101, 102, 103]

    def test_deterministic_output(self):
        """Same input produces identical output."""
        payload1 = _build_scenario_payload(VALID_EVENT_RECORD["failure_record"])
        payload2 = _build_scenario_payload(VALID_EVENT_RECORD["failure_record"])
        assert payload1 == payload2

    def test_map_file_paths_resolve_to_existing_files(self):
        """All emitted map_file paths must resolve to existing SVG maps."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        test_records = [
            VALID_EVENT_RECORD["failure_record"],
            VALID_SIDEWALK_RECORD["failure_record"],
        ]
        for record in test_records:
            payload = _build_scenario_payload(record)
            scenario = payload["scenarios"][0]
            map_file = scenario["map_file"]
            resolved = (repo_root / "output" / "failure_record_scenarios").resolve()
            expected_map = (resolved / map_file).resolve()
            assert expected_map.exists(), f"map_file {map_file!r} resolves to {expected_map} which does not exist"


class TestConvertFailureRecord:
    """Integration tests for the convert_failure_record function."""

    def test_convert_and_write_file(self):
        """Convert valid record and write to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.yaml"
            output_path = Path(tmpdir) / "output.yaml"

            with input_path.open("w") as f:
                yaml.dump(VALID_EVENT_RECORD, f)

            payload = convert_failure_record(input_path, output_path)

            assert payload is not None
            assert output_path.exists()

            with output_path.open("r") as f:
                loaded = yaml.safe_load(f)

            assert loaded["schema_version"] == "robot_sf.scenario_matrix.v1"
            assert len(loaded["scenarios"]) == 1

    def test_convert_file_not_found(self):
        """Missing input file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            convert_failure_record(Path("/nonexistent/path.yaml"))

    def test_convert_invalid_record(self):
        """Invalid record raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "invalid.yaml"
            with input_path.open("w") as f:
                yaml.dump({"schema_version": "wrong"}, f)

            with pytest.raises(ValueError) as exc_info:
                convert_failure_record(input_path)

            assert "Invalid failure record" in str(exc_info.value)

    def test_convert_stdout(self):
        """Convert with stdout output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.yaml"
            with input_path.open("w") as f:
                yaml.dump(VALID_EVENT_RECORD, f)

            payload = convert_failure_record(input_path, output_path=None)

            assert payload is not None
            assert "scenarios" in payload
