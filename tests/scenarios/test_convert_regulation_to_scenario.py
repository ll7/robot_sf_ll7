#!/usr/bin/env python3
"""Tests for convert_regulation_to_scenario.py.

These tests pin the three core behaviors required by issue #6054:

1. Deterministic compilation of a textual regulation excerpt into a
   parameterized scenario config.
2. Compilation/schema/scenario validity reported on SEPARATE axes, so a clean
   compile cannot be read as scenario validity.
3. Outputs marked as hypotheses until executed and reviewed.

The schema-validity axis reuses ``robot_sf.benchmark.scenario_schema``; these
tests do not modify it.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.scenario_schema import (
    SCENARIO_MATRIX_SCHEMA_VERSION,
    validate_scenario_list,
    validate_scenario_matrix_metadata,
)
from scripts.tools.convert_regulation_to_scenario import (
    DEFAULT_DENSITY_VALUE,
    GENERATION_METHOD,
    REGULATION_RECORD_SCHEMA_VERSION,
    TEMPLATE_TO_MAP,
    _build_scenario_payload,
    _map_file_for_output,
    _resolve_generated_map_file,
    _validate_regulation_record,
    assess_validity,
    compile_regulation_excerpt,
    convert_regulation_record,
)

VALID_RECORD = {
    "schema_version": REGULATION_RECORD_SCHEMA_VERSION,
    "regulation": {
        "id": "shared-space-001",
        "source": "Test source",
        "context": "shared space",
        "setting": "shared_space",
        "regulation_text": (
            "In shared spaces the robot's maximum speed shall not exceed 1.5 m/s. "
            "The robot must maintain a clearance of at least 1.0 m from pedestrians. "
            "Pedestrian density is expected to be high."
        ),
        "required_manual_review": True,
        "claim_boundary": "scenario hypothesis only; not evidence",
    },
}

VALID_RECORD_MINIMAL = {
    "schema_version": REGULATION_RECORD_SCHEMA_VERSION,
    "regulation": {
        "id": "minimal-001",
        "regulation_text": "The robot operates in a corridor.",
        "required_manual_review": True,
        "claim_boundary": "hypothesis; not evidence",
    },
}


# ---------------------------------------------------------------------------
# Input-record validation
# ---------------------------------------------------------------------------


class TestRegulationRecordValidation:
    """Tests for regulation-record.v1 envelope validation."""

    def test_valid_record_passes(self) -> None:
        errors = _validate_regulation_record(VALID_RECORD)
        assert errors == []

    def test_minimal_valid_record_passes(self) -> None:
        errors = _validate_regulation_record(VALID_RECORD_MINIMAL)
        assert errors == []

    def test_missing_schema_version(self) -> None:
        record = {"regulation": VALID_RECORD["regulation"]}
        errors = _validate_regulation_record(record)
        assert any("schema_version" in e for e in errors)

    def test_wrong_schema_version(self) -> None:
        record = copy.deepcopy(VALID_RECORD)
        record["schema_version"] = "wrong-version"
        errors = _validate_regulation_record(record)
        assert any("Unsupported schema_version" in e for e in errors)

    def test_missing_regulation_block(self) -> None:
        record = {"schema_version": REGULATION_RECORD_SCHEMA_VERSION}
        errors = _validate_regulation_record(record)
        assert any("Missing required field: regulation" in e for e in errors)

    def test_missing_required_regulation_fields(self) -> None:
        record = {
            "schema_version": REGULATION_RECORD_SCHEMA_VERSION,
            "regulation": {"id": "x"},
        }
        errors = _validate_regulation_record(record)
        joined = " ; ".join(errors)
        assert "regulation_text" in joined
        assert "required_manual_review" in joined
        assert "claim_boundary" in joined

    def test_empty_regulation_text_fails(self) -> None:
        record = copy.deepcopy(VALID_RECORD)
        record["regulation"]["regulation_text"] = "   "
        errors = _validate_regulation_record(record)
        assert any("must not be empty" in e for e in errors)

    def test_manual_review_must_be_true(self) -> None:
        record = copy.deepcopy(VALID_RECORD)
        record["regulation"]["required_manual_review"] = False
        errors = _validate_regulation_record(record)
        assert any("required_manual_review must be true" in e for e in errors)

    def test_claim_boundary_must_mention_evidence(self) -> None:
        record = copy.deepcopy(VALID_RECORD)
        record["regulation"]["claim_boundary"] = "some other boundary"
        errors = _validate_regulation_record(record)
        assert any("not evidence" in e for e in errors)

    def test_non_mapping_top_level(self) -> None:
        errors = _validate_regulation_record(["not", "a", "mapping"])  # type: ignore[arg-type]
        assert any("mapping" in e for e in errors)


# ---------------------------------------------------------------------------
# Compilation (deterministic parameter extraction)
# ---------------------------------------------------------------------------


class TestCompilation:
    """Tests for deterministic excerpt compilation."""

    def test_extracts_speed(self) -> None:
        params = compile_regulation_excerpt("The robot's maximum speed shall not exceed 1.5 m/s.")
        assert params.max_linear_speed == 1.5
        assert any(e["parameter"] == "max_linear_speed" for e in params.extracted)

    def test_extracts_clearance(self) -> None:
        params = compile_regulation_excerpt(
            "The robot must maintain a clearance of at least 1.0 m from pedestrians."
        )
        assert params.clearance_m == 1.0

    def test_extracts_density_word(self) -> None:
        params = compile_regulation_excerpt("Pedestrian density is expected to be high.")
        assert params.ped_density == 0.08

    def test_extracts_explicit_density(self) -> None:
        params = compile_regulation_excerpt("Pedestrian density is 0.06 per square meter.")
        assert params.ped_density == 0.06

    def test_explicit_density_out_of_range_falls_back(self) -> None:
        # 5.0 is outside [0, 1]; should not be taken as ped_density.
        params = compile_regulation_excerpt("Pedestrian density is 5.0 per square meter.")
        assert params.ped_density == DEFAULT_DENSITY_VALUE

    def test_zone_from_excerpt(self) -> None:
        params = compile_regulation_excerpt("In shared spaces the robot moves slowly.")
        assert params.zone_template == "shared_space"

    def test_zone_from_hint(self) -> None:
        params = compile_regulation_excerpt(
            "The robot moves slowly.", hints={"context": "corridor"}
        )
        assert params.zone_template == "corridor"

    def test_zone_default_when_unknown(self) -> None:
        params = compile_regulation_excerpt("The robot shall operate safely.")
        assert params.zone_template == "shared_space"

    def test_unmatched_clauses_recorded(self) -> None:
        params = compile_regulation_excerpt(
            "The maximum speed is 1.0 m/s. The robot shall provide an audible alert before moving."
        )
        assert any("audible alert" in c for c in params.unmatched_clauses)

    def test_european_decimal(self) -> None:
        params = compile_regulation_excerpt("The maximum speed is 1,5 m/s.")
        assert params.max_linear_speed == 1.5

    def test_multiple_speeds_warns(self) -> None:
        params = compile_regulation_excerpt(
            "The maximum speed is 1.5 m/s in open areas and 0.5 m/s near crowds."
        )
        # The first match is used, but a warning records the ambiguity.
        warnings_blob = " ".join(params.warnings) + " " + " ".join(str(e) for e in params.extracted)
        assert params.max_linear_speed in (1.5, 0.5)
        assert "multiple speed values" in warnings_blob

    def test_deterministic_output(self) -> None:
        text = VALID_RECORD["regulation"]["regulation_text"]
        p1 = compile_regulation_excerpt(text)
        p2 = compile_regulation_excerpt(text)
        assert p1 == p2

    def test_no_extraction_produces_warning(self) -> None:
        params = compile_regulation_excerpt("Be excellent to each other.")
        # Zone/density always get defaulted entries, but NOTHING meaningful is
        # extracted from such an excerpt, so the compiler warns and the
        # meaningful-extraction list is empty.
        assert params.meaningful_extractions == []
        assert any("No parameters" in w or "No maximum speed" in w for w in params.warnings)

    def test_episode_steps_hint(self) -> None:
        params = compile_regulation_excerpt("Max speed 1.0 m/s.", hints={"max_episode_steps": 250})
        assert params.max_episode_steps == 250

    def test_extracted_audit_records_zone_keyword(self) -> None:
        params = compile_regulation_excerpt("In shared spaces the robot moves slowly.")
        zone_entry = next(e for e in params.extracted if e["parameter"] == "zone_template")
        assert zone_entry["matched_keyword"] == "shared space"


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


class TestScenarioPayload:
    """Tests for the generated scenario matrix payload."""

    def test_payload_is_scenario_matrix_v1(self) -> None:
        params = compile_regulation_excerpt(VALID_RECORD["regulation"]["regulation_text"])
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params)
        assert payload["schema_version"] == SCENARIO_MATRIX_SCHEMA_VERSION
        assert len(payload["scenarios"]) == 1

    def test_speed_writes_robot_config(self) -> None:
        params = compile_regulation_excerpt("The maximum speed is 1.5 m/s.")
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params)
        assert payload["scenarios"][0]["robot_config"]["max_linear_speed"] == 1.5

    def test_no_speed_omits_robot_speed(self) -> None:
        params = compile_regulation_excerpt("Operate in a corridor.")
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params)
        robot_cfg = payload["scenarios"][0]["robot_config"]
        assert "max_linear_speed" not in robot_cfg

    def test_clearance_recorded_as_metadata_only(self) -> None:
        params = compile_regulation_excerpt("Maintain a clearance of at least 1.0 m.")
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params)
        meta = payload["scenarios"][0]["metadata"]
        assert meta["clearance_requirement_m"] == 1.0
        assert meta["clearance_enforcement"] == "metadata_only_not_runtime_enforced"

    def test_hypothesis_metadata_present(self) -> None:
        params = compile_regulation_excerpt("Max speed 1.0 m/s.")
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params)
        meta = payload["scenarios"][0]["metadata"]
        assert meta["hypothesis"] is True
        assert meta["required_manual_review"] is True
        assert meta["generation_method"] == GENERATION_METHOD
        assert meta["authoring"]["status"] == "draft"
        assert meta["authoring"]["benchmark_evidence"] is False
        assert "not executed evidence" in meta["claim_boundary"]

    def test_compiled_parameters_auditable(self) -> None:
        params = compile_regulation_excerpt(
            "In shared spaces max speed 1.5 m/s; maintain clearance of 1.0 m."
        )
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params)
        meta = payload["scenarios"][0]["metadata"]
        assert isinstance(meta["compiled_parameters"], list)
        assert len(meta["compiled_parameters"]) > 0
        assert isinstance(meta["unmatched_clauses"], list)

    def test_default_seeds(self) -> None:
        params = compile_regulation_excerpt("Max speed 1.0 m/s.")
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params)
        assert payload["scenarios"][0]["seeds"] == [6054, 6055, 6056]

    def test_deterministic_payload(self) -> None:
        params = compile_regulation_excerpt(VALID_RECORD["regulation"]["regulation_text"])
        p1 = _build_scenario_payload(VALID_RECORD["regulation"], params)
        p2 = _build_scenario_payload(VALID_RECORD["regulation"], params)
        assert p1 == p2

    def test_generated_payload_passes_schema(self) -> None:
        params = compile_regulation_excerpt(VALID_RECORD["regulation"]["regulation_text"])
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params)
        assert validate_scenario_matrix_metadata(payload) == []
        assert validate_scenario_list(payload["scenarios"]) == []


# ---------------------------------------------------------------------------
# The key feature: separate validity axes
# ---------------------------------------------------------------------------


class TestSeparateValidityAxes:
    """Compilation/schema/scenario validity must be reported separately."""

    def test_scenario_validity_always_not_assessed(self) -> None:
        params = compile_regulation_excerpt("Max speed 1.0 m/s.")
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params)
        report = assess_validity(compilation=params, payload=payload)
        assert report["scenario_validity"]["status"] == "not_assessed"
        assert "runner" in report["scenario_validity"]["reason"].lower()

    def test_schema_validity_uses_schema_validator(self) -> None:
        params = compile_regulation_excerpt("Max speed 1.0 m/s.")
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params)
        report = assess_validity(compilation=params, payload=payload)
        assert report["schema_validity"]["valid"] is True
        assert report["schema_validity"]["metadata_errors"] == []
        assert report["schema_validity"]["item_errors"] == []

    def test_schema_invalid_does_not_make_compilation_invalid(self) -> None:
        # A payload with a bad schema_version: compilation axis (independent of
        # payload) must remain valid while schema axis fails.
        params = compile_regulation_excerpt("Max speed 1.0 m/s.")
        bad_payload = {
            "schema_version": "bogus.version",
            "scenarios": [{"name": "x"}],  # missing required groups
        }
        report = assess_validity(compilation=params, payload=bad_payload)
        assert report["compilation_validity"]["valid"] is True
        assert report["schema_validity"]["valid"] is False
        # Scenario axis stays not_assessed regardless of schema problems.
        assert report["scenario_validity"]["status"] == "not_assessed"

    def test_compilation_invalid_does_not_affect_schema(self) -> None:
        # Nothing extracted -> compilation invalid, but a well-formed payload
        # still reports schema valid.
        params = CompiledParametersEmpty()
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params)
        report = assess_validity(compilation=params, payload=payload)
        assert report["compilation_validity"]["valid"] is False
        assert report["schema_validity"]["valid"] is True

    def test_report_keys_present(self) -> None:
        params = compile_regulation_excerpt("Max speed 1.0 m/s.")
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params)
        report = assess_validity(compilation=params, payload=payload)
        assert set(report.keys()) == {
            "compilation_validity",
            "schema_validity",
            "scenario_validity",
        }


def CompiledParametersEmpty():
    """Return CompiledParameters with no extracted entries (for axis tests)."""
    from scripts.tools.convert_regulation_to_scenario import CompiledParameters

    return CompiledParameters()


# ---------------------------------------------------------------------------
# Map path resolution
# ---------------------------------------------------------------------------


class TestMapFileResolution:
    """Tests for TEMPLATE_TO_MAP and map path resolution."""

    def test_all_template_to_map_entries_exist(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        for template, repo_rel in TEMPLATE_TO_MAP.items():
            svg = (repo_root / repo_rel).resolve()
            assert svg.is_file(), (
                f"TEMPLATE_TO_MAP[{template!r}] -> {repo_rel} resolves to {svg}, "
                "which does not exist"
            )

    def test_stdout_uses_repo_relative_paths(self) -> None:
        params = compile_regulation_excerpt("Max speed 1.0 m/s in shared spaces.")
        payload = _build_scenario_payload(VALID_RECORD["regulation"], params, output_path=None)
        map_file = payload["scenarios"][0]["map_file"]
        assert map_file.startswith("maps/svg_maps/"), (
            f"stdout map_file should be repo-relative, got {map_file!r}"
        )

    def test_output_yaml_uses_output_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "scenario.yaml"
            params = compile_regulation_excerpt("Max speed 1.0 m/s in shared spaces.")
            payload = _build_scenario_payload(
                VALID_RECORD["regulation"], params, output_path=output_path
            )
            map_file = payload["scenarios"][0]["map_file"]
            resolved = _resolve_generated_map_file(output_path, map_file)
            assert resolved.is_file()

    def test_map_file_for_output_returns_string(self) -> None:
        assert isinstance(_map_file_for_output("shared_space"), str)


# ---------------------------------------------------------------------------
# Integration (end-to-end convert_regulation_record)
# ---------------------------------------------------------------------------


class TestConvertRegulationRecord:
    """End-to-end integration tests."""

    def test_convert_and_write_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.yaml"
            output_path = Path(tmpdir) / "output.yaml"
            with input_path.open("w") as f:
                yaml.dump(VALID_RECORD, f)

            result = convert_regulation_record(input_path, output_path)

            assert result is not None
            assert output_path.exists()
            with output_path.open("r") as f:
                loaded = yaml.safe_load(f)
            assert loaded["schema_version"] == SCENARIO_MATRIX_SCHEMA_VERSION
            assert len(loaded["scenarios"]) == 1

    def test_convert_returns_validity_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.yaml"
            with input_path.open("w") as f:
                yaml.dump(VALID_RECORD, f)

            result = convert_regulation_record(input_path, output_path=None)
            assert result is not None
            validity = result["validity"]
            assert validity["compilation_validity"]["valid"] is True
            assert validity["schema_validity"]["valid"] is True
            assert validity["scenario_validity"]["status"] == "not_assessed"

    def test_convert_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            convert_regulation_record(Path("/nonexistent/path.yaml"))

    def test_convert_invalid_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "invalid.yaml"
            with input_path.open("w") as f:
                yaml.dump({"schema_version": "wrong"}, f)
            with pytest.raises(ValueError) as exc_info:
                convert_regulation_record(input_path)
            assert "Invalid regulation record" in str(exc_info.value)

    def test_stdout_payload_is_writable_and_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.yaml"
            with input_path.open("w") as f:
                yaml.dump(VALID_RECORD, f)
            result = convert_regulation_record(input_path, output_path=None)
            assert result is not None
            payload = result["payload"]
            assert validate_scenario_matrix_metadata(payload) == []
            assert validate_scenario_list(payload["scenarios"]) == []


# ---------------------------------------------------------------------------
# Checked-in artifacts load and validate
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_INPUT = REPO_ROOT / "configs/scenarios/contracts/issue_6054_regulation_source_example.yaml"
EXAMPLE_OUTPUT = REPO_ROOT / "configs/scenarios/single/issue_6054_regulation_to_scenario.yaml"


class TestCheckedInArtifacts:
    """The checked-in example input/output are consistent and schema-valid."""

    def test_example_input_is_valid_regulation_record(self) -> None:
        assert EXAMPLE_INPUT.is_file(), f"missing example input {EXAMPLE_INPUT}"
        with EXAMPLE_INPUT.open("r") as f:
            record = yaml.safe_load(f)
        assert _validate_regulation_record(record) == []

    def test_example_output_is_schema_valid(self) -> None:
        assert EXAMPLE_OUTPUT.is_file(), f"missing example output {EXAMPLE_OUTPUT}"
        with EXAMPLE_OUTPUT.open("r") as f:
            payload = yaml.safe_load(f)
        assert validate_scenario_matrix_metadata(payload) == []
        assert validate_scenario_list(payload["scenarios"]) == []

    def test_example_output_is_marked_hypothesis(self) -> None:
        with EXAMPLE_OUTPUT.open("r") as f:
            payload = yaml.safe_load(f)
        meta = payload["scenarios"][0]["metadata"]
        assert meta["hypothesis"] is True
        assert meta["required_manual_review"] is True
        assert meta["authoring"]["benchmark_evidence"] is False

    def test_example_output_matches_regeneration(self) -> None:
        """The checked-in output must equal a fresh compile (no drift)."""
        with EXAMPLE_INPUT.open("r") as f:
            record = yaml.safe_load(f)
        regulation = record["regulation"]
        hints = {
            k: regulation[k] for k in ("context", "setting", "max_episode_steps") if k in regulation
        }
        params = compile_regulation_excerpt(regulation["regulation_text"], hints=hints)
        regenerated = _build_scenario_payload(regulation, params, output_path=EXAMPLE_OUTPUT)
        with EXAMPLE_OUTPUT.open("r") as f:
            on_disk = yaml.safe_load(f)
        assert on_disk == regenerated, (
            "Checked-in example output differs from a fresh compile. "
            "Regenerate with the converter to remove drift."
        )
