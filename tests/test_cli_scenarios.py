"""Tests for the ``robot-sf scenarios`` discovery, inspection, and validation CLI (issue #8748)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from robot_sf.cli import main
from robot_sf.cli_scenarios import (
    REASON_DUPLICATE_SCENARIO_ID,
    REASON_EMPTY_FILE,
    REASON_FILE_NOT_FOUND,
    REASON_IS_A_DIRECTORY,
    REASON_MALFORMED_YAML,
    REASON_MAP_NOT_FOUND,
    REASON_PATH_TRAVERSAL,
    REASON_SCHEMA_VALIDATION_ERROR,
    REASON_UNSUPPORTED_SCENARIO,
    SCHEMA_DESCRIBE_VERSION,
    SCHEMA_LIST_VERSION,
    SCHEMA_VALIDATE_VERSION,
    STATUS_INVALID,
    STATUS_MISSING,
    STATUS_UNSUPPORTED,
    STATUS_VALID,
    describe_scenario_payload,
    list_scenarios_payload,
    validate_scenario_payload,
)


def test_list_scenarios_payload_structure() -> None:
    """``list_scenarios_payload`` returns a valid scenario_list.v1 payload."""
    payload = list_scenarios_payload()
    assert payload["schema_version"] == SCHEMA_LIST_VERSION
    assert payload["count"] > 0
    assert len(payload["scenarios"]) == payload["count"]

    # Verify deterministic sorting by identity
    identities = [s["identity"] for s in payload["scenarios"]]
    assert identities == sorted(identities)

    # Verify required known scenarios are present
    assert "quickstart_demo_crossing_basic" in identities
    assert "classic_crossing_low" in identities

    # Verify exclusions list
    exclusions = payload["exclusions"]
    assert len(exclusions) > 0
    assert any("archetype_validation_waivers.yaml" in exc["path"] for exc in exclusions)
    assert any("classic_density_tier_index.yaml" in exc["path"] for exc in exclusions)

    # Verify fields on each scenario summary
    sample = payload["scenarios"][0]
    for key in (
        "identity",
        "source_file",
        "duplicate_sources",
        "family",
        "status",
        "map_reference",
        "actor_counts",
        "seed_policy",
        "horizon_and_dt",
        "kinematics_and_observation",
        "validation_status",
    ):
        assert key in sample


def test_describe_scenario_by_exact_name() -> None:
    """``describe_scenario_payload`` returns details for an exact scenario id."""
    payload = describe_scenario_payload("quickstart_demo_crossing_basic")
    assert payload["schema_version"] == SCHEMA_DESCRIBE_VERSION
    assert payload["requested_query"] == "quickstart_demo_crossing_basic"
    assert payload["identity"] == "quickstart_demo_crossing_basic"
    assert "quickstart_demo.yaml" in payload["source_file"]
    assert payload["status"] == STATUS_VALID
    assert payload["map_reference"]["map_exists"] is True
    assert payload["validation_status"]["valid"] is True
    assert isinstance(payload["raw_metadata"], dict)


def test_describe_scenario_by_file_stem() -> None:
    """``describe_scenario_payload`` resolves single scenario files by stem."""
    payload = describe_scenario_payload("quickstart_demo")
    assert payload["schema_version"] == SCHEMA_DESCRIBE_VERSION
    assert payload["identity"] == "quickstart_demo_crossing_basic"


def test_describe_scenario_by_relative_path() -> None:
    """``describe_scenario_payload`` resolves direct file path queries."""
    payload = describe_scenario_payload("configs/scenarios/single/quickstart_demo.yaml")
    assert payload["schema_version"] == SCHEMA_DESCRIBE_VERSION
    assert payload["identity"] == "quickstart_demo_crossing_basic"


def test_describe_scenario_unknown_raises_keyerror() -> None:
    """Describing an unregistered scenario raises KeyError."""
    with pytest.raises(KeyError, match="Unknown scenario 'nonexistent_unknown_scenario'"):
        describe_scenario_payload("nonexistent_unknown_scenario")


def test_describe_scenario_ambiguous_raises_valueerror() -> None:
    """Describing an ambiguous scenario name raises ValueError with candidate files."""
    with pytest.raises(ValueError, match="is ambiguous"):
        describe_scenario_payload("issue_2756_occluded_emergence")


def test_validate_scenario_valid_file() -> None:
    """``validate_scenario_payload`` validates a known good scenario file."""
    payload = validate_scenario_payload("configs/scenarios/single/quickstart_demo.yaml")
    assert payload["schema_version"] == SCHEMA_VALIDATE_VERSION
    assert payload["valid"] is True
    assert payload["status"] == STATUS_VALID
    assert payload["num_scenarios"] == 1
    assert payload["errors"] == []


def test_validate_scenario_valid_fixture() -> None:
    """``validate_scenario_payload`` validates the valid test fixture."""
    payload = validate_scenario_payload("tests/fixtures/cli_scenarios/valid_scenario.yaml")
    assert payload["valid"] is True
    assert payload["status"] == STATUS_VALID
    assert payload["num_scenarios"] == 1
    assert payload["errors"] == []


def test_validate_scenario_missing_map() -> None:
    """``validate_scenario_payload`` catches non-existent map reference."""
    payload = validate_scenario_payload("tests/fixtures/cli_scenarios/missing_map.yaml")
    assert payload["valid"] is False
    assert payload["status"] == STATUS_MISSING
    assert any(e["code"] == REASON_MAP_NOT_FOUND for e in payload["errors"])


def test_validate_scenario_duplicate_id() -> None:
    """``validate_scenario_payload`` catches duplicate scenario IDs within a file."""
    payload = validate_scenario_payload("tests/fixtures/cli_scenarios/duplicate_id.yaml")
    assert payload["valid"] is False
    assert payload["status"] == STATUS_INVALID
    assert any(e["code"] == REASON_DUPLICATE_SCENARIO_ID for e in payload["errors"])


def test_validate_scenario_invalid_schema() -> None:
    """``validate_scenario_payload`` catches schema validation errors."""
    payload = validate_scenario_payload("tests/fixtures/cli_scenarios/invalid_schema.yaml")
    assert payload["valid"] is False
    assert payload["status"] == STATUS_INVALID
    assert any(e["code"] == REASON_SCHEMA_VALIDATION_ERROR for e in payload["errors"])


def test_validate_scenario_malformed_yaml() -> None:
    """``validate_scenario_payload`` catches malformed YAML syntax."""
    payload = validate_scenario_payload("tests/fixtures/cli_scenarios/malformed.yaml.invalid")
    assert payload["valid"] is False
    assert payload["status"] == STATUS_INVALID
    assert any(e["code"] == REASON_MALFORMED_YAML for e in payload["errors"])


def test_validate_scenario_unsupported_status() -> None:
    """``validate_scenario_payload`` reports unsupported scenario status."""
    payload = validate_scenario_payload("tests/fixtures/cli_scenarios/unsupported.yaml")
    assert payload["valid"] is True
    assert payload["status"] == STATUS_UNSUPPORTED
    assert any(w["code"] == REASON_UNSUPPORTED_SCENARIO for w in payload["warnings"])


def test_validate_scenario_file_not_found() -> None:
    """``validate_scenario_payload`` reports missing scenario file."""
    payload = validate_scenario_payload("configs/scenarios/does_not_exist_at_all.yaml")
    assert payload["valid"] is False
    assert payload["status"] == STATUS_MISSING
    assert any(e["code"] == REASON_FILE_NOT_FOUND for e in payload["errors"])


def test_validate_scenario_is_a_directory() -> None:
    """``validate_scenario_payload`` rejects directories."""
    payload = validate_scenario_payload("configs/scenarios")
    assert payload["valid"] is False
    assert payload["status"] == STATUS_INVALID
    assert any(e["code"] == REASON_IS_A_DIRECTORY for e in payload["errors"])


def test_validate_scenario_path_traversal() -> None:
    """``validate_scenario_payload`` rejects path traversal outside repository root."""
    payload = validate_scenario_payload("../../../../etc/passwd")
    assert payload["valid"] is False
    assert payload["status"] == STATUS_INVALID
    assert any(e["code"] == REASON_PATH_TRAVERSAL for e in payload["errors"])


def test_validate_scenario_empty_file() -> None:
    """``validate_scenario_payload`` rejects empty YAML files."""
    payload = validate_scenario_payload("tests/fixtures/cli_scenarios/empty.yaml")
    assert payload["valid"] is False
    assert payload["status"] == STATUS_INVALID
    assert any(e["code"] == REASON_EMPTY_FILE for e in payload["errors"])


def test_cli_scenarios_list_friendly_and_json(capsys) -> None:
    """Top-level ``scenarios list`` command exits 0 in friendly and json formats."""
    assert main(["scenarios", "list"]) == 0
    out_friendly = capsys.readouterr().out
    assert "Curated Scenarios" in out_friendly
    assert "quickstart_demo_crossing_basic" in out_friendly

    assert main(["scenarios", "list", "--format", "json"]) == 0
    out_json = capsys.readouterr().out
    data = json.loads(out_json)
    assert data["schema_version"] == SCHEMA_LIST_VERSION
    assert data["count"] > 0


def test_cli_scenarios_describe_friendly_and_json(capsys) -> None:
    """Top-level ``scenarios describe`` command exits 0 for valid and 2 for error."""
    assert main(["scenarios", "describe", "quickstart_demo_crossing_basic"]) == 0
    out_friendly = capsys.readouterr().out
    assert "Scenario: quickstart_demo_crossing_basic" in out_friendly
    assert "Source File: configs/scenarios/single/quickstart_demo.yaml" in out_friendly

    assert (
        main(["scenarios", "describe", "quickstart_demo_crossing_basic", "--format", "json"]) == 0
    )
    out_json = capsys.readouterr().out
    data = json.loads(out_json)
    assert data["schema_version"] == SCHEMA_DESCRIBE_VERSION
    assert data["identity"] == "quickstart_demo_crossing_basic"

    # Unknown exits 2
    assert main(["scenarios", "describe", "unknown_scenario_id"]) == 2
    err = capsys.readouterr().err
    assert "Unknown scenario 'unknown_scenario_id'" in err

    # Unknown with --format json exits 2 and outputs structured error
    assert main(["scenarios", "describe", "unknown_scenario_id", "--format", "json"]) == 2
    out_err_json = capsys.readouterr().out
    data_err = json.loads(out_err_json)
    assert data_err["status"] == "error"
    assert "unknown_scenario_id" in data_err["error"]


def test_cli_scenarios_validate_friendly_and_json(capsys) -> None:
    """Top-level ``scenarios validate`` command exits 0 for valid and 2 for invalid."""
    assert main(["scenarios", "validate", "configs/scenarios/single/quickstart_demo.yaml"]) == 0
    out = capsys.readouterr().out
    assert "Scenario validation: PASS" in out

    assert (
        main(
            [
                "scenarios",
                "validate",
                "configs/scenarios/single/quickstart_demo.yaml",
                "--format",
                "json",
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["valid"] is True

    # Missing file exits 2
    assert main(["scenarios", "validate", "nonexistent.yaml"]) == 2
    out_missing = capsys.readouterr().out
    assert "Scenario validation: FAIL" in out_missing

    # Malformed file exits 2
    assert (
        main(["scenarios", "validate", "tests/fixtures/cli_scenarios/malformed.yaml.invalid"]) == 2
    )


def test_cli_scenarios_import_light_isolated_subprocess() -> None:
    """Verify that importing cli_scenarios does not import heavy dependencies."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from robot_sf import cli_scenarios; "
            "forbidden = {'torch', 'stable_baselines3', 'carla'}; "
            "loaded = forbidden.intersection(sys.modules.keys()); "
            "assert not loaded, f'Forbidden modules imported: {loaded}'"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Import test failed: {result.stderr}"
