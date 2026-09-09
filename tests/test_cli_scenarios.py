"""Tests for the ``robot-sf scenarios`` discovery, inspection, and validation CLI (issue #8748)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from robot_sf import cli_scenarios as cli_scenarios_module
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
    assert payload["valid"] is False
    assert payload["status"] == STATUS_UNSUPPORTED
    assert any(e["code"] == REASON_UNSUPPORTED_SCENARIO for e in payload["errors"])


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


def test_cli_scenarios_unsupported_is_nonzero(capsys) -> None:
    """The CLI rejects a scenario explicitly marked ``supported: false``."""
    assert (
        main(
            [
                "scenarios",
                "validate",
                "tests/fixtures/cli_scenarios/unsupported.yaml",
                "--format",
                "json",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert payload["status"] == STATUS_UNSUPPORTED
    assert any(error["code"] == REASON_UNSUPPORTED_SCENARIO for error in payload["errors"])


def test_cli_scenarios_import_light_includes_public_cli_subparser() -> None:
    """Importing and registering the public scenario command stays import-light."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from robot_sf.cli import _build_scenarios_parser; "
            "_build_scenarios_parser(); "
            "forbidden = {'torch', 'stable_baselines3', 'carla'}; "
            "loaded = forbidden.intersection(sys.modules.keys()); "
            "assert not loaded, f'Forbidden modules imported: {loaded}'"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Import test failed: {result.stderr}"


def test_catalog_uses_canonical_loader_for_includes_selection_and_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery honors canonical include expansion, selection, and overrides."""
    scenarios_root = tmp_path / "configs" / "scenarios"
    input_root = tmp_path / "configs" / "scenario_inputs"
    scenarios_root.mkdir(parents=True)
    input_root.mkdir(parents=True)
    (input_root / "included.yaml").write_text(
        "scenarios:\n"
        "  - name: selected_from_include\n"
        "    simulation_config:\n"
        "      max_episode_steps: 10\n"
        "  - name: intentionally_not_selected\n"
        "    simulation_config:\n"
        "      max_episode_steps: 20\n",
        encoding="utf-8",
    )
    (scenarios_root / "composed.yaml").write_text(
        "includes:\n"
        "  - ../scenario_inputs/included.yaml\n"
        "select_scenarios:\n"
        "  - selected_from_include\n"
        "scenario_overrides:\n"
        "  simulation_config:\n"
        "    max_episode_steps: 42\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_scenarios_module, "_find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli_scenarios_module, "_CACHED_CATALOG", None)
    monkeypatch.setattr(cli_scenarios_module, "_CACHED_EXCLUSIONS", None)

    payload = cli_scenarios_module.list_scenarios_payload()

    assert payload["count"] == 1
    scenario = payload["scenarios"][0]
    assert scenario["identity"] == "selected_from_include"
    assert scenario["source_file"] == "configs/scenarios/composed.yaml"
    assert scenario["horizon_and_dt"]["max_episode_steps"] == 42


def test_catalog_fails_closed_on_malformed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed YAML is not silently dropped from catalog discovery."""
    scenarios_root = tmp_path / "configs" / "scenarios"
    scenarios_root.mkdir(parents=True)
    malformed = scenarios_root / "malformed.yaml"
    malformed.write_text("scenarios:\n  - name: [\n", encoding="utf-8")
    monkeypatch.setattr(cli_scenarios_module, "_find_repo_root", lambda: tmp_path)

    with pytest.raises(yaml.YAMLError):
        cli_scenarios_module._build_catalog()


def test_map_resolution_uses_registry_and_never_basename_fallback() -> None:
    """Map lookup rejects basename guesses and gives ``map_id`` precedence."""
    repo_root = cli_scenarios_module._find_repo_root()
    source_file = repo_root / "configs" / "scenarios" / "single" / "example.yaml"

    missing_path = cli_scenarios_module._resolve_map_reference(
        {"map_file": "missing/classic_crossing.svg"},
        source_file=source_file,
        repo_root=repo_root,
    )
    assert missing_path["map_exists"] is False
    assert missing_path["resolved_map_path"].endswith("missing/classic_crossing.svg")

    known_id = cli_scenarios_module._resolve_map_reference(
        {"map_id": "classic_crossing"},
        source_file=source_file,
        repo_root=repo_root,
    )
    assert known_id["map_exists"] is True
    assert known_id["resolved_map_path"] == "maps/svg_maps/classic_crossing.svg"

    unknown_id = cli_scenarios_module._resolve_map_reference(
        {
            "map_id": "not_in_canonical_registry",
            "map_file": "../../../maps/svg_maps/classic_crossing.svg",
        },
        source_file=source_file,
        repo_root=repo_root,
    )
    assert unknown_id["map_exists"] is False
    assert unknown_id["resolved_map_path"] is None


def test_validate_scenario_manifest_schema_version_uses_canonical_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Unsupported manifest metadata produces a schema error and CLI failure."""
    manifest = tmp_path / "manifest.yaml"
    map_file = tmp_path / "map.svg"
    map_file.write_text("<svg/>\n", encoding="utf-8")
    manifest.write_text(
        "schema_version: robot_sf.scenario_matrix.v0\n"
        "scenarios:\n"
        "  - name: metadata_version_mismatch\n"
        "    map_file: map.svg\n"
        "    simulation_config:\n"
        "      max_episode_steps: 10\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_scenarios_module, "_find_repo_root", lambda: tmp_path)

    payload = validate_scenario_payload(str(manifest))
    assert payload["valid"] is False
    assert payload["status"] == STATUS_INVALID
    metadata_errors = [
        error
        for error in payload["errors"]
        if error["code"] == REASON_SCHEMA_VALIDATION_ERROR and error["path"] == "/schema_version"
    ]
    assert len(metadata_errors) == 1

    assert main(["scenarios", "validate", str(manifest), "--format", "json"]) == 2
    cli_payload = json.loads(capsys.readouterr().out)
    assert any(error["path"] == "/schema_version" for error in cli_payload["errors"])


def test_scenario_docs_match_canonical_schema_and_status_contract() -> None:
    """The scenario guide names the canonical schema and only emitted statuses."""
    docs = Path("docs/SCENARIOS.md").read_text(encoding="utf-8")

    assert "robot_sf/benchmark/schemas/scenarios.schema.json" in docs
    assert "`schemas/scenarios.schema.json`" not in docs
    assert "`external_asset_dependent`" not in docs
    for status in ("valid", "missing", "invalid", "duplicate", "unsupported"):
        assert f"`{status}`" in docs


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
