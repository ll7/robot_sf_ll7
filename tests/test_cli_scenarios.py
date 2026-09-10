"""Tests for the ``robot-sf scenarios`` discovery, inspection, and validation CLI (issue #8748)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf import cli_scenarios as cli_scenarios_module
from robot_sf.cli import main
from robot_sf.cli_scenarios import (
    REASON_DUPLICATE_SCENARIO_ID,
    REASON_EMPTY_FILE,
    REASON_EXTERNAL_ASSET_DEPENDENT,
    REASON_FILE_NOT_FOUND,
    REASON_IS_A_DIRECTORY,
    REASON_LOAD_FAILURE,
    REASON_MALFORMED_YAML,
    REASON_MAP_NOT_FOUND,
    REASON_PATH_TRAVERSAL,
    REASON_SCHEMA_VALIDATION_ERROR,
    REASON_UNSUPPORTED_SCENARIO,
    SCHEMA_DESCRIBE_VERSION,
    SCHEMA_LIST_VERSION,
    SCHEMA_VALIDATE_VERSION,
    STATUS_EXTERNAL_ASSET_DEPENDENT,
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


def test_cli_scenarios_commands_are_import_light_in_isolated_process() -> None:
    """List/describe execution does not import optional frameworks or sensors."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import contextlib, io, sys\n"
            "from robot_sf.cli import main\n"
            "sink = io.StringIO()\n"
            "with contextlib.redirect_stdout(sink):\n"
            "    assert main(['scenarios', 'list', '--format', 'json']) == 0\n"
            "    assert main(['scenarios', 'describe', 'quickstart_demo_crossing_basic', '--format', 'json']) == 0\n"
            "forbidden_prefixes = ('matplotlib', 'scipy', 'robot_sf.sensor')\n"
            "loaded = sorted(name for name in sys.modules if name.startswith(forbidden_prefixes))\n"
            "assert not loaded, f'Optional modules imported: {loaded}'\n"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Command import-light test failed: {result.stderr}"


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


def test_catalog_rejects_out_of_repo_symlink_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A catalog symlink cannot make an external manifest appear bundled."""
    scenarios_root = tmp_path / "configs" / "scenarios"
    scenarios_root.mkdir(parents=True)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-manifest.yaml"
    outside.write_text(
        "scenarios:\n"
        "  - name: external_symlink_manifest\n"
        "    map_file: map.svg\n"
        "    simulation_config:\n"
        "      max_episode_steps: 10\n",
        encoding="utf-8",
    )
    linked = scenarios_root / "linked.yaml"
    linked.symlink_to(outside)
    monkeypatch.setattr(cli_scenarios_module, "_find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli_scenarios_module, "_CACHED_CATALOG", None)
    monkeypatch.setattr(cli_scenarios_module, "_CACHED_EXCLUSIONS", None)

    try:
        with pytest.raises(ValueError, match=REASON_EXTERNAL_ASSET_DEPENDENT):
            cli_scenarios_module.list_scenarios_payload()
    finally:
        outside.unlink(missing_ok=True)


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


def test_list_and_describe_surface_canonical_row_and_metadata_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """List/describe expose unknown-field, duplicate, and metadata failures."""
    scenarios_root = tmp_path / "configs" / "scenarios"
    map_root = tmp_path / "maps"
    scenarios_root.mkdir(parents=True)
    map_root.mkdir()
    (map_root / "local.svg").write_text("<svg/>\n", encoding="utf-8")
    manifest = scenarios_root / "diagnostics.yaml"
    manifest.write_text(
        "schema_version: robot_sf.scenario_matrix.v0\n"
        "scenarios:\n"
        "  - name: duplicate_diagnostic\n"
        "    map_file: ../../maps/local.svg\n"
        "    unknown_contract_field: true\n"
        "  - name: duplicate_diagnostic\n"
        "    map_file: ../../maps/local.svg\n"
        "  - name: shadowed_identity\n"
        "    map_file: ../../maps/local.svg\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_scenarios_module, "_find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli_scenarios_module, "_CACHED_CATALOG", None)
    monkeypatch.setattr(cli_scenarios_module, "_CACHED_EXCLUSIONS", None)

    listed = cli_scenarios_module.list_scenarios_payload()
    row = next(item for item in listed["scenarios"] if item["identity"] == "duplicate_diagnostic")
    listed_codes = {error["code"] for error in row["validation_status"]["errors"]}
    assert row["validation_status"]["valid"] is False
    assert REASON_SCHEMA_VALIDATION_ERROR in listed_codes
    assert REASON_DUPLICATE_SCENARIO_ID in listed_codes
    assert any(error["path"] == "/schema_version" for error in row["validation_status"]["errors"])

    described = describe_scenario_payload(str(manifest))
    described_codes = {error["code"] for error in described["validation_status"]["errors"]}
    assert described["validation_status"]["valid"] is False
    assert REASON_SCHEMA_VALIDATION_ERROR in described_codes
    assert REASON_DUPLICATE_SCENARIO_ID in described_codes
    assert any(
        error["path"] == "/schema_version" for error in described["validation_status"]["errors"]
    )

    direct_manifest = tmp_path / "direct.yaml"
    direct_manifest.write_text(
        "schema_version: robot_sf.scenario_matrix.v1\n"
        "scenarios:\n"
        "  - name: shadowed_identity\n"
        "    map_file: direct.svg\n"
        "    simulation_config:\n"
        "      max_episode_steps: 10\n"
        "    unknown_contract_field: true\n",
        encoding="utf-8",
    )
    (tmp_path / "direct.svg").write_text("<svg/>\n", encoding="utf-8")

    direct = describe_scenario_payload(str(direct_manifest))
    assert direct["identity"] == "shadowed_identity"
    assert direct["validation_status"]["valid"] is False
    assert any(
        error["path"] == "/unknown_contract_field"
        for error in direct["validation_status"]["errors"]
    )


def test_validate_mixed_malformed_rows_fails_closed_with_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed row is reported while valid neighboring rows remain inspectable."""
    scenarios_root = tmp_path / "configs" / "scenarios"
    scenarios_root.mkdir(parents=True)
    (tmp_path / "map.svg").write_text("<svg/>\n", encoding="utf-8")
    manifest = scenarios_root / "mixed.yaml"
    manifest.write_text(
        "scenarios:\n"
        "  - name: retained_valid_row\n"
        "    map_file: ../../map.svg\n"
        "  - this row is malformed\n"
        "  - name: retained_second_row\n"
        "    map_file: ../../map.svg\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_scenarios_module, "_find_repo_root", lambda: tmp_path)

    payload = validate_scenario_payload(str(manifest))

    assert payload["valid"] is False
    assert payload["num_scenarios"] == 3
    assert {row["identity"] for row in payload["scenarios"]} == {
        "retained_valid_row",
        "retained_second_row",
    }
    assert any(error["code"] == REASON_LOAD_FAILURE for error in payload["errors"])
    malformed_errors = [
        error
        for error in payload["errors"]
        if error["code"] == REASON_SCHEMA_VALIDATION_ERROR and error["path"] == "/scenarios/1"
    ]
    assert malformed_errors
    assert "must be a mapping" in malformed_errors[0]["message"]


def test_validate_nested_manifest_reports_metadata_schema_and_mixed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Included metadata and malformed rows retain provenance and valid neighbors."""
    scenarios_root = tmp_path / "configs" / "scenarios"
    inputs_root = tmp_path / "configs" / "scenario_inputs"
    scenarios_root.mkdir(parents=True)
    inputs_root.mkdir(parents=True)
    (tmp_path / "map.svg").write_text("<svg/>\n", encoding="utf-8")
    child = inputs_root / "mixed.yaml"
    child.write_text(
        "schema_version: robot_sf.scenario_matrix.v0\n"
        "scenarios:\n"
        "  - name: included_valid_neighbor\n"
        "    map_file: ../../map.svg\n"
        "    simulation_config:\n"
        "      max_episode_steps: 10\n"
        "  - malformed nested row\n"
        "  - name: included_schema_error\n"
        "    map_file: ../../map.svg\n"
        "    simulation_config:\n"
        "      max_episode_steps: 11\n"
        "    repeats: 0\n"
        "  - name: included_second_neighbor\n"
        "    map_file: ../../map.svg\n"
        "    simulation_config:\n"
        "      max_episode_steps: 12\n",
        encoding="utf-8",
    )
    manifest = scenarios_root / "root.yaml"
    manifest.write_text(
        "includes:\n"
        "  - ../scenario_inputs/mixed.yaml\n"
        "scenario_overrides:\n"
        "  simulation_config:\n"
        "    max_episode_steps: 13\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_scenarios_module, "_find_repo_root", lambda: tmp_path)

    payload = cli_scenarios_module.validate_scenario_payload(str(manifest))

    assert payload["valid"] is False
    assert payload["num_scenarios"] == 4
    identities = {row["identity"] for row in payload["scenarios"]}
    assert identities == {
        "included_valid_neighbor",
        "included_schema_error",
        "included_second_neighbor",
    }
    metadata_errors = [error for error in payload["errors"] if error["path"] == "/schema_version"]
    assert metadata_errors
    assert metadata_errors[0]["source_file"].endswith("configs/scenario_inputs/mixed.yaml")
    nested_row_errors = [
        error for error in payload["errors"] if error.get("path") == "/scenarios/1"
    ]
    assert any(error["code"] == REASON_SCHEMA_VALIDATION_ERROR for error in nested_row_errors)
    assert any(error["code"] == REASON_LOAD_FAILURE for error in nested_row_errors)
    assert any(
        error["code"] == REASON_SCHEMA_VALIDATION_ERROR and error["path"] == "/repeats"
        for error in payload["errors"]
    )
    repeats_errors = [error for error in payload["errors"] if error.get("path") == "/repeats"]
    assert repeats_errors[0]["source_file"].endswith("configs/scenario_inputs/mixed.yaml")
    included_schema_row = next(
        row for row in payload["scenarios"] if row["identity"] == "included_schema_error"
    )
    assert included_schema_row["map_reference"]["map_exists"] is True


def test_external_map_asset_is_explicitly_classified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing map outside the repository is not treated as valid."""
    scenarios_root = tmp_path / "configs" / "scenarios"
    scenarios_root.mkdir(parents=True)
    external_map = tmp_path.parent / f"{tmp_path.name}-external.svg"
    external_map.write_text("<svg/>\n", encoding="utf-8")
    manifest = scenarios_root / "external.yaml"
    manifest.write_text(
        f"scenarios:\n  - name: external_map_scenario\n    map_file: {external_map}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_scenarios_module, "_find_repo_root", lambda: tmp_path)

    payload = validate_scenario_payload(str(manifest))

    assert payload["valid"] is False
    assert payload["status"] == STATUS_EXTERNAL_ASSET_DEPENDENT
    assert payload["scenarios"][0]["status"] == STATUS_EXTERNAL_ASSET_DEPENDENT
    assert payload["scenarios"][0]["map_reference"]["external_asset_dependent"] is True
    assert any(error["code"] == REASON_EXTERNAL_ASSET_DEPENDENT for error in payload["errors"])


def test_external_route_asset_sets_consistent_row_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An external route is explicit in the map reference and row status."""
    scenarios_root = tmp_path / "configs" / "scenarios"
    scenarios_root.mkdir(parents=True)
    (tmp_path / "map.svg").write_text("<svg/>\n", encoding="utf-8")
    external_route = tmp_path.parent / f"{tmp_path.name}-external-route.yaml"
    manifest = scenarios_root / "external_route.yaml"
    manifest.write_text(
        "scenarios:\n"
        "  - name: external_route_scenario\n"
        "    map_file: ../../map.svg\n"
        f"    route_overrides_file: {external_route}\n"
        "    simulation_config:\n"
        "      max_episode_steps: 10\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_scenarios_module, "_find_repo_root", lambda: tmp_path)

    try:
        payload = cli_scenarios_module.validate_scenario_payload(str(manifest))
        row = payload["scenarios"][0]

        assert payload["status"] == STATUS_EXTERNAL_ASSET_DEPENDENT
        assert row["status"] == STATUS_EXTERNAL_ASSET_DEPENDENT
        assert row["validation_status"]["status"] == STATUS_EXTERNAL_ASSET_DEPENDENT
        assert row["map_reference"]["external_asset_dependent"] is True
        assert row["map_reference"]["external_map_asset_dependent"] is False
        assert row["map_reference"]["external_route_asset_dependent"] is True
        assert any(
            error["code"] == REASON_EXTERNAL_ASSET_DEPENDENT
            and error["path"] == "/route_overrides_file"
            for error in payload["errors"]
        )
    finally:
        external_route.unlink(missing_ok=True)


def test_external_manifest_include_is_rejected_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest includes outside the repository fail closed without loading them."""
    scenarios_root = tmp_path / "configs" / "scenarios"
    scenarios_root.mkdir(parents=True)
    external_manifest = tmp_path.parent / f"{tmp_path.name}-external-manifest.yaml"
    manifest = scenarios_root / "external_include.yaml"
    manifest.write_text(f"includes:\n  - {external_manifest}\n", encoding="utf-8")
    monkeypatch.setattr(cli_scenarios_module, "_find_repo_root", lambda: tmp_path)

    payload = validate_scenario_payload(str(manifest))

    assert payload["valid"] is False
    assert payload["status"] == STATUS_EXTERNAL_ASSET_DEPENDENT
    assert payload["num_scenarios"] == 0
    assert any(error["code"] == REASON_EXTERNAL_ASSET_DEPENDENT for error in payload["errors"])
    assert not any(error["code"] == REASON_LOAD_FAILURE for error in payload["errors"])


def test_nested_external_manifest_asset_is_rejected_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nested in-repository manifests cannot hide external search paths."""
    scenarios_root = tmp_path / "configs" / "scenarios"
    input_root = tmp_path / "configs" / "scenario_inputs"
    scenarios_root.mkdir(parents=True)
    input_root.mkdir(parents=True)
    external_root = tmp_path.parent / f"{tmp_path.name}-external-maps"
    included = input_root / "included.yaml"
    included.write_text(f"map_search_paths:\n  - {external_root}\n", encoding="utf-8")
    manifest = scenarios_root / "nested_external.yaml"
    manifest.write_text(
        "includes:\n  - ../scenario_inputs/included.yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_scenarios_module, "_find_repo_root", lambda: tmp_path)

    payload = validate_scenario_payload(str(manifest))

    assert payload["valid"] is False
    assert payload["status"] == STATUS_EXTERNAL_ASSET_DEPENDENT
    assert any(error["path"] == "/map_search_paths/0" for error in payload["errors"])
    assert not any(error["code"] == REASON_LOAD_FAILURE for error in payload["errors"])


def test_friendly_validation_includes_json_diagnostic_codes(capsys) -> None:
    """Friendly output exposes the same validation codes as the JSON contract."""
    path = "tests/fixtures/cli_scenarios/invalid_schema.yaml"
    assert main(["scenarios", "validate", path]) == 2
    friendly = capsys.readouterr().out
    assert "[SCHEMA_VALIDATION_ERROR]" in friendly

    assert main(["scenarios", "validate", path, "--format", "json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert any(error["code"] == REASON_SCHEMA_VALIDATION_ERROR for error in payload["errors"])


def test_friendly_list_and_validate_include_all_json_contract_facts() -> None:
    """Friendly projections expose every contract field represented in JSON."""
    listed = cli_scenarios_module.list_scenarios_payload()
    listed_text = cli_scenarios_module._format_list_scenarios(listed)
    sample = listed["scenarios"][0]
    assert listed["schema_version"] in listed_text
    assert str(listed["count"]) in listed_text
    for key in sample:
        assert key in listed_text
    for section in (
        "map_reference",
        "actor_counts",
        "seed_policy",
        "horizon_and_dt",
        "kinematics_and_observation",
        "validation_status",
    ):
        for key in sample[section]:
            assert key in listed_text

    validated = cli_scenarios_module.validate_scenario_payload(
        "tests/fixtures/cli_scenarios/valid_scenario.yaml"
    )
    validated_text = cli_scenarios_module._format_validate_scenario(validated)
    assert validated["schema_version"] in validated_text
    assert validated["resolved_path"] in validated_text
    assert "scenarios:" in validated_text
    assert "fixture_valid_crossing" in validated_text
    for key in validated["scenarios"][0]["kinematics_and_observation"]:
        assert key in validated_text


def test_unknown_map_id_keeps_stable_map_not_found_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown map_id is never hidden by a valid map_file fallback."""
    scenarios_root = tmp_path / "configs" / "scenarios"
    scenarios_root.mkdir(parents=True)
    (tmp_path / "map.svg").write_text("<svg/>\n", encoding="utf-8")
    manifest = scenarios_root / "unknown-map-id.yaml"
    manifest.write_text(
        "scenarios:\n"
        "  - name: unknown_map_id\n"
        "    map_id: not_registered\n"
        "    map_file: ../../map.svg\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_scenarios_module, "_find_repo_root", lambda: tmp_path)

    payload = validate_scenario_payload(str(manifest))

    map_errors = [error for error in payload["errors"] if error["code"] == REASON_MAP_NOT_FOUND]
    assert map_errors
    assert map_errors[0]["path"] == "/map_id"
    assert "not_registered" in map_errors[0]["message"]


def test_scenario_docs_match_canonical_schema_and_status_contract() -> None:
    """The scenario guide names the canonical schema and only emitted statuses."""
    docs = Path("docs/SCENARIOS.md").read_text(encoding="utf-8")

    assert "robot_sf/benchmark/schemas/scenarios.schema.json" in docs
    assert "`schemas/scenarios.schema.json`" not in docs
    for status in (
        "valid",
        "missing",
        "invalid",
        "duplicate",
        "unsupported",
        "external_asset_dependent",
    ):
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


def test_top_level_cli_lazy_dispatch_wrappers_are_exercised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep top-level lazy CLI dispatch covered in the fast validation lane."""
    from robot_sf import cli_datasets, cli_envs, cli_models, cli_planners, examples_cli
    from robot_sf.benchmark import doctor
    from robot_sf.recipes import cli as recipes_cli

    monkeypatch.setattr(cli_models, "list_models", lambda registry_path=None: [])
    monkeypatch.setattr(cli_datasets, "list_datasets", lambda: [])
    monkeypatch.setattr(cli_envs, "_handle_envs_list", lambda _args: 0)
    monkeypatch.setattr(cli_planners, "_handle_planners_list", lambda _args: 0)
    monkeypatch.setattr(examples_cli, "examples_cli_main", lambda _args: 0)
    monkeypatch.setattr(recipes_cli, "handle", lambda _args: 0)
    monkeypatch.setattr(doctor, "collect_doctor_report", lambda **_kwargs: {})
    monkeypatch.setattr(doctor, "doctor_exit_code", lambda _report: 0)

    assert main(["doctor", "--format", "json", "--skip-env-smoke", "--skip-quickstart-smoke"]) == 0
    assert main(["models", "list", "--format", "json"]) == 0
    assert main(["datasets", "list", "--format", "json"]) == 0
    assert main(["examples"]) == 0
    assert main(["envs", "list"]) == 0
    assert main(["planners", "list", "--format", "json"]) == 0
    assert main(["recipe", "list"]) == 0


def test_common_lazy_helpers_delegate_without_eager_optional_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the lightweight common facade's deferred helper imports."""
    from robot_sf import common
    from robot_sf.common import geometry, matplotlib_utils

    monkeypatch.setattr(geometry, "euclid_dist", lambda _first, _second: 42.0)
    monkeypatch.setattr(matplotlib_utils, "ensure_interactive_backend", lambda **_kwargs: True)
    monkeypatch.setattr(matplotlib_utils, "is_headless_environment", lambda: True)

    assert common.euclid_dist((0.0, 0.0), (3.0, 4.0)) == 42.0
    assert common.ensure_interactive_backend(verbose=True) is True
    assert common.is_headless_environment() is True


def test_scenario_loader_discovery_and_validation_preserve_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover auxiliary discovery, task bundles, and validation base-dir failures."""
    from robot_sf.training import scenario_loader, task_bundles

    auxiliary = tmp_path / "auxiliary.yaml"
    auxiliary.write_text("title: metadata-only\n", encoding="utf-8")
    assert scenario_loader.load_scenarios_for_discovery(auxiliary) is None

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate is empty"):
        scenario_loader.load_scenarios_for_discovery(empty)

    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("metadata-only\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a manifest"):
        scenario_loader.load_scenarios_for_discovery(scalar)

    fixture = Path("tests/fixtures/cli_scenarios/valid_scenario.yaml")
    discovered = scenario_loader.load_scenarios_for_discovery(fixture)
    assert discovered is not None
    assert discovered[0]["name"] == "fixture_valid_crossing"

    monkeypatch.setattr(task_bundles, "is_task_bundle_reference", lambda _reference: True)
    monkeypatch.setattr(
        task_bundles,
        "load_task_bundle_scenarios",
        lambda _reference: [{"name": "bundle_scenario"}],
    )
    bundled = scenario_loader.load_scenarios_for_validation("bundle:demo")
    assert bundled.load_error is None
    assert bundled.raw_entry_count == 1
    assert bundled.scenarios == [{"name": "bundle_scenario"}]

    monkeypatch.setattr(task_bundles, "is_task_bundle_reference", lambda _reference: False)
    missing_root = tmp_path / "missing-root"
    report = scenario_loader.load_scenarios_for_validation(
        tmp_path / "missing.yaml", base_dir=missing_root
    )
    assert report.scenarios == []
    assert report.load_error is not None
    assert "base_dir does not exist" in report.load_error


def test_scenario_loader_tolerant_paths_record_row_and_include_issues(tmp_path: Path) -> None:
    """Exercise tolerant include, selection, override, and normalization branches."""
    from robot_sf.training import scenario_loader

    root = tmp_path / "root.yaml"
    root.write_text("includes:\n  - missing.yaml\n", encoding="utf-8")
    report = scenario_loader.load_scenarios_for_validation(root)
    assert report.load_error is None
    assert report.scenarios == []
    assert report.load_issues
    assert report.load_issues[0].source == (tmp_path / "missing.yaml").resolve()

    source = tmp_path / "direct.yaml"
    collector = scenario_loader._ScenarioValidationCollector()
    selected = scenario_loader._apply_scenario_selection(
        [{}, {"name": "selected"}],
        data={"select_scenarios": ["selected"]},
        source=source,
        collector=collector,
    )
    assert selected == [{"name": "selected"}]
    assert len(collector.entry_issues) == 1

    collector = scenario_loader._ScenarioValidationCollector()
    overridden = scenario_loader._apply_scenario_overrides_by_name(
        [{}, {"name": "selected"}],
        data={"scenario_overrides_by_name": {"selected": {"metadata": {"tag": "ok"}}}},
        source=source,
        root=source,
        map_search_paths=[],
        collector=collector,
    )
    assert overridden[1]["metadata"] == {"tag": "ok"}
    assert len(collector.entry_issues) == 1

    with pytest.raises(ValueError, match="must be a mapping"):
        scenario_loader._normalize_scenarios(
            ["not-a-mapping"],
            source=source,
            root=source,
            map_search_paths=[],
        )

    collector = scenario_loader._ScenarioValidationCollector()
    assert (
        scenario_loader._normalize_scenarios(
            ["not-a-mapping"],
            source=source,
            root=source,
            map_search_paths=[],
            collector=collector,
        )
        == []
    )
    assert len(collector.entry_issues) == 1


def test_scenario_loader_deferred_override_imports_are_exercised() -> None:
    """Cover deferred map/config imports used by scenario override helpers."""
    from robot_sf.gym_env.unified_config import RobotSimulationConfig
    from robot_sf.nav.map_config import SinglePedestrianDefinition
    from robot_sf.training import scenario_loader

    config = RobotSimulationConfig()
    scenario_loader._set_simulation_override_attr(
        config,
        "ttc_predictive_force",
        {
            "ttc_predictive_force": {"force_scale": 2.0},
            "pedestrian_model": "hsfm_ttc_predictive_v1",
        },
    )
    assert config.sim_config.ttc_predictive_force.enabled is True
    assert config.sim_config.ttc_predictive_force.force_scale == 2.0

    ped = SinglePedestrianDefinition(id="ped", start=(0.0, 0.0), goal=(1.0, 1.0))
    updated = scenario_loader._apply_single_pedestrian_override(
        ped,
        {},
        SimpleNamespace(),
    )
    assert updated.id == "ped"
    assert updated.goal == (1.0, 1.0)

    rules = scenario_loader._parse_wait_overrides(
        [{"waypoint_index": 0, "wait_s": 0.5}],
        trajectory=[(0.0, 0.0)],
        trajectory_labels=None,
    )
    assert rules is not None
    assert rules[0].wait_s == pytest.approx(0.5)

    scenario_loader._apply_prf_config_override(config, {"force_multiplier": 2.0})
    assert config.sim_config.prf_config.force_multiplier == pytest.approx(2.0)
