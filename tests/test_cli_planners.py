"""Tests for the ``robot-sf planners`` list and describe CLI (issue #8747)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from robot_sf import cli_planners
from robot_sf.baselines import BASELINES
from robot_sf.cli import _HANDLERS, main
from robot_sf.cli_planners import (
    SCHEMA_DESCRIBE_VERSION,
    SCHEMA_LIST_VERSION,
    describe_planner_payload,
    list_planners_payload,
)


def test_planners_catalog_contains_all_baselines() -> None:
    """All keys in BASELINES must be resolvable through the planner catalog."""
    payload = list_planners_payload()
    canonical_names = {p["canonical_name"] for p in payload["planners"]}
    alias_map: dict[str, str] = {}
    for p in payload["planners"]:
        for a in p["aliases"]:
            alias_map[a.lower()] = p["canonical_name"]

    for baseline_key in BASELINES:
        norm = baseline_key.strip().lower()
        assert norm in alias_map, f"Baseline '{baseline_key}' missing from planner catalog"
        resolved_canonical = alias_map[norm]
        assert resolved_canonical in canonical_names


def test_planners_catalog_exclusions_present() -> None:
    """The exclusions list must include native_command with rationale."""
    payload = list_planners_payload()
    exclusions = {exc["key"]: exc for exc in payload["exclusions"]}
    assert "native_command" in exclusions
    assert exclusions["native_command"]["reason"] == "execution_arm"


def test_planners_list_payload_structure() -> None:
    """list_planners_payload must conform to schema contract."""
    payload = list_planners_payload()
    assert payload["schema_version"] == SCHEMA_LIST_VERSION
    assert payload["count"] == len(payload["planners"])
    assert payload["count"] >= 50

    # Planners should be deterministically sorted by canonical name
    canonical_order = [p["canonical_name"] for p in payload["planners"]]
    assert canonical_order == sorted(canonical_order)

    # Each entry must have all required contract fields
    required_fields = {
        "canonical_name",
        "aliases",
        "family",
        "tier",
        "status",
        "paper_baseline_eligible",
        "requires_explicit_opt_in",
        "execution_mode",
        "command_space",
        "compatible_robot_kinematics",
        "observation_mode_default",
        "observation_modes_supported",
        "observation_inputs",
        "required_extras",
        "required_artifacts",
        "metadata_source",
        "summary",
        "readiness_status",
        "availability_status",
        "counts_as_success_evidence",
        "metadata_completeness",
    }
    for entry in payload["planners"]:
        missing = required_fields - set(entry.keys())
        assert not missing, f"Planner {entry.get('canonical_name')} missing fields: {missing}"
        assert entry["compatible_robot_kinematics"]


def test_random_discovery_matches_declared_readiness_matrix() -> None:
    """Random remains a diagnostic, degraded, non-success discovery entry."""
    matrix_path = Path(__file__).parents[1] / "configs/benchmarks/planner_readiness_matrix_v1.yaml"
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    random_row = next(row for row in matrix["rows"] if row["planner_id"] == "random")

    payload = describe_planner_payload("random")
    assert payload["tier"] == random_row["tier"] == "diagnostic"
    assert payload["requires_explicit_opt_in"] is random_row["requires_explicit_opt_in"] is True
    assert payload["readiness_status"] == random_row["readiness_status"] == "degraded"
    assert payload["availability_status"] == random_row["availability_status"] == "not_available"
    assert (
        payload["counts_as_success_evidence"] is random_row["counts_as_success_evidence"] is False
    )
    assert payload["status"] == "diagnostic_opt_in"


def test_discovery_preserves_unknown_kinematics_and_marks_partial_metadata() -> None:
    """Absent compatibility metadata is explicit, never an empty capability list."""
    payload = describe_planner_payload("random")
    assert payload["compatible_robot_kinematics"] == ["unknown"]
    assert payload["metadata_completeness"] == "partial"


def test_discovery_preserves_explicit_not_declared_kinematics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit source marker is retained verbatim by discovery."""
    profile = dict(cli_planners._KINEMATICS_PROFILE_BY_CANONICAL["goal"])
    profile["compatible_robot_kinematics"] = ["not_declared"]
    monkeypatch.setitem(cli_planners._KINEMATICS_PROFILE_BY_CANONICAL, "goal", profile)

    entry = cli_planners._build_planner_entry(
        cli_planners._ALGORITHMS[0], frozenset(), ["goal"]
    ).to_dict()
    assert entry["compatible_robot_kinematics"] == ["not_declared"]
    assert entry["metadata_completeness"] == "partial"


def test_availability_is_not_a_local_dependency_probe() -> None:
    """Undeclared local availability stays unknown while matrix exclusions stay explicit."""
    assert describe_planner_payload("orca")["availability_status"] == "unknown"
    assert describe_planner_payload("random")["availability_status"] == "not_available"


def test_describe_planner_canonical() -> None:
    """describe_planner_payload with canonical key returns complete entry."""
    payload = describe_planner_payload("orca")
    assert payload["schema_version"] == SCHEMA_DESCRIBE_VERSION
    assert payload["canonical_name"] == "orca"
    assert payload["requested_key"] == "orca"
    assert payload["is_alias"] is False
    assert payload["alias_used"] is None
    assert payload["family"] == "classical"
    assert payload["execution_mode"] == "adapter"
    assert "rvo2" in payload["required_extras"]
    assert payload["paper_baseline_eligible"] is True


def test_describe_planner_aliases() -> None:
    """describe_planner_payload resolves aliases cleanly."""
    sf_payload = describe_planner_payload("sf")
    assert sf_payload["canonical_name"] == "social_force"
    assert sf_payload["requested_key"] == "sf"
    assert sf_payload["is_alias"] is True
    assert sf_payload["alias_used"] == "sf"

    baseline_sf_payload = describe_planner_payload("baseline_sf")
    assert baseline_sf_payload["canonical_name"] == "social_force"

    goal_payload = describe_planner_payload("simple")
    assert goal_payload["canonical_name"] == "goal"
    assert goal_payload["is_alias"] is True


def test_describe_planner_case_insensitivity() -> None:
    """describe_planner_payload handles case insensitivity and whitespace."""
    payload = describe_planner_payload("  ORCA  ")
    assert payload["canonical_name"] == "orca"


def test_describe_unknown_planner_raises_keyerror() -> None:
    """Unknown planner key raises KeyError with available candidates."""
    with pytest.raises(KeyError, match="Unknown planner"):
        describe_planner_payload("not_a_real_planner")


def test_cli_planners_list_friendly(capsys: pytest.CaptureFixture[str]) -> None:
    """Friendly list output formats entries and exits 0."""
    rc = cli_planners._handle_planners_list(argparse.Namespace(format="friendly"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Registered Planners (" in out
    assert "- orca" in out
    assert "- social_force" in out
    assert "Exclusions (" in out
    assert "native_command" in out
    assert "readiness status: degraded" in out
    assert "availability: not_available" in out
    assert "metadata completeness: partial" in out


def test_cli_planners_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    """JSON list output parses into valid payload matching schema."""
    rc = cli_planners._handle_planners_list(argparse.Namespace(format="json"))
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["schema_version"] == SCHEMA_LIST_VERSION
    assert isinstance(data["planners"], list)
    assert any(p["canonical_name"] == "orca" for p in data["planners"])


def test_cli_planners_describe_friendly(capsys: pytest.CaptureFixture[str]) -> None:
    """Friendly describe output formats detailed attributes and exits 0."""
    rc = cli_planners._handle_planners_describe(argparse.Namespace(key="orca", format="friendly"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Planner: orca" in out
    assert "Execution Mode: adapter" in out
    assert "Required Extras: rvo2" in out
    assert "Availability Status: unknown" in out


def test_cli_planners_describe_alias_friendly(capsys: pytest.CaptureFixture[str]) -> None:
    """Friendly describe output indicates alias resolution."""
    rc = cli_planners._handle_planners_describe(argparse.Namespace(key="sf", format="friendly"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Requested Alias: sf -> social_force" in out
    assert "Planner: social_force" in out


def test_cli_planners_describe_unknown_friendly_exits_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown planner friendly format exits 2 and writes error to stderr."""
    rc = cli_planners._handle_planners_describe(
        argparse.Namespace(key="unknown_planner", format="friendly")
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "Error:" in err
    assert "unknown_planner" in err


def test_cli_planners_describe_unknown_json_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    """Unknown planner JSON format exits 2 and writes error payload to stdout."""
    rc = cli_planners._handle_planners_describe(
        argparse.Namespace(key="unknown_planner", format="json")
    )
    out = capsys.readouterr().out
    assert rc == 2
    data = json.loads(out)
    assert data["status"] == "error"
    assert data["requested_key"] == "unknown_planner"
    assert "Unknown planner" in data["error"]


def test_top_level_cli_planners_dispatch() -> None:
    """Top-level main dispatches planners list and describe."""
    assert "planners" in _HANDLERS
    assert main(["planners", "list"]) == 0
    assert main(["planners", "list", "--format", "json"]) == 0
    assert main(["planners", "describe", "orca"]) == 0
    assert main(["planners", "describe", "sf", "--format", "json"]) == 0
    assert main(["planners", "describe", "unknown_planner"]) == 2


def test_import_light_cli_gate() -> None:
    """Importing robot_sf.cli must not import torch, stable_baselines3, or carla."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import robot_sf.cli; "
            "forbidden = {'torch', 'stable_baselines3', 'carla'}; "
            "loaded = forbidden & set(sys.modules); "
            "assert not loaded, f'Forbidden modules loaded: {loaded}'"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Import check failed: {res.stderr}"


def test_describe_planner_artifacts_and_extras() -> None:
    """describe_planner_payload captures required artifacts and extras."""
    sonic = describe_planner_payload("sonic_crowdnav")
    assert "checkpoint" in sonic["required_artifacts"]
    assert "training" in sonic["required_extras"]

    fast_pysf = describe_planner_payload("fast_pysf_planner")
    assert "fast-pysf" in fast_pysf["required_extras"]

    rvo = describe_planner_payload("rvo")
    assert rvo["status"] == "placeholder"

    risk_dwa = describe_planner_payload("risk_dwa")
    assert risk_dwa["requires_explicit_opt_in"] is True
    assert risk_dwa["status"] == "experimental_opt_in"


def test_top_level_cli_release_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The release subcommand dispatches through _handle_release."""
    from robot_sf import release_cli

    monkeypatch.setattr(release_cli, "handle", lambda args: 42)
    assert main(["release", "audit-published", "--tag", "v1", "--doi", "10.1234/test"]) == 42
