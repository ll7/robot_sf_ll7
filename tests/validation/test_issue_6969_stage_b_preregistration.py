"""Contract tests for the issue #6969 Stage B preregistration."""

# evidence-writer-exempt: tests write only temporary mutated summary fixtures under pytest tmp_path.

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from scripts.validation.check_issue_6969_stage_b_preregistration import (
    StageBPreregistrationError,
    load_preregistration_config,
    main,
    validate_preregistration_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / ("configs/benchmarks/issue_6969_lane_formation_stage_b_preregistration.yaml")
SUMMARY = REPO_ROOT / "docs/context/evidence/issue_6969_lane_formation_reference/summary.json"


def _packet() -> dict[str, object]:
    return load_preregistration_config(PACKET)


def _mutated_summary_packet(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> tuple[dict[str, object], Path]:
    packet = copy.deepcopy(_packet())
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    mutate(summary)
    source_root = tmp_path / "repo"
    summary_relative = "docs/context/evidence/issue_6969_lane_formation_reference/summary.json"
    summary_path = source_root / summary_relative
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_contracts = packet["source_contracts"]  # type: ignore[index]
    source_sha256 = packet["source_sha256"]  # type: ignore[index]
    source_contracts["stage_a_summary"] = summary_relative
    packet["source_sha256"]["stage_a_summary"] = hashlib.sha256(  # type: ignore[index]
        summary_path.read_bytes()
    ).hexdigest()
    for key in (
        "stage_a_parameter_screen",
        "stage_a_reference_contract",
        "stage_a_runner",
        "stage_a_tests",
    ):
        relative = source_contracts[key].split("::", maxsplit=1)[0]  # type: ignore[index]
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
        source_sha256[key] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return packet, source_root


def _stage_a(summary: dict[str, Any]) -> dict[str, Any]:
    stage_a = summary["stage_a"]
    assert isinstance(stage_a, dict)
    return stage_a


def _stage_a_design(summary: dict[str, Any]) -> dict[str, Any]:
    design = _stage_a(summary)["design"]
    assert isinstance(design, dict)
    return design


def _lhs_05_summary(summary: dict[str, Any]) -> dict[str, Any]:
    profiles = _stage_a(summary)["profile_summaries"]
    assert isinstance(profiles, list)
    matches = [
        profile
        for profile in profiles
        if isinstance(profile, dict) and profile.get("profile_id") == "lhs_05"
    ]
    assert len(matches) == 1
    return matches[0]


def test_packet_is_proposal_only_and_has_no_current_candidate() -> None:
    """The packet freezes the future design while refusing current execution."""
    report = validate_preregistration_config(_packet(), config_path=PACKET)

    assert report["status"] == "ok"
    assert report["stage_b_execution_allowed"] is False
    assert report["compute_submit_authorized"] is False
    assert report["stage_a_native_rows"] == 30
    assert report["held_out_seed_count"] == 10
    assert report["candidate_count"] == 0
    assert report["fidelity_surface_count"] == 6


def test_cli_json_reports_blocked_execution(capsys: pytest.CaptureFixture[str]) -> None:
    """The reusable validator emits machine-readable proposal status."""
    assert main(["--config", str(PACKET), "--json"]) == 0

    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["status"] == "ok"
    assert report["stage_b_execution_allowed"] is False


def test_execution_boundary_cannot_authorize_compute() -> None:
    """A packet mutation cannot turn preregistration into a launch authorization."""
    packet = copy.deepcopy(_packet())
    packet["execution_boundary"]["compute_submit_authorized"] = True  # type: ignore[index]

    with pytest.raises(StageBPreregistrationError, match="compute_submit_authorized"):
        validate_preregistration_config(packet, config_path=PACKET)


def test_source_digest_drift_fails_closed() -> None:
    """The Stage A source contract must remain byte-identical to the reviewed snapshot."""
    packet = copy.deepcopy(_packet())
    packet["source_sha256"]["stage_a_summary"] = "0" * 64  # type: ignore[index]

    with pytest.raises(StageBPreregistrationError, match="source_sha256.stage_a_summary"):
        validate_preregistration_config(packet, config_path=PACKET)


@pytest.mark.parametrize("source_path", [str(SUMMARY), "../summary.json"])
def test_non_relative_source_path_fails_closed(source_path: str) -> None:
    """The source contract cannot escape the repository-relative path boundary."""
    packet = copy.deepcopy(_packet())
    packet["source_contracts"]["stage_a_summary"] = source_path  # type: ignore[index]

    with pytest.raises(StageBPreregistrationError, match="repository-relative"):
        validate_preregistration_config(packet, config_path=PACKET)


def test_stage_a_test_digest_is_required() -> None:
    """The Stage A test contract must be covered by the same byte pinning as the code."""
    packet = copy.deepcopy(_packet())
    del packet["source_sha256"]["stage_a_tests"]  # type: ignore[index]

    with pytest.raises(StageBPreregistrationError, match="source_sha256.stage_a_tests"):
        validate_preregistration_config(packet, config_path=PACKET)


def test_symlinked_source_path_fails_closed(tmp_path: Path) -> None:
    """A source path cannot use a symlink hop to bypass the declared source root."""
    packet, source_root = _mutated_summary_packet(tmp_path, lambda _summary: None)
    summary_relative = "docs/context/evidence/issue_6969_lane_formation_reference/summary.json"
    summary_path = source_root / summary_relative
    summary_path.unlink()
    summary_path.symlink_to(SUMMARY)

    with pytest.raises(StageBPreregistrationError, match="must not traverse a symlink"):
        validate_preregistration_config(packet, config_path=PACKET, source_root=source_root)


def test_implementation_commit_must_exist_and_contain_source() -> None:
    """Implementation provenance must resolve to a commit containing the claimed source."""
    packet = copy.deepcopy(_packet())
    packet["stage_a_snapshot"]["implementation_commits"]["reference"] = "0" * 40  # type: ignore[index]

    with pytest.raises(
        StageBPreregistrationError, match="implementation_commits.reference is unavailable"
    ):
        validate_preregistration_config(packet, config_path=PACKET)


def test_stage_a_near_candidate_cannot_be_promoted() -> None:
    """The observed one-of-three hit remains ineligible for held-out execution."""
    packet = copy.deepcopy(_packet())
    packet["stage_a_snapshot"]["observed_decision"]["near_candidate"]["eligible_for_stage_b"] = True  # type: ignore[index]

    with pytest.raises(StageBPreregistrationError, match="one-of-three Stage A hit"):
        validate_preregistration_config(packet, config_path=PACKET)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda summary: _stage_a(summary).__setitem__("native_rows", 29),
            "native row count disagrees",
        ),
        (
            lambda summary: _stage_a(summary).__setitem__(
                "native_execution", "29/30 native:computed"
            ),
            "native execution status disagrees",
        ),
        (
            lambda summary: _stage_a_design(summary).__setitem__("space_filling_profiles", 7),
            "profile count disagrees",
        ),
        (
            lambda summary: _stage_a_design(summary).__setitem__("seeds", [5149, 5150, 5152]),
            "seed schedule disagrees",
        ),
        (
            lambda summary: _stage_a_design(summary).__setitem__("clear_threshold_lsi", 0.45),
            "clear threshold disagrees",
        ),
        (
            lambda summary: _stage_a(summary)["decision"].__setitem__(  # type: ignore[index, union-attr]
                "robust_clear_profile_found", True
            ),
            "robust candidate decision disagrees",
        ),
        (
            lambda summary: _lhs_05_summary(summary).__setitem__("clear_lsi_hits", 2),
            "lhs_05 near-hit disagrees",
        ),
    ],
)
def test_stage_a_summary_semantic_drift_fails_closed(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], match: str
) -> None:
    """The byte-pinned Stage A summary must agree with the preregistration snapshot."""
    packet, source_root = _mutated_summary_packet(tmp_path, mutate)

    with pytest.raises(StageBPreregistrationError, match=match):
        validate_preregistration_config(packet, config_path=PACKET, source_root=source_root)


def test_held_out_seed_overlap_fails_closed() -> None:
    """Held-out rows cannot reuse a Stage A seed."""
    packet = copy.deepcopy(_packet())
    packet["held_out_plan"]["seeds"][0] = 5151  # type: ignore[index]

    with pytest.raises(StageBPreregistrationError, match="held-out seed schedule drifted"):
        validate_preregistration_config(packet, config_path=PACKET)


def test_fidelity_surface_omission_fails_closed() -> None:
    """A candidate tradeoff cannot be reported while silently dropping a declared surface."""
    packet = copy.deepcopy(_packet())
    packet["fidelity_cost_surfaces"]["outcomes"].pop()  # type: ignore[index]

    with pytest.raises(StageBPreregistrationError, match="fidelity surface set drifted"):
        validate_preregistration_config(packet, config_path=PACKET)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        (
            "candidate_selection",
            "required_threshold_metric",
            "wrong_metric",
            "candidate threshold metric",
        ),
        (
            "held_out_plan",
            "protocol_source",
            "wrong.Protocol",
            "protocol source",
        ),
        (
            "held_out_plan",
            "missingness_policy",
            "drop missing rows",
            "missingness policy",
        ),
        (
            "fidelity_cost_surfaces",
            "report_effect_and_uncertainty",
            False,
            "effect and uncertainty",
        ),
    ],
)
def test_frozen_research_contract_fields_fail_closed(
    section: str, key: str, value: object, match: str
) -> None:
    """Frozen research design fields cannot drift without invalidating the packet."""
    packet = copy.deepcopy(_packet())
    packet[section][key] = value  # type: ignore[index]

    with pytest.raises(StageBPreregistrationError, match=match):
        validate_preregistration_config(packet, config_path=PACKET)


def test_yaml_packet_is_mapping() -> None:
    """The tracked packet remains directly parseable by the repository YAML toolchain."""
    payload = yaml.safe_load(PACKET.read_text(encoding="utf-8"))

    assert isinstance(payload, dict)
