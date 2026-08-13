"""Contract tests for the issue #6969 Stage B preregistration."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.validation.check_issue_6969_stage_b_preregistration import (
    StageBPreregistrationError,
    load_preregistration_config,
    main,
    validate_preregistration_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / ("configs/benchmarks/issue_6969_lane_formation_stage_b_preregistration.yaml")


def _packet() -> dict[str, object]:
    return load_preregistration_config(PACKET)


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


def test_stage_a_near_candidate_cannot_be_promoted() -> None:
    """The observed one-of-three hit remains ineligible for held-out execution."""
    packet = copy.deepcopy(_packet())
    packet["stage_a_snapshot"]["observed_decision"]["near_candidate"]["eligible_for_stage_b"] = True  # type: ignore[index]

    with pytest.raises(StageBPreregistrationError, match="one-of-three Stage A hit"):
        validate_preregistration_config(packet, config_path=PACKET)


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


def test_yaml_packet_is_mapping() -> None:
    """The tracked packet remains directly parseable by the repository YAML toolchain."""
    payload = yaml.safe_load(PACKET.read_text(encoding="utf-8"))

    assert isinstance(payload, dict)
