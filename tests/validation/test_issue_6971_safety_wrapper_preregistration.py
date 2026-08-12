"""Contract tests for the issue #6971 safety-wrapper preregistration."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.issue_6971_safety_wrapper_preregistration import (
    EXPECTED_EPISODES,
    SafetyWrapperPreregistrationError,
    build_validation_report,
    load_preregistration_config,
    validate_preregistration_config,
)
from scripts.validation.check_issue_6971_safety_wrapper_preregistration import main

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / ("configs/benchmarks/issue_6971_safety_wrapper_paired_preregistration.yaml")


def _packet() -> dict[str, object]:
    return load_preregistration_config(PACKET)


def test_preregistration_validates_as_proposal_only() -> None:
    """The frozen packet identifies the full design but authorizes no execution."""
    report = build_validation_report(PACKET)

    assert report["status"] == "ok"
    assert report["planned_episode_count"] == EXPECTED_EPISODES
    assert report["scenario_count"] == 48
    assert report["seed_count"] == 20
    assert report["retained_metric_count"] == 8
    assert report["execution_authorized"] is False
    assert report["compute_submit_authorized"] is False
    assert report["cost"]["reserved_wall_clock_hours"] == 6.0
    assert report["cost"]["reserved_worker_hours"] == 12.0


def test_cli_json_reports_success(capsys: pytest.CaptureFixture[str]) -> None:
    """The reusable validation entry point emits machine-readable proposal status."""
    assert main(["--config", str(PACKET), "--json"]) == 0

    output = capsys.readouterr().out.strip().splitlines()[-1]
    report = json.loads(output)
    assert report["status"] == "ok"
    assert report["benchmark_evidence"] is False


def test_seed_schedule_is_frozen() -> None:
    """Changing S20 would invalidate the declared precision and episode budget."""
    packet = copy.deepcopy(_packet())
    packet["design"]["seed_schedule"]["values"][-1] = 131  # type: ignore[index]

    with pytest.raises(SafetyWrapperPreregistrationError, match="seed schedule values"):
        validate_preregistration_config(packet)


def test_retained_field_path_is_exact() -> None:
    """A legacy alias must not silently replace the #6970 retained metric path."""
    packet = copy.deepcopy(_packet())
    packet["retained_field_manifest"]["fields"][0]["path"] = (  # type: ignore[index]
        "metric_values.legacy_collision_rate"
    )

    with pytest.raises(SafetyWrapperPreregistrationError, match="retained field path drifted"):
        validate_preregistration_config(packet)


def test_execution_boundary_is_fail_closed() -> None:
    """A packet mutation cannot authorize compute or campaign execution."""
    packet = copy.deepcopy(_packet())
    packet["execution_boundary"]["compute_submit_authorized"] = True  # type: ignore[index]

    with pytest.raises(SafetyWrapperPreregistrationError, match="compute_submit_authorized"):
        validate_preregistration_config(packet)


def test_source_digest_is_pinned() -> None:
    """Changing a source digest must block lineage drift before analysis is accepted."""
    packet = copy.deepcopy(_packet())
    packet["source_sha256"]["metric_contract"] = "0" * 64  # type: ignore[index]

    with pytest.raises(SafetyWrapperPreregistrationError, match="source_sha256.metric_contract"):
        validate_preregistration_config(packet)


def test_both_wrapper_arms_are_required() -> None:
    """The paired estimand is unidentified if either wrapper arm is removed."""
    packet = copy.deepcopy(_packet())
    packet["design"]["wrapper_arms"].pop()  # type: ignore[index]

    with pytest.raises(SafetyWrapperPreregistrationError, match="exactly two wrapper arms"):
        validate_preregistration_config(packet)


def test_yaml_packet_is_mapping() -> None:
    """The tracked packet remains directly parseable by the repository YAML toolchain."""
    payload = yaml.safe_load(PACKET.read_text(encoding="utf-8"))

    assert isinstance(payload, dict)
