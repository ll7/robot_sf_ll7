"""Tests for issue #4232 uncertainty-envelope claim pre-registration packet."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_4232_uncertainty_envelope_claim_packet.yaml"
SCRIPT = REPO_ROOT / "scripts/validation/check_issue_4232_uncertainty_envelope_claim_packet.py"

_SPEC = importlib.util.spec_from_file_location("_issue_4232_claim_packet_check", SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _load_packet() -> dict:
    return yaml.safe_load(PACKET.read_text(encoding="utf-8"))


def _assert_rejected(packet: dict, message: str) -> None:
    with pytest.raises(_MODULE.PacketError, match=message):
        _MODULE.validate_packet(packet)


def test_issue_4232_packet_passes_pre_registration_contract() -> None:
    """The checked-in packet defines claim scope without creating evidence claims."""
    summary = _MODULE.validate_packet(_load_packet())

    assert summary == {
        "ok": True,
        "issue": 4232,
        "evidence_status": "pre_registered_no_evidence",
        "planner_count": 3,
        "alpha_arm_count": 5,
        "compute_submit_authorized": False,
        "run_benchmark": False,
    }


def test_issue_4232_packet_rejects_missing_alpha_zero_regression_arm() -> None:
    """The alpha=0 enabled arm is required to prove regression equivalence."""
    packet = _load_packet()
    packet["alpha_arms"] = [
        arm for arm in packet["alpha_arms"] if arm["key"] != "envelope_on_alpha_0"
    ]

    _assert_rejected(packet, "missing envelope_on_alpha_0 regression arm")


def test_issue_4232_packet_rejects_negative_alpha() -> None:
    """Alpha arms must be non-negative physical inflation speeds."""
    packet = _load_packet()
    packet["alpha_arms"][2]["pedestrian_uncertainty_alpha_mps"] = -0.1

    _assert_rejected(packet, "alpha must be non-negative")


def test_issue_4232_packet_rejects_mismatched_seed_surface() -> None:
    """Paired alpha arms must share the same scenario surface and seed policy."""
    packet = _load_packet()
    packet["alpha_arms"][2]["seed_set_ref"] = "different_seed_policy"

    _assert_rejected(packet, "seed policy mismatch")


def test_issue_4232_packet_rejects_non_mpc_envelope_effect_claim() -> None:
    """Non-MPC planners cannot enter envelope-effect claims in this packet."""
    packet = _load_packet()
    packet["planner_families"].append(
        {
            "planner_id": "orca",
            "family": "reactive",
            "base_config_path": "configs/algos/reference_adapter.yaml",
            "claim_modes": ["envelope_effect"],
        }
    )

    _assert_rejected(packet, "cannot enter envelope-effect claims unless MPC-family")


def test_issue_4232_packet_allows_non_mpc_diagnostic_only_planner() -> None:
    """Non-MPC planners are allowed only when marked outside envelope-effect claims."""
    packet = _load_packet()
    packet["planner_families"].append(
        {
            "planner_id": "orca",
            "family": "reactive",
            "base_config_path": "configs/algos/reference_adapter.yaml",
            "claim_modes": ["diagnostic_only"],
        }
    )

    summary = _MODULE.validate_packet(packet)
    assert summary["planner_count"] == 4


def test_issue_4232_packet_rejects_missing_forbidden_claim_language() -> None:
    """The packet must keep conformal, deployment, and paper claims forbidden."""
    packet = _load_packet()
    packet["claim_modes"]["forbidden_without_followup"].remove("conformal_coverage_guarantee")

    _assert_rejected(packet, "missing required claim exclusions")


def test_issue_4232_packet_rejects_fallback_as_success_evidence() -> None:
    """Fallback/degraded execution cannot count as benchmark-strength evidence."""
    packet = _load_packet()
    packet["row_status_policy"]["benchmark_strength_success_values"].append("fallback")

    _assert_rejected(packet, "only successful_evidence can be success")


def test_issue_4232_packet_rejects_raw_artifacts_as_review_outputs() -> None:
    """Pre-registration must not put raw run products in durable docs evidence."""
    packet = _load_packet()
    packet["outputs"]["required_review_artifacts"].append("episodes.jsonl")

    _assert_rejected(packet, "raw episode JSONL must not be review artifact")


def test_issue_4232_packet_rejects_bad_planner_config_paths(tmp_path: Path) -> None:
    """Planner config paths must be repo-relative source files."""

    packet = _load_packet()
    packet["planner_families"][0]["base_config_path"] = str((tmp_path / "abs.yaml").resolve())
    with pytest.raises(_MODULE.PacketError, match="repository-relative"):
        _MODULE.validate_packet(packet, repo_root=REPO_ROOT)

    packet = _load_packet()
    packet["planner_families"][0]["base_config_path"] = "configs/algos"
    with pytest.raises(_MODULE.PacketError, match="must be a file"):
        _MODULE.validate_packet(packet, repo_root=REPO_ROOT)

    root = tmp_path / "repo"
    (root / "configs/algos").mkdir(parents=True)
    (root / "configs/scenarios").mkdir(parents=True)
    (root / "configs/scenarios/classic_interactions_francis2023.yaml").write_text(
        "scenario: unit\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside.yaml"
    outside.write_text("config: outside\n", encoding="utf-8")
    (root / "configs/algos/link.yaml").symlink_to(outside)
    packet = _load_packet()
    for planner in packet["planner_families"]:
        planner["base_config_path"] = "configs/algos/link.yaml"

    with pytest.raises(_MODULE.PacketError, match="symlinks"):
        _MODULE.validate_packet(packet, repo_root=root)
