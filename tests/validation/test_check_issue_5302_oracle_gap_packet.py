"""Tests for the issue #5302 oracle-gap analysis packet checker."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.validation import check_issue_5302_oracle_gap_packet as checker

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = REPO_ROOT / "configs/analysis/issue_5302_oracle_gap_packet.yaml"


def _packet() -> dict[str, object]:
    payload = yaml.safe_load(PACKET_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_shipped_packet_is_ready_without_running_campaign() -> None:
    """The contract validates against tracked sources while remaining no-submit."""
    result = checker.validate_packet(_packet(), repo_root=REPO_ROOT)
    assert result["status"] == "ok"
    assert result["planner_count"] == 5
    assert result["ceiling_count"] == 4
    assert result["campaign_execution_allowed"] is False


def test_rejects_roster_expansion() -> None:
    """Adding a registered planner must not silently change the pre-registered roster."""
    packet = _packet()
    roster = packet["planner_roster"]
    assert isinstance(roster, dict)
    required = roster["required"]
    assert isinstance(required, list)
    required.append({"planner_id": "goal", "role": "extra"})
    with pytest.raises(checker.PacketError, match="planner roster"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_rejects_split_leakage() -> None:
    """Family holdouts are mandatory rather than an optional reporting detail."""
    packet = _packet()
    inputs = packet["input_contract"]
    assert isinstance(inputs, dict)
    split = inputs["split_contract"]
    assert isinstance(split, dict)
    split["selection_and_evaluation_family_sets_must_be_disjoint"] = False
    with pytest.raises(
        checker.PacketError, match="selection_and_evaluation_family_sets_must_be_disjoint"
    ):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_rejects_transient_routing_state() -> None:
    """Host and queue routing belong to private ops/state, never this packet."""
    packet = _packet()
    packet["execution_boundary"]["target_host"] = "imech036"  # type: ignore[index]
    with pytest.raises(checker.PacketError, match="transient routing state"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_rejects_non_native_success_policy() -> None:
    """Fallback/degraded rows cannot be promoted by the analysis packet."""
    packet = _packet()
    policy = packet["row_status_policy"]
    assert isinstance(policy, dict)
    policy["eligible_execution_modes"] = ["native", "adapter"]
    with pytest.raises(checker.PacketError, match="only native"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_rejects_missing_provenance_path() -> None:
    """A missing canonical schema owner blocks readiness."""
    packet = _packet()
    inputs = packet["input_contract"]
    assert isinstance(inputs, dict)
    provenance = inputs["provenance"]
    assert isinstance(provenance, dict)
    provenance["required_paths"].append("missing/canonical_owner.py")
    with pytest.raises(checker.PacketError, match="provenance path missing"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)
