"""Contract tests for the issue #6942 ORCA adapter-hedge preregistration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validation import check_issue_6942_orca_adapter_preregistration as checker

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_6942_orca_adapter_hedge_preregistration.yaml"


def test_packet_is_valid_but_execution_blocked() -> None:
    """The packet freezes the diagnostic design without authorizing campaign execution."""
    result = checker.validate_packet(checker.load_packet(PACKET))

    assert result["status"] == "blocked"
    assert result["evidence_tier"] == "proposal"
    assert result["project_imports_performed"] is False
    assert result["execution_authorized"] is False
    assert result["domain_approval_status"] == "pending"
    assert result["scenario_count"] == 6
    assert result["seed_count"] == 30
    assert result["episode_cell_count"] == 180
    assert result["fallback_allowed"] is False


def test_cli_json_reports_blocked_without_running_campaign(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The checker emits structured blocked status for automation and issue handoff."""
    assert checker.main(["--config", str(PACKET), "--json"]) == 0
    result = json.loads(capsys.readouterr().out.strip())

    assert result["status"] == "blocked"
    assert result["checks"]["campaign_not_run"] is True


def test_packet_rejects_approval_status_change() -> None:
    """Approval cannot be silently treated as granted by editing the tracked packet."""
    packet = checker.load_packet(PACKET)
    packet["domain_approval"]["status"] = "approved"

    with pytest.raises(checker.PacketError, match="status must remain pending"):
        checker.validate_packet(packet)


def test_packet_rejects_fallback_success() -> None:
    """The preregistration retains the repository fallback exclusion."""
    packet = checker.load_packet(PACKET)
    packet["comparator_contract"]["native_solver"]["fallback_allowed"] = True

    with pytest.raises(checker.PacketError, match="native fallback must be disallowed"):
        checker.validate_packet(packet)


def test_packet_rejects_scenario_order_drift() -> None:
    """The fixed representative scenario suite cannot be reordered or replaced."""
    packet = checker.load_packet(PACKET)
    packet["baseline_protocol"]["selected_scenarios"][0]["scenario_id"] = "classic_merging_medium"

    with pytest.raises(checker.PacketError, match="scenario order mismatch"):
        checker.validate_packet(packet)


def test_packet_rejects_missing_trace_field() -> None:
    """Projection divergence cannot be computed when the trace schema is incomplete."""
    packet = checker.load_packet(PACKET)
    packet["trace_contract"]["required_fields"].remove("angle_error_rad")

    with pytest.raises(checker.PacketError, match="trace field order mismatch"):
        checker.validate_packet(packet)


def test_packet_rejects_transient_routing_state() -> None:
    """Tracked protocol files must not capture host, queue, or local-output state."""
    packet = checker.load_packet(PACKET)
    packet["preflight_contract"]["output_dir"] = "output/issue_6942"

    with pytest.raises(checker.PacketError, match="transient routing or local-output state"):
        checker.validate_packet(packet)


def test_missing_referenced_path_fails_closed() -> None:
    """A protocol that points at a missing source cannot pass metadata validation."""
    packet = checker.load_packet(PACKET)
    packet["preflight_contract"]["required_paths"][0] = "configs/does_not_exist.yaml"

    with pytest.raises(checker.PacketError, match="is missing"):
        checker.validate_packet(packet)


def test_packet_retains_complete_case_stop_rule() -> None:
    """A future campaign must provide all paired fixed-suite cells per arm."""
    packet = checker.load_packet(PACKET)

    assert any("complete_case_population_is_below_180" in rule for rule in packet["stop_rules"])
    assert checker.validate_packet(packet)["episode_cell_count"] == 180
