"""Contract tests for the issue #5592 cross-matrix pre-registration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from loguru import logger

from scripts.validation import check_issue_5592_cross_matrix_preregistration as checker

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_5592_cross_matrix_preregistration.yaml"


def test_packet_passes_matrix_metadata_and_roster_gate() -> None:
    """The packet resolves both matrices and the exact paired roster contract."""
    result = checker.validate_packet(checker.load_packet(PACKET))

    assert result["status"] == "ready"
    assert result["reference_scenario_count"] == 23
    assert result["candidate"]["scenario_count"] == 3
    assert result["candidate"]["seeds"] == [20, 21, 22, 23, 24]
    assert result["planner_count"] == 12
    assert result["agreement_table"] == "cross_matrix_agreement.csv"


def test_packet_rejects_transient_routing_state() -> None:
    """Tracked benchmark contracts must not encode host or queue routing state."""
    packet = checker.load_packet(PACKET)
    packet["target_host"] = "imech036"

    with pytest.raises(checker.PacketError, match="transient routing state"):
        checker.validate_packet(packet)


def test_packet_rejects_candidate_metadata_drift() -> None:
    """The selected-row table cannot silently drift from the loaded scenarios."""
    packet = checker.load_packet(PACKET)
    packet["candidate_contract"]["selected_rows"][1]["target_failure_mode"] = "oscillation"

    with pytest.raises(checker.PacketError, match="candidate selected-row metadata mismatch"):
        checker.validate_packet(packet)


def test_packet_rejects_seed_schedule_drift() -> None:
    """Paired matrices must retain all five pre-registered seeds."""
    packet = checker.load_packet(PACKET)
    packet["pairing_contract"]["seeds"] = [20, 21, 22]

    with pytest.raises(checker.PacketError, match="pairing seed schedule mismatch"):
        checker.validate_packet(packet)


def test_packet_requires_disagreement_rows_in_primary_output() -> None:
    """A future report must expose rank flips instead of hiding them in a merge."""
    packet = checker.load_packet(PACKET)
    packet["comparison_contract"]["must_emit_disagreement_rows"] = False

    with pytest.raises(checker.PacketError, match="disagreement rows"):
        checker.validate_packet(packet)


def test_cli_emits_machine_readable_ready_status(capsys) -> None:
    """The canonical checker command returns a compact JSON readiness result."""
    exit_code = checker.main(["--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "ready"
    assert payload["campaign_execution_allowed"] is False


def test_json_cli_preserves_caller_loguru_sink(capsys) -> None:
    """JSON validation must not remove process-global sinks installed by callers."""
    sink_id = logger.add(sys.stdout, format="{message}")
    try:
        exit_code = checker.main(["--json"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert json.loads(captured.out)["status"] == "ready"
        assert captured.err

        logger.info("caller sink remains active")
        assert capsys.readouterr().out.strip() == "caller sink remains active"
    finally:
        try:
            logger.remove(sink_id)
        except ValueError:
            pass
