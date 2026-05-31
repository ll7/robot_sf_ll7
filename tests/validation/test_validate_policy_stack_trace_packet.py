"""Tests for policy_stack_v1 arbitration trace packet validation."""

from __future__ import annotations

import json

from scripts.validation import validate_policy_stack_trace_packet as validator


def test_validate_policy_stack_trace_packet_cli(capsys) -> None:
    """The CLI should prove the tiny trace packet shape without benchmark metrics."""
    exit_code = validator.main([])

    assert exit_code == 0
    assert "OK policy_stack_v1 arbitration trace packet is valid" in capsys.readouterr().out


def test_validate_policy_stack_trace_packet_json(capsys) -> None:
    """The JSON mode should expose a complete packet for docs and fixtures."""
    exit_code = validator.main(["--json"])

    assert exit_code == 0
    packet = json.loads(capsys.readouterr().out)
    assert packet["schema_version"] == "policy_stack_v1.arbitration_trace_packet.v1"
    assert packet["training_enabled"] is False
    assert packet["command_contract"]["action_space"] == "unicycle_vw"
    assert set(packet["observation_contract"]["inference_available_features"]) == (
        validator.EXPECTED_INFERENCE_FEATURES
    )
    assert set(packet["observation_contract"]["leakage_exclusions"]) == (
        validator.EXPECTED_LEAKAGE_EXCLUSIONS
    )
    assert packet["trace"]["last_step"]["selected_proposal_key"] == "goal"
    assert packet["trace"]["last_step"]["candidate_ranking"][0]["rank"] == 1
