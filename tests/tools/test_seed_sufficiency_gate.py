"""Tests for the S5/S10/S20 seed-sufficiency gate."""

from __future__ import annotations

import json

from scripts.tools.seed_sufficiency_gate import SeedGateInput, decide_seed_gate, main


def test_seed_gate_escalates_when_uncertainty_exceeds_target() -> None:
    """Unstable S5 evidence should escalate to S10."""
    decision = decide_seed_gate(
        SeedGateInput(schedule="s5", ci_half_width=0.2, target_ci_half_width=0.1)
    )

    assert decision.decision == "escalate"
    assert decision.next_schedule == "s10"


def test_seed_gate_stops_when_frozen_rule_is_satisfied() -> None:
    """Stable evidence can stop instead of spending more compute."""
    decision = decide_seed_gate(
        SeedGateInput(schedule="s10", ci_half_width=0.05, target_ci_half_width=0.1)
    )

    assert decision.decision == "stop_confirmed"
    assert decision.next_schedule is None


def test_seed_gate_keeps_invalid_rows_diagnostic_only() -> None:
    """Invalid rows should not become stronger evidence through escalation."""
    decision = decide_seed_gate(
        SeedGateInput(
            schedule="s5",
            ci_half_width=0.05,
            target_ci_half_width=0.1,
            invalid_row_count=1,
        )
    )

    assert decision.decision == "diagnostic_only"
    assert decision.next_schedule is None


def test_seed_gate_keeps_unstable_s20_diagnostic_only() -> None:
    """S20 is the maximum schedule in the sprint gate."""
    decision = decide_seed_gate(
        SeedGateInput(
            schedule="s20",
            ci_half_width=0.2,
            target_ci_half_width=0.1,
            rank_flip_observed=True,
        )
    )

    assert decision.decision == "diagnostic_only"
    assert decision.next_schedule is None


def test_seed_gate_cli_ignores_extra_input_keys(tmp_path) -> None:
    """Evolving input payloads should not crash the CLI on unknown fields."""
    input_json = tmp_path / "seed-gate-input.json"
    input_json.write_text(
        json.dumps(
            {
                "schedule": "s5",
                "ci_half_width": 0.2,
                "target_ci_half_width": 0.1,
                "future_schema_field": "ignored",
            }
        ),
        encoding="utf-8",
    )

    assert main(["--input-json", str(input_json)]) == 0
