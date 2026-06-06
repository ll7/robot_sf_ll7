"""Tests for the issue #2472 intent-conditioned behavior contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_sf.benchmark.scenario_contract import (
    load_scenario_contracts,
    validate_scenario_contract_references,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    REPO_ROOT / "configs/scenarios/contracts/intent_conditioned_behavior_issue2472_contracts.yaml"
)


def _contract_payload() -> dict[str, Any]:
    """Load the single #2472 contract as a plain dictionary."""
    contracts = load_scenario_contracts(CONTRACT_PATH)
    assert len(contracts) == 1
    return contracts[0].to_dict()


def test_intent_conditioned_contract_resolves_existing_smoke_scenario() -> None:
    """The proposal contract should point at the existing intersection-wait scenario."""
    contract = load_scenario_contracts(CONTRACT_PATH)[0]

    assert contract.id == "francis2023_intersection_wait_intent_conditioned_contract"
    assert (
        contract.scenario_ref.source
        == "configs/scenarios/single/francis2023_intersection_wait.yaml"
    )
    assert contract.scenario_ref.scenario_name == "francis2023_intersection_wait"
    assert validate_scenario_contract_references(contract, repo_root=REPO_ROOT) == []


def test_intent_vocabulary_maps_to_existing_authoring_knobs() -> None:
    """Intent labels should map to current single-pedestrian knobs without new runtime claims."""
    payload = _contract_payload()
    extension = payload["extensions"]["intent_conditioned_behavior.v1"]
    vocabulary = extension["intent_vocabulary"]

    assert set(vocabulary) == {
        "crossing",
        "waiting",
        "following",
        "overtaking",
        "group_join",
    }
    assert {"trajectory", "speed_m_s"}.issubset(vocabulary["crossing"]["existing_knobs"])
    assert {"wait_at", "start_delay_s"}.issubset(vocabulary["waiting"]["existing_knobs"])
    assert {"role", "role_target_id", "role_offset"}.issubset(
        vocabulary["following"]["existing_knobs"]
    )
    assert {"role", "role_target_id"}.issubset(vocabulary["group_join"]["existing_knobs"])


def test_contract_declares_trace_gap_before_intent_interpretation() -> None:
    """The contract should name missing trace fields before any intent-conditioned claim."""
    payload = _contract_payload()
    extension = payload["extensions"]["intent_conditioned_behavior.v1"]

    assert extension["benchmark_evidence"] is False
    assert extension["status"] == "proposal_metadata_only"
    assert extension["first_smoke_scenario"] == {
        "source": "configs/scenarios/single/francis2023_intersection_wait.yaml",
        "scenario_name": "francis2023_intersection_wait",
        "selected_intents": ["waiting", "crossing"],
        "rationale": "Existing trajectory and wait_at annotations make this the smallest first smoke target.",
    }
    assert {"id", "position", "velocity"}.issubset(
        extension["current_trace_gap"]["current_trace_fields"]
    )
    assert {"intent_label", "intent_phase", "behavior_parameters"}.issubset(
        extension["current_trace_gap"]["missing_for_intent_interpretation"]
    )
    assert {"intent_label", "intent_phase", "release_event_step"}.issubset(
        extension["required_future_trace_fields"]
    )


def test_claim_boundary_rejects_realism_and_planner_ranking_claims() -> None:
    """The contract should remain proposal-tier and fail closed on interpretation."""
    payload = _contract_payload()
    claim_boundary = payload["benchmark_eligibility"]["claim_boundary"].lower()
    invariants = {invariant["id"] for invariant in payload["invariants"]}
    hooks = " ".join(payload["benchmark_eligibility"]["eligibility_hooks"]).lower()

    assert payload["benchmark_eligibility"]["intended_use"] == "exploratory"
    assert payload["certification"]["expected_eligibility"] == "unknown"
    assert "intent_metadata_not_runtime_model" in invariants
    assert "no_realism_claim_without_data" in invariants
    assert "not evidence" in claim_boundary
    assert "planner rankings" in claim_boundary
    assert "data-grounded validation" in hooks
