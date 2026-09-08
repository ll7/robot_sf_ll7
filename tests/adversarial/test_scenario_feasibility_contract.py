"""Unit tests for the versioned planner-free scenario-feasibility contract."""

from __future__ import annotations

import math

import pytest

from robot_sf.adversarial.feasibility_first import (
    CHECK_NAMES,
    SCENARIO_FEASIBILITY_CONTRACT_VERSION,
    FeasibilityFirstError,
    ScenarioFeasibilityContract,
    build_fixture_candidates,
    build_scenario_feasibility_ledger,
    evaluate_scenario_feasibility,
)


def _predicate(name: str, verdict: str, reason: str) -> dict[str, object]:
    """Build one hand-written predicate record."""
    return {
        "name": name,
        "verdict": verdict,
        "reason": reason,
        "evidence": {"fixture": "hand_built"} if verdict == "valid" else {},
    }


def _contract(candidate_id: str, verdict: str = "valid") -> dict[str, object]:
    """Build a hand-written four-dimension contract payload."""
    return {
        "contract_version": SCENARIO_FEASIBILITY_CONTRACT_VERSION,
        "candidate_id": candidate_id,
        "predicates": [_predicate(name, verdict, f"{name} is {verdict}") for name in CHECK_NAMES],
    }


@pytest.mark.parametrize("verdict", ["valid", "invalid", "missing", "contradictory"])
def test_hand_built_verdicts_are_explicit_and_deterministic(verdict: str) -> None:
    """Each supported verdict is preserved and non-valid records are rejected."""
    first = ScenarioFeasibilityContract.from_mapping(_contract("hand-built", verdict))
    second = ScenarioFeasibilityContract.from_mapping(_contract("hand-built", verdict))

    assert first == second
    assert {predicate.verdict for predicate in first.predicates} == {verdict}
    assert first.feasible is (verdict == "valid")
    if verdict == "valid":
        assert first.rejection_reasons == ()
    else:
        assert len(first.rejection_reasons) == len(CHECK_NAMES)
        assert all(
            reason.startswith(f"{name}:{verdict}:")
            for name, reason in zip(CHECK_NAMES, first.rejection_reasons, strict=True)
        )


def test_static_set_excludes_every_rejected_candidate_from_denominator() -> None:
    """The ledger keeps only fully valid identities in its denominator."""
    records = [
        _contract("valid-row"),
        _contract("invalid-row", "invalid"),
        _contract("missing-row", "missing"),
        _contract("contradictory-row", "contradictory"),
    ]

    ledger = build_scenario_feasibility_ledger(records)
    repeated = build_scenario_feasibility_ledger(list(reversed(records)))

    assert ledger.to_dict() == repeated.to_dict()
    assert ledger.accepted_candidate_ids == ("valid-row",)
    assert set(ledger.rejected_candidate_ids).isdisjoint(ledger.safety_denominator_candidate_ids)
    assert ledger.safety_denominator_candidate_ids == ("valid-row",)
    assert ledger.safety_denominator_candidate_ids == ledger.accepted_candidate_ids
    assert all(contract.rejection_reasons for contract in ledger.rejected_contracts)
    serialized = ledger.to_dict()
    assert serialized["invalid_candidates_excluded_from_safety_denominators"] is True
    assert all(entry["reasons"] for entry in serialized["rejection_ledger"])


def test_existing_candidate_checks_project_without_changing_their_schema() -> None:
    """The facade reuses canonical checks and maps unavailable to missing."""
    candidate = build_fixture_candidates()[6]
    contract = ScenarioFeasibilityContract.from_candidate(candidate)

    assert contract.candidate_id == candidate.candidate_id
    assert [predicate.name for predicate in contract.predicates] == list(CHECK_NAMES)
    assert contract.predicates[-1].verdict == "missing"
    assert candidate.to_dict()["checks"][3]["status"] == "unavailable"
    assert build_scenario_feasibility_ledger([candidate.to_dict()]).to_dict() == (
        build_scenario_feasibility_ledger([candidate]).to_dict()
    )


def test_evaluate_accepts_canonical_checks_and_rejects_wrong_shape() -> None:
    """The pure evaluator accepts existing checks while requiring all dimensions."""
    candidate = build_fixture_candidates()[0]
    contract = evaluate_scenario_feasibility(candidate.candidate_id, candidate.checks)

    assert contract.feasible is True
    with pytest.raises(FeasibilityFirstError, match="4 entries"):
        evaluate_scenario_feasibility(candidate.candidate_id, candidate.checks[:3])


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.pop("contract_version"), "missing"),
        (lambda payload: payload["predicates"][0].pop("evidence"), "evidence mapping"),
        (lambda payload: payload["predicates"][0].update({"evidence": {"x": math.nan}}), "finite"),
        (lambda payload: payload["predicates"][0].update({"verdict": "unknown"}), "unsupported"),
        (
            lambda payload: payload["predicates"][0].update(
                {"status": "invalid", "verdict": "valid"}
            ),
            "contradictory",
        ),
        (lambda payload: payload.update({"feasible": False}), "contradicts predicates"),
    ],
)
def test_malformed_nonfinite_and_contradictory_payloads_fail_closed(mutator, message: str) -> None:
    """Malformed evidence never becomes an implicitly admissible record."""
    payload = _contract("bad-row")
    mutator(payload)

    with pytest.raises(FeasibilityFirstError, match=message):
        ScenarioFeasibilityContract.from_mapping(payload)


def test_rejection_reason_is_required() -> None:
    """A rejected predicate without a reason cannot enter the contract."""
    payload = _contract("no-reason", "invalid")
    payload["predicates"][2]["reason"] = ""

    with pytest.raises(FeasibilityFirstError, match="non-empty reason"):
        ScenarioFeasibilityContract.from_mapping(payload)
