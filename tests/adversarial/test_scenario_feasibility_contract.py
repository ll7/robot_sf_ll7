"""Unit tests for the versioned planner-free scenario-feasibility contract."""

from __future__ import annotations

import math

import pytest

from robot_sf.adversarial.feasibility_first import (
    CHECK_NAMES,
    SCENARIO_FEASIBILITY_CONTRACT_VERSION,
    FeasibilityCheck,
    FeasibilityFirstError,
    ScenarioFeasibilityContract,
    ScenarioFeasibilityPredicate,
    ScenarioFeasibilityRejectionLedger,
    build_fixture_candidates,
    build_scenario_feasibility_ledger,
    evaluate_scenario_feasibility,
    validate_scenario_feasibility_contract,
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


def test_predicate_parser_rejects_shape_and_identity_drift() -> None:
    """Predicate parsing fails closed for malformed records and renamed dimensions."""
    expected = CHECK_NAMES[0]
    cases = [
        (None, "must be a mapping"),
        ({"verdict": "valid", "reason": "ok", "evidence": {}, "extra": True}, "unknown"),
        (
            {"name": CHECK_NAMES[1], "verdict": "valid", "reason": "ok", "evidence": {}},
            "must be",
        ),
        ({"reason": "ok", "evidence": {}}, "requires a verdict"),
        ({"verdict": "valid", "reason": 1, "evidence": {}}, "string reason"),
        ({"verdict": "valid", "reason": "ok", "evidence": []}, "evidence must be"),
    ]
    for payload, message in cases:
        with pytest.raises(FeasibilityFirstError, match=message):
            ScenarioFeasibilityPredicate.from_mapping(payload, expected_name=expected)

    with pytest.raises(FeasibilityFirstError, match="predicate name"):
        ScenarioFeasibilityPredicate("not-a-dimension", "valid", "ok", {"fixture": "direct"})
    with pytest.raises(FeasibilityFirstError, match="unsupported verdict"):
        ScenarioFeasibilityPredicate(expected, "unknown", "ok", {})
    with pytest.raises(FeasibilityFirstError, match="requires evidence"):
        ScenarioFeasibilityPredicate(expected, "valid", "ok", {})


def test_contract_constructor_and_parser_reject_invariants() -> None:
    """Direct construction and deserialization enforce version, type, and order invariants."""
    predicates = tuple(
        ScenarioFeasibilityPredicate(name, "valid", "ok", {"fixture": "direct"})
        for name in CHECK_NAMES
    )
    assert ScenarioFeasibilityContract("candidate", predicates).predicate_verdicts == predicates
    with pytest.raises(FeasibilityFirstError, match="contract_version"):
        ScenarioFeasibilityContract("candidate", predicates, contract_version="old")
    with pytest.raises(FeasibilityFirstError, match="candidate_id"):
        ScenarioFeasibilityContract("", predicates)
    with pytest.raises(FeasibilityFirstError, match="4 entries"):
        ScenarioFeasibilityContract("candidate", ())
    with pytest.raises(FeasibilityFirstError, match="invalid record types"):
        ScenarioFeasibilityContract("candidate", (object(),) * len(CHECK_NAMES))
    with pytest.raises(FeasibilityFirstError, match="canonical order"):
        ScenarioFeasibilityContract("candidate", (predicates[1], predicates[0], *predicates[2:]))

    with pytest.raises(FeasibilityFirstError, match="must be a mapping"):
        ScenarioFeasibilityContract.from_mapping(None)
    payload = _contract("drift")
    payload.pop("contract_version")
    payload["unexpected"] = True
    with pytest.raises(FeasibilityFirstError, match="unknown"):
        ScenarioFeasibilityContract.from_mapping(payload)

    for predicates_payload, message in (
        ("not-a-sequence", "must be a sequence"),
        ([], "4 entries"),
    ):
        malformed = _contract("predicate-shape")
        malformed["predicates"] = predicates_payload
        with pytest.raises(FeasibilityFirstError, match=message):
            ScenarioFeasibilityContract.from_mapping(malformed)


def test_contract_derived_fields_and_validation_round_trip() -> None:
    """Derived booleans and rejection reasons are validated against predicates."""
    valid = ScenarioFeasibilityContract.from_mapping(_contract("valid"))
    validate_scenario_feasibility_contract(valid.to_dict())

    rejected = ScenarioFeasibilityContract.from_mapping(_contract("rejected", "invalid"))
    serialized = rejected.to_dict()
    validate_scenario_feasibility_contract(serialized)

    for field in ("feasible", "safety_denominator_eligible"):
        invalid_type = dict(serialized)
        invalid_type[field] = 1
        with pytest.raises(FeasibilityFirstError, match="must be boolean"):
            ScenarioFeasibilityContract.from_mapping(invalid_type)

    for reasons, message in (
        ("not-a-sequence", "must be a sequence"),
        ([1], "must contain"),
        (["wrong"], "contradict"),
    ):
        invalid_reasons = dict(serialized)
        invalid_reasons["rejection_reasons"] = reasons
        with pytest.raises(FeasibilityFirstError, match=message):
            ScenarioFeasibilityContract.from_mapping(invalid_reasons)


def test_evaluator_accepts_all_record_forms_and_rejects_drift() -> None:
    """The evaluator accepts checks, predicate objects, and mappings with strict names."""
    candidate = build_fixture_candidates()[0]
    direct = tuple(
        ScenarioFeasibilityPredicate(name, "valid", "ok", {"fixture": "direct"})
        for name in CHECK_NAMES
    )
    assert evaluate_scenario_feasibility(candidate.candidate_id, direct).feasible
    assert evaluate_scenario_feasibility(
        candidate.candidate_id, [_predicate(name, "valid", "ok") for name in CHECK_NAMES]
    ).feasible

    wrong_check = list(candidate.checks)
    wrong_check[0] = FeasibilityCheck(CHECK_NAMES[1], "pass", "ok", {"fixture": "check"})
    with pytest.raises(FeasibilityFirstError, match="must be"):
        evaluate_scenario_feasibility(candidate.candidate_id, wrong_check)
    with pytest.raises(FeasibilityFirstError, match="must be"):
        evaluate_scenario_feasibility(candidate.candidate_id, (direct[1], direct[0], *direct[2:]))
    with pytest.raises(FeasibilityFirstError, match="must be a check"):
        evaluate_scenario_feasibility(candidate.candidate_id, [object()] * len(CHECK_NAMES))
    with pytest.raises(FeasibilityFirstError, match="must be a sequence"):
        evaluate_scenario_feasibility(candidate.candidate_id, "not-a-sequence")


def test_ledger_constructor_and_builder_reject_mixed_or_unsupported_records() -> None:
    """Ledger construction rejects duplicates and mixed record families."""
    contract = ScenarioFeasibilityContract.from_mapping(_contract("one"))
    assert isinstance(ScenarioFeasibilityRejectionLedger([contract]).contracts, tuple)
    assert build_scenario_feasibility_ledger([contract]).accepted_candidate_ids == ("one",)
    with pytest.raises(FeasibilityFirstError, match="ledger contract_version"):
        ScenarioFeasibilityRejectionLedger((contract,), contract_version="old")
    with pytest.raises(FeasibilityFirstError, match="invalid record types"):
        ScenarioFeasibilityRejectionLedger((object(),))
    mixed = ScenarioFeasibilityContract.from_mapping(_contract("mixed"))
    object.__setattr__(mixed, "contract_version", "old")
    with pytest.raises(FeasibilityFirstError, match="mixed"):
        ScenarioFeasibilityRejectionLedger((mixed,))
    with pytest.raises(FeasibilityFirstError, match="unique"):
        ScenarioFeasibilityRejectionLedger((contract, contract))
    with pytest.raises(FeasibilityFirstError, match="must be a sequence"):
        build_scenario_feasibility_ledger(None)
    with pytest.raises(FeasibilityFirstError, match="candidates, contracts"):
        build_scenario_feasibility_ledger([object()])
