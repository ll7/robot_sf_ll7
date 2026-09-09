"""Tests for fail-closed research answerability and yield reporting."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from robot_sf.adversarial.feasibility_first import (
    CHECK_NAMES,
    FeasibilityCheck,
    build_scenario_feasibility_ledger,
    evaluate_scenario_feasibility,
)
from robot_sf.benchmark.research_answerability import (
    ADVERSARIAL_FALSIFICATION_PACKET_OUTCOMES,
    AdversarialFalsificationPacketError,
    answerability_from_manifest,
    compute_adversarial_falsification_packet_digest,
    evaluate_answerability,
    load_adversarial_falsification_packet,
    load_adversarial_falsification_packet_schema,
    validate_adversarial_falsification_packet,
)
from scripts.analysis.report_research_yield import (
    ResearchYieldError,
    build_research_yield_report,
    load_snapshot,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST = REPO_ROOT / "configs/benchmarks/research_campaign_manifest.example.yaml"
ISSUE_6474_FIXTURE = REPO_ROOT / "tests/fixtures/research_answerability/issue_6474_bounded.json"
YIELD_FIXTURE = REPO_ROOT / "tests/fixtures/research_yield_snapshot.v1.json"
ADVERSARIAL_PACKET = (
    REPO_ROOT / "configs/adversarial/issue_8570_bounded_falsification_answerability_v1.yaml"
)


def _example_contract() -> dict[str, object]:
    payload = yaml.safe_load(EXAMPLE_MANIFEST.read_text(encoding="utf-8"))
    return copy.deepcopy(payload["answerability"])


def test_example_contract_is_diagnostic_only() -> None:
    """The canonical example is executable only as a bounded diagnostic packet."""
    result = evaluate_answerability(_example_contract())

    assert result.state == "diagnostic_only"
    assert result.as_dict()["decision_capable"] is False


def test_optional_unavailable_metric_is_preserved_without_blocking() -> None:
    """A bounded fixture may keep optional unavailable metrics explicit."""
    contract = json.loads(ISSUE_6474_FIXTURE.read_text(encoding="utf-8"))

    result = evaluate_answerability(contract)

    assert result.state == "answerable"
    assert result.warnings
    assert "secondary_realism_metric" in result.warnings[0]


@pytest.mark.parametrize(
    ("section", "field", "value", "expected"),
    [
        ("producers", "status", "missing", "blocked_missing_producer"),
        ("producers", "execution_mode", "fallback", "blocked_missing_producer"),
        ("design", "power_status", "underpowered", "blocked_underpowered"),
        ("analysis", "dry_run_status", "failed", "blocked_analysis_contract"),
        ("analysis", "comparability_status", "mismatched", "blocked_noncomparable_rows"),
        ("artifacts", "durability_status", "blocked", "blocked_artifact_plan"),
    ],
)
def test_known_answerability_blockers_are_fail_closed(
    section: str, field: str, value: str, expected: str
) -> None:
    """Known campaign failure classes map to explicit non-answerable states."""
    contract = _example_contract()
    if section == "producers":
        contract[section][0][field] = value
    else:
        contract[section][field] = value

    assert evaluate_answerability(contract).state == expected


def test_malformed_contract_is_invalid() -> None:
    """Missing schema fields cannot be mistaken for an underpowered campaign."""
    contract = _example_contract()
    del contract["estimand"]["primary"]

    result = evaluate_answerability(contract)

    assert result.state == "invalid_contract"
    assert "primary" in result.reasons[0]


@pytest.mark.parametrize(
    ("case_id", "mutator", "expected"),
    [
        (
            "6970_missing_normalized_producer",
            lambda contract: contract["producers"][0].update(
                {"status": "missing", "field": "normalized_reference_value"}
            ),
            "blocked_missing_producer",
        ),
        (
            "6849_underpowered_held_out_design",
            lambda contract: contract["design"].update({"power_status": "underpowered"}),
            "blocked_underpowered",
        ),
        (
            "6980_missing_reference_exposure",
            lambda contract: contract["analysis"].update({"comparability_status": "mismatched"}),
            "blocked_noncomparable_rows",
        ),
        (
            "6814_missing_durable_provenance",
            lambda contract: contract["artifacts"].update({"durability_status": "blocked"}),
            "blocked_artifact_plan",
        ),
    ],
)
def test_known_failure_cases_have_explicit_states(case_id: str, mutator, expected: str) -> None:
    """Known issue failure classes cannot be silently promoted to answerable."""
    contract = _example_contract()
    mutator(contract)

    result = evaluate_answerability(contract)

    assert case_id
    assert result.state == expected


def test_manifest_without_answerability_is_not_declared() -> None:
    """Existing manifests remain loadable but can be gated explicitly."""
    manifest = {"campaign": {}}

    result = answerability_from_manifest(manifest)

    assert result["state"] == "not_declared"
    assert result["decision_capable"] is False


def _issue_8570_packet() -> dict[str, object]:
    """Load the checked-in packet for mutation-based validator tests."""
    return load_adversarial_falsification_packet(ADVERSARIAL_PACKET)


def _refresh_packet_digest(packet: dict[str, object]) -> None:
    """Keep a mutated fixture self-consistent so semantic checks are reached."""
    packet["self_digest"] = compute_adversarial_falsification_packet_digest(packet)


def test_issue_8570_packet_is_schema_valid_and_compute_blocked() -> None:
    """The checked-in design is source-bound, diagnostic-only, and not executable."""
    schema = load_adversarial_falsification_packet_schema()
    Draft202012Validator.check_schema(schema)
    packet = _issue_8570_packet()

    assert packet["self_digest"] == compute_adversarial_falsification_packet_digest(packet)
    assert packet["claim_eligible"] is False
    assert packet["answerability"]["schema_version"] == "research_answerability.v1"
    assert evaluate_answerability(packet["answerability"]).state == "blocked_missing_producer"
    assert packet["compute_authorization"]["status"] == "blocked"
    assert packet["compute_authorization"]["authorized"] is False
    delay = packet["variable_map"]["pedestrian_delay_s"]
    assert delay["runtime_effective"] is False
    assert delay["binding_status"] == "provenance_only"


def test_issue_8570_packet_rejects_self_digest_tampering() -> None:
    """Changing packet bytes without updating the self-digest fails closed."""
    packet = _issue_8570_packet()
    packet["self_digest"] = "0" * 64

    with pytest.raises(AdversarialFalsificationPacketError, match="self_digest"):
        validate_adversarial_falsification_packet(packet)


def test_issue_8570_packet_rejects_source_hash_tampering() -> None:
    """A raw source-byte hash mismatch is detected after a valid packet digest update."""
    packet = _issue_8570_packet()
    packet["source"]["inputs"]["search_space"]["sha256"] = "0" * 64
    _refresh_packet_digest(packet)

    with pytest.raises(AdversarialFalsificationPacketError, match="source.inputs.search_space"):
        validate_adversarial_falsification_packet(packet)


def test_issue_8570_packet_rejects_owner_hash_tampering() -> None:
    """Code-owner provenance is bound to the same fail-closed source manifest."""
    packet = _issue_8570_packet()
    packet["source"]["owners"][0]["sha256"] = "0" * 64
    _refresh_packet_digest(packet)

    with pytest.raises(AdversarialFalsificationPacketError, match=r"source\.owners\[0\]"):
        validate_adversarial_falsification_packet(packet)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda packet: packet["source"]["search_space"]["semantic"]["variables"][
                "start_x"
            ].update({"max": 99.0}),
            "source.search_space.semantic",
        ),
        (
            lambda packet: packet["variable_map"]["start_x"]["bounds"].update({"max": 99.0}),
            "variable_map.start_x.bounds",
        ),
    ],
)
def test_issue_8570_packet_rejects_source_or_bound_mismatch(mutation, message: str) -> None:
    """Source semantics and the explicit variable map cannot drift independently."""
    packet = _issue_8570_packet()
    mutation(packet)
    _refresh_packet_digest(packet)

    with pytest.raises(AdversarialFalsificationPacketError, match=message):
        validate_adversarial_falsification_packet(packet)


def test_issue_8570_packet_accounts_for_equal_budgets_and_disjoint_seeds() -> None:
    """CMA-ES and both controls use the approved 64-by-3 search allocation."""
    packet = _issue_8570_packet()
    seed_policy = packet["seed_policy"]
    search_seeds = seed_policy["search_seeds"]
    confirmation_seeds = seed_policy["confirmation_seeds"]
    budget = packet["budget"]

    assert len(search_seeds) == 3
    assert len(confirmation_seeds) == 5
    assert set(search_seeds).isdisjoint(confirmation_seeds)
    assert budget["search_candidate_rows_per_arm"] == 64 * len(search_seeds)
    assert budget["search_candidate_rows_all_arms"] == 64 * len(search_seeds) * 3
    assert {
        method["budget_per_seed"]
        for method in [
            packet["search_methods"]["primary"],
            *packet["search_methods"]["controls"],
        ]
    } == {64}

    tampered = copy.deepcopy(packet)
    tampered["seed_policy"]["confirmation_seeds"][0] = search_seeds[0]
    _refresh_packet_digest(tampered)
    with pytest.raises(AdversarialFalsificationPacketError, match="disjoint"):
        validate_adversarial_falsification_packet(tampered)

    tampered = copy.deepcopy(packet)
    tampered["budget"]["search_candidate_rows_all_arms"] -= 1
    _refresh_packet_digest(tampered)
    with pytest.raises(AdversarialFalsificationPacketError, match="all_arms"):
        validate_adversarial_falsification_packet(tampered)


def test_issue_8570_packet_has_nonadaptive_budget_stop_rule() -> None:
    """Search and held-out budgets stop deterministically without replacement rows."""
    packet = _issue_8570_packet()
    assert packet["stop_rule"] == {
        "search": {
            "action": "stop",
            "candidates_per_arm_per_seed": 64,
            "search_seed_count": 3,
            "no_early_stop_on_objective": True,
        },
        "confirmation": {
            "action": "stop",
            "held_out_seed_count": 5,
            "execution_status": "not_authorized",
        },
        "contract_failure": {
            "action": "stop_before_compute",
            "outcomes": ["invalid", "unavailable", "inconclusive", "blocked"],
        },
        "replacement_rows_allowed": False,
    }

    tampered = copy.deepcopy(packet)
    tampered["stop_rule"]["search"]["candidates_per_arm_per_seed"] = 63
    _refresh_packet_digest(tampered)
    with pytest.raises(AdversarialFalsificationPacketError, match="stop_rule.search"):
        validate_adversarial_falsification_packet(tampered)


def test_issue_8570_packet_has_complete_no_result_vocabulary() -> None:
    """Result, null, and every explicit no-result state remain distinct."""
    packet = _issue_8570_packet()
    assert set(packet["outcome_vocabulary"]) == set(ADVERSARIAL_FALSIFICATION_PACKET_OUTCOMES)

    tampered = copy.deepcopy(packet)
    del tampered["outcome_vocabulary"]["null"]
    _refresh_packet_digest(tampered)
    with pytest.raises(AdversarialFalsificationPacketError, match="outcome_vocabulary"):
        validate_adversarial_falsification_packet(tampered)


def test_issue_8570_packet_rejects_template_delay_as_runtime_effective() -> None:
    """Metadata cannot authorize the #7340 template-mode pedestrian delay."""
    packet = _issue_8570_packet()
    delay = packet["variable_map"]["pedestrian_delay_s"]
    delay["runtime_effective"] = True
    _refresh_packet_digest(packet)

    with pytest.raises(AdversarialFalsificationPacketError, match="runtime_effective"):
        validate_adversarial_falsification_packet(packet)


def _fixture_feasibility_contract(
    candidate_id: str,
    rejected_name: str | None = None,
    rejected_status: str = "fail",
):
    """Build a pure-data contract for deterministic rejection-ledger accounting."""
    checks = tuple(
        FeasibilityCheck(
            name,
            rejected_status if name == rejected_name else "pass",
            "fixture rejection" if name == rejected_name else "fixture evidence",
            {"source": "test fixture"},
        )
        for name in CHECK_NAMES
    )
    return evaluate_scenario_feasibility(candidate_id, checks)


def test_existing_feasibility_rejection_accounting_is_deterministic() -> None:
    """The packet composes the existing sorted ledger and denominator boundary."""
    records = [
        _fixture_feasibility_contract("candidate-b", "geometry_traffic"),
        _fixture_feasibility_contract("candidate-a"),
        _fixture_feasibility_contract("candidate-c", "simulator_validity", "unavailable"),
    ]

    first = build_scenario_feasibility_ledger(records)
    second = build_scenario_feasibility_ledger(list(reversed(records)))

    assert first.to_dict() == second.to_dict()
    assert first.accepted_candidate_ids == ("candidate-a",)
    assert first.rejected_candidate_ids == ("candidate-b", "candidate-c")
    assert first.safety_denominator_candidate_ids == ("candidate-a",)
    assert first.rejection_counts == {"geometry_traffic": 1, "simulator_validity": 1}
    assert first.to_dict()["invalid_candidates_excluded_from_safety_denominators"] is True


def test_research_yield_report_separates_empirical_and_infrastructure() -> None:
    """Yield dimensions remain separate and carry the frozen source digest."""
    snapshot = load_snapshot(YIELD_FIXTURE)
    report = build_research_yield_report(snapshot, source_path=YIELD_FIXTURE)

    assert report["records_total"] == 5
    assert report["empirical_answers"] == {
        "records": 3,
        "statuses": {"completed": 1, "inconclusive": 2},
    }
    assert report["infrastructure_throughput"]["records"] == 2
    assert report["lag_days"]["approval_to_first_result"]["median_days"] == 2.0
    assert report["source_snapshot"]["sha256"]
    assert "closure" in report["definitions"]["empirical_answers"]
    assert "## Empirical Answers" in render_markdown(report)


def test_research_yield_report_rejects_unknown_kind(tmp_path: Path) -> None:
    """The report must not silently classify an unknown workflow record."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["records"][0]["kind"] = "merged_issue"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="kind is unsupported"):
        load_snapshot(path)
