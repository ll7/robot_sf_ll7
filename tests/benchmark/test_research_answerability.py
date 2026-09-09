"""Tests for fail-closed research answerability and yield reporting."""

from __future__ import annotations

import copy
import hashlib
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
from robot_sf.benchmark import research_answerability as answerability_module
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
    ("mutator", "message"),
    [
        (lambda contract: contract.update({"question": None}), "must be a mapping"),
        (lambda contract: contract.update({"producers": []}), "must be a non-empty list"),
        (
            lambda contract: contract["question"].update({"decision_vocabulary": [1]}),
            "only non-empty strings",
        ),
        (
            lambda contract: contract["question"].update({"decision_vocabulary": ["pause"]}),
            "unsupported values",
        ),
        (
            lambda contract: contract["producers"][0].update({"status": "unknown"}),
            "status must be one of",
        ),
        (
            lambda contract: contract["producers"][0].update({"execution_mode": "unknown"}),
            "execution_mode must be one of",
        ),
        (
            lambda contract: contract["producers"][0].update({"required": "yes"}),
            "required must be a boolean",
        ),
        (
            lambda contract: contract["analysis"].update({"dry_run_status": "unknown-value"}),
            "dry_run_status must be",
        ),
        (
            lambda contract: contract["analysis"].update({"comparability_status": "unknown-value"}),
            "comparability_status must be",
        ),
        (
            lambda contract: contract["design"].update({"mode": "unknown-value"}),
            "design.mode must be",
        ),
        (
            lambda contract: contract["design"].update({"power_status": "unknown-value"}),
            "design.power_status must be",
        ),
        (
            lambda contract: contract["artifacts"].update({"checksums": []}),
            "checksums must be",
        ),
        (
            lambda contract: contract["artifacts"].update({"durability_status": "unknown-value"}),
            "durability_status must be",
        ),
    ],
)
def test_answerability_contract_validation_rejects_invalid_values(mutator, message: str) -> None:
    """Structural answerability fields fail closed with actionable messages."""
    contract = _example_contract()
    mutator(contract)

    result = evaluate_answerability(contract)

    assert result.state == "invalid_contract"
    assert message in result.reasons[0]


def test_answerability_validation_rejects_non_mapping_and_wrong_schema() -> None:
    """The evaluator distinguishes non-mappings and unsupported schema versions."""
    assert evaluate_answerability([]).state == "invalid_contract"

    contract = _example_contract()
    contract["schema_version"] = "research_answerability.v0"
    result = evaluate_answerability(contract)

    assert result.state == "invalid_contract"
    assert "schema_version" in result.reasons[0]


def test_unknown_power_status_is_a_blocked_underpowered_design() -> None:
    """A declared but unknown power classification cannot become answerable."""
    contract = _example_contract()
    contract["design"]["power_status"] = "unknown"

    result = evaluate_answerability(contract)

    assert result.state == "blocked_underpowered"
    assert "unknown" in result.reasons[0]


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


def test_manifest_with_non_mapping_answerability_is_invalid() -> None:
    """A malformed optional manifest section is reported, not ignored."""
    result = answerability_from_manifest({"answerability": []})

    assert result["state"] == "invalid_contract"
    assert result["decision_capable"] is False


def test_manifest_answerability_delegates_to_contract_evaluator() -> None:
    """A declared manifest section uses the same answerability state machine."""
    result = answerability_from_manifest({"answerability": _example_contract()})

    assert result["state"] == "diagnostic_only"
    assert result["schema_version"] == "research_answerability.v1"


def _issue_8570_packet() -> dict[str, object]:
    """Load the checked-in packet for mutation-based validator tests."""
    return load_adversarial_falsification_packet(ADVERSARIAL_PACKET)


def _refresh_packet_digest(packet: dict[str, object]) -> None:
    """Keep a mutated fixture self-consistent so semantic checks are reached."""
    packet["self_digest"] = compute_adversarial_falsification_packet_digest(packet)


def _source_semantics() -> dict[str, object]:
    """Load the frozen search-space semantics used by private binding checks."""
    path = REPO_ROOT / "configs/adversarial/issue_7340_station_platform_search_space_v1.yaml"
    return answerability_module._packet_source_semantics(path)


def _assert_private_packet_error(
    packet: dict[str, object], validator, mutation, message: str
) -> None:
    """Apply one semantic mutation and assert its fail-closed private validator."""
    mutation(packet)
    with pytest.raises(AdversarialFalsificationPacketError, match=message):
        validator(packet)


def _scenario_source_packet(tmp_path: Path, payload: object) -> dict[str, object]:
    """Point a packet at a temporary scenario-template payload for source checks."""
    packet = _issue_8570_packet()
    scenario_path = tmp_path / "scenario-template.yaml"
    scenario_path.write_text(
        payload if isinstance(payload, str) else yaml.safe_dump(payload), encoding="utf-8"
    )
    packet["source"]["inputs"]["scenario_template"]["sha256"] = hashlib.sha256(
        scenario_path.read_bytes()
    ).hexdigest()
    original_resolver = answerability_module._resolve_packet_source_path

    def resolve_source(value, *, repo_root: Path, field: str) -> Path:
        if field == "source.inputs.scenario_template.path":
            return scenario_path
        return original_resolver(value, repo_root=repo_root, field=field)

    packet["_scenario_resolver"] = resolve_source
    return packet


def _validate_scenario_source_packet(packet: dict[str, object]) -> None:
    """Validate a temporary scenario payload using the packet source checker."""
    resolver = packet.pop("_scenario_resolver")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(answerability_module, "_resolve_packet_source_path", resolver)
        answerability_module._validate_packet_sources(packet, repo_root=REPO_ROOT)


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


def test_issue_8570_packet_rejects_missing_runtime_effectiveness_note() -> None:
    """Provenance-only variables must explain why they are not runtime inputs."""
    packet = _issue_8570_packet()
    packet["variable_map"]["pedestrian_delay_s"]["effectiveness_note"] = (
        "This value is documented for analysis."
    )
    _refresh_packet_digest(packet)

    with pytest.raises(AdversarialFalsificationPacketError, match="explain its missing"):
        validate_adversarial_falsification_packet(packet)


def test_packet_digest_and_schema_loaders_fail_closed(tmp_path: Path, monkeypatch) -> None:
    """Digest and schema loader failures remain explicit instead of being swallowed."""
    with pytest.raises(AdversarialFalsificationPacketError, match="packet must be a mapping"):
        compute_adversarial_falsification_packet_digest([])
    with pytest.raises(AdversarialFalsificationPacketError, match="canonically serialized"):
        compute_adversarial_falsification_packet_digest({"value": object()})

    malformed_schema = tmp_path / "malformed-schema.json"
    malformed_schema.write_text("{", encoding="utf-8")
    monkeypatch.setattr(answerability_module, "_PACKET_SCHEMA_FILE", malformed_schema)
    with pytest.raises(AdversarialFalsificationPacketError, match="cannot load"):
        answerability_module.load_adversarial_falsification_packet_schema()

    non_object_schema = tmp_path / "non-object-schema.json"
    non_object_schema.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(answerability_module, "_PACKET_SCHEMA_FILE", non_object_schema)
    with pytest.raises(AdversarialFalsificationPacketError, match="must be a JSON object"):
        answerability_module.load_adversarial_falsification_packet_schema()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "must be a non-empty path"),
        ("/absolute/input.yaml", "must be repository-relative"),
        ("../outside.yaml", "escapes the repository root"),
        ("missing/input.yaml", "does not resolve to a file"),
    ],
)
def test_packet_source_path_resolution_is_fail_closed(
    tmp_path: Path, value: str, message: str
) -> None:
    """Packet source paths cannot be empty, absolute, escaping, or missing."""
    with pytest.raises(AdversarialFalsificationPacketError, match=message):
        answerability_module._resolve_packet_source_path(
            value, repo_root=tmp_path, field="source.input.path"
        )


def test_packet_source_validation_rejects_malformed_digest() -> None:
    """Source digest syntax is checked before reading source bytes."""
    packet = _issue_8570_packet()
    packet["source"]["inputs"]["search_space"]["sha256"] = "not-a-sha"

    with pytest.raises(AdversarialFalsificationPacketError, match="lowercase SHA-256"):
        answerability_module._validate_packet_sources(packet, repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[", "cannot load source scenario template"),
        (["item"], "scenario template must be a mapping"),
        ({"scenarios": []}, "first scenario mapping"),
    ],
)
def test_packet_source_scenario_shape_is_validated(
    tmp_path: Path, payload: object, message: str
) -> None:
    """Scenario-template parsing fails closed for malformed source content."""
    packet = _scenario_source_packet(tmp_path, payload)

    with pytest.raises(AdversarialFalsificationPacketError, match=message):
        _validate_scenario_source_packet(packet)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda scenario: scenario.update({"name": "other-scenario"}), "scenario_name"),
        (lambda scenario: scenario.update({"single_pedestrians": []}), "pedestrian binding"),
        (lambda scenario: scenario.pop("map_file"), "map_file is missing"),
        (
            lambda scenario: scenario.update({"map_file": "../../maps/other.svg"}),
            "does not match the scenario template map_file",
        ),
    ],
)
def test_packet_source_scenario_bindings_are_validated(
    tmp_path: Path, mutation, message: str
) -> None:
    """Scenario name, pedestrian, and map bindings cannot drift from the packet."""
    source_path = REPO_ROOT / "configs/adversarial/issue_7340_station_platform_medium_v1.yaml"
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    payload["scenarios"][0]["map_file"] = str(
        REPO_ROOT / "maps/svg_maps/classic_station_platform.svg"
    )
    mutation(payload["scenarios"][0])
    packet = _scenario_source_packet(tmp_path, payload)

    with pytest.raises(AdversarialFalsificationPacketError, match=message):
        _validate_scenario_source_packet(packet)


def test_packet_source_binding_rejects_route_mode_drift(tmp_path: Path) -> None:
    """The packet's declared pedestrian route mode remains bound to source semantics."""
    source_path = REPO_ROOT / "configs/adversarial/issue_7340_station_platform_medium_v1.yaml"
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    payload["scenarios"][0]["map_file"] = str(
        REPO_ROOT / "maps/svg_maps/classic_station_platform.svg"
    )
    packet = _scenario_source_packet(tmp_path, payload)
    packet["source"]["scenario_template"]["pedestrian_route_mode"] = "waypoint"

    with pytest.raises(AdversarialFalsificationPacketError, match="route_mode"):
        _validate_scenario_source_packet(packet)


def test_packet_source_validation_rejects_malformed_owner_digest() -> None:
    """Code-owner digests must be syntactically valid before byte comparison."""
    packet = _issue_8570_packet()
    packet["source"]["owners"][0]["sha256"] = "not-a-sha"

    with pytest.raises(AdversarialFalsificationPacketError, match=r"owners\[0\].sha256"):
        answerability_module._validate_packet_sources(packet, repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda packet: packet.update({"variable_order": ["start_y"]}),
            "variable_order",
        ),
        (
            lambda packet: packet["variable_map"].update({"unexpected": {}}),
            "variable_map keys",
        ),
        (
            lambda packet: packet["variable_map"]["start_x"].update({"actor": "pedestrian"}),
            "variable_map.start_x.actor",
        ),
        (
            lambda packet: packet["variable_map"]["start_x"].update({"effectiveness_note": ""}),
            "effectiveness_note",
        ),
    ],
)
def test_packet_variable_bindings_fail_closed(mutation, message: str) -> None:
    """Variable order, keys, bindings, and effectiveness notes are immutable contracts."""
    packet = _issue_8570_packet()

    _assert_private_packet_error(
        packet,
        lambda candidate: answerability_module._validate_packet_variable_map(
            candidate, search_space_semantics=_source_semantics()
        ),
        mutation,
        message,
    )


def test_packet_variable_binding_requires_provenance_note_for_delay() -> None:
    """Template-mode delay metadata must explain its provenance-only status."""
    packet = _issue_8570_packet()
    packet["variable_map"]["pedestrian_delay_s"]["effectiveness_note"] = (
        "This is a runtime delay input."
    )
    _refresh_packet_digest(packet)

    with pytest.raises(AdversarialFalsificationPacketError, match="explain its missing"):
        validate_adversarial_falsification_packet(packet)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda packet: packet["feasibility"].update({"contract_version": "v0"}),
            "contract_version",
        ),
        (
            lambda packet: packet["feasibility"].update({"predicate_order": ["wrong"]}),
            "predicate_order",
        ),
        (
            lambda packet: packet["feasibility"]["predicates"][0].update({"name": "wrong"}),
            "predicates",
        ),
        (
            lambda packet: packet["feasibility"]["rejection_accounting"].update(
                {"owner": "other.owner"}
            ),
            "ledger owner",
        ),
        (
            lambda packet: packet["feasibility"]["rejection_accounting"].update(
                {"pre_simulation": False}
            ),
            "pre-simulation",
        ),
    ],
)
def test_packet_feasibility_contract_is_reused_exactly(mutation, message: str) -> None:
    """The packet cannot fork predicate order or rejection-ledger ownership."""
    packet = _issue_8570_packet()

    _assert_private_packet_error(
        packet, answerability_module._validate_packet_feasibility, mutation, message
    )


def test_packet_feasibility_contract_detects_owner_order_drift(monkeypatch) -> None:
    """A changed shared feasibility vocabulary invalidates the packet."""
    packet = _issue_8570_packet()
    monkeypatch.setattr(
        "robot_sf.adversarial.feasibility_first.CHECK_NAMES",
        ("changed",),
    )

    with pytest.raises(AdversarialFalsificationPacketError, match="CHECK_NAMES"):
        answerability_module._validate_packet_feasibility(packet)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda packet: packet["objective"]["ordering"][0].update({"name": "wrong"}),
            "objective.ordering",
        ),
        (
            lambda packet: packet["objective"].update({"scalarization": "weighted"}),
            "lexicographic",
        ),
        (
            lambda packet: packet["seed_policy"].update(
                {"search_seeds": packet["seed_policy"]["search_seeds"][:2]}
            ),
            "three search and five",
        ),
        (
            lambda packet: packet["seed_policy"].update({"search_seeds": [301, 301, 302]}),
            "unique",
        ),
        (
            lambda packet: packet["seed_policy"].update({"candidate_seed_mode": "random"}),
            "index_derived",
        ),
        (
            lambda packet: packet["budget"].update({"search_candidate_rows_per_arm": 1}),
            "per_arm",
        ),
        (
            lambda packet: packet["compute_ceiling"].update({"max_search_candidate_rows": 1}),
            "max_search_candidate_rows",
        ),
        (
            lambda packet: packet["compute_ceiling"].update({"max_confirmation_seeds": 1}),
            "max_confirmation_seeds",
        ),
        (
            lambda packet: packet["compute_ceiling"].update({"max_steps_per_rollout": 1}),
            "max_steps_per_rollout",
        ),
        (
            lambda packet: packet["search_methods"].update(
                {"primary": {**packet["search_methods"]["primary"], "id": "other"}}
            ),
            "search_methods",
        ),
        (
            lambda packet: packet["search_methods"]["primary"].update({"budget_per_seed": 1}),
            "equal-budget",
        ),
        (
            lambda packet: packet["search_methods"].update({"equal_budget": False}),
            "equal_budget",
        ),
        (
            lambda packet: packet["budget"].update({"rollouts_per_candidate": 2}),
            "rollout",
        ),
    ],
)
def test_packet_design_budget_contract_fails_closed(mutation, message: str) -> None:
    """Objective, seed, budget, and search-method drift cannot authorize compute."""
    packet = _issue_8570_packet()

    _assert_private_packet_error(
        packet, answerability_module._validate_packet_design, mutation, message
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda packet: packet["stop_rule"]["search"].update({"action": "continue"}),
            "stop_rule.search",
        ),
        (
            lambda packet: packet["stop_rule"]["confirmation"].update(
                {"execution_status": "authorized"}
            ),
            "confirmation",
        ),
        (
            lambda packet: packet["stop_rule"]["contract_failure"].update({"action": "continue"}),
            "contract_failure",
        ),
        (
            lambda packet: packet["stop_rule"].update({"replacement_rows_allowed": True}),
            "replacement_rows_allowed",
        ),
    ],
)
def test_packet_stop_rule_is_nonadaptive_and_fail_closed(mutation, message: str) -> None:
    """No-result outcomes and fixed budgets cannot be replaced adaptively."""
    packet = _issue_8570_packet()

    _assert_private_packet_error(
        packet, answerability_module._validate_packet_stop_rule, mutation, message
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda packet: packet["outcome_vocabulary"].pop("null"),
            "outcome_vocabulary",
        ),
        (
            lambda packet: packet["outcome_vocabulary"].update({"null": ""}),
            "meanings",
        ),
    ],
)
def test_packet_outcome_vocabulary_remains_complete(mutation, message: str) -> None:
    """Explicit result and no-result states cannot be removed or left undefined."""
    packet = _issue_8570_packet()

    _assert_private_packet_error(
        packet, answerability_module._validate_packet_outcomes, mutation, message
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda packet: packet["compute_authorization"].update({"gate": "other.v1"}),
            "compute_authorization.gate",
        ),
        (
            lambda packet: packet["compute_authorization"].update(
                {"evaluated_state": "answerable"}
            ),
            "evaluated_state",
        ),
        (
            lambda packet: packet["compute_authorization"].update({"authorized": True}),
            "derived fail-closed",
        ),
        (
            lambda packet: packet["compute_authorization"].update({"status": "authorized"}),
            "status does not match",
        ),
        (
            lambda packet: packet["compute_authorization"].update({"blocking_reasons": []}),
            "blocking_reasons",
        ),
        (
            lambda packet: packet["compute_authorization"].update(
                {"blocking_reasons": ["some other reason"]}
            ),
            "must name the non-runtime",
        ),
        (
            lambda packet: packet["compute_authorization"].update({"future_adapter_condition": ""}),
            "future_adapter_condition",
        ),
    ],
)
def test_packet_compute_authorization_is_derived_and_explicit(mutation, message: str) -> None:
    """The compute gate must agree with answerability and runtime effectiveness."""
    packet = _issue_8570_packet()

    _assert_private_packet_error(
        packet, answerability_module._validate_packet_compute_gate, mutation, message
    )


def test_packet_loader_rejects_invalid_yaml_and_non_mapping(tmp_path: Path) -> None:
    """Packet loading reports malformed YAML and non-object documents."""
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("[", encoding="utf-8")
    with pytest.raises(AdversarialFalsificationPacketError, match="cannot load"):
        load_adversarial_falsification_packet(malformed)

    non_mapping = tmp_path / "scalar.yaml"
    non_mapping.write_text("- item\n", encoding="utf-8")
    with pytest.raises(AdversarialFalsificationPacketError, match="must be a mapping"):
        load_adversarial_falsification_packet(non_mapping)


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
