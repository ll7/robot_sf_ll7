"""Tests for the ``scenario_contract.v1`` governance schema."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from robot_sf.benchmark.scenario_contract import (
    CONTRACT_SCHEMA_VERSION,
    ScenarioContractValidationError,
    load_scenario_contract_schema,
    load_scenario_contracts,
    scenario_contract_from_dict,
    validate_scenario_contract_references,
)

FIXTURE_PATH = Path(
    "configs/scenarios/contracts/station_platform_candidate_pack_issue736_contracts.yaml"
)


def test_load_fixture_contract_as_typed_governance_payload() -> None:
    """Representative contracts should validate independently of benchmark execution."""

    contracts = load_scenario_contracts(FIXTURE_PATH)

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.schema_version == CONTRACT_SCHEMA_VERSION
    assert contract.id == "station_platform_waiting_passengers_medium_contract"
    assert (
        contract.scenario_ref.source
        == "configs/scenarios/sets/station_platform_candidate_pack_issue736.yaml"
    )
    assert contract.scenario_ref.scenario_name == "station_platform_waiting_passengers_medium"
    assert contract.certification.schema_version == "scenario_cert.v1"
    assert contract.certification.required_before_benchmark_claim is True
    assert contract.benchmark_eligibility.intended_use == "exploratory"
    assert contract.observables[0].metric == "collision_rate"

    jsonschema.validate(contract.to_dict(), load_scenario_contract_schema())


def test_invalid_contract_sections_fail_closed_with_json_paths() -> None:
    """Invalid actor, invariant, observable, and termination definitions fail closed."""

    payload = load_scenario_contracts(FIXTURE_PATH)[0].to_dict()
    invalid_cases = [
        (
            lambda candidate: candidate["actors"][0].__setitem__(
                "kind",
                "unsupported_actor_kind",
            ),
            "/actors/0/kind",
            "unsupported_actor_kind",
        ),
        (
            lambda candidate: candidate["invariants"][0].pop("description"),
            "/invariants/0",
            "description",
        ),
        (
            lambda candidate: candidate["observables"][0].__setitem__("required", "yes"),
            "/observables/0/required",
            "'yes' is not of type 'boolean'",
        ),
        (
            lambda candidate: candidate["termination_conditions"][0].__setitem__(
                "reason",
                "timeout",
            ),
            "/termination_conditions/0/reason",
            "timeout",
        ),
    ]

    for mutate, expected_path, expected_fragment in invalid_cases:
        candidate = deepcopy(payload)
        mutate(candidate)
        with pytest.raises(ScenarioContractValidationError) as exc_info:
            scenario_contract_from_dict(candidate, source=FIXTURE_PATH)
        message = str(exc_info.value)
        assert expected_path in message
        assert expected_fragment in message


def test_fixture_contract_resolves_existing_scenario_surface() -> None:
    """Contract fixtures should point at existing scenario YAML entries, not synthetic names."""

    contract = load_scenario_contracts(FIXTURE_PATH)[0]

    assert validate_scenario_contract_references(contract, repo_root=Path(".")) == []

    missing = replace(
        contract,
        scenario_ref=replace(contract.scenario_ref, scenario_name="missing_station_platform_case"),
    )
    assert validate_scenario_contract_references(missing, repo_root=Path(".")) == [
        "scenario_ref.scenario_name 'missing_station_platform_case' was not found in "
        "configs/scenarios/sets/station_platform_candidate_pack_issue736.yaml",
    ]
