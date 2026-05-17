"""Tests for the ``odd_contract.v1`` benchmark evidence-boundary schema."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from robot_sf.benchmark.odd_contract import (
    ODD_CONTRACT_SCHEMA_VERSION,
    OddContractValidationError,
    classify_odd_claim_boundary,
    load_odd_contract_schema,
    load_odd_contracts,
    odd_contract_from_dict,
    validate_odd_contract_references,
)
from robot_sf.benchmark.scenario_contract import (
    load_scenario_contracts,
    validate_scenario_odd_contract_reference,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmarks"
    / "odd_contracts"
    / "low_speed_public_space_v1.yaml"
)

SCENARIO_CONTRACT_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "scenarios"
    / "contracts"
    / "station_platform_candidate_pack_issue736_contracts.yaml"
)


def test_load_fixture_as_typed_odd_contract() -> None:
    """Representative ODD contracts should validate independently of benchmark runs."""

    contracts = load_odd_contracts(FIXTURE_PATH)

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.schema_version == ODD_CONTRACT_SCHEMA_VERSION
    assert contract.id == "low_speed_public_space_v1"
    assert contract.operating_context.environment_types == [
        "indoor_public_transit",
        "outdoor_walkway",
        "synthetic_atomic",
    ]
    assert contract.speed_limits.max_robot_speed_mps == pytest.approx(2.5)
    assert contract.pedestrian_density.density_bins == ["low", "medium", "high"]
    assert "safety_certification" in contract.claim_boundaries.non_claims

    jsonschema.validate(contract.to_dict(), load_odd_contract_schema())


def test_invalid_odd_contract_sections_fail_closed_with_json_paths() -> None:
    """Malformed ODD declarations should fail with actionable JSON-pointer paths."""

    payload = load_odd_contracts(FIXTURE_PATH)[0].to_dict()
    invalid_cases = [
        (
            lambda candidate: candidate.__setitem__("schema_version", "odd_contract.v2"),
            "/schema_version",
            "odd_contract.v1",
        ),
        (
            lambda candidate: candidate["speed_limits"].__setitem__("max_robot_speed_mps", 0),
            "/speed_limits/max_robot_speed_mps",
            "greater than 0",
        ),
        (
            lambda candidate: candidate["pedestrian_density"].__setitem__("density_bins", []),
            "/pedestrian_density/density_bins",
            "should be non-empty",
        ),
        (
            lambda candidate: candidate["claim_boundaries"].__setitem__(
                "evidence_status",
                "certified",
            ),
            "/claim_boundaries/evidence_status",
            "certified",
        ),
    ]

    for mutate, expected_path, expected_fragment in invalid_cases:
        candidate = deepcopy(payload)
        mutate(candidate)
        with pytest.raises(OddContractValidationError) as exc_info:
            odd_contract_from_dict(candidate, source=FIXTURE_PATH)
        message = str(exc_info.value)
        assert expected_path in message
        assert expected_fragment in message


def test_odd_contract_reference_validation_names_missing_files_and_ids() -> None:
    """Reference validation should distinguish missing files from missing contract IDs."""

    contract = load_odd_contracts(FIXTURE_PATH)[0]

    assert (
        validate_odd_contract_references(
            source="configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml",
            contract_id=contract.id,
            repo_root=Path("."),
        )
        == []
    )

    assert validate_odd_contract_references(
        source="configs/benchmarks/odd_contracts/missing.yaml",
        contract_id=contract.id,
        repo_root=Path("."),
    ) == ["odd_contract_ref.source 'configs/benchmarks/odd_contracts/missing.yaml' does not exist"]

    assert validate_odd_contract_references(
        source="configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml",
        contract_id="missing_odd",
        repo_root=Path("."),
    ) == [
        "odd_contract_ref.contract_id 'missing_odd' was not found in "
        "configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml",
    ]


def test_odd_contract_classifies_supported_excluded_and_unknown_claims() -> None:
    """ODD metadata should make out-of-scope claims executable, not only prose."""

    contract = load_odd_contracts(FIXTURE_PATH)[0]

    assert classify_odd_claim_boundary(contract, "benchmark_evidence_boundary") == "supported"
    assert classify_odd_claim_boundary(contract, "safety_certification") == "excluded"
    assert classify_odd_claim_boundary(contract, "public-road autonomy") == "excluded"
    assert classify_odd_claim_boundary(contract, "real_world_deployment_readiness") == "excluded"
    assert classify_odd_claim_boundary(contract, "untracked_claim") == "unknown"


def test_scenario_contract_fixture_can_reference_odd_declaration() -> None:
    """Scenario contracts should be able to point at ODD metadata by reference."""

    contract = load_scenario_contracts(SCENARIO_CONTRACT_FIXTURE)[0]

    assert contract.odd_contract_ref is not None
    assert contract.odd_contract_ref.contract_id == "low_speed_public_space_v1"
    assert validate_scenario_odd_contract_reference(contract, repo_root=Path(".")) == []

    missing = replace(
        contract,
        odd_contract_ref=replace(contract.odd_contract_ref, contract_id="missing_odd"),
    )
    assert validate_scenario_odd_contract_reference(missing, repo_root=Path(".")) == [
        "odd_contract_ref.contract_id 'missing_odd' was not found in "
        "configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml",
    ]
