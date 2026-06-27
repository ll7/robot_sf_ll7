"""Tests for the dataset-backed scenario-prior staging-contract checker (issue #3161)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from robot_sf.research.scenario_prior_staging_contract import (
    CONTRACT_STATUS_BLOCKED_EXTERNAL,
    CONTRACT_STATUS_INVALID,
    CONTRACT_STATUS_READY,
    SCENARIO_PRIOR_STAGING_CONTRACT_SCHEMA_VERSION,
    STAGING_CONTRACT_EVIDENCE_BOUNDARY,
    ScenarioPriorStagingContractError,
    check_scenario_prior_staging_contract,
    load_scenario_prior_staging_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONTRACT_PATH = (
    REPO_ROOT / "configs" / "research" / "scenario_prior_staging_contract_issue_3161.yaml"
)

# Canonical comparison parameter groups used as the allowed distribution-field vocabulary in tests.
ALLOWED_GROUPS = {
    "pedestrian_density",
    "pedestrian_speed",
    "timing_offset_s",
    "spatial_offset_m",
    "route_coordinate_m",
    "clearance_distance_m",
    "episode_horizon_steps",
}


def _dataset(staging_status: str = "blocked-external-input", **overrides: object) -> dict:
    """Return one valid dataset staging entry, overridable per-test."""
    dataset = {
        "dataset_id": "sdd",
        "asset_id": "sdd",
        "title": "Stanford Drone Dataset annotations",
        "staging_status": staging_status,
        "redistribution": "none",
        "blocker_issues": [1497],
        "provenance": {
            "source_url": "https://cvgl.stanford.edu/projects/uav_data/",
            "license": "CC BY-NC-SA 3.0",
            "license_status": "license-gated",
            "citation": "Robicquet et al., ECCV 2016.",
        },
        "distribution_fields": ["pedestrian_density", "pedestrian_speed"],
    }
    dataset.update(overrides)
    return dataset


def _contract(datasets: list[dict] | None = None) -> dict:
    """Return a minimal valid staging contract."""
    return {
        "schema_version": SCENARIO_PRIOR_STAGING_CONTRACT_SCHEMA_VERSION,
        "contract_id": "test_contract",
        "issue": 3161,
        "claim_boundary": "staging contract metadata only",
        "authored_baseline": "configs/research/scenario_prior_cards_issue_2917.yaml",
        "comparison_harness": "scripts/analysis/compare_scenario_priors_issue_2919.py",
        "datasets": datasets if datasets is not None else [_dataset()],
    }


def test_blocked_external_input_contract_is_blocked_not_ready() -> None:
    """A well-formed contract with no staged dataset reports blocked-external-input."""
    report = check_scenario_prior_staging_contract(
        _contract(), allowed_distribution_groups=ALLOWED_GROUPS
    )
    assert report.contract_status == CONTRACT_STATUS_BLOCKED_EXTERNAL
    assert report.dataset_backed_comparison_allowed is False
    assert report.comparison_ready_datasets == []
    assert report.evidence_boundary == STAGING_CONTRACT_EVIDENCE_BOUNDARY
    assert report.blockers == []


def test_staged_clean_dataset_unlocks_comparison() -> None:
    """A staged, blocker-free dataset makes the contract ready and comparison-allowed."""
    report = check_scenario_prior_staging_contract(
        _contract([_dataset(staging_status="staged")]),
        allowed_distribution_groups=ALLOWED_GROUPS,
    )
    assert report.contract_status == CONTRACT_STATUS_READY
    assert report.dataset_backed_comparison_allowed is True
    assert report.comparison_ready_datasets == ["sdd"]


def test_missing_dataset_is_not_comparison_ready() -> None:
    """A 'missing' dataset is well-formed but not comparison-ready."""
    report = check_scenario_prior_staging_contract(
        _contract([_dataset(staging_status="missing")]),
        allowed_distribution_groups=ALLOWED_GROUPS,
    )
    assert report.contract_status == CONTRACT_STATUS_BLOCKED_EXTERNAL
    assert report.comparison_ready_datasets == []
    dataset = report.datasets[0]
    assert dataset.effective_staged is False
    assert dataset.blockers == []


def test_unknown_distribution_field_fails_closed() -> None:
    """A distribution field outside the comparison vocabulary is an invalid-contract blocker."""
    report = check_scenario_prior_staging_contract(
        _contract([_dataset(distribution_fields=["pedestrian_speed", "not_a_real_group"])]),
        allowed_distribution_groups=ALLOWED_GROUPS,
    )
    assert report.contract_status == CONTRACT_STATUS_INVALID
    assert report.datasets[0].unknown_distribution_fields == ["not_a_real_group"]
    assert any("not_a_real_group" in blocker for blocker in report.blockers)


def test_distribution_field_check_skipped_without_vocabulary() -> None:
    """Without a vocabulary, unknown fields are tolerated (no drift check)."""
    report = check_scenario_prior_staging_contract(
        _contract([_dataset(distribution_fields=["anything_goes"])]),
        allowed_distribution_groups=None,
    )
    assert report.datasets[0].unknown_distribution_fields == []
    assert report.contract_status == CONTRACT_STATUS_BLOCKED_EXTERNAL


def test_blocked_dataset_without_blocker_issue_is_invalid() -> None:
    """A blocked-external-input dataset must name at least one blocker issue."""
    report = check_scenario_prior_staging_contract(
        _contract([_dataset(blocker_issues=[])]),
        allowed_distribution_groups=ALLOWED_GROUPS,
    )
    assert report.contract_status == CONTRACT_STATUS_INVALID
    assert any("blocker_issues" in blocker for blocker in report.blockers)


def test_declared_staged_but_live_missing_fails_closed() -> None:
    """A dataset declared 'staged' whose live probe is missing fails closed."""
    report = check_scenario_prior_staging_contract(
        _contract([_dataset(staging_status="staged")]),
        allowed_distribution_groups=ALLOWED_GROUPS,
        live_staging_status={"sdd": "missing"},
    )
    assert report.contract_status == CONTRACT_STATUS_INVALID
    dataset = report.datasets[0]
    assert dataset.effective_staged is False
    assert dataset.comparison_ready is False
    assert any("live probe" in blocker for blocker in report.blockers)


def test_declared_staged_and_live_available_is_ready() -> None:
    """A dataset declared 'staged' and live-available reconciles to comparison-ready."""
    report = check_scenario_prior_staging_contract(
        _contract([_dataset(staging_status="staged")]),
        allowed_distribution_groups=ALLOWED_GROUPS,
        live_staging_status={"sdd": "available"},
    )
    assert report.contract_status == CONTRACT_STATUS_READY
    assert report.datasets[0].live_staging_status == "available"
    assert report.comparison_ready_datasets == ["sdd"]


def test_schema_violation_raises() -> None:
    """An invalid payload (bad enum) raises a schema error."""
    bad = _contract([_dataset(staging_status="totally-invalid")])
    with pytest.raises(ScenarioPriorStagingContractError):
        check_scenario_prior_staging_contract(bad, allowed_distribution_groups=ALLOWED_GROUPS)


def test_missing_required_field_raises() -> None:
    """A dataset missing a required provenance field raises a schema error."""
    bad = _contract()
    del bad["datasets"][0]["provenance"]["citation"]
    with pytest.raises(ScenarioPriorStagingContractError):
        check_scenario_prior_staging_contract(bad, allowed_distribution_groups=ALLOWED_GROUPS)


def test_report_to_dict_is_json_serializable() -> None:
    """The report serializes to JSON (CLI contract)."""
    report = check_scenario_prior_staging_contract(
        _contract(), allowed_distribution_groups=ALLOWED_GROUPS
    )
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["schema_version"] == SCENARIO_PRIOR_STAGING_CONTRACT_SCHEMA_VERSION
    assert payload["datasets"][0]["dataset_id"] == "sdd"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Loading a non-existent contract path raises an actionable error."""
    with pytest.raises(ScenarioPriorStagingContractError):
        load_scenario_prior_staging_contract(tmp_path / "nope.yaml")


def test_load_roundtrip(tmp_path: Path) -> None:
    """A contract written to disk loads and validates."""
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(_contract(), sort_keys=False), encoding="utf-8")
    loaded = load_scenario_prior_staging_contract(path)
    assert loaded["contract_id"] == "test_contract"


def test_repo_example_contract_is_valid_and_blocked() -> None:
    """The shipped #3161 example contract validates and is blocked-external-input today."""
    contract = load_scenario_prior_staging_contract(EXAMPLE_CONTRACT_PATH)
    report = check_scenario_prior_staging_contract(
        contract, allowed_distribution_groups=ALLOWED_GROUPS
    )
    assert report.contract_status == CONTRACT_STATUS_BLOCKED_EXTERNAL
    assert report.dataset_backed_comparison_allowed is False
    assert report.blockers == []
    assert {d.dataset_id for d in report.datasets} == {"sdd", "socnavbench_eth", "amv"}


def test_repo_example_distribution_fields_match_live_harness_vocabulary() -> None:
    """Every example distribution field is a real #2919 comparison parameter group.

    This guards against the example contract drifting from the comparison harness
    it feeds; it imports the live PARAMETER_GROUPS rather than the test constant.
    """
    from scripts.analysis.compare_scenario_priors_issue_2919 import PARAMETER_GROUPS

    contract = load_scenario_prior_staging_contract(EXAMPLE_CONTRACT_PATH)
    report = check_scenario_prior_staging_contract(
        contract, allowed_distribution_groups=set(PARAMETER_GROUPS)
    )
    for dataset in report.datasets:
        assert dataset.unknown_distribution_fields == []


def test_example_contract_matches_repo_path() -> None:
    """The example contract ships at the path the CLI defaults to."""
    assert EXAMPLE_CONTRACT_PATH.is_file()
    payload = yaml.safe_load(EXAMPLE_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCENARIO_PRIOR_STAGING_CONTRACT_SCHEMA_VERSION


def test_multiple_datasets_one_staged_is_ready() -> None:
    """With several datasets, one staged-and-clean entry unlocks the comparison."""
    datasets = [
        _dataset(dataset_id="sdd", asset_id="sdd", staging_status="staged"),
        _dataset(
            dataset_id="amv",
            asset_id="amv-calibration",
            staging_status="blocked-external-input",
            blocker_issues=[2000],
            distribution_fields=["pedestrian_speed", "timing_offset_s"],
        ),
    ]
    report = check_scenario_prior_staging_contract(
        _contract(datasets), allowed_distribution_groups=ALLOWED_GROUPS
    )
    assert report.contract_status == CONTRACT_STATUS_READY
    assert report.comparison_ready_datasets == ["sdd"]


def test_copy_independence() -> None:
    """The checker does not mutate the input contract mapping."""
    contract = _contract()
    snapshot = copy.deepcopy(contract)
    check_scenario_prior_staging_contract(contract, allowed_distribution_groups=ALLOWED_GROUPS)
    assert contract == snapshot
