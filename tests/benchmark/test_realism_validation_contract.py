"""Contract tests for the interaction-conditioned realism validation plan."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.pedestrian_realism_validation import INTERACTION_CLASSES
from robot_sf.benchmark.realism_validation_contract import (
    CONSTANT_VELOCITY_BASELINE,
    REALISM_VALIDATION_CONTRACT_SCHEMA_VERSION,
    RealismValidationContractError,
    evaluate_interaction_event_counts,
    load_realism_validation_contract,
    realism_validation_contract_from_mapping,
)
from robot_sf.sim.pedestrian_model_variants import SOCIAL_FORCE_DEFAULT, SUPPORTED_PEDESTRIAN_MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs" / "benchmark" / "realism_validation_contract.v1.yaml"


def test_shipped_contract_freezes_held_out_realism_validation_plan() -> None:
    """The checked-in plan names disjoint scenes, valid baselines, and all event floors."""

    contract = load_realism_validation_contract(CONTRACT_PATH)

    assert contract.schema_version == REALISM_VALIDATION_CONTRACT_SCHEMA_VERSION
    assert contract.status == "blocked-external-input"
    assert set(contract.calibration_scenes).isdisjoint(contract.held_out_scenes)
    assert contract.baseline_arms[0] == "constant_velocity"
    assert CONSTANT_VELOCITY_BASELINE in contract.baseline_arms
    assert SOCIAL_FORCE_DEFAULT in contract.baseline_arms
    assert set(SUPPORTED_PEDESTRIAN_MODELS).issubset(contract.baseline_arms)
    assert set(contract.minimum_event_counts) == set(INTERACTION_CLASSES)
    assert contract.promotion_rule.comparator_arm == SOCIAL_FORCE_DEFAULT
    assert contract.promotion_rule.held_out_only is True
    assert contract.segmentation.frame_window_s > 0.0


def test_overlapping_calibration_and_held_out_scenes_fail_closed() -> None:
    """A scene cannot contribute to both calibration and confirmation partitions."""

    contract = load_realism_validation_contract(CONTRACT_PATH)
    payload = contract.to_dict()
    payload["held_out_scenes"] = [contract.calibration_scenes[0]]

    with pytest.raises(RealismValidationContractError, match="overlap"):
        realism_validation_contract_from_mapping(payload)


def test_interaction_event_floors_keep_insufficient_classes_explicit() -> None:
    """A sparse class is reported as insufficient instead of being averaged away."""

    contract = load_realism_validation_contract(CONTRACT_PATH)
    counts = dict.fromkeys(INTERACTION_CLASSES, 20)
    counts["crossing_conflict"] = 2

    report = evaluate_interaction_event_counts(counts, contract.minimum_event_counts)

    assert report["status"] == "insufficient_events"
    assert report["rows"]["crossing_conflict"] == {
        "observed": 2,
        "minimum": 10,
        "status": "insufficient_events",
    }
    assert report["rows"]["free_walking"]["status"] == "sufficient"


def test_orca_cannot_enter_the_pedestrian_model_baseline_hierarchy() -> None:
    """Planner names are rejected where the contract requires pedestrian models."""

    contract = load_realism_validation_contract(CONTRACT_PATH)
    payload = contract.to_dict()
    payload["baseline_arms"] = [*contract.baseline_arms, "orca"]

    with pytest.raises(RealismValidationContractError, match="ORCA"):
        realism_validation_contract_from_mapping(payload)


def test_contract_rejects_unknown_lifecycle_status() -> None:
    """A typo cannot make a preregistration appear ready or blocked."""

    contract = load_realism_validation_contract(CONTRACT_PATH)
    payload = contract.to_dict()
    payload["status"] = "blocked_external_input"

    with pytest.raises(RealismValidationContractError, match="status must be one of"):
        realism_validation_contract_from_mapping(payload)


def test_contract_requires_constant_velocity_comparator_baseline() -> None:
    """The declared hierarchy must retain the constant-velocity reference arm."""

    contract = load_realism_validation_contract(CONTRACT_PATH)
    payload = contract.to_dict()
    payload["baseline_arms"] = [
        arm for arm in contract.baseline_arms if arm != CONSTANT_VELOCITY_BASELINE
    ]

    with pytest.raises(RealismValidationContractError, match="constant_velocity"):
        realism_validation_contract_from_mapping(payload)
