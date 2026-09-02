"""Contract tests for the interaction-conditioned realism validation plan."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.pedestrian_realism_validation import INTERACTION_CLASSES
from robot_sf.benchmark.realism_validation_contract import (
    CONSTANT_VELOCITY_BASELINE,
    REALISM_VALIDATION_CONTRACT_SCHEMA_VERSION,
    RealismSyntheticClassMixRule,
    RealismValidationContractError,
    evaluate_interaction_event_counts,
    evaluate_synthetic_class_mix_recall,
    load_realism_validation_contract,
    realism_validation_contract_from_mapping,
    validate_realism_validation_contract,
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
    assert contract.synthetic_class_mix.expected_event_counts == dict.fromkeys(
        INTERACTION_CLASSES, 1
    )
    assert contract.synthetic_class_mix.minimum_per_class_recall == 1.0
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


def test_synthetic_class_mix_recall_is_an_explicit_diagnostic_acceptance_rule() -> None:
    """Known planted class counts pass only when every declared class is recovered."""

    contract = load_realism_validation_contract(CONTRACT_PATH)
    expected = contract.synthetic_class_mix.expected_event_counts

    accepted = evaluate_synthetic_class_mix_recall(expected, contract.synthetic_class_mix)

    assert accepted["status"] == "sufficient"
    assert accepted["evidence_status"] == "diagnostic-only"
    assert accepted["minimum_per_class_recall"] == 1.0
    assert all(row["recall"] == 1.0 for row in accepted["rows"].values())

    missed = dict(expected)
    missed["crossing_conflict"] = 0
    rejected = evaluate_synthetic_class_mix_recall(missed, contract.synthetic_class_mix)

    assert rejected["status"] == "insufficient_recall"
    assert rejected["rows"]["crossing_conflict"] == {
        "observed": 0,
        "expected": 1,
        "recall": 0.0,
        "minimum_recall": 1.0,
        "status": "insufficient_recall",
    }


def test_synthetic_class_mix_contract_rejects_incomplete_or_vacuous_rules() -> None:
    """Missing classes and a zero recall threshold cannot weaken the preregistration."""

    contract = load_realism_validation_contract(CONTRACT_PATH)

    missing_class = contract.to_dict()
    del missing_class["synthetic_class_mix"]["expected_event_counts"]["group"]
    with pytest.raises(RealismValidationContractError, match="expected_event_counts.*exactly"):
        realism_validation_contract_from_mapping(missing_class)

    zero_threshold = contract.to_dict()
    zero_threshold["synthetic_class_mix"]["minimum_per_class_recall"] = 0.0
    with pytest.raises(RealismValidationContractError, match="greater than 0"):
        realism_validation_contract_from_mapping(zero_threshold)


def test_synthetic_class_mix_evaluation_rejects_unknown_or_non_integer_counts() -> None:
    """Malformed observations fail closed instead of being silently ignored."""

    contract = load_realism_validation_contract(CONTRACT_PATH)
    rule = contract.synthetic_class_mix

    with pytest.raises(RealismValidationContractError, match="unknown interaction classes"):
        evaluate_synthetic_class_mix_recall({"unknown": 1}, rule)

    observed = dict.fromkeys(INTERACTION_CLASSES, 1)
    observed["group"] = True
    with pytest.raises(RealismValidationContractError, match="non-negative integer"):
        evaluate_synthetic_class_mix_recall(observed, rule)


def test_synthetic_class_mix_recall_is_bounded_and_uses_typed_rule() -> None:
    """A typed rule reports fractional recall without allowing it above one."""

    rule = RealismSyntheticClassMixRule(
        expected_event_counts=dict.fromkeys(INTERACTION_CLASSES, 2),
        minimum_per_class_recall=0.5,
    )
    observed = dict.fromkeys(INTERACTION_CLASSES, 1)
    observed["group"] = 3

    report = evaluate_synthetic_class_mix_recall(observed, rule)

    assert report["status"] == "sufficient"
    assert report["rows"]["free_walking"]["recall"] == 0.5
    assert report["rows"]["group"]["recall"] == 1.0


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


def test_validation_alias_accepts_the_shipped_contract() -> None:
    """The public mapping validator follows the same strict path as the loader."""

    contract = load_realism_validation_contract(CONTRACT_PATH)

    validated = validate_realism_validation_contract(contract.to_dict(), source="fixture")

    assert validated.to_dict() == contract.to_dict()


def test_loader_rejects_non_mapping_payload(tmp_path: Path) -> None:
    """A YAML list cannot masquerade as a realism validation contract."""

    path = tmp_path / "invalid.yaml"
    path.write_text("- not-a-contract\n", encoding="utf-8")

    with pytest.raises(RealismValidationContractError, match="must be a mapping"):
        load_realism_validation_contract(path)


def test_loader_rejects_missing_contract_file(tmp_path: Path) -> None:
    """A missing preregistration fails closed with its source path."""

    path = tmp_path / "missing.yaml"

    with pytest.raises(RealismValidationContractError, match="cannot be read"):
        load_realism_validation_contract(path)
