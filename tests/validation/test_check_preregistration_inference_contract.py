"""Tests for the shared preregistration inference-contract checker."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from scripts.validation.check_preregistration_inference_contract import (
    InferenceContractError,
    check_inference_contract,
    check_yaml_file,
    main,
)


def _minimal_valid_packet() -> dict[str, object]:
    """Smallest packet that passes inference_contract validation."""
    return {
        "schema_version": "test.v1",
        "inference_contract": {
            "resampling_unit": {
                "method": "scenario-clustered hierarchical bootstrap",
                "rationale": (
                    "Treat scenarios as outer resampling unit because between-scenario "
                    "heterogeneity is large and we generalize to unseen scenarios."
                ),
            },
            "inference_population": {
                "type": "sampled_population",
                "rationale": (
                    "The 48 scenarios represent a sample from a broader scenario population, "
                    "not a fixed benchmark suite."
                ),
            },
            "estimand": {
                "type": "paired_delta",
                "description": (
                    "Paired per-episode delta between treatment and control, "
                    "aggregated via hierarchical bootstrap."
                ),
            },
            "decision_rule": {
                "rule": "CI-excludes-zero",
                "threshold": (
                    "The 95% hierarchical bootstrap CI for the paired delta must exclude "
                    "zero for the hypothesis to be supported."
                ),
            },
            "primary_metrics": {
                "metrics": ["collision_free_completion", "near_miss_exposure_normalized"],
                "ordered_by_importance": True,
            },
            "multiplicity_handling": {
                "strategy": (
                    "Holm-Bonferroni adjustment across the two primary metrics to "
                    "control family-wise error rate at alpha=0.05."
                ),
                "rationale": (
                    "Multiple primary metrics increase Type I error risk without "
                    "adjustment; Holm procedure maintains power over Bonferroni."
                ),
            },
        },
    }


def _fixed_suite_packet() -> dict[str, object]:
    """A packet that uses a fixed-suite inference population."""
    pkt = _minimal_valid_packet()
    pkt["inference_contract"]["inference_population"]["type"] = "fixed_suite"
    return pkt


def test_minimal_valid_packet_passes() -> None:
    """A complete inference contract with sampled population validates correctly."""
    result = check_inference_contract(_minimal_valid_packet())
    assert result["status"] == "ok"
    assert result["population_type"] == "sampled_population"
    assert result["metric_count"] == 2


def test_fixed_suite_population_passes() -> None:
    """A fixed-suite inference population is accepted."""
    result = check_inference_contract(_fixed_suite_packet())
    assert result["status"] == "ok"
    assert result["population_type"] == "fixed_suite"


def test_all_valid_estimand_types_pass() -> None:
    """Each valid estimand type (paired_delta, per_arm, ratio, both) is accepted."""
    for est_type in ("paired_delta", "per_arm_interval", "ratio", "paired_delta_and_per_arm"):
        pkt = _minimal_valid_packet()
        pkt["inference_contract"]["estimand"]["type"] = est_type
        result = check_inference_contract(pkt)
        assert result["status"] == "ok", f"estimand type {est_type} should pass"


def test_optional_fields_accepted() -> None:
    """Optional fields (bootstrap_confidence, resampling_order, adjustment_method) pass."""
    pkt = _minimal_valid_packet()
    pkt["inference_contract"]["resampling_unit"]["bootstrap_confidence"] = 0.95
    pkt["inference_contract"]["resampling_unit"]["resampling_order"] = "scenario-seed"
    pkt["inference_contract"]["multiplicity_handling"]["adjustment_method"] = "holm_bonferroni"
    result = check_inference_contract(pkt)
    assert result["status"] == "ok"


def test_missing_inference_contract_section() -> None:
    """A packet without inference_contract raises an error."""
    pkt = {"schema_version": "test.v1"}
    with pytest.raises(InferenceContractError, match="inference_contract"):
        check_inference_contract(pkt)


def test_inference_contract_not_a_mapping() -> None:
    """A string inference_contract value raises a type error."""
    pkt = {"schema_version": "test.v1", "inference_contract": "not a mapping"}
    with pytest.raises(InferenceContractError, match="inference_contract.*must be a mapping"):
        check_inference_contract(pkt)


def test_missing_resampling_unit() -> None:
    """An empty resampling_unit dict raises an error."""
    pkt = _minimal_valid_packet()
    pkt["inference_contract"]["resampling_unit"] = {}
    with pytest.raises(InferenceContractError, match="resampling_unit.method"):
        check_inference_contract(pkt)


def test_empty_resampling_unit_method() -> None:
    """A blank resampling_unit method is rejected."""
    pkt = _minimal_valid_packet()
    pkt["inference_contract"]["resampling_unit"]["method"] = ""
    pkt["inference_contract"]["resampling_unit"]["rationale"] = "some rationale text here"
    with pytest.raises(InferenceContractError, match="resampling_unit.method"):
        check_inference_contract(pkt)


def test_short_resampling_unit_rationale() -> None:
    """A rationale shorter than five characters is rejected."""
    pkt = _minimal_valid_packet()
    pkt["inference_contract"]["resampling_unit"]["rationale"] = "OK"
    with pytest.raises(InferenceContractError, match="resampling_unit.rationale"):
        check_inference_contract(pkt)


def test_invalid_bootstrap_confidence() -> None:
    """A bootstrap_confidence outside [0.8, 0.99] is rejected."""
    pkt = _minimal_valid_packet()
    pkt["inference_contract"]["resampling_unit"]["bootstrap_confidence"] = 0.5
    with pytest.raises(InferenceContractError, match="bootstrap_confidence"):
        check_inference_contract(pkt)


def test_missing_inference_population() -> None:
    """Deleting inference_population raises an error."""
    pkt = _minimal_valid_packet()
    del pkt["inference_contract"]["inference_population"]
    with pytest.raises(InferenceContractError, match="inference_population"):
        check_inference_contract(pkt)


def test_invalid_population_type() -> None:
    """An unknown inference_population type is rejected."""
    pkt = _minimal_valid_packet()
    pkt["inference_contract"]["inference_population"]["type"] = "unknown_type"
    with pytest.raises(InferenceContractError, match="inference_population.type"):
        check_inference_contract(pkt)


def test_missing_estimand() -> None:
    """Deleting estimand raises an error."""
    pkt = _minimal_valid_packet()
    del pkt["inference_contract"]["estimand"]
    with pytest.raises(InferenceContractError, match="estimand"):
        check_inference_contract(pkt)


def test_invalid_estimand_type() -> None:
    """An unknown estimand type is rejected."""
    pkt = _minimal_valid_packet()
    pkt["inference_contract"]["estimand"]["type"] = "unknown_type"
    with pytest.raises(InferenceContractError, match="estimand.type"):
        check_inference_contract(pkt)


def test_missing_decision_rule() -> None:
    """Deleting decision_rule raises an error."""
    pkt = _minimal_valid_packet()
    del pkt["inference_contract"]["decision_rule"]
    with pytest.raises(InferenceContractError, match="decision_rule"):
        check_inference_contract(pkt)


def test_empty_decision_rule_threshold() -> None:
    """A threshold shorter than five characters is rejected."""
    pkt = _minimal_valid_packet()
    pkt["inference_contract"]["decision_rule"]["threshold"] = "OK"
    with pytest.raises(InferenceContractError, match="decision_rule.threshold"):
        check_inference_contract(pkt)


def test_missing_primary_metrics() -> None:
    """Deleting primary_metrics raises an error."""
    pkt = _minimal_valid_packet()
    del pkt["inference_contract"]["primary_metrics"]
    with pytest.raises(InferenceContractError, match="primary_metrics"):
        check_inference_contract(pkt)


def test_empty_metrics_list() -> None:
    """An empty metrics list is rejected."""
    pkt = _minimal_valid_packet()
    pkt["inference_contract"]["primary_metrics"]["metrics"] = []
    with pytest.raises(InferenceContractError, match="primary_metrics"):
        check_inference_contract(pkt)


def test_metric_not_string() -> None:
    """A non-string metric entry is rejected."""
    pkt = _minimal_valid_packet()
    pkt["inference_contract"]["primary_metrics"]["metrics"] = [123]  # type: ignore[list-item]
    with pytest.raises(InferenceContractError, match="metric must be a non-empty string"):
        check_inference_contract(pkt)


def test_missing_multiplicity_handling() -> None:
    """Deleting multiplicity_handling raises an error."""
    pkt = _minimal_valid_packet()
    del pkt["inference_contract"]["multiplicity_handling"]
    with pytest.raises(InferenceContractError, match="multiplicity_handling"):
        check_inference_contract(pkt)


def test_invalid_adjustment_method() -> None:
    """An unknown adjustment_method is rejected."""
    pkt = _minimal_valid_packet()
    pkt["inference_contract"]["multiplicity_handling"]["adjustment_method"] = "made_up_method"
    with pytest.raises(InferenceContractError, match="adjustment_method"):
        check_inference_contract(pkt)


def test_check_yaml_file_missing_file(tmp_path: Path) -> None:
    """A missing YAML file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="preregistration config not found"):
        check_yaml_file(tmp_path / "missing.yaml")


def test_check_yaml_file_no_schema(tmp_path: Path) -> None:
    """A YAML without schema_version is rejected."""
    path = tmp_path / "no_schema.yaml"
    path.write_text("inference_contract: {}\n", encoding="utf-8")
    with pytest.raises(InferenceContractError, match="schema_version"):
        check_yaml_file(path)


def test_check_yaml_file_valid(tmp_path: Path) -> None:
    """A valid YAML file passes through check_yaml_file."""
    path = tmp_path / "valid.yaml"
    path.write_text(yaml.safe_dump(_minimal_valid_packet()), encoding="utf-8")
    result = check_yaml_file(path)
    assert result["status"] == "ok"


def test_custom_section_key() -> None:
    """The checker accepts a custom section key for non-standard layouts."""
    pkt = _minimal_valid_packet()
    contract = pkt.pop("inference_contract")
    pkt["analysis_plan"] = contract
    result = check_inference_contract(pkt, section_key="analysis_plan")
    assert result["status"] == "ok"


def test_config_listing_includes_all_research_yaml(capsys: pytest.CaptureFixture[str]) -> None:
    """Discovery includes research configs whose names omit ``preregistration``."""
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "configs/research/prediction_mpc_factorial_v1.yaml" in payload["configs"]
