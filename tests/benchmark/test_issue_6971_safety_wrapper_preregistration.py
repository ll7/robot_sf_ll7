"""Contract tests for the issue #6971 safety-wrapper preregistration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from robot_sf.benchmark.paired_effect_metric_contract import load_paired_effect_metric_contract
from robot_sf.benchmark.runner import load_scenario_matrix
from scripts.validation.check_preregistration_inference_contract import check_yaml_file

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / (
    "configs/benchmarks/issue_6971_safety_wrapper_paired_preregistration.yaml"
)
METRIC_CONTRACT_PATH = REPO_ROOT / "configs/benchmarks/paired_effect_metric_contract_v1.yaml"
RELATED_CONTEXT_PATH = REPO_ROOT / "docs/context/issue_3501_safety_wrapper.md"
CONTEXT_INDEX_PATH = REPO_ROOT / "docs/context/INDEX.md"


def _config() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_preregistration_has_a_valid_shared_inference_contract() -> None:
    """The packet cannot omit the analysis choices required before execution."""
    result = check_yaml_file(CONFIG_PATH, repo_root=REPO_ROOT)

    assert result["status"] == "ok"
    assert result["population_type"] == "fixed_suite"
    assert result["estimand_type"] == "paired_delta"
    assert result["metric_count"] == 1


def test_preregistration_freezes_matrix_hash_and_resolved_row_counts() -> None:
    """The source matrix and complete factorial denominator are predeclared."""
    config = _config()
    source_contracts = config["source_contracts"]
    design = config["design"]
    scenario_contract = design["scenario_contract"]
    pairing = design["pairing"]

    scenario_path = REPO_ROOT / source_contracts["scenario_matrix"]
    digest = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    assert digest == source_contracts["scenario_matrix_sha256"]
    scenarios = load_scenario_matrix(scenario_path)
    assert len(scenarios) == scenario_contract["resolved_scenario_count"]
    scenario_ids = [scenario.get("scenario_id") or scenario["name"] for scenario in scenarios]
    resolved_ids_digest = hashlib.sha256(
        json.dumps(scenario_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert resolved_ids_digest == scenario_contract["resolved_scenario_ids_sha256"]

    assert pairing["pair_count_per_planner"] == 48 * 3
    assert pairing["total_pair_count"] == 48 * 3 * 3
    assert pairing["retained_row_count"] == 48 * 3 * 3 * 2


def test_preregistration_retains_exactly_the_6970_metric_paths() -> None:
    """Every declared field is sourced from the merged normalized metric contract."""
    config = _config()
    contract = load_paired_effect_metric_contract(METRIC_CONTRACT_PATH)
    declared = config["retained_field_manifest"]["required_metric_fields"]

    assert [entry["name"] for entry in declared] == contract["required_metric_names"]
    assert [entry["path"] for entry in declared] == [
        f"metric_values.{name}" for name in contract["required_metric_names"]
    ]
    assert config["retained_field_manifest"]["metric_contract_schema"] == contract["schema_version"]


def test_preregistration_freezes_primary_cost_and_promotion_boundaries() -> None:
    """A future result cannot silently change the primary or turn cost into a footnote."""
    config = _config()
    outcomes = config["outcomes"]
    inference = config["inference_contract"]
    execution = config["execution_boundary"]
    promotion = config["promotion_criteria"]
    acceptance = config["acceptance"]

    assert outcomes["primary"] == [
        {
            "name": "exact_collision_probability",
            "path": "metric_values.exact_collision_probability",
            "role": "one_primary_safety_outcome",
            "unit": "probability [0, 1]",
        }
    ]
    assert inference["primary_metrics"]["metrics"] == ["exact_collision_probability"]
    assert "completion_probability" in str(outcomes["task_performance_cost"])
    assert "progress_at_timeout" in str(outcomes["task_performance_cost"])
    assert execution["campaign_execution_allowed_in_this_pr"] is False
    assert execution["compute_submit_authorized"] is False
    assert promotion["measured_safety_gain"]["required"]
    assert "width <= 0.10" in inference["decision_rule"]["threshold"]
    assert promotion["no_gain"]["required"]
    assert promotion["inconclusive"]["required"]
    assert acceptance["retained_schema_identifiability"]["status"] == (
        "confirmed_after_independent_review"
    )
    assert acceptance["explicit_no_submission"] is True
    assert config["cost_estimate"]["estimated_compute_hours"] > 0
    assert config["cost_estimate"]["estimated_wall_clock_hours"] > 0
    assert config["cost_estimate"]["estimated_storage_gib"] > 0


def test_preregistration_is_linked_from_the_safety_wrapper_context() -> None:
    """The parent safety-wrapper context points readers to the new preregistration."""
    text = RELATED_CONTEXT_PATH.read_text(encoding="utf-8")
    index_text = CONTEXT_INDEX_PATH.read_text(encoding="utf-8")

    assert "issue_6971_safety_wrapper_paired_preregistration.yaml" in text
    assert "issue_6971_safety_wrapper_paired_preregistration.md" in text
    assert "issue_6971_safety_wrapper_paired_preregistration.md" in index_text
