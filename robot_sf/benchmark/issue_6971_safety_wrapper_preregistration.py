"""Validator for the issue #6971 safety-wrapper paired-campaign preregistration.

This module owns the frozen, metric-capable campaign design that follows the
retained-row contract from issue #6970.  It validates provenance and analysis
choices without executing episodes, submitting compute, or making a safety
effectiveness claim.  The future campaign remains a separate maintainer
go/no-go decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from robot_sf.benchmark.paired_effect_metric_contract import (
    REQUIRED_METRIC_NAMES,
    validate_paired_effect_metric_contract,
)
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.check_preregistration_inference_contract import (
    InferenceContractError,
    check_inference_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / (
    "configs/benchmarks/issue_6971_safety_wrapper_paired_preregistration.yaml"
)
SCHEMA_VERSION = "issue_6971_safety_wrapper_paired_preregistration.v1"
EXPECTED_PARENT_ISSUES = (3501, 4598, 4830, 6970)
EXPECTED_PLANNER_KEYS = ("orca", "social_force", "prediction_planner")
EXPECTED_SEEDS = tuple(range(111, 131))
EXPECTED_SCENARIO_COUNT = 48
EXPECTED_EPISODES_PER_PLANNER = EXPECTED_SCENARIO_COUNT * len(EXPECTED_SEEDS) * 2
EXPECTED_EPISODES = EXPECTED_EPISODES_PER_PLANNER * len(EXPECTED_PLANNER_KEYS)
EXPECTED_PAIRING_KEY = ("planner", "scenario_id", "seed")
EXPECTED_SECONDARY_METRICS = (
    "near_miss_probability",
    "min_predicted_separation_m",
    "completion_probability",
    "false_positive_stop_rate",
    "stop_yield_latency_s",
    "wrapper_intervention_rate",
)
EXPECTED_SOURCE_PATHS = {
    "campaign_config": "configs/benchmarks/issue_4830_safety_wrapper_factorial_v1.yaml",
    "design_config": "configs/research/safety_wrapper_ablation_v1.yaml",
    "metric_contract": "configs/benchmarks/paired_effect_metric_contract_v1.yaml",
    "scenario_matrix": "configs/scenarios/classic_interactions_francis2023.yaml",
    "seed_sets": "configs/benchmarks/seed_sets_v1.yaml",
    "runtime_validator": "robot_sf/benchmark/safety/safety_wrapper_runtime.py",
    "report_builder": "robot_sf/benchmark/safety/safety_wrapper_factorial_report.py",
    "timing_reference": (
        "docs/context/evidence/camera_ready_all_planners_2026-05-04/reports/campaign_summary.json"
    ),
}
EXPECTED_SOURCE_SHA256 = {
    "campaign_config": "1f4c958c5c1d97f37127f1925041cec3ca0ff0267233c6bc572e7302592011d6",
    "design_config": "635262978e20bf1427b4545dff7d1e6e9f315b2d19cc5b0c9b38cdb90836fef3",
    "metric_contract": "cc423c03fac128f21681504fdda85f18606eb66ea292f5c145181ec3c63dc309",
    "scenario_matrix": "d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5",
    "seed_sets": "3aaab9171517b8d33bafc679d4a2c740864db0f96650e24d75c4c7e927d239e6",
    "runtime_validator": "a8941e1344c7d566636d0a9938275dafc55659635bd62b18e5f00d0d05f3b155",
    "report_builder": "bbaceebbf38d9b427a7c8b7fe9ed7bd04357bcdbb4684eb3c5610b8269c89ada",
    "timing_reference": "2211bf0c13815bf6af34afa2798b50da8559153be76d7b84650f8c970c2539b6",
}
EXPECTED_REFERENCE_RATES = {
    "orca": 0.9360,
    "social_force": 1.3297,
    "prediction_planner": 0.1431,
}
FORBIDDEN_TRANSIENT_KEYS = {
    "host",
    "target_host",
    "slurm",
    "job_id",
    "queue",
    "queue_route",
    "worktree",
    "output_dir",
    "run_id",
}


class SafetyWrapperPreregistrationError(ValueError):
    """Raised when the issue #6971 preregistration is incomplete or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SafetyWrapperPreregistrationError(message)


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    _require(isinstance(value, Mapping), f"{key} must be a mapping")
    return value


def _sequence(parent: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = parent.get(key)
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{key} must be a list",
    )
    return value


def _nonempty_string(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be a non-empty string")
    return value.strip()


def _repo_relative_path(value: Any, field: str) -> str:
    path_text = _nonempty_string(value, field)
    path = PurePosixPath(path_text)
    _require(not path.is_absolute() and ".." not in path.parts, f"{field} must be repo-relative")
    return path_text


def _walk_for_transient_keys(value: Any, path: str = "packet") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(
                str(key).lower() not in FORBIDDEN_TRANSIENT_KEYS,
                f"{path}.{key} is transient routing or local-output state",
            )
            _walk_for_transient_keys(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _walk_for_transient_keys(child, f"{path}[{index}]")


def _resolve_file(root: Path, value: Any, field: str) -> Path:
    relative = _repo_relative_path(value, field)
    path = root / relative
    _require(path.is_file(), f"{field} is missing: {relative}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_payload(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            return json.loads(text)
        return yaml.safe_load(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SafetyWrapperPreregistrationError(
            f"cannot read source contract {path}: {exc}"
        ) from exc


def load_preregistration_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the issue #6971 preregistration YAML as a mapping.

    Returns:
        Parsed YAML mapping.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SafetyWrapperPreregistrationError(
            f"cannot read preregistration {config_path}: {exc}"
        ) from exc
    _require(isinstance(payload, dict), "preregistration must be a YAML mapping")
    return payload


def _validate_header(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    _require(config.get("issue") == 6971, "issue must be 6971")
    _require(
        tuple(config.get("parent_issues", ())) == EXPECTED_PARENT_ISSUES,
        f"parent_issues must be {list(EXPECTED_PARENT_ISSUES)!r}",
    )
    _require(config.get("status") == "preregistration", "status must remain preregistration")
    _require(config.get("evidence_tier") == "proposal", "evidence_tier must remain proposal")
    _require(config.get("benchmark_evidence") is False, "benchmark_evidence must be false")
    _require(config.get("paper_facing") is False, "paper_facing must be false")
    claim_boundary = _nonempty_string(config.get("claim_boundary"), "claim_boundary").lower()
    for phrase in ("preregistration", "does not run", "not benchmark evidence", "paper-facing"):
        _require(phrase in claim_boundary, f"claim_boundary must mention {phrase}")


def _validate_execution_boundary(config: Mapping[str, Any]) -> None:
    execution = _mapping(config, "execution_boundary")
    for key in (
        "campaign_execution_in_this_pr",
        "compute_submit_authorized",
        "slurm_submission_in_this_pr",
        "gpu_submission_in_this_pr",
        "paper_or_dissertation_claim_in_this_pr",
        "metric_semantics_changes",
        "fallback_or_degraded_success_allowed",
    ):
        _require(execution.get(key) is False, f"execution_boundary.{key} must be false")
    approval = _mapping(config, "domain_approval")
    _require(approval.get("required") is True, "domain_approval.required must be true")
    _require(approval.get("status") == "pending", "domain_approval.status must remain pending")
    decisions = _sequence(approval, "required_decisions")
    _require(len(decisions) >= 5, "domain_approval.required_decisions is incomplete")


def _validate_source_contracts(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    contracts = _mapping(config, "source_contracts")
    hashes = _mapping(config, "source_sha256")
    _require(set(contracts) == set(EXPECTED_SOURCE_PATHS), "source_contracts keys drifted")
    _require(set(hashes) == set(EXPECTED_SOURCE_SHA256), "source_sha256 keys drifted")
    payloads: dict[str, Any] = {}
    for key, expected_path in EXPECTED_SOURCE_PATHS.items():
        _require(contracts.get(key) == expected_path, f"source_contracts.{key} path drifted")
        path = _resolve_file(root, contracts.get(key), f"source_contracts.{key}")
        observed_hash = _sha256(path)
        _require(
            hashes.get(key) == EXPECTED_SOURCE_SHA256[key],
            f"source_sha256.{key} is not the pinned digest",
        )
        _require(
            observed_hash == hashes.get(key),
            f"source_sha256.{key} does not match {expected_path}",
        )
        if key in {
            "campaign_config",
            "design_config",
            "metric_contract",
            "scenario_matrix",
            "seed_sets",
            "timing_reference",
        }:
            payloads[key] = _load_source_payload(path)
    return payloads


def _validate_lineage_sources(payloads: Mapping[str, Any]) -> None:
    campaign = payloads.get("campaign_config")
    design = payloads.get("design_config")
    metric = payloads.get("metric_contract")
    seed_sets = payloads.get("seed_sets")
    _require(isinstance(campaign, Mapping), "campaign source must be a mapping")
    _require(isinstance(design, Mapping), "design source must be a mapping")
    _require(isinstance(metric, Mapping), "metric source must be a mapping")
    _require(isinstance(seed_sets, Mapping), "seed source must be a mapping")

    _require(
        campaign.get("schema_version") == "issue-4830-safety-wrapper-factorial-v1",
        "campaign schema drifted",
    )
    _require(campaign.get("parent_issue") == 4830, "campaign parent issue drifted")
    _require(
        campaign.get("design_contract") == EXPECTED_SOURCE_PATHS["design_config"],
        "campaign design contract path drifted",
    )
    _require(
        campaign.get("scenario_matrix") == EXPECTED_SOURCE_PATHS["scenario_matrix"],
        "campaign scenario matrix path drifted",
    )
    campaign_seed_policy = _mapping(campaign, "seed_policy")
    _require(
        tuple(campaign_seed_policy.get("seeds", ())) == (111, 112, 113), "campaign S3 seeds drifted"
    )
    _require(
        campaign_seed_policy.get("seed_sets_path") == EXPECTED_SOURCE_PATHS["seed_sets"],
        "campaign seed source path drifted",
    )
    report_contract = _mapping(campaign, "report_contract")
    _require(
        tuple(report_contract.get("pairing_key_fields", ())) == EXPECTED_PAIRING_KEY,
        "campaign pairing key drifted",
    )
    _require(
        tuple(report_contract.get("expected_wrapper_arms", ())) == ("wrapper_off", "wrapper_on"),
        "campaign wrapper arms drifted",
    )
    _require(
        campaign.get("retained_metric_contract")
        == Path(EXPECTED_SOURCE_PATHS["metric_contract"]).name,
        "campaign metric contract drifted",
    )
    planners = _sequence(campaign, "planners")
    expected_arm_keys = tuple(
        f"{planner}__{arm}"
        for planner in EXPECTED_PLANNER_KEYS
        for arm in ("wrapper_off", "wrapper_on")
    )
    _require(
        tuple(row.get("key") for row in planners if isinstance(row, Mapping)) == expected_arm_keys,
        "campaign planner arm roster drifted",
    )
    _require(len(planners) == len(expected_arm_keys), "campaign planner arm count drifted")
    expected_algos = {
        f"{planner}__{arm}": planner
        for planner in EXPECTED_PLANNER_KEYS
        for arm in ("wrapper_off", "wrapper_on")
    }
    for row in planners:
        _require(isinstance(row, Mapping), "campaign planner rows must be mappings")
        key = str(row.get("key"))
        _require(row.get("algo") == expected_algos[key], f"campaign algo drifted for {key}")
        wrapper = _mapping(row, "safety_wrapper")
        expected_arm = key.rsplit("__", 1)[-1]
        _require(wrapper.get("arm_key") == expected_arm, f"campaign wrapper arm drifted for {key}")
        _require(
            wrapper.get("enabled") is (expected_arm == "wrapper_on"),
            f"campaign wrapper enablement drifted for {key}",
        )

    _require(design.get("schema_version") == "safety-wrapper-ablation.v1", "design schema drifted")
    _require(design.get("issue") == 3501, "design issue drifted")
    fixed_scope = _mapping(design, "fixed_scope")
    _require(tuple(fixed_scope.get("seeds", ())) == (111, 112, 113), "design S3 seeds drifted")
    _require(
        tuple(fixed_scope.get("planner_groups", ())) == EXPECTED_PLANNER_KEYS,
        "design planner roster drifted",
    )
    design_arms = _sequence(design, "wrapper_arms")
    _require(
        tuple(row.get("key") for row in design_arms if isinstance(row, Mapping))
        == ("wrapper_off", "wrapper_on"),
        "design wrapper arms drifted",
    )
    _require(
        design.get("retained_metric_contract") == EXPECTED_SOURCE_PATHS["metric_contract"],
        "design metric contract drifted",
    )

    try:
        validated_metric = validate_paired_effect_metric_contract(metric, source="metric_contract")
    except ValueError as exc:
        raise SafetyWrapperPreregistrationError(str(exc)) from exc
    _require(
        tuple(validated_metric["required_metric_names"]) == REQUIRED_METRIC_NAMES,
        "retained metric names drifted",
    )
    _require(
        tuple(seed_sets.get("paper_eval_s20", ())) == EXPECTED_SEEDS,
        "paper_eval_s20 seed source drifted",
    )


def _validate_scenario_source(payloads: Mapping[str, Any], root: Path) -> None:
    scenario_path = root / EXPECTED_SOURCE_PATHS["scenario_matrix"]
    try:
        scenarios = load_scenarios(scenario_path)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SafetyWrapperPreregistrationError(
            f"scenario matrix cannot be resolved: {exc}"
        ) from exc
    _require(
        len(scenarios) == EXPECTED_SCENARIO_COUNT, "scenario matrix must resolve to 48 scenarios"
    )
    scenario_ids = [str(row.get("name", row.get("scenario_id", ""))) for row in scenarios]
    _require(all(scenario_ids), "scenario matrix entries must have stable names")
    _require(len(set(scenario_ids)) == EXPECTED_SCENARIO_COUNT, "scenario names must be unique")
    _require(
        isinstance(payloads.get("scenario_matrix"), Mapping), "scenario source must be a mapping"
    )


def _validate_timing_reference(payloads: Mapping[str, Any]) -> None:
    """Reconcile declared planning rates with the durable historical summary."""
    reference = payloads.get("timing_reference")
    _require(isinstance(reference, Mapping), "timing reference must be a mapping")
    campaign = _mapping(reference, "campaign")
    _require(campaign.get("total_episodes") == 1008, "timing reference episode count drifted")
    _require(campaign.get("benchmark_success") is False, "timing reference success status drifted")
    rows = _sequence(reference, "planner_rows")
    by_planner = {str(row.get("planner_key")): row for row in rows if isinstance(row, Mapping)}
    for planner, expected_rate in EXPECTED_REFERENCE_RATES.items():
        _require(planner in by_planner, f"timing reference is missing {planner}")
        _require(
            math.isclose(
                float(by_planner[planner].get("episodes_per_second")),
                expected_rate,
                rel_tol=0.0,
                abs_tol=1.0e-4,
            ),
            f"timing reference rate drifted for {planner}",
        )


def _validate_field_manifest(config: Mapping[str, Any], metric_payload: Mapping[str, Any]) -> None:
    manifest = _mapping(config, "retained_field_manifest")
    _require(
        manifest.get("contract_path") == EXPECTED_SOURCE_PATHS["metric_contract"],
        "retained_field_manifest.contract_path drifted",
    )
    fields = _sequence(manifest, "fields")
    source_fields = _sequence(metric_payload, "fields")
    _require(len(fields) == len(REQUIRED_METRIC_NAMES), "retained field count drifted")
    _require(
        len(source_fields) == len(REQUIRED_METRIC_NAMES), "source retained field count drifted"
    )
    for index, (field, source_field) in enumerate(zip(fields, source_fields, strict=True)):
        _require(
            isinstance(field, Mapping), f"retained_field_manifest.fields[{index}] must be a mapping"
        )
        _require(
            isinstance(source_field, Mapping), f"metric contract fields[{index}] must be a mapping"
        )
        _require(
            field.get("name") == REQUIRED_METRIC_NAMES[index],
            f"retained field name drifted at {index}",
        )
        _require(
            field.get("path") == f"metric_values.{REQUIRED_METRIC_NAMES[index]}",
            f"retained field path drifted at {index}",
        )
        _require(
            field.get("name") == source_field.get("name"),
            f"retained field source mismatch at {index}",
        )
        _require(
            field.get("path") == source_field.get("path"),
            f"retained field path source mismatch at {index}",
        )


def _validate_design(config: Mapping[str, Any]) -> None:
    design = _mapping(config, "design")
    _require(
        design.get("scenario_matrix") == EXPECTED_SOURCE_PATHS["scenario_matrix"],
        "design scenario matrix drifted",
    )
    _require(
        design.get("scenario_selection") == "all_resolved_scenarios_no_filter",
        "scenario selection must remain all resolved scenarios",
    )
    _require(
        design.get("scenario_count") == EXPECTED_SCENARIO_COUNT, "design scenario count drifted"
    )
    _require(
        tuple(design.get("planner_keys", ())) == EXPECTED_PLANNER_KEYS,
        "design planner keys drifted",
    )
    roster = _sequence(design, "planner_roster")
    _require(
        tuple(row.get("key") for row in roster if isinstance(row, Mapping))
        == EXPECTED_PLANNER_KEYS,
        "preregistration planner roster drifted",
    )
    _require(
        len(roster) == len(EXPECTED_PLANNER_KEYS), "preregistration planner roster count drifted"
    )
    for row in roster:
        _require(isinstance(row, Mapping), "preregistration planner roster rows must be mappings")
        planner = str(row.get("key"))
        _require(
            tuple(row.get("source_arm_keys", ()))
            == (f"{planner}__wrapper_off", f"{planner}__wrapper_on"),
            f"source arms drifted for {planner}",
        )
        _nonempty_string(row.get("algorithm"), f"planner_roster.{planner}.algorithm")
        _nonempty_string(row.get("role"), f"planner_roster.{planner}.role")

    seed_schedule = _mapping(design, "seed_schedule")
    _require(seed_schedule.get("name") == "paper_eval_s20", "seed schedule name drifted")
    _require(
        seed_schedule.get("source") == f"{EXPECTED_SOURCE_PATHS['seed_sets']}::paper_eval_s20",
        "seed schedule source drifted",
    )
    _require(
        tuple(seed_schedule.get("values", ())) == EXPECTED_SEEDS, "seed schedule values drifted"
    )
    _require(seed_schedule.get("count") == len(EXPECTED_SEEDS), "seed schedule count drifted")
    _require(tuple(design.get("pairing_key", ())) == EXPECTED_PAIRING_KEY, "pairing key drifted")
    _require(
        design.get("planned_episode_count") == EXPECTED_EPISODES, "planned episode count drifted"
    )
    arms = _sequence(design, "wrapper_arms")
    _require(len(arms) == 2, "design must declare exactly two wrapper arms")
    _require(
        tuple(row.get("key") for row in arms if isinstance(row, Mapping))
        == ("wrapper_off", "wrapper_on"),
        "wrapper arm order drifted",
    )
    _require(
        arms[0].get("enabled") is False and arms[0].get("baseline") is True,
        "wrapper_off arm drifted",
    )
    _require(
        arms[1].get("enabled") is True and arms[1].get("baseline") is False,
        "wrapper_on arm drifted",
    )

    runner = _mapping(config, "fixed_runner")
    expected_runner = {
        "kinematics": ["differential_drive"],
        "horizon_steps": 100,
        "dt_seconds": 0.1,
        "workers": 2,
        "arm_isolation": "subprocess",
        "resume": True,
        "record_forces": True,
        "videos": False,
    }
    for key, expected in expected_runner.items():
        _require(runner.get(key) == expected, f"fixed_runner.{key} drifted")


def _validate_outcomes(config: Mapping[str, Any]) -> None:
    estimand = _mapping(config, "estimand")
    _require(
        estimand.get("primary_metric") == "exact_collision_probability", "primary metric drifted"
    )
    _require(estimand.get("unit") == "probability [0, 1]", "primary metric unit drifted")
    _require(estimand.get("contrast") == "wrapper_on_minus_wrapper_off", "contrast drifted")
    _require(
        tuple(estimand.get("pairing_key", ())) == EXPECTED_PAIRING_KEY,
        "estimand pairing key drifted",
    )
    _nonempty_string(estimand.get("definition"), "estimand.definition")
    _nonempty_string(estimand.get("safety_gain_definition"), "estimand.safety_gain_definition")
    non_claims = _sequence(estimand, "non_claims")
    _require(len(non_claims) >= 3, "estimand.non_claims is incomplete")
    non_claim_text = " ".join(str(item).lower() for item in non_claims)
    _require(
        "universal" in non_claim_text and "transfer" in non_claim_text,
        "estimand non-claims must reject universal and transfer claims",
    )
    secondary = _sequence(config, "secondary_outcomes")
    _require(
        tuple(row.get("metric") for row in secondary if isinstance(row, Mapping))
        == EXPECTED_SECONDARY_METRICS,
        "secondary outcome roster drifted",
    )
    for index, row in enumerate(secondary):
        _require(isinstance(row, Mapping), f"secondary_outcomes[{index}] must be a mapping")
        _require(row.get("role") == "secondary", f"secondary_outcomes[{index}] role drifted")
        _nonempty_string(row.get("definition"), f"secondary_outcomes[{index}].definition")
    task_cost = _mapping(config, "task_performance_cost")
    _require(task_cost.get("metric") == "progress_at_timeout", "task-performance metric drifted")
    _nonempty_string(task_cost.get("definition"), "task_performance_cost.definition")
    _require(
        task_cost.get("maximum_acceptable_mean_delta") == -0.05,
        "task-performance tolerance drifted",
    )


def _validate_analysis(config: Mapping[str, Any]) -> None:
    inference = config.get("inference_contract")
    _require(isinstance(inference, Mapping), "inference_contract must be a mapping")
    try:
        check_inference_contract(dict(config))
    except InferenceContractError as exc:
        raise SafetyWrapperPreregistrationError(str(exc)) from exc
    precision = _mapping(config, "power_precision")
    _require(
        precision.get("smallest_practically_meaningful_absolute_delta") == 0.05,
        "meaningful delta drifted",
    )
    _require(precision.get("target_ci_half_width") == 0.03, "target CI half-width drifted")
    _require(precision.get("target_ci_width") == 0.06, "target CI width drifted")
    _require(
        precision.get("interval_width_not_significance_threshold") is True,
        "power precision must be interval-width based",
    )
    _nonempty_string(precision.get("seed_budget"), "power_precision.seed_budget")
    _nonempty_string(precision.get("rationale"), "power_precision.rationale")

    analysis = _mapping(config, "analysis_plan")
    for key in (
        "paired_estimator",
        "aggregation",
        "ties_and_degenerate_episodes",
        "missingness_policy",
        "stop_rule",
    ):
        _nonempty_string(analysis.get(key), f"analysis_plan.{key}")
    _require(analysis.get("bootstrap_samples") == 1000, "bootstrap sample count drifted")
    _require(analysis.get("bootstrap_confidence") == 0.95, "bootstrap confidence drifted")
    _require(analysis.get("bootstrap_seed") == 123, "bootstrap seed drifted")
    stop_rule = str(analysis["stop_rule"]).lower()
    _require(
        "post-hoc" in stop_rule and "substitut" in stop_rule,
        "stop rule must reject post-hoc substitution",
    )

    criteria = _mapping(config, "promotion_criteria")
    for key in ("measured_safety_gain", "no_gain", "inconclusive", "reporting_commitment"):
        _nonempty_string(criteria.get(key), f"promotion_criteria.{key}")
    _require(
        "inconclusive" in str(criteria["inconclusive"]).lower(),
        "inconclusive criterion must be explicit",
    )


def _validate_cost(config: Mapping[str, Any]) -> dict[str, float]:
    cost = _mapping(config, "cost_estimate")
    _require(cost.get("planned_episodes") == EXPECTED_EPISODES, "cost episode count drifted")
    _require(
        cost.get("episodes_per_planner") == EXPECTED_EPISODES_PER_PLANNER,
        "cost per-planner episode count drifted",
    )
    _require(
        cost.get("reference_source") == EXPECTED_SOURCE_PATHS["timing_reference"],
        "timing reference path drifted",
    )
    rates = _mapping(cost, "reference_episodes_per_second")
    _require(set(rates) == set(EXPECTED_REFERENCE_RATES), "timing reference planner roster drifted")
    for planner, expected_rate in EXPECTED_REFERENCE_RATES.items():
        _require(
            math.isclose(float(rates[planner]), expected_rate, rel_tol=0.0, abs_tol=1.0e-9),
            f"timing rate drifted for {planner}",
        )
    projected = {
        planner: EXPECTED_EPISODES_PER_PLANNER / rate
        for planner, rate in EXPECTED_REFERENCE_RATES.items()
    }
    declared_projected = _mapping(cost, "projected_seconds_per_planner")
    for planner, expected_seconds in projected.items():
        _require(
            math.isclose(
                float(declared_projected[planner]), expected_seconds, rel_tol=0.0, abs_tol=0.1
            ),
            f"projected seconds drifted for {planner}",
        )
    sequential_seconds = sum(projected.values())
    ideal_parallel_seconds = max(projected.values())
    _require(
        math.isclose(
            float(cost.get("projected_sequential_seconds")),
            sequential_seconds,
            rel_tol=0.0,
            abs_tol=0.2,
        ),
        "sequential timing estimate drifted",
    )
    _require(
        math.isclose(
            float(cost.get("projected_sequential_wall_hours")),
            sequential_seconds / 3600,
            rel_tol=0.0,
            abs_tol=0.01,
        ),
        "sequential wall estimate drifted",
    )
    _require(
        math.isclose(
            float(cost.get("ideal_parallel_wall_hours")),
            ideal_parallel_seconds / 3600,
            rel_tol=0.0,
            abs_tol=0.01,
        ),
        "parallel wall estimate drifted",
    )
    headroom = float(cost.get("headroom_fraction"))
    setup_hours = float(cost.get("fixed_setup_overhead_hours"))
    required_hours = ideal_parallel_seconds / 3600 * (1.0 + headroom) + setup_hours
    _require(
        math.isclose(
            float(cost.get("required_wall_hours")), required_hours, rel_tol=0.0, abs_tol=0.02
        ),
        "required wall estimate drifted",
    )
    _require(
        float(cost.get("reserved_wall_clock_hours")) == 6.0,
        "reserved wall-clock hours must remain 6",
    )
    slots = int(cost.get("concurrent_worker_slots"))
    _require(slots == 2, "concurrent worker slots must remain 2")
    _require(
        float(cost.get("reserved_worker_hours")) == 12.0, "reserved worker-hours must remain 12"
    )
    storage = _mapping(cost, "storage")
    _require(storage.get("assumed_raw_episode_kib") == 64, "raw episode storage assumption drifted")
    _require(storage.get("report_log_headroom_mib") == 128, "report/log storage headroom drifted")
    modeled_mib = EXPECTED_EPISODES * 64 / 1024 + 128
    _require(
        math.isclose(float(storage.get("modeled_mib")), modeled_mib, rel_tol=0.0, abs_tol=0.1),
        "modeled storage drifted",
    )
    _require(
        float(storage.get("reserved_storage_gib")) == 1.0, "reserved storage must remain 1 GiB"
    )
    _nonempty_string(cost.get("caveat"), "cost_estimate.caveat")
    return {
        "planned_episodes": float(EXPECTED_EPISODES),
        "sequential_wall_hours": sequential_seconds / 3600,
        "ideal_parallel_wall_hours": ideal_parallel_seconds / 3600,
        "required_wall_hours": required_hours,
        "reserved_wall_clock_hours": 6.0,
        "reserved_worker_hours": 12.0,
        "modeled_storage_mib": modeled_mib,
    }


def validate_preregistration_config(
    config: Mapping[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Validate the frozen issue #6971 packet without executing the campaign.

    Returns:
        The input configuration as a plain dictionary after all fail-closed checks pass.
    """
    _require(isinstance(config, Mapping), "preregistration must be a mapping")
    normalized = dict(config)
    root = (repo_root or REPO_ROOT).resolve()
    _walk_for_transient_keys(normalized)
    _validate_header(normalized)
    _validate_execution_boundary(normalized)
    payloads = _validate_source_contracts(normalized, root)
    _validate_lineage_sources(payloads)
    _validate_scenario_source(payloads, root)
    _validate_timing_reference(payloads)
    _validate_field_manifest(normalized, payloads["metric_contract"])
    _validate_design(normalized)
    _validate_outcomes(normalized)
    _validate_analysis(normalized)
    _validate_cost(normalized)
    readiness = _mapping(normalized, "readiness_decision")
    _require(
        readiness.get("status") == "blocked_pending_maintainer_go_no_go", "readiness status drifted"
    )
    _require(
        readiness.get("campaign_execution_allowed") is False,
        "campaign execution must remain disallowed",
    )
    _require(
        readiness.get("compute_submit_authorized") is False,
        "compute submission must remain disallowed",
    )
    _require(
        len(_sequence(readiness, "future_run_requires")) >= 3, "future run gates are incomplete"
    )
    return normalized


def build_validation_report(
    path: str | Path = DEFAULT_CONFIG, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Validate a packet and return a compact non-evidence report.

    Returns:
        Machine-readable proposal-level validation summary.
    """
    config = load_preregistration_config(path)
    validate_preregistration_config(config, repo_root=repo_root)
    cost = _validate_cost(config)
    return {
        "schema_version": "issue_6971_safety_wrapper_preregistration_validation.v1",
        "issue": 6971,
        "status": "ok",
        "evidence_tier": "proposal",
        "benchmark_evidence": False,
        "execution_authorized": False,
        "compute_submit_authorized": False,
        "planned_episode_count": EXPECTED_EPISODES,
        "planner_count": len(EXPECTED_PLANNER_KEYS),
        "scenario_count": EXPECTED_SCENARIO_COUNT,
        "seed_count": len(EXPECTED_SEEDS),
        "retained_metric_count": len(REQUIRED_METRIC_NAMES),
        "cost": cost,
        "claim_boundary": (
            "Preregistration contract only; no campaign episodes, compute submission, "
            "benchmark result, or paper-facing safety claim."
        ),
    }


__all__ = [
    "DEFAULT_CONFIG",
    "EXPECTED_EPISODES",
    "EXPECTED_EPISODES_PER_PLANNER",
    "EXPECTED_PLANNER_KEYS",
    "EXPECTED_SCENARIO_COUNT",
    "EXPECTED_SEEDS",
    "EXPECTED_SOURCE_PATHS",
    "REQUIRED_METRIC_NAMES",
    "SCHEMA_VERSION",
    "SafetyWrapperPreregistrationError",
    "build_validation_report",
    "load_preregistration_config",
    "validate_preregistration_config",
]
