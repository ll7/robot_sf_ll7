#!/usr/bin/env python3
"""Validate the issue #6969 Stage B preregistration without running compute.

The validator checks the frozen candidate-selection, held-out seed, fidelity-cost,
result-packet, provenance, and fail-closed execution boundaries. A successful
validation is proposal-level readiness only; it never launches Stage B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from scripts.validation.check_preregistration_inference_contract import (
    InferenceContractError,
    check_inference_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / (
    "configs/benchmarks/issue_6969_lane_formation_stage_b_preregistration.yaml"
)
SCHEMA_VERSION = "issue_6969_lane_formation_stage_b_preregistration.v1"
SOURCE_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
LOCAL_ONLY_ROOTS = frozenset({".git", ".venv", "output", "results"})
EXPECTED_MISSINGNESS_POLICY = (
    "Missing, non-finite, unavailable, fallback, degraded, or non-native rows remain visible "
    "and invalidate the corresponding candidate comparison; never impute, drop, or replace a "
    "row after seeing its outcome."
)
EXPECTED_FIDELITY_SURFACES = (
    "lane_formation",
    "exit_arching",
    "doorway_oscillation",
    "throughput",
    "collision_overlap",
    "planner_facing_interaction",
)
EXPECTED_PERSISTENCE_RULE = {
    "steady_state_window": "the declared post-warmup observation window from the Stage A protocol",
    "per_seed_clear_condition": (
        "lane_segregation_index >= 0.5 for the complete steady_state_window"
    ),
    "confirmation_rule": "at least 8 of 10 held-out seeds satisfy per_seed_clear_condition",
    "report_all_seed_values": True,
}
EXPECTED_FIDELITY_SURFACE_FIELDS = {
    "lane_formation": {
        "metric": "lane_segregation_index",
        "unit": [0, 1],
        "role": "primary",
        "direction": "larger_is_more_lane_segregation",
        "source_contract": "robot_sf.research.lane_formation_reference",
    },
    "exit_arching": {
        "metric": "exit_density_ratio",
        "unit": "dimensionless_ratio",
        "role": "fidelity_cost",
        "direction": "positive_delta_is_arching_cost",
        "scenario_source": "robot_sf.research.emergent_phenomena.default_scenario_set",
    },
    "doorway_oscillation": {
        "metric": "oscillation_flips",
        "unit": "flips_per_observation_window",
        "role": "fidelity_cost",
        "direction": "positive_delta_is_oscillation_cost",
        "scenario_source": "robot_sf.research.emergent_phenomena.default_scenario_set",
    },
    "throughput": {
        "metric": "completed_agents_per_second",
        "unit": "agents_per_second",
        "role": "fidelity_cost",
        "direction": "negative_delta_is_throughput_cost",
        "scenario_source": "robot_sf.research.emergent_phenomena.default_scenario_set",
    },
    "collision_overlap": {
        "metric": "collision_and_overlap_rate",
        "unit": "proportion_of_valid_rows",
        "role": "fidelity_cost",
        "direction": "positive_delta_is_collision_overlap_cost",
        "scenario_source": "native_runner_metric_contract_to_be_named_before_launch",
    },
    "planner_facing_interaction": {
        "metric": ["completion_rate", "progress_at_timeout", "collision_rate"],
        "unit": "declared_per_metric",
        "role": "downstream_diagnostic",
        "direction": "report_signed_deltas_without_a_universal_planner_claim",
        "scenario_source": "named_planner_facing_suite_required_before_launch",
    },
}
EXPECTED_RESULT_PACKET_FIELDS = (
    "research_question_and_hypothesis",
    "evidence_tier_and_admission_state",
    "source_artifact_ids_paths_digests_generation_commit_command",
    "analysis_population_inclusion_exclusion_and_attrition",
    "native_adapter_fallback_degraded_unavailable_rejected_accounting",
    "primary_estimand_analysis_unit_pairing_key_comparator_and_contrast_direction",
    "metric_units_support_denominators_missingness_and_uncertainty",
    "fidelity_surface_effects_and_sensitivity_results",
    "figure_ids_visual_contracts_and_sample_size_display",
    "structured_caption_assertions_with_observed_inferred_unavailable_status",
    "claim_boundary_forbidden_claims_and_exact_decision_vocabulary",
    "independent_review_identity_status_digest_and_findings",
)
EXPECTED_FORBIDDEN_RESULT_FIELDS = (
    "dissertation_admission",
    "paper_facing_claim",
    "released_default_recommendation",
    "universal_social_force_model_claim",
)
EXPECTED_STOP_RULES = (
    "Stop before compute if the Stage A source digest or implementation contract drifts.",
    "Stop with no candidate if the frozen eligibility rule returns an empty set.",
    "Stop and classify incomplete if any required row is non-native, missing, non-finite, fallback, or degraded.",
    "Stop if a seed, surface, comparator, or parameter is substituted after outcomes are inspected.",
    "Stop before interpretation if the result packet cannot bind every number, figure, caption, and claim to exact artifacts.",
    "Never change released defaults in this issue.",
)


class StageBPreregistrationError(ValueError):
    """Raised when the #6969 Stage B contract is incomplete or drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageBPreregistrationError(message)


def _strict_equal(actual: Any, expected: Any) -> bool:
    """Compare YAML values without allowing Python's bool/int equality aliasing."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _strict_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _strict_string_members_match(actual: Any, expected: tuple[str, ...]) -> bool:
    """Require the exact string members of a semantic list, independent of order."""

    return (
        isinstance(actual, list)
        and all(type(item) is str for item in actual)
        and len(actual) == len(expected)
        and len(set(actual)) == len(actual)
        and set(actual) == set(expected)
    )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be a mapping")
    return value


def _list(value: Any, path: str) -> list[Any]:
    _require(isinstance(value, list) and value, f"{path} must be a non-empty list")
    return value


def _relative_source_path(value: str, *, source_root: Path = REPO_ROOT) -> Path:
    """Resolve a source path only when it stays inside the declared source root."""

    raw_path = value.split("::", maxsplit=1)[0].strip()
    candidate = Path(raw_path)
    _require(
        not candidate.is_absolute() and ".." not in candidate.parts,
        f"source path must be repository-relative without traversal: {value}",
    )
    _require(
        candidate.parts
        and candidate.parts[0] not in LOCAL_ONLY_ROOTS
        and ".worktrees" not in candidate.parts,
        f"source path is local-only: {value}",
    )
    root = source_root.resolve()
    unresolved = root / candidate
    current = root
    try:
        for part in candidate.parts:
            current /= part
            _require(not current.is_symlink(), f"source path must not traverse a symlink: {value}")
        resolved = unresolved.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise StageBPreregistrationError(f"source path cannot be resolved: {value}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StageBPreregistrationError(
            f"source path resolves outside the repository: {value}"
        ) from exc
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBPreregistrationError(f"{label} must be readable JSON: {exc}") from exc
    return _mapping(payload, label)


def load_preregistration_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the tracked YAML packet."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "preregistration config must be a YAML mapping")
    validate_preregistration_config(payload, config_path=config_path)
    return payload


def validate_preregistration_config(
    payload: Mapping[str, Any],
    *,
    config_path: str | Path | None = DEFAULT_CONFIG,
    source_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate the frozen Stage B contract and return a normalized copy."""
    config = dict(payload)
    _require(
        config.get("schema_version") == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}"
    )
    _require(config.get("issue") == 6969, "issue must be 6969")
    _require(config.get("status") == "preregistration", "status must remain preregistration")
    _require(config.get("benchmark_evidence") is False, "benchmark_evidence must be false")
    _require(config.get("paper_facing") is False, "paper_facing must be false")

    execution = _mapping(config.get("execution_boundary"), "execution_boundary")
    for key in (
        "stage_b_execution_in_this_pr",
        "compute_submit_authorized",
        "slurm_submission_in_this_pr",
        "gpu_submission_in_this_pr",
        "released_default_change",
        "new_evidence_generation",
        "paper_or_dissertation_claim_in_this_pr",
        "metric_semantics_changes",
    ):
        _require(execution.get(key) is False, f"execution_boundary.{key} must be false")
    _require(
        execution.get("fallback_or_degraded_success_allowed") is False,
        "fallback_or_degraded_success_allowed must be false",
    )

    approval = _mapping(config.get("domain_approval"), "domain_approval")
    _require(approval.get("required") is True, "domain_approval.required must be true")
    _require(approval.get("status") == "pending", "domain_approval.status must be pending")
    _list(approval.get("required_decisions"), "domain_approval.required_decisions")

    source_root_path = Path(source_root)
    _validate_sources(config, config_path=config_path, source_root=source_root_path)
    _validate_implementation_commits(config)
    _validate_stage_a_snapshot(config)
    _validate_stage_a_snapshot_matches_summary(config, source_root=source_root_path)
    _validate_candidate_selection(config)
    _validate_held_out_plan(config)
    _validate_fidelity_surfaces(config)

    try:
        inference_report = check_inference_contract(config, repo_root=REPO_ROOT)
    except (InferenceContractError, TypeError, ValueError) as exc:
        raise StageBPreregistrationError(str(exc)) from exc

    packet = _mapping(config.get("result_packet_skeleton"), "result_packet_skeleton")
    _require(
        packet.get("status") == "planned_not_generated",
        "result packet must remain planned_not_generated",
    )
    _require(
        packet.get("upstream_contract_status") == "pending_issue_7029_review_and_merge",
        "result packet must remain pending the #7029 contract review",
    )
    required_packet_fields = _list(
        packet.get("required_fields"), "result_packet_skeleton.required_fields"
    )
    _require(
        _strict_string_members_match(required_packet_fields, EXPECTED_RESULT_PACKET_FIELDS),
        "result_packet_skeleton.required_fields drifted",
    )
    forbidden_packet_fields = _list(
        packet.get("forbidden_until_reviewed"), "result_packet_skeleton.forbidden_until_reviewed"
    )
    _require(
        _strict_string_members_match(forbidden_packet_fields, EXPECTED_FORBIDDEN_RESULT_FIELDS),
        "result_packet_skeleton.forbidden_until_reviewed drifted",
    )

    stop_rules = _list(config.get("stop_rules"), "stop_rules")
    _require(len(stop_rules) >= 5, "stop_rules must retain the full fail-closed stop contract")
    _require(_strict_string_members_match(stop_rules, EXPECTED_STOP_RULES), "stop_rules drifted")

    readiness = _mapping(config.get("readiness_decision"), "readiness_decision")
    _require(
        str(readiness.get("status", "")).startswith("blocked_pending_"),
        "readiness_decision.status must remain blocked_pending_*",
    )
    for key in (
        "stage_b_execution_allowed",
        "compute_submit_authorized",
        "released_default_change",
        "publication_or_admission_allowed",
    ):
        _require(readiness.get(key) is False, f"readiness_decision.{key} must be false")

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 6969,
        "status": "ok",
        "benchmark_evidence": False,
        "stage_b_execution_allowed": False,
        "compute_submit_authorized": False,
        "stage_a_native_rows": 30,
        "held_out_seed_count": len(_mapping(config["held_out_plan"], "held_out_plan")["seeds"]),
        "candidate_count": len(
            _mapping(config["candidate_selection"], "candidate_selection")["current_selection"]
        ),
        "fidelity_surface_count": len(EXPECTED_FIDELITY_SURFACES),
        "inference_contract": inference_report,
    }


def _validate_sources(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None,
    source_root: Path,
) -> None:
    sources = _mapping(config.get("source_contracts"), "source_contracts")
    required = (
        "stage_a_summary",
        "stage_a_parameter_screen",
        "stage_a_reference_contract",
        "stage_a_runner",
        "stage_a_tests",
        "result_packet_contract",
    )
    for key in required:
        _require(
            isinstance(sources.get(key), str) and sources[key].strip(),
            f"source_contracts.{key} is required",
        )

    digests = _mapping(config.get("source_sha256"), "source_sha256")
    for key in (
        "stage_a_summary",
        "stage_a_parameter_screen",
        "stage_a_reference_contract",
        "stage_a_runner",
        "stage_a_tests",
    ):
        digest = digests.get(key)
        _require(
            isinstance(digest, str) and SOURCE_DIGEST_RE.fullmatch(digest),
            f"source_sha256.{key} must be a SHA-256 digest",
        )
        source_path = _relative_source_path(str(sources[key]), source_root=source_root)
        _require(source_path.is_file(), f"source path does not exist: {sources[key]}")
        _require(
            _sha256(source_path) == digest, f"source_sha256.{key} does not match {sources[key]}"
        )

    if config_path is not None:
        _require(Path(config_path).is_file(), f"config path does not exist: {config_path}")


def _validate_implementation_commits(config: Mapping[str, Any]) -> None:
    """Verify the implementation commits and source paths recorded by Stage A."""

    commits = _mapping(
        _mapping(config.get("stage_a_snapshot"), "stage_a_snapshot").get("implementation_commits"),
        "stage_a_snapshot.implementation_commits",
    )
    paths = {
        "reference": "robot_sf/research/lane_formation_reference.py",
        "parameter_screen": "robot_sf/research/lane_formation_parameter_screen.py",
    }
    for key, source_path in paths.items():
        commit = commits.get(key)
        _require(
            isinstance(commit, str) and COMMIT_RE.fullmatch(commit),
            f"stage_a_snapshot.implementation_commits.{key} must be a git revision",
        )
        _require(
            _git_object_exists(f"{commit}^{{commit}}"),
            f"stage_a_snapshot.implementation_commits.{key} is unavailable: {commit}",
        )
        _require(
            _git_object_exists(f"{commit}:{source_path}"),
            f"stage_a_snapshot.implementation_commits.{key} does not contain {source_path}",
        )


def _git_object_exists(object_name: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "-e", object_name],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _validate_stage_a_snapshot(config: Mapping[str, Any]) -> None:
    snapshot = _mapping(config.get("stage_a_snapshot"), "stage_a_snapshot")
    _require(snapshot.get("native_rows") == 30, "stage_a_snapshot.native_rows must remain 30")
    _require(
        snapshot.get("execution_status") == "30/30 native:computed",
        "Stage A execution status drifted",
    )
    profiles = _mapping(snapshot.get("profiles"), "stage_a_snapshot.profiles")
    _require(profiles.get("space_filling_count") == 8, "Stage A profile count must remain 8")
    _require(profiles.get("profile_seed") == 6969, "Stage A profile seed must remain 6969")
    _require(profiles.get("seeds") == [5149, 5150, 5151], "Stage A seeds must remain frozen")
    observed = _mapping(snapshot.get("observed_decision"), "stage_a_snapshot.observed_decision")
    _require(
        observed.get("robust_candidate_count") == 0,
        "observed Stage A candidate count must remain zero",
    )
    near = _mapping(
        observed.get("near_candidate"), "stage_a_snapshot.observed_decision.near_candidate"
    )
    _require(near.get("profile_id") == "lhs_05", "Stage A near-candidate record drifted")
    _require(
        near.get("clear_hits") == 1 and near.get("clear_total") == 3,
        "Stage A near-candidate hit count drifted",
    )
    _require(
        near.get("eligible_for_stage_b") is False, "one-of-three Stage A hit cannot be eligible"
    )


def _validate_stage_a_snapshot_matches_summary(
    config: Mapping[str, Any],
    *,
    source_root: Path,
) -> None:
    sources = _mapping(config.get("source_contracts"), "source_contracts")
    summary_source = sources.get("stage_a_summary")
    _require(
        isinstance(summary_source, str) and summary_source.strip(),
        "source_contracts.stage_a_summary is required",
    )
    summary = _load_json_mapping(
        _relative_source_path(summary_source, source_root=source_root),
        "source_contracts.stage_a_summary",
    )
    stage_a = _mapping(summary.get("stage_a"), "stage_a_summary.stage_a")

    snapshot = _mapping(config.get("stage_a_snapshot"), "stage_a_snapshot")
    implementation_commits = _mapping(
        snapshot.get("implementation_commits"), "stage_a_snapshot.implementation_commits"
    )
    reference_stage = _mapping(summary.get("reference_stage"), "stage_a_summary.reference_stage")
    _require(
        reference_stage.get("implementation_commit") == implementation_commits.get("reference"),
        "Stage A reference implementation commit disagrees with preregistration snapshot",
    )
    _require(
        stage_a.get("implementation_commit") == implementation_commits.get("parameter_screen"),
        "Stage A parameter-screen implementation commit disagrees with preregistration snapshot",
    )
    _require(
        stage_a.get("native_rows") == snapshot.get("native_rows"),
        "Stage A summary native row count disagrees with preregistration snapshot",
    )
    _require(
        stage_a.get("native_execution") == snapshot.get("execution_status"),
        "Stage A summary native execution status disagrees with preregistration snapshot",
    )

    design = _mapping(stage_a.get("design"), "stage_a_summary.stage_a.design")
    profiles = _mapping(snapshot.get("profiles"), "stage_a_snapshot.profiles")
    _require(
        design.get("space_filling_profiles") == profiles.get("space_filling_count"),
        "Stage A summary profile count disagrees with preregistration snapshot",
    )
    _require(
        design.get("profile_seed") == profiles.get("profile_seed"),
        "Stage A summary profile seed disagrees with preregistration snapshot",
    )
    _require(
        design.get("seeds") == profiles.get("seeds"),
        "Stage A summary seed schedule disagrees with preregistration snapshot",
    )
    threshold = _mapping(
        profiles.get("clear_threshold"), "stage_a_snapshot.profiles.clear_threshold"
    )
    _require(
        design.get("clear_threshold_lsi") == threshold.get("threshold"),
        "Stage A summary clear threshold disagrees with preregistration snapshot",
    )

    profile_summaries = _list(
        stage_a.get("profile_summaries"), "stage_a_summary.stage_a.profile_summaries"
    )
    selection = _mapping(config.get("candidate_selection"), "candidate_selection")
    required_hit_count = selection.get("required_clear_hit_count")
    required_seed_count = selection.get("required_clear_seed_count")
    _require(
        isinstance(required_hit_count, int)
        and not isinstance(required_hit_count, bool)
        and isinstance(required_seed_count, int)
        and not isinstance(required_seed_count, bool),
        "candidate hit-count rule must be integer-valued",
    )

    expected_space_filling_ids = _list(
        profiles.get("profile_ids"), "stage_a_snapshot.profiles.profile_ids"
    )
    _require(
        all(
            isinstance(profile_id, str) and profile_id.strip()
            for profile_id in expected_space_filling_ids
        ),
        "Stage A space-filling profile IDs must be non-empty strings",
    )
    _require(
        len(set(expected_space_filling_ids)) == len(expected_space_filling_ids),
        "Stage A space-filling profile IDs must be unique",
    )
    fixed_anchors = _list(profiles.get("fixed_anchors"), "stage_a_snapshot.profiles.fixed_anchors")
    _require(
        all(isinstance(profile_id, str) and profile_id.strip() for profile_id in fixed_anchors),
        "Stage A fixed-anchor profile IDs must be non-empty strings",
    )
    expected_profile_ids = set(expected_space_filling_ids) | set(fixed_anchors)
    _require(
        len(expected_profile_ids) == len(expected_space_filling_ids) + len(fixed_anchors),
        "Stage A profile IDs must be unique across space-filling and fixed-anchor rows",
    )

    rows_by_id: dict[str, Mapping[str, Any]] = {}
    computed_eligible_ids: list[str] = []
    for raw_profile in profile_summaries:
        profile = _mapping(raw_profile, "stage_a_summary.stage_a.profile_summaries[]")
        profile_id = profile.get("profile_id")
        _require(
            isinstance(profile_id, str) and profile_id.strip(),
            "Stage A profile summary must contain a non-empty profile_id",
        )
        _require(
            profile_id not in rows_by_id,
            f"Stage A profile summary has duplicate profile_id: {profile_id}",
        )
        rows_by_id[profile_id] = profile
        clear_hits = profile.get("clear_lsi_hits")
        clear_total = profile.get("clear_lsi_total")
        _require(
            isinstance(clear_hits, int)
            and not isinstance(clear_hits, bool)
            and isinstance(clear_total, int)
            and not isinstance(clear_total, bool)
            and 0 <= clear_hits <= clear_total,
            f"Stage A profile {profile_id} has malformed clear hit counts",
        )
        _require(
            clear_total == required_seed_count,
            f"Stage A profile {profile_id} clear total disagrees with frozen seed count",
        )
        if profile_id in expected_space_filling_ids and clear_hits == required_hit_count:
            computed_eligible_ids.append(profile_id)

    _require(
        set(rows_by_id) == expected_profile_ids,
        "Stage A profile summary identity set disagrees with preregistration snapshot",
    )
    computed_eligible_ids = sorted(computed_eligible_ids)
    frozen_selection = selection.get("current_selection")
    _require(
        isinstance(frozen_selection, list)
        and all(isinstance(profile_id, str) for profile_id in frozen_selection),
        "candidate_selection.current_selection must be a list of profile IDs",
    )
    _require(
        computed_eligible_ids == frozen_selection,
        "computed Stage A eligible profile IDs disagree with frozen current selection",
    )

    decision = _mapping(stage_a.get("decision"), "stage_a_summary.stage_a.decision")
    observed = _mapping(snapshot.get("observed_decision"), "stage_a_snapshot.observed_decision")
    _require(
        observed.get("robust_candidate_count") == len(computed_eligible_ids),
        "computed Stage A eligible profile count disagrees with preregistration decision",
    )
    _require(
        decision.get("robust_clear_profile_found") is bool(computed_eligible_ids),
        "Stage A summary robust candidate decision disagrees with computed eligibility",
    )

    lhs_05_rows = [rows_by_id["lhs_05"]] if "lhs_05" in rows_by_id else []
    _require(
        len(lhs_05_rows) == 1,
        "Stage A summary must contain exactly one lhs_05 profile summary",
    )
    near = _mapping(
        observed.get("near_candidate"), "stage_a_snapshot.observed_decision.near_candidate"
    )
    lhs_05 = lhs_05_rows[0]
    _require(
        near.get("profile_id") == lhs_05.get("profile_id")
        and near.get("clear_hits") == lhs_05.get("clear_lsi_hits")
        and near.get("clear_total") == lhs_05.get("clear_lsi_total"),
        "Stage A summary lhs_05 near-hit disagrees with preregistration snapshot",
    )
    _require(
        lhs_05.get("clear_lsi_hits") == 1 and lhs_05.get("clear_lsi_total") == 3,
        "Stage A summary lhs_05 must remain a one-of-three near-hit",
    )


def _validate_candidate_selection(config: Mapping[str, Any]) -> None:
    selection = _mapping(config.get("candidate_selection"), "candidate_selection")
    _require(
        selection.get("selection_is_frozen_before_held_out_execution") is True,
        "candidate selection must be frozen before held-out execution",
    )
    _require(
        selection.get("eligible_design_role") == "space_filling_stage_a",
        "candidate design role drifted",
    )
    _require(
        selection.get("required_execution_mode") == "native",
        "candidate execution mode must be native",
    )
    _require(
        selection.get("required_execution_status") == "computed",
        "candidate execution status must be computed",
    )
    _require(
        selection.get("required_clear_hit_count") == 3,
        "candidate clear-hit rule must require all three Stage A seeds",
    )
    _require(
        selection.get("required_clear_seed_count") == 3,
        "candidate clear-seed rule must require three seeds",
    )
    _require(selection.get("required_threshold") == 0.5, "candidate threshold must remain 0.5")
    _require(
        selection.get("required_threshold_metric") == "lane_segregation_index",
        "candidate threshold metric must remain lane_segregation_index",
    )
    _require(
        selection.get("no_response_dependent_ranking") is True,
        "candidate selection cannot rank by response",
    )
    _require(
        selection.get("selection_order") == "profile_id_lexicographic_after_eligibility",
        "candidate selection order must remain frozen",
    )
    _require(
        selection.get("no_candidate_action") == "blocked_no_stage_b_compute",
        "no-candidate action must block compute",
    )
    current = selection.get("current_selection")
    _require(
        isinstance(current, list) and current == [], "current Stage A selection must remain empty"
    )
    _require(
        selection.get("maximum_candidate_count") == 2, "maximum candidate count must remain two"
    )


def _validate_held_out_plan(config: Mapping[str, Any]) -> None:
    held_out = _mapping(config.get("held_out_plan"), "held_out_plan")
    _require(
        held_out.get("scenario_id") == "bidirectional_corridor",
        "held-out scenario must remain bidirectional_corridor",
    )
    _require(
        held_out.get("protocol_source")
        == "robot_sf.research.lane_formation_reference.ReferenceProtocol",
        "held-out protocol source must remain the Stage A ReferenceProtocol",
    )
    seeds = held_out.get("seeds")
    _require(
        isinstance(seeds, list) and len(seeds) == 10,
        "held_out_plan.seeds must contain exactly ten seeds",
    )
    _require(
        all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds),
        "held-out seeds must be integers",
    )
    _require(len(set(seeds)) == len(seeds), "held-out seeds must be unique")
    _require(seeds == list(range(5152, 5162)), "held-out seed schedule drifted")
    _require(held_out.get("seed_count") == 10, "held_out_plan.seed_count must be ten")
    _require(held_out.get("disjoint_from_stage_a_seeds") is True, "held-out seeds must be disjoint")
    _require(
        held_out.get("no_seed_substitution") is True, "held-out seed substitution must be forbidden"
    )
    _require(
        held_out.get("same_protocol_as_stage_a") is True,
        "held-out plan must use the Stage A protocol",
    )
    _require(
        held_out.get("paired_comparator") == "anchor_released_default",
        "released default must remain comparator",
    )
    _require(
        held_out.get("planned_cells")
        == "candidate_count_times_10_held_out_seeds_plus_10_released_default_rows",
        "held-out planned cell formula drifted",
    )
    _require(
        held_out.get("execution_mode") == "native_only",
        "held-out execution mode must remain native_only",
    )
    _require(
        held_out.get("missingness_policy") == EXPECTED_MISSINGNESS_POLICY,
        "held-out missingness policy must remain fail-closed",
    )
    persistence = _mapping(held_out.get("persistence_rule"), "held_out_plan.persistence_rule")
    for key, expected in EXPECTED_PERSISTENCE_RULE.items():
        _require(
            _strict_equal(persistence.get(key), expected),
            f"held_out_plan.persistence_rule.{key} drifted",
        )


def _validate_fidelity_surfaces(config: Mapping[str, Any]) -> None:
    surfaces = _mapping(config.get("fidelity_cost_surfaces"), "fidelity_cost_surfaces")
    _require(
        surfaces.get("comparison") == "candidate_minus_anchor_released_default",
        "fidelity comparator drifted",
    )
    _require(
        surfaces.get("pairing_key") == ["surface_id", "candidate_profile_id", "seed"],
        "fidelity pairing key drifted",
    )
    entries = _list(surfaces.get("outcomes"), "fidelity_cost_surfaces.outcomes")
    by_id: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        surface = _mapping(entry, "fidelity outcome")
        surface_id = surface.get("surface_id")
        _require(
            isinstance(surface_id, str) and surface_id.strip(),
            "fidelity surface must contain a non-empty surface_id",
        )
        _require(
            surface_id not in by_id,
            f"duplicate fidelity surface_id: {surface_id}",
        )
        by_id[surface_id] = surface
    _require(set(by_id) == set(EXPECTED_FIDELITY_SURFACES), "fidelity surface set drifted")
    for surface_id in EXPECTED_FIDELITY_SURFACES:
        entry = _mapping(by_id[surface_id], f"fidelity_cost_surfaces.{surface_id}")
        for key, expected in EXPECTED_FIDELITY_SURFACE_FIELDS[surface_id].items():
            _require(
                _strict_equal(entry.get(key), expected),
                f"fidelity surface {surface_id}.{key} drifted",
            )
    _require(
        surfaces.get("decision_policy")
        == "report_tradeoffs; do not pre-authorize a recalibration or default change",
        "fidelity decision policy must remain report-only",
    )
    _require(
        surfaces.get("report_effect_and_uncertainty") is True,
        "fidelity surfaces must report effect and uncertainty",
    )
    _require(
        surfaces.get("incomplete_surface_action") == "inconclusive_and_block_publication",
        "incomplete fidelity surfaces must block publication",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the fail-closed validator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        report = validate_preregistration_config(
            yaml.safe_load(args.config.read_text(encoding="utf-8")),
            config_path=args.config,
        )
    except (OSError, StageBPreregistrationError, TypeError, yaml.YAMLError) as exc:
        result = {
            "schema_version": "issue_6969_lane_formation_stage_b_preregistration_validation.v1",
            "issue": 6969,
            "status": "blocked",
            "error": str(exc),
            "stage_b_execution_allowed": False,
            "compute_submit_authorized": False,
        }
        if args.as_json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(f"BLOCKED: {exc}")
        return 2
    if args.as_json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "OK: issue #6969 Stage B preregistration validated; "
            f"{report['held_out_seed_count']} held-out seeds planned, no compute authorized."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
