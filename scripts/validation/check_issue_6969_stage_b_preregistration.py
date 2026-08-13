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
EXPECTED_FIDELITY_SURFACES = (
    "lane_formation",
    "exit_arching",
    "doorway_oscillation",
    "throughput",
    "collision_overlap",
    "planner_facing_interaction",
)


class StageBPreregistrationError(ValueError):
    """Raised when the #6969 Stage B contract is incomplete or drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageBPreregistrationError(message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be a mapping")
    return value


def _list(value: Any, path: str) -> list[Any]:
    _require(isinstance(value, list) and value, f"{path} must be a non-empty list")
    return value


def _relative_source_path(value: str) -> Path:
    return (REPO_ROOT / value.split("::", maxsplit=1)[0]).resolve()


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

    _validate_sources(config, config_path=config_path)
    _validate_stage_a_snapshot(config)
    _validate_stage_a_snapshot_matches_summary(config)
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
        len(set(required_packet_fields)) == len(required_packet_fields),
        "result_packet_skeleton.required_fields must be unique",
    )
    _list(packet.get("forbidden_until_reviewed"), "result_packet_skeleton.forbidden_until_reviewed")

    stop_rules = _list(config.get("stop_rules"), "stop_rules")
    _require(len(stop_rules) >= 5, "stop_rules must retain the full fail-closed stop contract")

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


def _validate_sources(config: Mapping[str, Any], *, config_path: str | Path | None) -> None:
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
    ):
        digest = digests.get(key)
        _require(
            isinstance(digest, str) and SOURCE_DIGEST_RE.fullmatch(digest),
            f"source_sha256.{key} must be a SHA-256 digest",
        )
        source_path = _relative_source_path(str(sources[key]))
        _require(source_path.is_file(), f"source path does not exist: {sources[key]}")
        _require(
            _sha256(source_path) == digest, f"source_sha256.{key} does not match {sources[key]}"
        )

    if config_path is not None:
        _require(Path(config_path).is_file(), f"config path does not exist: {config_path}")


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


def _validate_stage_a_snapshot_matches_summary(config: Mapping[str, Any]) -> None:
    sources = _mapping(config.get("source_contracts"), "source_contracts")
    summary_source = sources.get("stage_a_summary")
    _require(
        isinstance(summary_source, str) and summary_source.strip(),
        "source_contracts.stage_a_summary is required",
    )
    summary = _load_json_mapping(
        _relative_source_path(summary_source), "source_contracts.stage_a_summary"
    )
    stage_a = _mapping(summary.get("stage_a"), "stage_a_summary.stage_a")

    snapshot = _mapping(config.get("stage_a_snapshot"), "stage_a_snapshot")
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

    decision = _mapping(stage_a.get("decision"), "stage_a_summary.stage_a.decision")
    observed = _mapping(snapshot.get("observed_decision"), "stage_a_snapshot.observed_decision")
    _require(
        decision.get("robust_clear_profile_found") is False
        and observed.get("robust_candidate_count") == 0,
        "Stage A summary robust candidate decision disagrees with preregistration snapshot",
    )

    profile_summaries = _list(
        stage_a.get("profile_summaries"), "stage_a_summary.stage_a.profile_summaries"
    )
    lhs_05_rows = [
        _mapping(profile, "stage_a_summary.stage_a.profile_summaries[]")
        for profile in profile_summaries
        if _mapping(profile, "stage_a_summary.stage_a.profile_summaries[]").get("profile_id")
        == "lhs_05"
    ]
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
        selection.get("no_response_dependent_ranking") is True,
        "candidate selection cannot rank by response",
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
        held_out.get("paired_comparator") == "anchor_released_default",
        "released default must remain comparator",
    )
    persistence = _mapping(held_out.get("persistence_rule"), "held_out_plan.persistence_rule")
    _require(
        persistence.get("confirmation_rule")
        == "at least 8 of 10 held-out seeds satisfy per_seed_clear_condition",
        "held-out confirmation rule drifted",
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
    by_id = {str(_mapping(entry, "fidelity outcome").get("surface_id")): entry for entry in entries}
    _require(set(by_id) == set(EXPECTED_FIDELITY_SURFACES), "fidelity surface set drifted")
    for surface_id in EXPECTED_FIDELITY_SURFACES:
        entry = _mapping(by_id[surface_id], f"fidelity_cost_surfaces.{surface_id}")
        _require(entry.get("metric"), f"fidelity surface {surface_id} must name a metric")
        _require(entry.get("unit"), f"fidelity surface {surface_id} must name units")
        _require(entry.get("direction"), f"fidelity surface {surface_id} must name direction")
    _require(
        surfaces.get("decision_policy")
        == "report_tradeoffs; do not pre-authorize a recalibration or default change",
        "fidelity decision policy must remain report-only",
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
