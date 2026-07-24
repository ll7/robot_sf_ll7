#!/usr/bin/env python3
"""Validate the issue #4232 uncertainty-envelope claim pre-registration packet."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PACKET = Path("configs/benchmarks/issue_4232_uncertainty_envelope_claim_packet.yaml")
SCHEMA_VERSION = "uncertainty-envelope-claim-packet.v1"
MPC_FAMILIES = {"mpc", "prediction_mpc", "prediction_mpc_cbf", "nmpc_social"}
REQUIRED_FORBIDDEN_CLAIMS = {
    "conformal_coverage_guarantee",
    "real_world_safety_claim",
    "deployment_certification",
    "generalized_planner_superiority",
    "paper_or_dissertation_claim",
}
REQUIRED_INELIGIBLE_ROW_STATUSES = {
    "diagnostic_only",
    "fallback",
    "degraded",
    "not_available",
    "failed",
    "blocked",
}
RAW_ARTIFACT_MARKERS = {
    "raw_episode_jsonl",
    "videos",
    "slurm_logs",
    "checkpoints",
    "model_caches",
}


class PacketError(ValueError):
    """Raised when the pre-registration packet would fail closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    _require(isinstance(value, Mapping), f"{key} must be mapping")
    return value


def _require_sequence(payload: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = payload.get(key)
    _require(
        isinstance(value, Sequence) and not isinstance(value, str) and len(value) > 0,
        f"{key} must be non-empty sequence",
    )
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PacketError("packet must be YAML mapping")
    return payload


def _require_repo_file(value: Any, field_name: str, repo_root: Path) -> Path:
    _require(isinstance(value, str) and value.strip() != "", f"{field_name} must be non-empty path")
    path = Path(value)
    _require(not path.is_absolute(), f"{field_name} must be repository-relative")
    _require(".." not in path.parts, f"{field_name} must not contain '..'")
    root = repo_root.resolve()
    candidate = root / path
    _require(
        not _has_symlink_component(candidate, root=root), f"{field_name} must not use symlinks"
    )
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PacketError(f"{field_name} not found: {value}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PacketError(f"{field_name} escapes repository root: {value}") from exc
    _require(resolved.is_file(), f"{field_name} must be a file")
    return resolved


def _has_symlink_component(path: Path, *, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _require_claim_boundary(packet: Mapping[str, Any]) -> None:
    boundary = str(packet.get("claim_boundary", "")).lower()
    for phrase in (
        "planning packet only",
        "does not establish safety",
        "performance",
        "conformal",
        "dissertation",
        "paper",
        "real-world",
        "deployment",
        "generalized planner",
    ):
        _require(phrase in boundary, f"claim_boundary must mention {phrase!r}")

    claim_modes = _require_mapping(packet, "claim_modes")
    forbidden = set(claim_modes.get("forbidden_without_followup") or [])
    _require(
        REQUIRED_FORBIDDEN_CLAIMS <= forbidden,
        "claim_modes.forbidden_without_followup missing required claim exclusions",
    )


def _require_execution_boundary(packet: Mapping[str, Any]) -> None:
    execution = _require_mapping(packet, "execution_boundary")
    _require(execution.get("run_benchmark") is False, "run_benchmark must be false")
    _require(
        execution.get("compute_submit_authorized") is False,
        "compute_submit_authorized must be false",
    )
    _require(
        execution.get("local_submission_allowed") is False,
        "local_submission_allowed must be false",
    )
    _require(
        str(execution.get("slurm_job_id", "")).lower() == "not_submitted",
        "slurm_job_id must be not_submitted",
    )


def _require_scenario_and_seeds(packet: Mapping[str, Any], *, repo_root: Path) -> None:
    surface = _require_mapping(packet, "scenario_surface")
    _require(
        surface.get("matrix_path") == "configs/scenarios/classic_interactions_francis2023.yaml",
        "scenario_surface.matrix_path must pin classic_interactions_francis2023.yaml",
    )
    _require_repo_file(surface.get("matrix_path"), "scenario_surface.matrix_path", repo_root)
    _require(
        surface.get("selection_policy") == "predeclared_before_run",
        "scenario selection must be predeclared before run",
    )
    _require(surface.get("max_episode_steps") == 600, "max_episode_steps must be 600")
    _require(surface.get("dt") == 0.1, "scenario_surface.dt must be 0.1")
    kinematics = list(_require_sequence(surface, "kinematics_matrix"))
    _require("differential_drive" in kinematics, "differential_drive kinematics required")

    seed_policy = _require_mapping(packet, "seed_policy")
    _require(seed_policy.get("mode") == "fixed-list", "seed_policy.mode must be fixed-list")
    seeds = list(_require_sequence(seed_policy, "seeds"))
    _require(all(isinstance(seed, int) for seed in seeds), "all seeds must be integers")
    _require(len(seeds) == len(set(seeds)), "seeds must be unique")


def _require_planners(packet: Mapping[str, Any], *, repo_root: Path) -> int:
    planners = _require_sequence(packet, "planner_families")
    for planner in planners:
        _require(isinstance(planner, Mapping), "planner entries must be mappings")
        planner_id = str(planner.get("planner_id", ""))
        _require(planner_id != "", "planner_id is required")
        _require_repo_file(
            planner.get("base_config_path"),
            f"{planner_id}.base_config_path",
            repo_root,
        )
        family = str(planner.get("family", "")).lower()
        claim_modes = set(planner.get("claim_modes") or [])
        if "envelope_effect" in claim_modes:
            _require(
                family in MPC_FAMILIES or planner_id in MPC_FAMILIES,
                f"{planner_id} cannot enter envelope-effect claims unless MPC-family",
            )
    return len(planners)


def _require_alpha_arms(packet: Mapping[str, Any]) -> int:
    surface = _require_mapping(packet, "scenario_surface")
    expected_kinematics = list(surface.get("kinematics_matrix") or [])
    arms = _require_sequence(packet, "alpha_arms")
    keys: set[str] = set()
    nonzero_alpha = False

    for arm in arms:
        _require(isinstance(arm, Mapping), "alpha arm entries must be mappings")
        key = str(arm.get("key", ""))
        _require(key != "", "alpha arm key is required")
        _require(key not in keys, f"duplicate alpha arm key: {key}")
        keys.add(key)
        enabled = arm.get("pedestrian_uncertainty_envelope_enabled")
        _require(isinstance(enabled, bool), f"{key} enabled flag must be boolean")
        alpha = arm.get("pedestrian_uncertainty_alpha_mps")
        _require(isinstance(alpha, int | float), f"{key} alpha must be numeric")
        _require(alpha >= 0.0, f"{key} alpha must be non-negative")
        nonzero_alpha = nonzero_alpha or alpha > 0.0
        _require(arm.get("scenario_surface_ref") == "scenario_surface", f"{key} scenario mismatch")
        _require(arm.get("seed_set_ref") == "seed_policy", f"{key} seed policy mismatch")
        _require(
            arm.get("max_episode_steps") == surface.get("max_episode_steps"),
            f"{key} max_episode_steps mismatch",
        )
        _require(arm.get("dt") == surface.get("dt"), f"{key} dt mismatch")
        _require(
            list(arm.get("kinematics_matrix") or []) == expected_kinematics,
            f"{key} kinematics mismatch",
        )

    _require("envelope_off_alpha_0" in keys, "missing envelope_off_alpha_0 control arm")
    _require("envelope_on_alpha_0" in keys, "missing envelope_on_alpha_0 regression arm")
    _require(nonzero_alpha, "at least one nonzero alpha arm is required")
    return len(arms)


def _require_metrics_and_stop_rules(packet: Mapping[str, Any]) -> None:
    metrics = _require_mapping(packet, "metrics")
    primary = set(_require_sequence(metrics, "primary"))
    for metric in (
        "collision_rate",
        "near_miss_rate",
        "minimum_clearance",
        "success_rate",
        "runtime_per_episode",
        "fallback_rate",
    ):
        _require(metric in primary, f"primary metric missing: {metric}")
    secondary = set(_require_sequence(metrics, "secondary_diagnostic"))
    _require("snqi_diagnostic_only" in secondary, "SNQI must be diagnostic-only")
    _require(
        "effective_radius_used_by_planner" in secondary,
        "envelope activation diagnostic missing",
    )

    stop_rules = _require_sequence(packet, "stop_rules")
    stop_ids = {str(rule.get("id", "")) for rule in stop_rules if isinstance(rule, Mapping)}
    for stop_id in (
        "alpha_zero_equivalence",
        "nonzero_alpha_fallback_only",
        "safety_runtime_tradeoff_revision",
        "claim_review_entry",
    ):
        _require(stop_id in stop_ids, f"stop rule missing: {stop_id}")


def _require_row_status_and_outputs(packet: Mapping[str, Any]) -> None:
    row_status = _require_mapping(packet, "row_status_policy")
    success_values = set(row_status.get("benchmark_strength_success_values") or [])
    _require(success_values == {"successful_evidence"}, "only successful_evidence can be success")
    ineligible = set(row_status.get("ineligible_benchmark_strength_values") or [])
    _require(
        REQUIRED_INELIGIBLE_ROW_STATUSES <= ineligible,
        "fallback/degraded/unavailable rows must be ineligible benchmark evidence",
    )
    fallback_policy = str(row_status.get("fallback_policy", "")).lower()
    _require(
        "cannot count as successful benchmark-strength evidence" in fallback_policy,
        "fallback policy must exclude fallback/degraded rows from success evidence",
    )

    outputs = _require_mapping(packet, "outputs")
    bundle = str(outputs.get("compact_review_bundle", ""))
    _require(
        bundle.startswith("docs/context/evidence/issue_4232_"),
        "compact_review_bundle must be a durable docs/context/evidence path",
    )
    required_artifacts = set(outputs.get("required_review_artifacts") or [])
    for artifact in (
        "metadata.json",
        "pre_registration_packet.json",
        "row_status_audit.csv",
        "claim_readiness.md",
        "SHA256SUMS",
    ):
        _require(artifact in required_artifacts, f"required review artifact missing: {artifact}")
    prohibited = set(outputs.get("prohibited_git_artifacts") or [])
    _require(
        RAW_ARTIFACT_MARKERS <= prohibited,
        "raw JSONL/videos/logs/checkpoints/caches must be prohibited git artifacts",
    )
    for artifact in required_artifacts:
        lower = str(artifact).lower()
        _require(not lower.endswith(".jsonl"), "raw episode JSONL must not be review artifact")
        _require("video" not in lower, "videos must not be review artifacts")
        _require("slurm" not in lower and "log" not in lower, "logs must not be review artifacts")
        _require("checkpoint" not in lower, "checkpoints must not be review artifacts")


def _require_provenance_and_claim_gates(packet: Mapping[str, Any]) -> None:
    provenance = _require_mapping(packet, "artifact_provenance")
    for key in (
        "require_public_commit_hash",
        "require_packet_sha256",
        "require_scenario_matrix_sha256",
        "require_resolved_seed_manifest",
        "require_planner_alpha_roster",
        "require_row_status_audit",
    ):
        _require(provenance.get(key) is True, f"artifact_provenance.{key} must be true")
    _require(
        provenance.get("raw_artifacts_git_policy") == "keep_out_of_git",
        "raw artifacts must stay out of git",
    )

    gates = _require_mapping(packet, "claim_update_conditions")
    _require(
        "maintainer_claim_boundary_review" in str(gates.get("claim_map_update_allowed", "")),
        "claim map update must require maintainer claim-boundary review",
    )
    _require(
        "separate_pr" in str(gates.get("paper_or_dissertation_update_allowed", "")),
        "paper/dissertation updates must require separate PR",
    )
    _require(
        gates.get("conformal_calibration_claim_allowed") is False,
        "conformal calibration claims must be disallowed",
    )


def validate_packet(packet: Mapping[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    """Validate the issue #4232 pre-registration packet and return a summary."""
    root = Path.cwd() if repo_root is None else Path(repo_root)
    _require(packet.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    _require(packet.get("issue") == 4232, "issue must be 4232")
    _require(packet.get("parent_runtime_issue") == 4141, "parent_runtime_issue must be 4141")
    _require(packet.get("runtime_completion_pr") == 4229, "runtime_completion_pr must be 4229")
    _require(
        packet.get("evidence_status") == "pre_registered_no_evidence",
        "evidence_status must remain pre_registered_no_evidence",
    )

    _require_claim_boundary(packet)
    _require_execution_boundary(packet)
    _require_scenario_and_seeds(packet, repo_root=root)
    planner_count = _require_planners(packet, repo_root=root)
    alpha_arm_count = _require_alpha_arms(packet)
    _require_metrics_and_stop_rules(packet)
    _require_row_status_and_outputs(packet)
    _require_provenance_and_claim_gates(packet)

    return {
        "ok": True,
        "issue": 4232,
        "evidence_status": packet.get("evidence_status"),
        "planner_count": planner_count,
        "alpha_arm_count": alpha_arm_count,
        "compute_submit_authorized": False,
        "run_benchmark": False,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line packet checker."""
    args = _parse_args(argv)
    try:
        summary = validate_packet(
            _load_yaml(args.packet), repo_root=args.packet.resolve().parents[2]
        )
    except PacketError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"BLOCK: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "PASS: issue #4232 uncertainty-envelope claim packet is pre-registered and claim-gated."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
