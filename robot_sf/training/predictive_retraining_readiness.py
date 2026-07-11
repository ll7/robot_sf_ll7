"""Fail-closed readiness checks for predictive retraining decision packets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robot_sf.errors import RobotSfError

SCHEMA_VERSION = "predictive-retraining-readiness.v1"
DECISION_BLOCKED = "retraining_blocked"
DECISION_READY = "retraining_launch_ready"
DEFAULT_PACKET = "configs/training/predictive/predictive_retraining_readiness_issue_3214.yaml"

_NEGATIVE_PRIOR_STATUSES = {"verified_negative", "negative_result"}
_REQUIRED_RERUN_ITEMS = {
    "control_law_change_config",
    "checkpoint_provenance_plan",
    "hard_seed_evaluation_plan",
}
_REQUIRED_PACKET_SECTIONS = {
    "data_prerequisites",
    "expected_checkpoint_lineage",
    "evaluation_config",
    "output_roots",
}
_REQUIRED_PIPELINE_SCENARIOS = {
    "scenario_matrix",
    "hard_seed_manifest",
    "planner_grid",
}
_REQUIRED_PIPELINE_COLLECTIONS = {
    "base_collection",
    "hardcase_collection",
}
_REQUIRED_PIPELINE_PROVENANCE = {
    "checkpoint_path",
    "checkpoint_provenance_path",
    "hard_seed_evaluation_summary",
}


class PredictiveRetrainingReadinessError(RobotSfError, ValueError):
    """Raised when a predictive retraining readiness packet cannot be evaluated."""


def load_readiness_packet(packet_path: Path) -> dict[str, Any]:
    """Load a readiness packet from YAML.

    Returns:
        Parsed readiness packet mapping.
    """

    if not packet_path.is_file():
        raise PredictiveRetrainingReadinessError(f"readiness packet is not a file: {packet_path}")
    payload = yaml.safe_load(packet_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PredictiveRetrainingReadinessError(
            f"readiness packet must be a mapping: {packet_path}"
        )
    return payload


def evaluate_retraining_readiness(
    packet_path: Path | str = DEFAULT_PACKET,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate whether a predictive retraining packet is launch-ready.

    The checker performs only static public-repository checks. It never submits compute, fetches
    private artifacts, or upgrades diagnostic training evidence into a benchmark or paper claim.

    Returns:
        Static readiness report with packet completeness, launch decision, and blockers.
    """

    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    packet = _resolve_path(packet_path, root)
    payload = load_readiness_packet(packet)

    blockers: list[str] = []
    warnings: list[str] = []
    _validate_static_contract(payload, packet, root, blockers, warnings)

    prior_result = _mapping(payload.get("prior_result"))
    launch_decision = _mapping(payload.get("launch_decision"))
    prior_status = str(prior_result.get("status") or "").strip()
    decision_state = str(launch_decision.get("state") or "").strip()
    compute_submit_allowed = bool(launch_decision.get("compute_submit_allowed", False))

    if prior_status in _NEGATIVE_PRIOR_STATUSES:
        if decision_state != "blocked_until_control_law_change":
            blockers.append(
                "launch_decision.state must be blocked_until_control_law_change after a "
                "verified negative prior retraining result"
            )
        _validate_required_rerun_items(launch_decision, blockers)

    launch_ready = (
        not blockers and compute_submit_allowed and prior_status not in _NEGATIVE_PRIOR_STATUSES
    )
    decision = DECISION_READY if launch_ready else DECISION_BLOCKED

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "issue": payload.get("issue"),
        "candidate_id": payload.get("candidate_id"),
        "packet": str(packet),
        "packet_complete": not blockers,
        "launch_ready": launch_ready,
        "decision": decision,
        "decision_state": decision_state,
        "prior_result_status": prior_status,
        "compute_submit_allowed": compute_submit_allowed,
        "blockers": blockers,
        "warnings": warnings,
        "evidence_boundary": payload.get("claim_boundary"),
    }


def _validate_static_contract(
    payload: dict[str, Any],
    packet_path: Path,
    repo_root: Path,
    blockers: list[str],
    warnings: list[str],
) -> None:
    """Validate packet fields that make a public launch decision reproducible."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        blockers.append(f"schema_version must be {SCHEMA_VERSION!r}")
    issue = str(payload.get("issue") or "").strip()
    if issue not in {"#3214", "3214"}:
        blockers.append("issue must be #3214")
    _require_non_empty_string(payload, "candidate_id", blockers)
    claim_boundary = str(payload.get("claim_boundary") or "")
    for required_text in ("no Slurm", "no full benchmark", "no paper"):
        if required_text.lower() not in claim_boundary.lower():
            blockers.append(f"claim_boundary must state {required_text!r}")

    inputs = _mapping(payload.get("inputs"))
    _require_existing_repo_path(inputs, "weighting_spec", packet_path, repo_root, blockers)
    _require_existing_repo_path(inputs, "pipeline_config", packet_path, repo_root, blockers)
    _validate_launch_packet_sections(payload, blockers)
    _validate_pipeline_config(payload, inputs, packet_path, repo_root, blockers)

    prior_result = _mapping(payload.get("prior_result"))
    _require_non_empty_string(prior_result, "status", blockers, prefix="prior_result")
    _require_non_empty_string(prior_result, "source_issue", blockers, prefix="prior_result")
    evidence_status = str(prior_result.get("evidence_status") or "")
    if "paper" not in evidence_status.lower() or "not" not in evidence_status.lower():
        warnings.append("prior_result.evidence_status should explicitly avoid paper-claim status")

    launch_decision = _mapping(payload.get("launch_decision"))
    if launch_decision.get("fail_closed_on_missing") is not True:
        blockers.append("launch_decision.fail_closed_on_missing must be true")
    if launch_decision.get("compute_submit_allowed") is not False:
        blockers.append("launch_decision.compute_submit_allowed must be false for this PR scope")


def _validate_required_rerun_items(launch_decision: dict[str, Any], blockers: list[str]) -> None:
    required = launch_decision.get("required_before_rerun")
    if not isinstance(required, list):
        blockers.append("launch_decision.required_before_rerun must be a list")
        return
    observed = {str(item).strip() for item in required if str(item).strip()}
    missing = sorted(_REQUIRED_RERUN_ITEMS - observed)
    if missing:
        blockers.append(
            "launch_decision.required_before_rerun missing required items: " + ", ".join(missing)
        )


def _validate_launch_packet_sections(payload: dict[str, Any], blockers: list[str]) -> None:
    """Require the issue #3214 packet to name each launch-preflight concern explicitly."""

    missing = sorted(section for section in _REQUIRED_PACKET_SECTIONS if section not in payload)
    if missing:
        blockers.append("readiness packet missing sections: " + ", ".join(missing))

    data_prerequisites = _mapping(payload.get("data_prerequisites"))
    for key in ("base_collection", "hardcase_collection", "weighting_spec"):
        _require_non_empty_string(data_prerequisites, key, blockers, prefix="data_prerequisites")

    checkpoint_lineage = _mapping(payload.get("expected_checkpoint_lineage"))
    for key in ("candidate_model_id", "checkpoint_path", "checkpoint_provenance_path"):
        _require_non_empty_string(
            checkpoint_lineage,
            key,
            blockers,
            prefix="expected_checkpoint_lineage",
        )
    for key in ("checkpoint_path", "checkpoint_provenance_path"):
        _require_packet_output_path(
            checkpoint_lineage, key, blockers, "expected_checkpoint_lineage"
        )

    evaluation_config = _mapping(payload.get("evaluation_config"))
    for key in ("hard_seed_benchmark", "planner_grid", "summary_path"):
        _require_non_empty_string(evaluation_config, key, blockers, prefix="evaluation_config")
    for key in ("hard_seed_benchmark", "planner_grid"):
        _require_packet_config_path(evaluation_config, key, blockers, "evaluation_config")
    _require_packet_output_path(evaluation_config, "summary_path", blockers, "evaluation_config")

    output_roots = _mapping(payload.get("output_roots"))
    for key in ("pipeline_root", "training_root", "evaluation_root"):
        _require_non_empty_string(output_roots, key, blockers, prefix="output_roots")
        _require_packet_output_path(output_roots, key, blockers, "output_roots")


def _validate_pipeline_config(
    packet: dict[str, Any],
    inputs: dict[str, Any],
    packet_path: Path,
    repo_root: Path,
    blockers: list[str],
) -> None:
    """Validate the public predictive retraining pipeline config referenced by the packet."""

    pipeline_value = inputs.get("pipeline_config")
    if not isinstance(pipeline_value, str) or not pipeline_value.strip():
        return
    pipeline_path = _resolve_path(pipeline_value, repo_root, base=packet_path.parent)
    if not pipeline_path.is_file():
        return
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    if not isinstance(pipeline, dict):
        blockers.append(f"inputs.pipeline_config must be mapping YAML: {pipeline_value}")
        return

    _validate_pipeline_output_contract(pipeline, blockers)
    _validate_packet_pipeline_summary_contract(packet, pipeline, blockers)
    _validate_pipeline_scenario_paths(pipeline, pipeline_path, repo_root, blockers)

    for key in _REQUIRED_PIPELINE_COLLECTIONS:
        if not isinstance(pipeline.get(key), dict):
            blockers.append(f"pipeline_config.{key} must be mapping")

    training = _mapping(pipeline.get("training"))
    _require_non_empty_string(training, "model_id", blockers, prefix="pipeline_config.training")

    evaluation = _mapping(pipeline.get("evaluation"))
    for key in ("horizon", "dt", "workers", "campaign_workers"):
        if key not in evaluation:
            blockers.append(f"pipeline_config.evaluation.{key} must be set")


def _validate_pipeline_output_contract(
    pipeline: dict[str, Any],
    blockers: list[str],
) -> None:
    """Validate output root and expected-missing checkpoint provenance contract."""

    output = _mapping(pipeline.get("output"))
    root_value = output.get("root")
    if not isinstance(root_value, str) or not root_value.strip():
        blockers.append("pipeline_config.output.root must be non-empty path string")
    elif not _is_repo_output_path(root_value):
        blockers.append("pipeline_config.output.root must stay under output/")

    provenance = _mapping(output.get("provenance") or pipeline.get("provenance"))
    for key in _REQUIRED_PIPELINE_PROVENANCE:
        value = provenance.get(key)
        if not isinstance(value, str) or not value.strip():
            blockers.append(
                f"pipeline_config.output.provenance.{key} must be non-empty path string"
            )
        elif not _is_repo_output_path(value):
            blockers.append(f"pipeline_config.output.provenance.{key} must stay under output/")
    if provenance.get("status") != "expected_missing_until_training":
        blockers.append(
            "pipeline_config.output.provenance.status must be expected_missing_until_training"
        )


def _validate_packet_pipeline_summary_contract(
    packet: dict[str, Any],
    pipeline: dict[str, Any],
    blockers: list[str],
) -> None:
    """Require packet and pipeline to point at the same hard-seed summary artifact."""

    evaluation_config = _mapping(packet.get("evaluation_config"))
    packet_summary = evaluation_config.get("summary_path")
    provenance = _mapping(
        _mapping(pipeline.get("output")).get("provenance") or pipeline.get("provenance")
    )
    pipeline_summary = provenance.get("hard_seed_evaluation_summary")
    if not isinstance(packet_summary, str) or not isinstance(pipeline_summary, str):
        return
    if packet_summary != pipeline_summary:
        blockers.append(
            "evaluation_config.summary_path must match "
            "pipeline_config.output.provenance.hard_seed_evaluation_summary"
        )


def _validate_pipeline_scenario_paths(
    pipeline: dict[str, Any],
    pipeline_path: Path,
    repo_root: Path,
    blockers: list[str],
) -> None:
    """Validate public scenario and benchmark config inputs used by the launch packet."""

    scenarios = _mapping(pipeline.get("scenarios"))
    for key in _REQUIRED_PIPELINE_SCENARIOS:
        value = scenarios.get(key)
        if not isinstance(value, str) or not value.strip():
            blockers.append(f"pipeline_config.scenarios.{key} must be non-empty path string")
            continue
        resolved = _resolve_path(value, repo_root, base=pipeline_path.parent)
        if not resolved.is_file():
            blockers.append(f"pipeline_config.scenarios.{key} does not exist: {value}")


def _is_repo_output_path(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and path.parts[:1] == ("output",)


def _require_packet_output_path(
    mapping: dict[str, Any],
    key: str,
    blockers: list[str],
    prefix: str,
) -> None:
    value = mapping.get(key)
    if isinstance(value, str) and value.strip() and not _is_repo_output_path(value):
        blockers.append(f"{prefix}.{key} must stay under output/")


def _require_packet_config_path(
    mapping: dict[str, Any],
    key: str,
    blockers: list[str],
    prefix: str,
) -> None:
    value = mapping.get(key)
    if isinstance(value, str) and value.strip() and not value.startswith("configs/"):
        blockers.append(f"{prefix}.{key} must point under configs/")


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _require_non_empty_string(
    mapping: dict[str, Any],
    key: str,
    blockers: list[str],
    *,
    prefix: str | None = None,
) -> None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        field = f"{prefix}.{key}" if prefix else key
        blockers.append(f"{field} must be a non-empty string")


def _require_existing_repo_path(
    mapping: dict[str, Any],
    key: str,
    packet_path: Path,
    repo_root: Path,
    blockers: list[str],
) -> None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        blockers.append(f"inputs.{key} must be a non-empty path string")
        return
    resolved = _resolve_path(value, repo_root, base=packet_path.parent)
    if not resolved.is_file():
        blockers.append(f"inputs.{key} does not exist: {value}")


def _resolve_path(path: Path | str, repo_root: Path, *, base: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    if str(path).startswith(("configs/", "scripts/", "docs/", "robot_sf/", "tests/")):
        return (repo_root / candidate).resolve()
    return ((base or repo_root) / candidate).resolve()


__all__ = [
    "DECISION_BLOCKED",
    "DECISION_READY",
    "DEFAULT_PACKET",
    "PredictiveRetrainingReadinessError",
    "evaluate_retraining_readiness",
    "load_readiness_packet",
]
