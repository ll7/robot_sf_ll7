"""Fail-closed readiness checks for predictive retraining decision packets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

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


class PredictiveRetrainingReadinessError(ValueError):
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
