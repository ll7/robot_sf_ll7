"""Fail-closed campaign-readiness aggregator for the learned-risk model v1 Slurm run.

Issue #1472 ("research: run learned risk model v1 Slurm campaign") is blocked until
its prerequisites are mechanically satisfied. Two canonical owners already validate
the separate halves of those prerequisites:

- :mod:`robot_sf.training.learned_risk_launch_packet` proves the *launch packet*:
  trace input contract against a tracked fixture, baseline packet, safety policy,
  and execution boundary (config/route metadata).
- :mod:`robot_sf.training.learned_risk_trace_manifest` proves the *durable trace
  manifest*: whether the durable trace + baseline artifact pointers are resolvable
  or training must fail closed.

This module owns neither contract. It is the thin *campaign-level* aggregator that
runs both owners and folds their results into one fail-closed launch decision so
#1472 has a single mechanical gate ("is the campaign launch-ready, or blocked, and
on what?") instead of two separate exit codes that a caller has to reconcile.

Decision boundary:

- Missing/unreadable config *files* raise :class:`LearnedRiskCampaignReadinessError`
  (operator error: nothing to evaluate).
- Every other failure -- an invalid launch packet, a structurally invalid manifest,
  or a well-formed-but-unresolved manifest -- is folded into a fail-closed
  ``campaign_blocked`` decision with the underlying blockers surfaced per gate. The
  decision is ``campaign_launch_ready`` only when *both* owners report ready.

It performs no Slurm submission, no training, no network fetch, and no artifact
promotion; a ready decision means "locally contract-complete", not "job launched".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_sf.training.learned_risk_launch_packet import (
    LearnedRiskLaunchPacketError,
    validate_launch_packet,
)
from robot_sf.training.learned_risk_trace_manifest import (
    DECISION_READY,
    LearnedRiskTraceManifestError,
    validate_trace_manifest,
)

# Default checked-in campaign inputs for the #1472 learned-risk model v1 run.
DEFAULT_LAUNCH_PACKET = "configs/training/learned_risk_model_issue_1395_launch_packet.yaml"
DEFAULT_TRACE_MANIFEST = "configs/training/learned_risk_trace_manifest_issue_2312.yaml"

# Campaign decision vocabulary. ``campaign_blocked`` is fail-closed: it is the
# decision for any non-ready gate so an unresolved or invalid input can never be
# mistaken for a launchable campaign.
CAMPAIGN_READY = "campaign_launch_ready"
CAMPAIGN_BLOCKED = "campaign_blocked"

_GATE_PASSED = "passed"
_GATE_BLOCKED = "blocked"


class LearnedRiskCampaignReadinessError(ValueError):
    """Raised when a campaign-readiness input file is missing or unreadable.

    This is reserved for operator error (a config path that does not point at a
    file). Contract failures inside an existing file are reported as fail-closed
    blocked gates, not raised, so the campaign decision stays evaluable.
    """


def evaluate_campaign_readiness(
    launch_packet_path: Path | str = DEFAULT_LAUNCH_PACKET,
    trace_manifest_path: Path | str = DEFAULT_TRACE_MANIFEST,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Aggregate the launch-packet and trace-manifest owners into one launch gate.

    Args:
        launch_packet_path: Learned-risk launch-packet YAML path (config metadata).
        trace_manifest_path: Durable trace-manifest YAML path (artifact metadata).
        repo_root: Repository root for resolving relative paths.

    Returns:
        Compact report with a per-gate breakdown, the aggregated
        ``campaign_state`` (``campaign_launch_ready`` only when both gates pass),
        the list of ``blocking_gates``, and the flattened ``blockers``.

    Raises:
        LearnedRiskCampaignReadinessError: If either config path is not a file, so
            the campaign cannot be evaluated at all.
    """
    root = (repo_root or Path.cwd()).resolve()
    launch_packet_path = _resolve_existing(launch_packet_path, root, "launch packet")
    trace_manifest_path = _resolve_existing(trace_manifest_path, root, "trace manifest")

    gates = [
        _evaluate_launch_packet_gate(launch_packet_path, root),
        _evaluate_trace_manifest_gate(trace_manifest_path, root),
    ]

    blocking_gates = [gate["name"] for gate in gates if gate["status"] != _GATE_PASSED]
    blockers = [
        f"{gate['name']}: {blocker}" for gate in gates for blocker in gate.get("blockers", [])
    ]
    campaign_state = CAMPAIGN_BLOCKED if blocking_gates else CAMPAIGN_READY
    return {
        "status": "ok",
        "candidate_id": "learned_risk_model_v1",
        "issue": 1472,
        "launch_packet": str(launch_packet_path),
        "trace_manifest": str(trace_manifest_path),
        "gates": gates,
        "campaign_ready": not blocking_gates,
        "campaign_state": campaign_state,
        "blocking_gates": blocking_gates,
        "blockers": blockers,
    }


def _resolve_existing(path: Path | str, repo_root: Path, label: str) -> Path:
    """Resolve a config path and require that it points at an existing file.

    Returns:
        The resolved absolute path.

    Raises:
        LearnedRiskCampaignReadinessError: If the resolved path is not a file.
    """
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
    if not resolved.is_file():
        raise LearnedRiskCampaignReadinessError(f"{label} is not a file: {path}")
    return resolved


def _evaluate_launch_packet_gate(launch_packet_path: Path, root: Path) -> dict[str, Any]:
    """Run the launch-packet owner and fold its result into a campaign gate.

    Returns:
        A gate mapping with ``name``, ``status``, ``summary``, and ``blockers``.
    """
    try:
        report = validate_launch_packet(launch_packet_path, repo_root=root)
    except LearnedRiskLaunchPacketError as exc:
        return {
            "name": "launch_packet",
            "status": _GATE_BLOCKED,
            "summary": "launch packet failed validation",
            "blockers": _split_error(exc),
        }
    return {
        "name": "launch_packet",
        "status": _GATE_PASSED,
        "summary": f"launch packet valid for {report['candidate_id']}",
        "blockers": [],
    }


def _evaluate_trace_manifest_gate(trace_manifest_path: Path, root: Path) -> dict[str, Any]:
    """Run the trace-manifest owner and fold its result into a campaign gate.

    A structurally invalid manifest (which the owner raises on) is treated as a
    fail-closed blocker at the campaign level rather than re-raised, so a single
    bad input never masks the other gate's status.

    Returns:
        A gate mapping with ``name``, ``status``, ``summary``, and ``blockers``.
    """
    try:
        report = validate_trace_manifest(trace_manifest_path, repo_root=root)
    except LearnedRiskTraceManifestError as exc:
        return {
            "name": "trace_manifest",
            "status": _GATE_BLOCKED,
            "summary": "trace manifest is structurally invalid",
            "blockers": _split_error(exc),
        }
    decision = report["training_readiness_decision"]
    status = _GATE_PASSED if decision == DECISION_READY else _GATE_BLOCKED
    return {
        "name": "trace_manifest",
        "status": status,
        "summary": f"trace manifest decision: {decision}",
        "blockers": list(report.get("blockers", [])),
    }


def _split_error(exc: Exception) -> list[str]:
    """Split a multi-line validation error into a flat list of blocker lines.

    Returns:
        Non-empty, bullet-stripped lines from the exception message.
    """
    text = str(exc)
    lines = [line.strip(" -") for line in text.splitlines()]
    return [line for line in lines if line]


__all__ = [
    "CAMPAIGN_BLOCKED",
    "CAMPAIGN_READY",
    "DEFAULT_LAUNCH_PACKET",
    "DEFAULT_TRACE_MANIFEST",
    "LearnedRiskCampaignReadinessError",
    "evaluate_campaign_readiness",
]
