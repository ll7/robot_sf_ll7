"""Read-only readiness/preflight surface for the ORCA-residual learned-policy lane.

Issue #1358 is a parent/umbrella coordination issue: it asks whether a small learned
residual policy trained with ORCA command/risk features can improve progress while
staying inside Robot SF's safety and comparability gates. The parent does not execute
training directly; the actual learned-residual training campaign is SLURM-gated and
flows through child issue #1475 (smoke/nominal lineage job) and decision child #2445.

This module provides a *bounded, read-only* inventory of the lane so an operator (or an
autonomous routing pass) can answer three questions without touching planner behavior,
launching training, or submitting SLURM:

1. **Prerequisites** -- which local scaffolding surfaces (lineage packet, candidate
   configs, candidate-registry entries, policy-search runner, observed-evidence report)
   actually exist and validate.
2. **Routes** -- the canonical commands the lane uses (lineage validation, smoke
   candidate, SLURM hand-off), reported as informational strings, never executed here.
3. **Blockers** -- the declared external gates that keep the lane from completing
   autonomously (child #1475 continue/revise/stop classification, ``resource:slurm``
   training, and pending durable dataset/checkpoint artifacts).

The diagnostics contract and the lineage-packet schema are *not* re-implemented here.
This surface reuses :mod:`robot_sf.training.orca_residual_lineage_packet` as the single
source of truth (``REQUIRED_ORCA_RESIDUAL_DIAGNOSTICS`` and ``validate_launch_packet``),
per the AGENTS.md canonical-owner rule.

Status semantics (fail-closed on missing local scaffolding):

- ``prerequisites_incomplete`` -- a required local surface is absent or the lineage packet
  fails validation. This is a real, actionable failure; the lane cannot be handed off.
- ``blocked_on_followup`` -- every local prerequisite is present and valid, and the only
  remaining blockers are the declared external gates (child #1475 + SLURM + durable
  artifacts). This is the expected healthy state for a coordination-only parent lane.

There is intentionally no ``ready`` state: the parent lane stays blocked until child
#1475 classifies the learned-residual path continue/revise/stop with durable evidence.
This surface only confirms the local scaffolding is complete enough to hand off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.training.orca_residual_lineage_packet import (
    REQUIRED_ORCA_RESIDUAL_DIAGNOSTICS,
    OrcaResidualLineagePacketError,
    validate_launch_packet,
)

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = "orca-residual-lane-readiness.v1"

#: Local scaffolding the lane needs before a learned-residual training job can be handed
#: off to the SLURM child. Each path is repository-root-relative.
LINEAGE_PACKET_PATH = "configs/training/orca_residual/orca_residual_bc_issue_1428.yaml"

#: Candidate ids that must be registered (with ``training_required: true``) in the
#: policy-search candidate registry for the learned-residual lane.
REQUIRED_CANDIDATE_IDS = (
    "orca_residual_guarded_ppo_v0",
    "orca_residual_guarded_ppo_progress_v1",
)

CANDIDATE_REGISTRY_PATH = "docs/context/policy_search/candidate_registry.yaml"

ISSUE_1475_REQUIRED_SMOKE_EVIDENCE_FIELDS = (
    "residual_clipping_rate",
    "guard_veto_rate",
    "fallback_degraded_status",
    "artifact_pointer_status",
)

ISSUE_1475_LOCAL_PREFLIGHT_COMMANDS = (
    (
        "LOGURU_LEVEL=WARNING uv run python "
        "scripts/validation/validate_orca_residual_lineage_packet.py "
        "--config configs/training/orca_residual/orca_residual_bc_issue_1428.yaml --json"
    ),
    "scripts/dev/sbatch_orca_residual_bc_issue1475.sh --dry-run --no-status",
)


@dataclass(frozen=True)
class LanePrerequisite:
    """A local file the ORCA-residual lane needs before SLURM hand-off.

    Attributes:
        key: Stable identifier used in the readiness report.
        description: Human-readable role of the surface.
        path: Repository-root-relative path that must exist.
        kind: Coarse category for grouping in the report.
    """

    key: str
    description: str
    path: str
    kind: str


@dataclass(frozen=True)
class LaneBlocker:
    """A declared external gate that keeps the lane from completing autonomously.

    Attributes:
        key: Stable identifier used in the readiness report.
        reason: Why the lane cannot finish here.
        gate: The gating issue, resource, or artifact class.
    """

    key: str
    reason: str
    gate: str


@dataclass(frozen=True)
class LaneRoute:
    """An informational command shape the lane uses (never executed here).

    Attributes:
        key: Stable identifier used in the readiness report.
        description: What the command does.
        command_shape: Canonical command string (illustrative, not run).
    """

    key: str
    description: str
    command_shape: str


# Required local scaffolding surfaces. Missing any of these means the lane is not yet
# handoff-ready (``prerequisites_incomplete``).
REQUIRED_PREREQUISITES: tuple[LanePrerequisite, ...] = (
    LanePrerequisite(
        key="lineage_packet",
        description="Behavior-cloning lineage packet defining objective, observation "
        "contract, residual bounds, artifact lineage, and diagnostics.",
        path=LINEAGE_PACKET_PATH,
        kind="lineage_packet",
    ),
    LanePrerequisite(
        key="smoke_pretrain_config",
        description="Bounded smoke/pretrain training config routed through child #1475.",
        path="configs/training/orca_residual/orca_residual_bc_issue_1475_smoke_pretrain.yaml",
        kind="config",
    ),
    LanePrerequisite(
        key="candidate_v0",
        description="ORCA-residual guarded-PPO v0 benchmark-surface candidate config.",
        path="configs/policy_search/candidates/orca_residual_guarded_ppo_v0.yaml",
        kind="candidate_config",
    ),
    LanePrerequisite(
        key="candidate_progress_v1",
        description="ORCA-residual guarded-PPO progress-probe v1 candidate config.",
        path="configs/policy_search/candidates/orca_residual_guarded_ppo_progress_v1.yaml",
        kind="candidate_config",
    ),
    LanePrerequisite(
        key="candidate_registry",
        description="Policy-search candidate registry; must register the residual candidates.",
        path=CANDIDATE_REGISTRY_PATH,
        kind="registry",
    ),
    LanePrerequisite(
        key="policy_search_runner",
        description="Policy-search candidate runner used for the smoke/nominal stages.",
        path="scripts/validation/run_policy_search_candidate.py",
        kind="runner",
    ),
    LanePrerequisite(
        key="lineage_validator",
        description="Fail-closed CLI validator for the lineage packet.",
        path="scripts/validation/validate_orca_residual_lineage_packet.py",
        kind="runner",
    ),
    LanePrerequisite(
        key="materialize_script",
        description="Materializes the ORCA-residual candidate for a training job.",
        path="scripts/tools/materialize_orca_residual_candidate.py",
        kind="runner",
    ),
    LanePrerequisite(
        key="observed_evidence_report",
        description="2026-05-05 learning-hybrid report grounding the lane's research question.",
        path="docs/context/policy_search/reports/2026-05-05_learning_hybrid_policy_search.md",
        kind="report",
    ),
)

# Declared external gates. These are *expected* for a coordination-only parent lane and do
# not make the surface fail; they are the residual risk the operator hands off.
LANE_BLOCKERS: tuple[LaneBlocker, ...] = (
    LaneBlocker(
        key="child_classification_gate",
        reason="Parent #1358 stays blocked until child #1475 classifies the learned-residual "
        "path continue/revise/stop with durable evidence (decision child #2445).",
        gate="issue:1475",
    ),
    LaneBlocker(
        key="slurm_training_required",
        reason="Learned-residual training is resource:slurm and must run on an approved worker, "
        "not the shared local host. This surface never submits SLURM.",
        gate="resource:slurm",
    ),
    LaneBlocker(
        key="durable_artifacts_pending",
        reason="Durable residual dataset, checkpoint, and diagnostic-report artifacts are "
        "produced by the #1475 SLURM job; the lineage packet currently points at planned "
        "(not yet materialized) durable artifact URIs.",
        gate="artifact:durable_dataset_checkpoint",
    ),
)

# Canonical command shapes for the lane. Reported for operator orientation; never executed.
LANE_ROUTES: tuple[LaneRoute, ...] = (
    LaneRoute(
        key="validate_lineage_packet",
        description="Fail-closed validation of the lineage packet before any hand-off.",
        command_shape=(
            "LOGURU_LEVEL=WARNING uv run python "
            "scripts/validation/validate_orca_residual_lineage_packet.py --json"
        ),
    ),
    LaneRoute(
        key="smoke_candidate",
        description="Bounded policy-search smoke stage for the residual candidate (child #1475).",
        command_shape=(
            "LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 uv run python "
            "scripts/validation/run_policy_search_candidate.py "
            "--candidate orca_residual_guarded_ppo_progress_v1 --stage smoke --workers 1"
        ),
    ),
    LaneRoute(
        key="slurm_handoff",
        description="Learned-residual training/evaluation is submitted from child #1475, "
        "not from this parent.",
        command_shape="<approved SLURM submit path via issue #1475 / private ops>",
    ),
)


def _load_yaml(path: Path) -> Any:
    """Load a YAML file.

    Returns:
        Parsed YAML content, or None if the file is absent or unreadable.
    """
    if not path.is_file():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def _resolve(repo_root: Path, rel_path: str) -> Path:
    """Resolve a repository-root-relative path.

    Returns:
        The absolute resolved path.
    """
    return (repo_root / rel_path).resolve()


@dataclass
class PrerequisiteResult:
    """Per-prerequisite presence result.

    Attributes:
        key: Prerequisite identifier.
        present: Whether the file exists.
        path: Repository-root-relative path checked.
        kind: Prerequisite category.
        messages: Any actionable problems (missing file, registry gaps).
    """

    key: str
    present: bool
    path: str
    kind: str
    messages: list[str] = field(default_factory=list)


def _check_candidate_registry(
    repo_root: Path,
    required_ids: tuple[str, ...],
) -> list[str]:
    """Check the candidate registry for the required residual candidates.

    The candidates must be registered with ``training_required: true`` so the lane is
    routed as a learned (not config-only) campaign.

    Returns:
        Actionable messages for any missing or misconfigured candidate (empty when OK).
    """
    registry_path = _resolve(repo_root, CANDIDATE_REGISTRY_PATH)
    data = _load_yaml(registry_path)
    if not isinstance(data, dict):
        return [f"candidate registry could not be loaded: {CANDIDATE_REGISTRY_PATH}"]
    candidates = data.get("candidates")
    if not isinstance(candidates, dict):
        return ["candidate registry has no 'candidates' mapping"]
    messages: list[str] = []
    for candidate_id in required_ids:
        entry = candidates.get(candidate_id)
        if not isinstance(entry, dict):
            messages.append(f"candidate registry missing required candidate: {candidate_id}")
            continue
        if entry.get("training_required") is not True:
            messages.append(
                f"candidate {candidate_id} must set training_required: true "
                "(learned-residual lane is not config-only)"
            )
    return messages


def _validate_lineage_packet(repo_root: Path) -> list[str]:
    """Validate the lineage packet via the canonical validator.

    Returns:
        Error messages from the canonical validator (empty when the packet is valid).
    """
    packet_path = _resolve(repo_root, LINEAGE_PACKET_PATH)
    if not packet_path.is_file():
        return [f"lineage packet not found: {LINEAGE_PACKET_PATH}"]
    try:
        validate_launch_packet(packet_path, repo_root=repo_root)
    except OrcaResidualLineagePacketError as exc:
        return [f"lineage packet failed validation: {exc}"]
    return []


def _build_issue_1475_decision_packet(
    *,
    overall_status: str,
    errors: list[str],
) -> dict[str, Any]:
    """Build the read-only #1475 smoke/nominal readiness packet.

    Returns:
        JSON-serializable packet describing local readiness, no-submit scope,
        required smoke evidence, and nominal-escalation gates.
    """
    local_ready = overall_status == "blocked_on_followup"
    return {
        "issue": 1475,
        "task_class": "orca_residual_bc_smoke_decision_packet",
        "decision_status": (
            "ready_for_single_smoke_handoff" if local_ready else "local_packet_incomplete"
        ),
        "submission_allowed_from_this_checker": False,
        "run_nominal": False,
        "claim_boundary": (
            "Read-only launch/readiness packet only. This checker does not submit SLURM, "
            "does not run training, does not run a full benchmark campaign, and does not "
            "promote paper or dissertation claims."
        ),
        "local_preflight_commands": list(ISSUE_1475_LOCAL_PREFLIGHT_COMMANDS),
        "smoke_submission_command_shape": (
            "scripts/dev/sbatch_orca_residual_bc_issue1475.sh --episodes 3 "
            '--seeds "111:112:113" --no-status'
        ),
        "nominal_escalation_gate": {
            "allowed_by_packet": False,
            "required_before_nominal": [
                "single bounded smoke job completed on an approved SLURM worker",
                "summary.json and smoke JSONL present for the ORCA-residual candidate",
                "success_rate > 0.0 with no timeout_low_progress stop classification",
                "collision_rate == 0.0",
                "required smoke evidence fields present",
                "fallback_degraded_status excludes fallback/degraded success evidence",
                "artifact_pointer_status records durable retrieval or explicit non-durable status",
            ],
        },
        "required_smoke_evidence_fields": list(ISSUE_1475_REQUIRED_SMOKE_EVIDENCE_FIELDS),
        "local_blockers": errors,
    }


def _build_integration_report(
    *,
    overall_status: str,
    prerequisites: list[PrerequisiteResult],
    errors: list[str],
) -> dict[str, Any]:
    """Build a compact parent-level integration report for issue #1358.

    The report is intentionally derived from the same readiness primitives as the
    preflight checker. It gives reviewers and routing agents one coherent status
    surface without turning this parent issue into a local training or benchmark
    execution path.

    Returns:
        JSON-serializable parent integration status.
    """
    missing_or_invalid = [p.key for p in prerequisites if p.messages]
    local_handoff_ready = overall_status == "blocked_on_followup"
    return {
        "issue": 1358,
        "integration_status": (
            "local_handoff_ready_parent_blocked"
            if local_handoff_ready
            else "local_contract_incomplete"
        ),
        "new_capability": "parent_integration_status_report",
        "local_contract": {
            "prerequisites_total": len(prerequisites),
            "prerequisites_ready": sum(1 for p in prerequisites if p.present and not p.messages),
            "missing_or_invalid_prerequisites": missing_or_invalid,
        },
        "remaining_blocker_keys": [b.key for b in LANE_BLOCKERS],
        "next_empirical_action": (
            "Use child #1475 to collect one bounded smoke training/evaluation artifact "
            "set on an approved SLURM worker, then classify continue/revise/stop before "
            "any nominal escalation."
        ),
        "non_claims": [
            "no local training was run",
            "no SLURM job was submitted",
            "no benchmark campaign was run",
            "no paper or dissertation claim is promoted",
        ],
        "claim_boundary": (
            "Integration report only. A handoff-ready local contract is not learned-residual "
            "evidence and does not unblock parent #1358 without child smoke/nominal artifacts."
        ),
        "errors": errors,
    }


def assess_lane_readiness(
    repo_root: Path,
    *,
    validate_packet: bool = True,
) -> dict[str, Any]:
    """Assess ORCA-residual lane readiness as a bounded, read-only report.

    Args:
        repo_root: Repository root for resolving relative paths.
        validate_packet: When True, run the canonical lineage-packet validator. Disable
            only for fast structural checks where packet validity is asserted elsewhere.

    Returns:
        A JSON-serializable readiness report. ``overall_status`` is
        ``prerequisites_incomplete`` when any required local surface is missing or the
        lineage packet is invalid, otherwise ``blocked_on_followup`` (local scaffolding
        complete; only the declared external gates remain).
    """
    repo_root = repo_root.resolve()
    prerequisites: list[PrerequisiteResult] = []

    for prereq in REQUIRED_PREREQUISITES:
        path = _resolve(repo_root, prereq.path)
        present = path.is_file()
        messages: list[str] = []
        if not present:
            messages.append(f"missing required surface: {prereq.path}")
        prerequisites.append(
            PrerequisiteResult(
                key=prereq.key,
                present=present,
                path=prereq.path,
                kind=prereq.kind,
                messages=messages,
            )
        )

    # Registry-content check is attached to the candidate_registry prerequisite when present.
    registry_result = next((p for p in prerequisites if p.key == "candidate_registry"), None)
    if registry_result is not None and registry_result.present:
        registry_result.messages.extend(
            _check_candidate_registry(repo_root, REQUIRED_CANDIDATE_IDS)
        )

    # Lineage-packet validity is attached to the lineage_packet prerequisite when present.
    packet_messages: list[str] = []
    if validate_packet:
        lineage_result = next((p for p in prerequisites if p.key == "lineage_packet"), None)
        if lineage_result is not None and lineage_result.present:
            packet_messages = _validate_lineage_packet(repo_root)
            lineage_result.messages.extend(packet_messages)

    errors = [msg for p in prerequisites for msg in p.messages]
    overall_status = "prerequisites_incomplete" if errors else "blocked_on_followup"

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 1358,
        "overall_status": overall_status,
        "claim_boundary": (
            "Read-only inventory of local lane scaffolding, routes, and blockers. Does not "
            "submit SLURM, train policies, alter planner behavior, run benchmarks, or assert "
            "any benchmark/paper result. A 'blocked_on_followup' status only means the local "
            "scaffolding is handoff-complete; the lane remains gated by child #1475."
        ),
        "prerequisites": [
            {
                "key": p.key,
                "present": p.present,
                "path": p.path,
                "kind": p.kind,
                "messages": p.messages,
            }
            for p in prerequisites
        ],
        "required_diagnostics_contract": list(REQUIRED_ORCA_RESIDUAL_DIAGNOSTICS),
        "blockers": [{"key": b.key, "reason": b.reason, "gate": b.gate} for b in LANE_BLOCKERS],
        "routes": [
            {"key": r.key, "description": r.description, "command_shape": r.command_shape}
            for r in LANE_ROUTES
        ],
        "issue_1475_decision_packet": _build_issue_1475_decision_packet(
            overall_status=overall_status,
            errors=errors,
        ),
        "integration_report": _build_integration_report(
            overall_status=overall_status,
            prerequisites=prerequisites,
            errors=errors,
        ),
        "errors": errors,
    }


__all__ = [
    "CANDIDATE_REGISTRY_PATH",
    "ISSUE_1475_LOCAL_PREFLIGHT_COMMANDS",
    "ISSUE_1475_REQUIRED_SMOKE_EVIDENCE_FIELDS",
    "LANE_BLOCKERS",
    "LANE_ROUTES",
    "LINEAGE_PACKET_PATH",
    "REQUIRED_CANDIDATE_IDS",
    "REQUIRED_PREREQUISITES",
    "SCHEMA_VERSION",
    "LaneBlocker",
    "LanePrerequisite",
    "LaneRoute",
    "PrerequisiteResult",
    "assess_lane_readiness",
]
