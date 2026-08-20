"""Issue-specific registries for the declarative acceptance-audit runner.

The YAML contracts describe the stable acceptance surface.  This module owns
the small amount of repository-specific evidence loading and criterion logic
that cannot be represented declaratively.  Compatibility wrappers call the
two public builders below; they do not import one another.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

import yaml

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json
from robot_sf.benchmark.orca_residual_lane_readiness import assess_lane_readiness
from robot_sf.training.orca_residual_lineage_packet import (
    OrcaResidualLineagePacketError,
    validate_smoke_nominal_gate,
)
from scripts.validation.acceptance_audit_runner import (
    ContextBuilder,
    ContractValidationError,
    CriterionAudit,
    CriterionDefinition,
    CriterionEvaluator,
    ReportResolver,
    check_contract,
    run_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIR = Path(__file__).resolve().parent / "contracts"
CONTRACT_PATHS = {
    1358: CONTRACT_DIR / "issue_1358_acceptance_audit.v1.yaml",
    1475: CONTRACT_DIR / "issue_1475_acceptance_audit.v1.yaml",
}
DEFAULT_SMOKE_SUMMARY = Path(
    "docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17/summary.json"
)
DEFAULT_SOURCE_CHECKSUMS = Path(
    "docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17/"
    "source_slurm_checksum_manifest.sha256"
)
DEFAULT_1475_CLOSURE_AUDIT = Path("docs/context/evidence/issue_1475_closure_audit_2026-07-06.md")
DEFAULT_1475_STATE_SURFACE = Path("docs/context/issue_1475_state.yaml")
DEFAULT_1358_CLOSURE_AUDIT = Path("docs/context/issue_1358_closure_audit_2026-07-07.md")
DEFAULT_1358_STATE_SURFACE = Path("docs/context/issue_1358_state.yaml")


def _rooted(repo_root: Path, path: Path) -> Path:
    """Resolve a compatibility input path using the historical CLI contract."""

    return path if path.is_absolute() else repo_root / path


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"failed parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a YAML mapping")
    return data


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc


def _contains_checksum(checksums: str, suffix: str) -> bool:
    return any(line.strip().endswith(suffix) for line in checksums.splitlines())


def _field(summary: dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary[key]
    required = summary.get("required_smoke_evidence")
    if isinstance(required, dict):
        value = required.get(key)
        if value is not None:
            return value
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        return metrics.get(key)
    return None


def _smoke_gate_input(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten tracked closeout summaries into the smoke gate's public contract."""

    gate_summary = dict(summary)
    for key in (
        "success_rate",
        "collision_rate",
        "residual_clipping_rate",
        "guard_veto_rate",
        "fallback_degraded_status",
        "artifact_pointer_status",
    ):
        gate_summary.setdefault(key, _field(summary, key))
    return gate_summary


def _issue_1475_context(repo_root: Path, inputs: Mapping[str, Path]) -> Mapping[str, Any]:
    smoke_summary_path = inputs["smoke_summary_path"]
    source_checksums_path = inputs["source_checksums_path"]
    closure_audit_path = inputs["closure_audit_path"]
    state_surface_path = inputs["state_surface_path"]
    smoke_summary = _load_json(_rooted(repo_root, smoke_summary_path))
    source_checksums = _read_text(_rooted(repo_root, source_checksums_path))
    closure_audit = _read_text(_rooted(repo_root, closure_audit_path))

    try:
        smoke_gate = validate_smoke_nominal_gate(_smoke_gate_input(smoke_summary))
        smoke_gate_status = smoke_gate["status"]
        smoke_gate_error = ""
    except OrcaResidualLineagePacketError as exc:
        smoke_gate_status = "invalid"
        smoke_gate_error = str(exc)

    artifact_pointer_status = _field(smoke_summary, "artifact_pointer_status")
    return {
        "smoke_summary_path": smoke_summary_path,
        "source_checksums_path": source_checksums_path,
        "closure_audit_path": closure_audit_path,
        "state_surface_path": state_surface_path,
        "smoke_summary": smoke_summary,
        "source_checksums": source_checksums,
        "closure_audit": closure_audit,
        "smoke_gate_status": smoke_gate_status,
        "smoke_gate_error": smoke_gate_error,
        "smoke_gate": {"status": smoke_gate_status, "error": smoke_gate_error},
        "dataset_npz_recorded": _contains_checksum(
            source_checksums,
            "benchmarks/expert_trajectories/issue_1428_orca_residual_bc_progress_v1_smoke.npz",
        ),
        "checkpoint_recorded": _contains_checksum(
            source_checksums,
            "benchmarks/expert_policies/issue_1428_orca_residual_bc_progress_v1_policy_smoke.zip",
        ),
        "artifact_pointer_status": artifact_pointer_status,
        "nominal_escalation_allowed": bool(smoke_summary.get("nominal_escalation_allowed")),
        "missing_smoke_fields": [
            field
            for field in (
                "residual_clipping_rate",
                "guard_veto_rate",
                "fallback_degraded_status",
                "artifact_pointer_status",
            )
            if _field(smoke_summary, field) in (None, "")
        ],
        "closure_audit_contains_issue_1475": "Issue #1475" in closure_audit,
    }


def _issue_1475_residual_dataset(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    return CriterionAudit(
        criterion=definition.criterion,
        status="partially_met" if context["dataset_npz_recorded"] else "not_met",
        evidence=(
            f"{context['source_checksums_path']} records the smoke NPZ checksum; "
            f"smoke artifact_pointer_status={context['artifact_pointer_status']!r}, so a durable pointer "
            "is still not proven by the tracked smoke summary."
        ),
    )


def _issue_1475_residual_checkpoint(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    return CriterionAudit(
        criterion=definition.criterion,
        status="partially_met" if context["checkpoint_recorded"] else "not_met",
        evidence=(
            f"{context['source_checksums_path']} records the smoke checkpoint checksum; "
            f"smoke artifact_pointer_status={context['artifact_pointer_status']!r}, so durable checkpoint "
            "pointer completion remains unproven."
        ),
    )


def _issue_1475_diagnostics(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    missing = context["missing_smoke_fields"]
    return CriterionAudit(
        criterion=definition.criterion,
        status="not_met" if missing else "met",
        evidence=(
            f"{context['smoke_summary_path']} missing required smoke evidence fields: "
            f"{missing or 'none'}."
        ),
    )


def _issue_1475_fallback_exclusion(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    return CriterionAudit(
        criterion=definition.criterion,
        status="met",
        evidence=(
            f"validate_smoke_nominal_gate status={context['smoke_gate_status']}; "
            "tracked claim boundary is failed-closed smoke evidence only."
        ),
    )


def _issue_1475_smoke_before_nominal(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    summary = context["smoke_summary"]
    return CriterionAudit(
        criterion=definition.criterion,
        status="met",
        evidence=(
            f"{context['smoke_summary_path']} records status={summary.get('status')!r}, "
            f"success_rate={_field(summary, 'success_rate')!r}, "
            f"nominal_escalation_allowed={context['nominal_escalation_allowed']!r}."
        ),
    )


def _issue_1475_nominal_classification(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    return CriterionAudit(
        criterion=definition.criterion,
        status="not_met",
        evidence=(
            "No nominal result exists in tracked evidence; smoke gate remains "
            f"{context['smoke_gate_status']!r}. Gate error: "
            f"{context['smoke_gate_error'] or 'none'}"
        ),
    )


def _issue_1475_integration_report(
    context: Mapping[str, Any], criteria: list[CriterionAudit]
) -> dict[str, Any]:
    blockers_remaining = [
        {
            "criterion": item.criterion,
            "status": item.status,
            "why_blocking": item.evidence,
        }
        for item in criteria
        if item.status in {"not_met", "partially_met"}
    ]
    return {
        "status": "blocked" if blockers_remaining else "complete",
        "evidence_grade": "tracked_cpu_audit_plus_retrieved_failed_closed_smoke",
        "fragmentation_guard_response": (
            "Integration slice: consolidate executable audit, canonical state row, "
            "and remaining empirical action after multiple issue #1475 audit/state PRs."
        ),
        "blockers_remaining": blockers_remaining,
        "blockers_new": [],
        "blockers_intentional": [
            {
                "blocker": "No Slurm/GPU submission in this PR.",
                "why_intentional": (
                    "Current authorization forbids compute_submit and local.machine.md "
                    "sets allow_slurm_submission: false."
                ),
            },
            {
                "blocker": "No nominal escalation while smoke gate is invalid.",
                "why_intentional": (
                    "The issue smoke-to-nominal contract requires a passing smoke "
                    "summary before nominal work can count as evidence."
                ),
            },
        ],
        "smoke_gate": {
            "status": context["smoke_gate_status"],
            "error": context["smoke_gate_error"],
        },
        "canonical_state_surface": str(context["state_surface_path"]),
        "next_empirical_action": (
            "Run one bounded ORCA-residual BC smoke rerun on a Slurm-capable host; "
            "only if validate_smoke_nominal_gate passes, escalate nominal and "
            "classify #1358 continuation/revise/stop."
        ),
    }


def _context_smoke_gate(
    context: Mapping[str, Any], criteria: list[CriterionAudit]
) -> dict[str, Any]:
    del criteria
    return context["smoke_gate"]


def _context_closure_audit_contains_issue_1475(
    context: Mapping[str, Any], criteria: list[CriterionAudit]
) -> bool:
    del criteria
    return context["closure_audit_contains_issue_1475"]


ISSUE_1475_EVALUATORS: dict[str, CriterionEvaluator] = {
    "issue_1475_residual_dataset": _issue_1475_residual_dataset,
    "issue_1475_residual_checkpoint": _issue_1475_residual_checkpoint,
    "issue_1475_diagnostics": _issue_1475_diagnostics,
    "issue_1475_fallback_exclusion": _issue_1475_fallback_exclusion,
    "issue_1475_smoke_before_nominal": _issue_1475_smoke_before_nominal,
    "issue_1475_nominal_classification": _issue_1475_nominal_classification,
}
ISSUE_1475_RESOLVERS: dict[str, ReportResolver] = {
    "context_smoke_gate": _context_smoke_gate,
    "issue_1475_integration_report": _issue_1475_integration_report,
    "context_closure_audit_contains_issue_1475": (_context_closure_audit_contains_issue_1475),
}


def _stable_issue_1475_audit_summary(issue_1475_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Return child facts without volatile state-row metadata."""

    state_surface = issue_1475_audit["state_surface"]
    return {
        "status": issue_1475_audit["status"],
        "closure_call": issue_1475_audit["closure_call"],
        "remaining_criteria": issue_1475_audit["remaining_criteria"],
        "state_surface": {
            "path": state_surface["path"],
            "status": state_surface["status"],
            "errors": state_surface["errors"],
            "integration_report_status": state_surface.get("integration_report_status"),
        },
    }


def _issue_1358_context(repo_root: Path, inputs: Mapping[str, Path]) -> Mapping[str, Any]:
    readiness = assess_lane_readiness(repo_root, validate_packet=True)
    issue_1475_audit = build_issue_1475_audit(repo_root=repo_root)
    integration = readiness["integration_report"]
    local_handoff_ready = (
        readiness["overall_status"] == "blocked_on_followup"
        and integration["integration_status"] == "local_handoff_ready_parent_blocked"
        and not readiness["errors"]
    )
    child_1475_blocked = (
        issue_1475_audit["issue"] == 1475
        and issue_1475_audit["status"] == "blocked"
        and issue_1475_audit["closure_call"] == "keep_open"
    )
    state_1475_valid = issue_1475_audit["state_surface"]["status"] == "valid"
    return {
        "closure_audit_path": inputs["closure_audit_path"],
        "state_surface_path": inputs["state_surface_path"],
        "readiness": readiness,
        "integration": integration,
        "issue_1475_audit": issue_1475_audit,
        "local_handoff_ready": local_handoff_ready,
        "child_1475_blocked": child_1475_blocked,
        "state_1475_valid": state_1475_valid,
    }


def _issue_1358_candidate_design(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    readiness = context["readiness"]
    integration = context["integration"]
    return CriterionAudit(
        criterion=definition.criterion,
        status="met" if context["local_handoff_ready"] else "not_met",
        evidence=(
            "Readiness report status="
            f"{readiness['overall_status']!r}, prerequisites "
            f"{integration['local_contract']['prerequisites_ready']}/"
            f"{integration['local_contract']['prerequisites_total']} ready; "
            "merged PRs #1409/#1875/#3770 provide the local handoff surface."
        ),
    )


def _issue_1358_training_config(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    return CriterionAudit(
        criterion=definition.criterion,
        status="met" if context["local_handoff_ready"] else "not_met",
        evidence=(
            "Readiness report route set includes lineage validation, smoke candidate, "
            "and SLURM handoff command shapes; no commands are executed by this audit."
        ),
    )


def _issue_1358_checkpoint_lineage(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    child = context["issue_1475_audit"]
    return CriterionAudit(
        criterion=definition.criterion,
        status="not_met",
        evidence=(
            "Issue #1475 audit status="
            f"{child['status']!r}; remaining criteria include durable "
            "dataset/checkpoint/report artifacts from the next Slurm smoke rerun."
        ),
    )


def _issue_1358_smoke_nominal_sanity(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    child = context["issue_1475_audit"]
    return CriterionAudit(
        criterion=definition.criterion,
        status="not_met",
        evidence=(
            "Issue #1475 audit keeps fallback/degraded success fail-closed, but tracked smoke "
            f"gate status remains {child['smoke_gate']['status']!r}; no nominal result exists."
        ),
    )


def _issue_1358_comparator_report(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    del context
    return CriterionAudit(
        criterion=definition.criterion,
        status="not_met",
        evidence=(
            "No trained residual checkpoint and no scenario-stratified nominal report exist; "
            "comparison remains blocked on child #1475 durable evidence."
        ),
    )


def _issue_1358_result_classification(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    del context
    return CriterionAudit(
        criterion=definition.criterion,
        status="not_met",
        evidence=(
            "Issue #2445 closed an earlier progress-probe decision, but parent #1358 thread "
            "still requires #1475 durable evidence before continue/revise/stop classification."
        ),
    )


def _issue_1358_parent_stays_open(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    child = context["issue_1475_audit"]
    return CriterionAudit(
        criterion=definition.criterion,
        status=(
            "met" if context["child_1475_blocked"] and context["state_1475_valid"] else "not_met"
        ),
        evidence=(
            "Issue #1475 executable audit closure_call="
            f"{child['closure_call']!r}; state surface status="
            f"{child['state_surface']['status']!r}."
        ),
    )


def _issue_1358_no_training_children(
    definition: CriterionDefinition, context: Mapping[str, Any]
) -> CriterionAudit:
    del context
    return CriterionAudit(
        criterion=definition.criterion,
        status="met",
        evidence=(
            "This audit is CPU-only evidence generation; it does not add children, submit "
            "Slurm/GPU work, run training, or mutate planner behavior."
        ),
    )


def _context_readiness(
    context: Mapping[str, Any], criteria: list[CriterionAudit]
) -> dict[str, Any]:
    del criteria
    readiness = context["readiness"]
    integration = context["integration"]
    return {
        "overall_status": readiness["overall_status"],
        "integration_status": integration["integration_status"],
        "remaining_blocker_keys": integration["remaining_blocker_keys"],
        "errors": readiness["errors"],
    }


def _context_child_1475_summary(
    context: Mapping[str, Any], criteria: list[CriterionAudit]
) -> dict[str, Any]:
    del criteria
    return _stable_issue_1475_audit_summary(context["issue_1475_audit"])


ISSUE_1358_EVALUATORS: dict[str, CriterionEvaluator] = {
    "issue_1358_candidate_design": _issue_1358_candidate_design,
    "issue_1358_training_config": _issue_1358_training_config,
    "issue_1358_checkpoint_lineage": _issue_1358_checkpoint_lineage,
    "issue_1358_smoke_nominal_sanity": _issue_1358_smoke_nominal_sanity,
    "issue_1358_comparator_report": _issue_1358_comparator_report,
    "issue_1358_result_classification": _issue_1358_result_classification,
    "issue_1358_parent_stays_open": _issue_1358_parent_stays_open,
    "issue_1358_no_training_children": _issue_1358_no_training_children,
}
ISSUE_1358_RESOLVERS: dict[str, ReportResolver] = {
    "context_readiness": _context_readiness,
    "context_child_1475_summary": _context_child_1475_summary,
}

CONTEXT_BUILDERS: dict[str, ContextBuilder] = {
    "issue_1358": _issue_1358_context,
    "issue_1475": _issue_1475_context,
}


def build_issue_1475_audit(
    *,
    repo_root: Path,
    smoke_summary_path: Path = DEFAULT_SMOKE_SUMMARY,
    source_checksums_path: Path = DEFAULT_SOURCE_CHECKSUMS,
    closure_audit_path: Path = DEFAULT_1475_CLOSURE_AUDIT,
    state_surface_path: Path = DEFAULT_1475_STATE_SURFACE,
) -> dict[str, Any]:
    """Build issue #1475 through the shared declarative runner."""

    return run_contract(
        contract_path=CONTRACT_PATHS[1475],
        repo_root=repo_root,
        input_paths={
            "smoke_summary_path": smoke_summary_path,
            "source_checksums_path": source_checksums_path,
            "closure_audit_path": closure_audit_path,
            "state_surface_path": state_surface_path,
        },
        context_builders={"issue_1475": _issue_1475_context},
        evaluators=ISSUE_1475_EVALUATORS,
        report_resolvers=ISSUE_1475_RESOLVERS,
    )


def build_issue_1358_audit(
    *,
    repo_root: Path,
    closure_audit_path: Path = DEFAULT_1358_CLOSURE_AUDIT,
    state_surface_path: Path = DEFAULT_1358_STATE_SURFACE,
) -> dict[str, Any]:
    """Build issue #1358 through the shared declarative runner."""

    return run_contract(
        contract_path=CONTRACT_PATHS[1358],
        repo_root=repo_root,
        input_paths={
            "closure_audit_path": closure_audit_path,
            "state_surface_path": state_surface_path,
        },
        context_builders={"issue_1358": _issue_1358_context},
        evaluators=ISSUE_1358_EVALUATORS,
        report_resolvers=ISSUE_1358_RESOLVERS,
    )


def check_issue_contract(issue: int) -> None:
    """Validate one issue contract without reading its evidence inputs."""

    if issue == 1475:
        check_contract(
            CONTRACT_PATHS[issue],
            evaluator_names=ISSUE_1475_EVALUATORS,
            context_builder_names={"issue_1475"},
            report_resolver_names=ISSUE_1475_RESOLVERS,
        )
        return
    if issue == 1358:
        check_contract(
            CONTRACT_PATHS[issue],
            evaluator_names=ISSUE_1358_EVALUATORS,
            context_builder_names={"issue_1358"},
            report_resolver_names=ISSUE_1358_RESOLVERS,
        )
        return
    raise ContractValidationError(f"unsupported acceptance-audit issue #{issue}")
