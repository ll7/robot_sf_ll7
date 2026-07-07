#!/usr/bin/env python3
"""Build the issue #1475 acceptance-criteria audit from tracked evidence."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.training.orca_residual_lineage_packet import (
    OrcaResidualLineagePacketError,
    validate_smoke_nominal_gate,
)

SCHEMA_VERSION = "issue-1475-acceptance-audit.v1"
DEFAULT_SMOKE_SUMMARY = Path(
    "docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17/summary.json"
)
DEFAULT_SOURCE_CHECKSUMS = Path(
    "docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17/"
    "source_slurm_checksum_manifest.sha256"
)
DEFAULT_CLOSURE_AUDIT = Path("docs/context/evidence/issue_1475_closure_audit_2026-07-06.md")
DEFAULT_OUTPUT = Path("docs/context/evidence/issue_1475_acceptance_audit_2026-07-06.json")
DEFAULT_STATE_SURFACE = Path("docs/context/issue_1475_state.yaml")


@dataclass(frozen=True)
class CriterionAudit:
    """One issue #1475 acceptance criterion mapped to repository evidence."""

    criterion: str
    status: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-friendly representation."""
        return {
            "criterion": self.criterion,
            "status": self.status,
            "evidence": self.evidence,
        }


def _repo_root_from(path: Path) -> Path:
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return data


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


def _state_surface_check(
    *,
    repo_root: Path,
    state_surface_path: Path,
    acceptance_evidence: list[CriterionAudit],
    closure_call: str,
) -> dict[str, Any]:
    """Check the append-only high-churn state row matches this audit."""

    state_surface = _load_yaml(repo_root / state_surface_path)
    entries = state_surface.get("entries")
    if not isinstance(entries, list) or not entries:
        return {
            "path": str(state_surface_path),
            "status": "invalid",
            "errors": ["state surface has no entries"],
        }
    latest = max(
        (entry for entry in entries if isinstance(entry, dict)),
        key=lambda entry: str(entry.get("recorded_at_utc", "")),
        default={},
    )
    errors: list[str] = []
    if state_surface.get("issue") != 1475:
        errors.append(f"issue must be 1475, got {state_surface.get('issue')!r}")
    closure_boundary = latest.get("closure_boundary", {})
    if closure_boundary.get("closure_call_for_this_pr") != closure_call:
        errors.append(
            "latest entry closure_call_for_this_pr "
            f"{closure_boundary.get('closure_call_for_this_pr')!r} != {closure_call!r}"
        )
    state_by_criterion = {
        item.get("criterion"): item.get("status")
        for item in latest.get("acceptance_evidence", [])
        if isinstance(item, dict)
    }
    for item in acceptance_evidence:
        if state_by_criterion.get(item.criterion) != item.status:
            errors.append(
                f"{item.criterion!r} status "
                f"{state_by_criterion.get(item.criterion)!r} != {item.status!r}"
            )
    return {
        "path": str(state_surface_path),
        "status": "valid" if not errors else "invalid",
        "latest_recorded_at_utc": latest.get("recorded_at_utc"),
        "entry_status": latest.get("status"),
        "integration_report_status": latest.get("integration_report", {}).get("status"),
        "errors": errors,
    }


def _build_integration_report(
    *,
    criteria: list[CriterionAudit],
    smoke_gate_status: str,
    smoke_gate_error: str,
    state_surface_path: Path,
) -> dict[str, Any]:
    """Summarize current closure blockers without transient queue state."""

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
            "status": smoke_gate_status,
            "error": smoke_gate_error,
        },
        "canonical_state_surface": str(state_surface_path),
        "next_empirical_action": (
            "Run one bounded ORCA-residual BC smoke rerun on a Slurm-capable host; "
            "only if validate_smoke_nominal_gate passes, escalate nominal and "
            "classify #1358 continuation/revise/stop."
        ),
    }


def _build_merged_pr_evidence() -> list[dict[str, Any]]:
    """Return merged PR evidence for the current issue #1475 closure audit."""

    return [
        {
            "pr": "#4561",
            "title": "Issue #1475 add ORCA-residual smoke nominal gate",
            "merged_at_utc": "2026-07-05T02:58:35Z",
            "evidence": (
                "Added CPU-only validate_smoke_nominal_gate() and CLI validation for "
                "required smoke telemetry, fallback/degraded status, durable artifact "
                "status, and smoke thresholds before nominal escalation."
            ),
            "criteria_supported": [
                "Fallback/degraded rows are not counted learned-residual success evidence.",
                "Smoke result recorded before nominal escalation.",
            ],
            "closure_effect": "partial",
        },
        {
            "pr": "#4661",
            "title": "Issue #1475: add executable acceptance audit",
            "merged_at_utc": "2026-07-06T15:44:15Z",
            "evidence": (
                "Added the executable criterion-to-evidence audit over tracked issue #1475 "
                "smoke artifacts and fail-closed closure_call=keep_open behavior."
            ),
            "criteria_supported": [
                "All acceptance criteria mapped to repository evidence.",
            ],
            "closure_effect": "partial",
        },
        {
            "pr": "#4667",
            "title": "Issue #1475: add high-churn closure state surface",
            "merged_at_utc": "2026-07-06T16:52:33Z",
            "evidence": (
                "Added docs/context/issue_1475_state.yaml as the canonical durable "
                "state surface for high-churn issue #1475 audit propagation."
            ),
            "criteria_supported": [
                "High-churn issue state recorded in one canonical durable surface.",
            ],
            "closure_effect": "partial",
        },
        {
            "pr": "#4678",
            "title": "Issue #1475: verify closure state surface",
            "merged_at_utc": "2026-07-06T18:12:15Z",
            "evidence": (
                "Verified the executable audit and canonical state surface stay aligned "
                "on acceptance statuses and blocked closure boundary."
            ),
            "criteria_supported": [
                "State surface agrees with executable acceptance evidence.",
            ],
            "closure_effect": "partial",
        },
        {
            "pr": "#4721",
            "title": "Issue #1475: add acceptance audit integration report",
            "merged_at_utc": "2026-07-07T02:26:36Z",
            "evidence": (
                "Added integration report fields naming remaining blockers, intentional "
                "compute exclusions, and the next empirical Slurm smoke action."
            ),
            "criteria_supported": [
                "Fragmentation guard satisfied by a consolidation/integration slice.",
                "Remaining blockers and next empirical action recorded.",
            ],
            "closure_effect": "partial_keep_open",
        },
        {
            "pr": "#4726",
            "title": "Issue #1475: add merged PR acceptance evidence",
            "merged_at_utc": "2026-07-07T03:10:15Z",
            "evidence": (
                "Added ordered merged-pr_evidence records for #4561/#4661/"
                "#4667/#4678/#4721 and posted the live issue state update "
                "that the audit still keeps #1475 open."
            ),
            "criteria_supported": [
                "Merged PR acceptance evidence mapped through PR #4721.",
                "Post-merge issue propagation recorded keep-open closure call.",
            ],
            "closure_effect": "post_merge_audit_keep_open",
        },
    ]


def build_audit(
    *,
    repo_root: Path,
    smoke_summary_path: Path = DEFAULT_SMOKE_SUMMARY,
    source_checksums_path: Path = DEFAULT_SOURCE_CHECKSUMS,
    closure_audit_path: Path = DEFAULT_CLOSURE_AUDIT,
    state_surface_path: Path = DEFAULT_STATE_SURFACE,
) -> dict[str, Any]:
    """Build a fail-closed issue #1475 acceptance audit."""

    smoke_summary = _load_json(repo_root / smoke_summary_path)
    source_checksums = _read_text(repo_root / source_checksums_path)
    closure_audit = _read_text(repo_root / closure_audit_path)

    try:
        smoke_gate = validate_smoke_nominal_gate(_smoke_gate_input(smoke_summary))
        smoke_gate_status = smoke_gate["status"]
        smoke_gate_error = ""
    except OrcaResidualLineagePacketError as exc:
        smoke_gate_status = "invalid"
        smoke_gate_error = str(exc)

    dataset_npz_recorded = _contains_checksum(
        source_checksums,
        "benchmarks/expert_trajectories/issue_1428_orca_residual_bc_progress_v1_smoke.npz",
    )
    checkpoint_recorded = _contains_checksum(
        source_checksums,
        "benchmarks/expert_policies/issue_1428_orca_residual_bc_progress_v1_policy_smoke.zip",
    )
    artifact_pointer_status = _field(smoke_summary, "artifact_pointer_status")
    nominal_escalation_allowed = bool(smoke_summary.get("nominal_escalation_allowed"))

    missing_smoke_fields = [
        field
        for field in (
            "residual_clipping_rate",
            "guard_veto_rate",
            "fallback_degraded_status",
            "artifact_pointer_status",
        )
        if _field(smoke_summary, field) in (None, "")
    ]

    criteria = [
        CriterionAudit(
            criterion="Residual dataset manifest NPZ recorded durable artifacts.",
            status="partially_met" if dataset_npz_recorded else "not_met",
            evidence=(
                f"{source_checksums_path} records the smoke NPZ checksum; "
                f"smoke artifact_pointer_status={artifact_pointer_status!r}, so a durable pointer "
                "is still not proven by the tracked smoke summary."
            ),
        ),
        CriterionAudit(
            criterion="Learned residual checkpoint pointer durable included in completion update.",
            status="partially_met" if checkpoint_recorded else "not_met",
            evidence=(
                f"{source_checksums_path} records the smoke checkpoint checksum; "
                f"smoke artifact_pointer_status={artifact_pointer_status!r}, so durable checkpoint "
                "pointer completion remains unproven."
            ),
        ),
        CriterionAudit(
            criterion=(
                "Diagnostics report includes ORCA command, raw residual, bounded residual, final "
                "guarded command, residual clipping rate, guard veto rate, fallback/degraded status."
            ),
            status="not_met" if missing_smoke_fields else "met",
            evidence=(
                f"{smoke_summary_path} missing required smoke evidence fields: "
                f"{missing_smoke_fields or 'none'}."
            ),
        ),
        CriterionAudit(
            criterion="Fallback/degraded rows are not counted learned-residual success evidence.",
            status="met",
            evidence=(
                f"validate_smoke_nominal_gate status={smoke_gate_status}; "
                "tracked claim boundary is failed-closed smoke evidence only."
            ),
        ),
        CriterionAudit(
            criterion="Smoke result recorded before nominal escalation.",
            status="met",
            evidence=(
                f"{smoke_summary_path} records status={smoke_summary.get('status')!r}, "
                f"success_rate={_field(smoke_summary, 'success_rate')!r}, "
                f"nominal_escalation_allowed={nominal_escalation_allowed!r}."
            ),
        ),
        CriterionAudit(
            criterion="Nominal result classified ready #1358 continuation, revise, stop.",
            status="not_met",
            evidence=(
                "No nominal result exists in tracked evidence; smoke gate remains "
                f"{smoke_gate_status!r}. Gate error: {smoke_gate_error or 'none'}"
            ),
        ),
    ]

    unmet_statuses = {"not_met", "partially_met"}
    status = "complete" if all(item.status == "met" for item in criteria) else "blocked"
    closure_call = "close" if status == "complete" else "keep_open"
    state_surface = _state_surface_check(
        repo_root=repo_root,
        state_surface_path=state_surface_path,
        acceptance_evidence=criteria,
        closure_call=closure_call,
    )
    integration_report = _build_integration_report(
        criteria=criteria,
        smoke_gate_status=smoke_gate_status,
        smoke_gate_error=smoke_gate_error,
        state_surface_path=state_surface_path,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 1475,
        "status": status,
        "closure_call": closure_call,
        "claim_boundary": (
            "CPU-only acceptance audit over tracked evidence; no Slurm/GPU submission, "
            "benchmark campaign, training run, or paper-facing claim."
        ),
        "checked_paths": [
            str(smoke_summary_path),
            str(source_checksums_path),
            str(closure_audit_path),
            str(state_surface_path),
        ],
        "smoke_gate": {
            "status": smoke_gate_status,
            "error": smoke_gate_error,
        },
        "acceptance_evidence": [item.to_dict() for item in criteria],
        "remaining_criteria": [
            item.to_dict() for item in criteria if item.status in unmet_statuses
        ],
        "integration_report": integration_report,
        "merged_pr_evidence": _build_merged_pr_evidence(),
        "state_surface": state_surface,
        "next_empirical_action": (
            "Run one bounded ORCA-residual BC smoke rerun on a Slurm-capable host; only if "
            "validate_smoke_nominal_gate passes, escalate nominal and classify #1358 "
            "continuation/revise/stop."
        ),
        "source_thread_summary": (
            "Issue #1475 live thread reviewed via gh api on 2026-07-07; latest issue "
            "comment is the 2026-07-07 post-PR #4726 state update keeping the issue open."
        ),
        "closure_audit_contains_issue_1475": "Issue #1475" in closure_audit,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Build the CPU-only issue #1475 criterion-to-evidence acceptance audit."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--smoke-summary", type=Path, default=DEFAULT_SMOKE_SUMMARY)
    parser.add_argument("--source-checksums", type=Path, default=DEFAULT_SOURCE_CHECKSUMS)
    parser.add_argument("--closure-audit", type=Path, default=DEFAULT_CLOSURE_AUDIT)
    parser.add_argument("--state-surface", type=Path, default=DEFAULT_STATE_SURFACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true", help="Write the JSON artifact to --output.")
    return parser


def main() -> int:
    """Build and print or write the audit artifact."""

    args = build_parser().parse_args()
    repo_root = _repo_root_from(args.repo_root)
    report = build_audit(
        repo_root=repo_root,
        smoke_summary_path=args.smoke_summary,
        source_checksums_path=args.source_checksums,
        closure_audit_path=args.closure_audit,
        state_surface_path=args.state_surface,
    )

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write:
        output_path = repo_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0 if report["status"] in {"complete", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
