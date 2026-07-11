#!/usr/bin/env python3
"""Build a public-safe SLURM-to-claim trace checklist.

The checker is metadata-only. It traces one selected job through a compact
retrieval/source manifest, tracked evidence directory, optional finalizer output,
optional reconciler output, and a spine-citable pointer. Missing ordinary inputs
become explicit fail-closed blockers instead of simulated success.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json
from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256
from scripts.tools.reconcile_slurm_evidence import _load_finalizer_report

SCHEMA_VERSION = "slurm-to-claim-trace-checklist.v1"
ALLOWED_DECISIONS = {"promote", "keep_diagnostic", "block", "stop"}
DURABLE_POINTER_PREFIXES = (
    "wandb://",
    "wandb-artifact://",
    "https://",
    "http://",
    "s3://",
    "gs://",
    "dvc://",
)


@dataclass(frozen=True)
class ChecklistCheck:
    """One checklist row with fail-closed remediation."""

    name: str
    status: str
    detail: str
    remediation: str | None = None


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    """Render paths without leaking host-specific absolute prefixes."""

    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_decision(value: str | None) -> str | None:
    """Normalize bounded #3425 claim decision labels."""

    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def _find_source_run(payload: dict[str, Any], job_id: str) -> dict[str, Any] | None:
    """Find a job run inside a compact source manifest."""

    runs = payload.get("runs")
    if not isinstance(runs, list):
        return None
    for run in runs:
        if isinstance(run, dict) and str(run.get("job_id", "")).strip() == job_id:
            return run
    return None


def _check_source_manifest(
    source_manifest: Path,
    *,
    job_id: str,
    repo_root: Path,
) -> tuple[ChecklistCheck, dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Check retrieval/source manifest and return selected run metadata."""

    if not source_manifest.exists():
        return (
            ChecklistCheck(
                "retrieval_source_manifest",
                "blocked",
                f"source manifest not found: {_repo_relative(source_manifest, repo_root=repo_root)}",
                "retrieve compact h600 source manifest before finalizer trace",
            ),
            None,
            None,
            None,
        )
    try:
        payload = _load_json(source_manifest)
    except RuntimeError as exc:
        return (
            ChecklistCheck(
                "retrieval_source_manifest",
                "blocked",
                str(exc),
                "regenerate source manifest as valid compact JSON",
            ),
            None,
            None,
            None,
        )
    run = _find_source_run(payload, job_id)
    if run is None:
        return (
            ChecklistCheck(
                "retrieval_source_manifest",
                "blocked",
                f"job {job_id} not listed in {_repo_relative(source_manifest, repo_root=repo_root)}",
                "refresh source manifest so the target job is traceable",
            ),
            payload,
            None,
            None,
        )
    digest = _sha256(source_manifest)
    return (
        ChecklistCheck(
            "retrieval_source_manifest",
            "pass",
            f"job {job_id} listed in {_repo_relative(source_manifest, repo_root=repo_root)}",
        ),
        payload,
        run,
        digest,
    )


def _parse_sha256s(path: Path) -> dict[str, str]:
    """Parse a SHA256SUMS file into path-to-digest mapping."""

    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, _, relpath = stripped.partition("  ")
        if not relpath:
            digest, _, relpath = stripped.partition(" ")
        entries[relpath.strip()] = digest.strip()
    return entries


def _check_evidence_dir(
    evidence_dir: Path, *, repo_root: Path
) -> tuple[list[ChecklistCheck], list[str]]:
    """Check tracked compact evidence directory and checksum spine."""

    rel_dir = _repo_relative(evidence_dir, repo_root=repo_root)
    if not evidence_dir.is_dir():
        return [
            ChecklistCheck(
                "evidence_directory",
                "blocked",
                f"evidence directory not found: {rel_dir}",
                "promote compact evidence files under docs/context/evidence",
            )
        ], []

    checks = [
        ChecklistCheck("evidence_directory", "pass", f"evidence directory present: {rel_dir}")
    ]
    checksum_file = evidence_dir / "SHA256SUMS"
    if not checksum_file.exists():
        checks.append(
            ChecklistCheck(
                "evidence_checksums",
                "blocked",
                f"missing checksum spine: {_repo_relative(checksum_file, repo_root=repo_root)}",
                "write SHA256SUMS for compact evidence files before citation",
            )
        )
        return checks, []

    entries = _parse_sha256s(checksum_file)
    mismatches: list[str] = []
    present: list[str] = []
    for relpath, expected in sorted(entries.items()):
        artifact = repo_root / relpath
        if not artifact.exists():
            mismatches.append(f"{relpath}: missing")
            continue
        actual = _sha256(artifact)
        if actual != expected:
            mismatches.append(f"{relpath}: checksum mismatch")
        else:
            present.append(relpath)
    if mismatches:
        checks.append(
            ChecklistCheck(
                "evidence_checksums",
                "blocked",
                "; ".join(mismatches),
                "regenerate compact evidence checksums after evidence changes",
            )
        )
    else:
        checks.append(
            ChecklistCheck(
                "evidence_checksums",
                "pass",
                f"{len(present)} checksum entries verified",
            )
        )
    return checks, present


def _check_finalizer(
    finalizer_manifest: Path | None,
    *,
    job_id: str,
    issue: int,
    repo_root: Path,
) -> tuple[list[ChecklistCheck], dict[str, Any] | None, str | None, str | None]:
    """Check finalizer manifest for the selected job."""

    if finalizer_manifest is None:
        return (
            [
                ChecklistCheck(
                    "finalizer_manifest",
                    "blocked",
                    "no finalizer manifest provided",
                    "run scripts/tools/slurm_job_finalize.py for the completed job",
                )
            ],
            None,
            None,
            None,
        )
    if not finalizer_manifest.exists():
        return (
            [
                ChecklistCheck(
                    "finalizer_manifest",
                    "blocked",
                    f"finalizer manifest not found: {_repo_relative(finalizer_manifest, repo_root=repo_root)}",
                    "preserve or regenerate the finalizer manifest",
                )
            ],
            None,
            None,
            None,
        )
    try:
        reports = _load_finalizer_report(finalizer_manifest)
    except RuntimeError as exc:
        return (
            [
                ChecklistCheck(
                    "finalizer_manifest",
                    "blocked",
                    str(exc),
                    "regenerate finalizer manifest with the supported schema",
                )
            ],
            None,
            None,
            None,
        )

    selected = next((report for report in reports if report.job_id == job_id), None)
    if selected is None:
        return (
            [
                ChecklistCheck(
                    "finalizer_manifest",
                    "blocked",
                    f"job {job_id} not found in {_repo_relative(finalizer_manifest, repo_root=repo_root)}",
                    "use the finalizer manifest produced for this job",
                )
            ],
            None,
            None,
            None,
        )

    checks = [
        ChecklistCheck(
            "finalizer_manifest",
            "pass",
            f"job {job_id} finalizer record loaded",
        )
    ]
    if selected.issue_number != issue:
        checks.append(
            ChecklistCheck(
                "finalizer_issue_traceability",
                "blocked",
                f"finalizer issue {selected.issue_number} does not match #{issue}",
                "regenerate finalizer manifest with the public issue number",
            )
        )
    else:
        checks.append(
            ChecklistCheck(
                "finalizer_issue_traceability",
                "pass",
                f"finalizer links to issue #{issue}",
            )
        )
    if selected.classification == "success" and not selected.durable_pointer:
        checks.append(
            ChecklistCheck(
                "finalizer_durable_pointer",
                "blocked",
                "successful finalizer lacks durable pointer",
                "promote compact artifacts to durable storage and record durable_uri",
            )
        )
    elif selected.classification == "success":
        checks.append(
            ChecklistCheck(
                "finalizer_durable_pointer",
                "pass",
                "successful finalizer carries a durable pointer",
            )
        )
    else:
        checks.append(
            ChecklistCheck(
                "finalizer_durable_pointer",
                "pass",
                f"finalizer classification is {selected.classification}; durable pointer not required",
            )
        )

    return checks, asdict(selected), selected.claim_decision, selected.claim_boundary


def _check_reconciliation(
    reconciliation: Path | None,
    *,
    job_id: str,
    repo_root: Path,
) -> tuple[list[ChecklistCheck], str | None, str | None]:
    """Check reconciler output for selected job."""

    if reconciliation is None:
        return (
            [
                ChecklistCheck(
                    "reconciliation_output",
                    "blocked",
                    "no reconciliation output provided",
                    "run scripts/tools/reconcile_slurm_evidence.py with the finalizer manifest",
                )
            ],
            None,
            None,
        )
    if not reconciliation.exists():
        return (
            [
                ChecklistCheck(
                    "reconciliation_output",
                    "blocked",
                    f"reconciliation output not found: {_repo_relative(reconciliation, repo_root=repo_root)}",
                    "preserve reconciler JSON output before claiming the slice",
                )
            ],
            None,
            None,
        )
    try:
        payload = _load_json(reconciliation)
    except RuntimeError as exc:
        return (
            [
                ChecklistCheck(
                    "reconciliation_output",
                    "blocked",
                    str(exc),
                    "regenerate reconciler output as valid JSON",
                )
            ],
            None,
            None,
        )

    checks: list[ChecklistCheck] = []
    errors = payload.get("errors", [])
    if errors:
        checks.append(
            ChecklistCheck(
                "reconciliation_errors",
                "blocked",
                "; ".join(str(error) for error in errors),
                "resolve reconciler errors before treating the trace as complete",
            )
        )
    else:
        checks.append(ChecklistCheck("reconciliation_errors", "pass", "reconciler errors empty"))

    bridge = payload.get("finalizer_bridge", {})
    rows = bridge.get("rows", []) if isinstance(bridge, dict) else []
    selected = next(
        (row for row in rows if isinstance(row, dict) and str(row.get("job_id", "")) == job_id),
        None,
    )
    if selected is None:
        checks.append(
            ChecklistCheck(
                "reconciliation_finalizer_bridge",
                "blocked",
                f"job {job_id} not present in finalizer_bridge rows",
                "rerun reconciler with the selected job finalizer manifest",
            )
        )
        return checks, None, None

    checks.append(
        ChecklistCheck(
            "reconciliation_finalizer_bridge",
            "pass",
            f"job {job_id} present in finalizer_bridge rows",
        )
    )
    return (
        checks,
        _normalize_decision(selected.get("claim_decision")),
        selected.get("claim_boundary"),
    )


def _check_spine_pointer(pointer: str | None, *, repo_root: Path) -> ChecklistCheck:
    """Check whether a pointer can be cited without local host state."""

    if pointer is None or not pointer.strip():
        return ChecklistCheck(
            "spine_citable_pointer",
            "blocked",
            "no spine-citable pointer provided",
            "point at a tracked docs/context/evidence artifact or durable URI",
        )
    value = pointer.strip()
    if value.startswith(DURABLE_POINTER_PREFIXES):
        return ChecklistCheck("spine_citable_pointer", "pass", f"durable URI pointer: {value}")
    path = (repo_root / value).resolve()
    try:
        rel = path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return ChecklistCheck(
            "spine_citable_pointer",
            "blocked",
            f"pointer escapes repository: {value}",
            "use a tracked docs/context/evidence path or durable URI",
        )
    if not rel.startswith("docs/context/evidence/"):
        return ChecklistCheck(
            "spine_citable_pointer",
            "blocked",
            f"repository pointer is outside docs/context/evidence: {rel}",
            "use a compact evidence artifact path under docs/context/evidence",
        )
    if not path.exists():
        return ChecklistCheck(
            "spine_citable_pointer",
            "blocked",
            f"pointer path does not exist: {rel}",
            "create the compact evidence artifact before citation",
        )
    return ChecklistCheck("spine_citable_pointer", "pass", f"tracked evidence pointer: {rel}")


def build_checklist(  # noqa: PLR0913
    *,
    job_id: str,
    issue: int,
    source_manifest: Path,
    evidence_dir: Path,
    finalizer_manifest: Path | None,
    reconciliation: Path | None,
    spine_pointer: str | None,
    decision: str | None = None,
    claim_boundary: str | None = None,
    generated_at: str | None = None,
    repo_root: Path = Path.cwd(),
) -> dict[str, Any]:
    """Build one fail-closed trace checklist."""

    repo_root = repo_root.resolve()
    checks: list[ChecklistCheck] = []

    source_check, source_payload, source_run, source_digest = _check_source_manifest(
        source_manifest,
        job_id=job_id,
        repo_root=repo_root,
    )
    checks.append(source_check)

    evidence_checks, verified_files = _check_evidence_dir(evidence_dir, repo_root=repo_root)
    checks.extend(evidence_checks)

    finalizer_checks, finalizer_report, finalizer_decision, finalizer_boundary = _check_finalizer(
        finalizer_manifest,
        job_id=job_id,
        issue=issue,
        repo_root=repo_root,
    )
    checks.extend(finalizer_checks)

    reconciliation_checks, reconciler_decision, reconciler_boundary = _check_reconciliation(
        reconciliation,
        job_id=job_id,
        repo_root=repo_root,
    )
    checks.extend(reconciliation_checks)
    checks.append(_check_spine_pointer(spine_pointer, repo_root=repo_root))

    selected_decision = _normalize_decision(decision) or reconciler_decision or finalizer_decision
    invalid_decision = selected_decision is not None and selected_decision not in ALLOWED_DECISIONS
    if invalid_decision:
        checks.append(
            ChecklistCheck(
                "claim_decision",
                "blocked",
                f"unsupported claim decision: {selected_decision}",
                "use promote, keep_diagnostic, block, or stop",
            )
        )
    elif selected_decision is None:
        checks.append(
            ChecklistCheck(
                "claim_decision",
                "blocked",
                "no claim decision found",
                "record the bounded decision before handoff",
            )
        )
    else:
        checks.append(
            ChecklistCheck("claim_decision", "pass", f"claim decision: {selected_decision}")
        )

    blockers = [asdict(check) for check in checks if check.status == "blocked"]
    status = "pass" if not blockers else "blocked"
    if status == "blocked" and selected_decision is None:
        selected_decision = "block"

    source_run_summary = None
    if source_run is not None:
        source_run_summary = {
            "job_id": str(source_run.get("job_id", "")),
            "run_label": source_run.get("run_label"),
            "campaign_id": (source_run.get("campaign") or {}).get("campaign_id")
            if isinstance(source_run.get("campaign"), dict)
            else None,
            "planner_keys": source_run.get("planner_keys", []),
            "source_sha256": source_run.get("source_sha256", {}),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "issue": issue,
        "job_id": job_id,
        "status": status,
        "claim_decision": selected_decision,
        "claim_boundary": claim_boundary or reconciler_boundary or finalizer_boundary,
        "checks": [asdict(check) for check in checks],
        "blockers": blockers,
        "retrieval": {
            "source_manifest": _repo_relative(source_manifest, repo_root=repo_root),
            "source_manifest_sha256": source_digest,
            "source_manifest_schema": source_payload.get("schema_version")
            if source_payload is not None
            else None,
            "run": source_run_summary,
        },
        "evidence": {
            "directory": _repo_relative(evidence_dir, repo_root=repo_root),
            "verified_checksum_files": verified_files,
            "spine_pointer": spine_pointer,
        },
        "finalizer": {
            "manifest": _repo_relative(finalizer_manifest, repo_root=repo_root)
            if finalizer_manifest is not None
            else None,
            "report": finalizer_report,
        },
        "reconciliation": {
            "path": _repo_relative(reconciliation, repo_root=repo_root)
            if reconciliation is not None
            else None,
        },
        "claim_boundary_note": (
            "Checklist status is a workflow/tooling trace only. It does not promote "
            "planner ranking, benchmark, paper, or dissertation claims."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render checklist report as compact Markdown."""

    lines = [
        f"# Issue #{report['issue']} SLURM-to-Claim Trace Checklist",
        "",
        f"- Job: `{report['job_id']}`",
        f"- Status: `{report['status']}`",
        f"- Claim decision: `{report['claim_decision']}`",
        f"- Claim boundary: {report.get('claim_boundary') or 'not recorded'}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        detail = str(check["detail"]).replace("\n", " ")
        lines.append(f"| `{check['name']}` | `{check['status']}` | {detail} |")
    if report["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            remediation = blocker.get("remediation") or "resolve blocker"
            lines.append(f"- `{blocker['name']}`: {remediation}")
    lines.extend(
        [
            "",
            "## Citation Boundary",
            "",
            report["claim_boundary_note"],
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True, help="Job id to trace.")
    parser.add_argument("--issue", type=int, default=3425, help="Expected issue number.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--finalizer-manifest", type=Path)
    parser.add_argument("--reconciliation", type=Path)
    parser.add_argument("--spine-pointer")
    parser.add_argument("--decision")
    parser.add_argument("--claim-boundary")
    parser.add_argument("--generated-at")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--json", action="store_true", help="Also print JSON to stdout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = _parse_args(argv)
    report = build_checklist(
        job_id=args.job_id,
        issue=args.issue,
        source_manifest=args.source_manifest,
        evidence_dir=args.evidence_dir,
        finalizer_manifest=args.finalizer_manifest,
        reconciliation=args.reconciliation,
        spine_pointer=args.spine_pointer,
        decision=args.decision,
        claim_boundary=args.claim_boundary,
        generated_at=args.generated_at,
        repo_root=args.repo_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
