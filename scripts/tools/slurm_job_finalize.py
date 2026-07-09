#!/usr/bin/env python3
"""Finalize a completed SLURM job into compact artifact evidence.

The helper is intentionally metadata-only: it classifies a known job state,
checks expected artifact paths, computes checksums for present files, and
writes small JSON/Markdown summaries. It does not submit jobs, upload
artifacts, or copy raw ``output/`` trees into git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "robot-sf-slurm-job-finalization.v1"
CONTROL_PLANE_RUN_ARTIFACTS = (
    "run_manifest.json",
    "episodes.parquet",
    "summary.json",
    "checksums.sha256",
    "stdout.log",
    "stderr.log",
    "environment.json",
)

SUCCESS_STATES = {"COMPLETED", "COMPLETING"}
FAILED_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}
INCOMPLETE_STATES = {"PENDING", "CONFIGURING", "RUNNING", "SUSPENDED", "REQUEUED"}
UNAVAILABLE_STATES = {"NOT_AVAILABLE", "UNKNOWN", "MISSING"}

# Recognized durable artifact-store URI schemes. The approved durable backend for sprint
# studies is Weights & Biases (``wandb://`` / ``wandb-artifact://``); see
# docs/context/issue_3075_durable_artifact_backend.md. Other schemes are accepted so a run
# may point at an equivalent durable store, but a bare local path is never durable.
DURABLE_URI_SCHEMES = (
    "wandb://",
    "wandb-artifact://",
    "https://",
    "http://",
    "s3://",
    "gs://",
    "dvc://",
)
ALLOWED_CLAIM_DECISIONS = {"promote", "keep_diagnostic", "block", "stop"}


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Compact metadata for one expected or optional artifact path."""

    path: str
    role: str
    required: bool
    exists: bool
    kind: str
    size_bytes: int | None
    sha256: str | None


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    """Return a stable display path relative to the repository when possible."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    """Resolve an artifact path relative to the repository root."""
    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _directory_digest(path: Path) -> tuple[int, str]:
    """Compute total byte size and deterministic digest for a directory tree."""
    digest = hashlib.sha256()
    total_size = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                total_size += len(chunk)
                digest.update(chunk)
        digest.update(b"\0")
    return total_size, digest.hexdigest()


def artifact_record(
    path: str | Path,
    *,
    repo_root: Path,
    role: str = "artifact",
    required: bool = True,
) -> ArtifactRecord:
    """Build compact metadata for one artifact path without copying it."""
    resolved = _resolve_path(path, repo_root=repo_root)
    display_path = _repo_relative(resolved, repo_root=repo_root)
    if not resolved.exists():
        return ArtifactRecord(
            path=display_path,
            role=role,
            required=required,
            exists=False,
            kind="missing",
            size_bytes=None,
            sha256=None,
        )
    if resolved.is_file():
        return ArtifactRecord(
            path=display_path,
            role=role,
            required=required,
            exists=True,
            kind="file",
            size_bytes=resolved.stat().st_size,
            sha256=_sha256_file(resolved),
        )
    if resolved.is_dir():
        size_bytes, digest = _directory_digest(resolved)
        return ArtifactRecord(
            path=display_path,
            role=role,
            required=required,
            exists=True,
            kind="directory",
            size_bytes=size_bytes,
            sha256=digest,
        )
    return ArtifactRecord(
        path=display_path,
        role=role,
        required=required,
        exists=True,
        kind="other",
        size_bytes=None,
        sha256=None,
    )


def normalize_job_state(job_state: str) -> str:
    """Normalize a SLURM-like job state token."""
    return job_state.strip().upper().replace("-", "_") or "UNKNOWN"


def validate_durable_uri(value: str | None) -> str | None:
    """Return a stripped durable-store URI, or raise ``ValueError`` for an unknown scheme.

    A blank or absent value is treated as "no durable URI yet" (the run stays
    pending-durable); a non-empty value must use a recognized durable scheme so a bare
    local ``output/`` path can never masquerade as durable evidence.
    """
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if not candidate.startswith(DURABLE_URI_SCHEMES):
        raise ValueError(
            f"durable URI {candidate!r} must use one of {DURABLE_URI_SCHEMES}; "
            "the approved durable backend is Weights & Biases (wandb://entity/project/artifact:ver)."
        )
    return candidate


def normalize_claim_decision(value: str | None) -> str | None:
    """Normalize bounded #3425 claim-decision labels."""
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    if normalized not in ALLOWED_CLAIM_DECISIONS:
        allowed = ", ".join(sorted(ALLOWED_CLAIM_DECISIONS))
        raise ValueError(f"claim decision {value!r} must be one of: {allowed}")
    return normalized


def classify_durable_status(classification: str, durable_uri: str | None) -> str:
    """Classify durable-store readiness, fail-closed.

    Only a ``success`` run with a recorded durable URI is ``durable``; a successful run
    without a durable pointer stays ``pending_durable`` (its artifacts still need
    promotion before they can be cited), and non-success runs are ``not_applicable``.
    """
    if classification != "success":
        return "not_applicable"
    return "durable" if durable_uri else "pending_durable"


def classify_finalization(
    *,
    job_state: str,
    artifacts: list[ArtifactRecord],
    manual_decision: bool = False,
) -> str:
    """Classify a SLURM job finalization using fail-closed artifact checks."""
    state = normalize_job_state(job_state)
    if manual_decision:
        return "manual_decision_required"
    if state in INCOMPLETE_STATES:
        return "incomplete"
    if state in FAILED_STATES:
        return "failed"
    if state in UNAVAILABLE_STATES:
        return "not_available"
    required = [artifact for artifact in artifacts if artifact.required]
    if not required:
        return "manual_decision_required"
    if any(not artifact.exists for artifact in required):
        return "missing_artifacts"
    if state in SUCCESS_STATES:
        return "success"
    return "manual_decision_required"


def _artifact_status(artifacts: list[ArtifactRecord]) -> str:
    """Summarize artifact availability."""
    required = [artifact for artifact in artifacts if artifact.required]
    if not required:
        return "no_required_artifacts_declared"
    if all(artifact.exists for artifact in required):
        return "all_required_present"
    if any(artifact.exists for artifact in required):
        return "partial_required_present"
    return "required_missing"


def issue_update_markdown(report: dict[str, Any]) -> str:
    """Return issue-ready Markdown summarizing the finalization result."""
    lines = [
        f"SLURM finalization for job `{report['job_id']}`:",
        "",
        f"- classification: `{report['classification']}`",
        f"- job state: `{report['job_state']}`",
        f"- artifact status: `{report['artifact_status']}`",
        f"- durable status: `{report.get('durable_status', 'not_applicable')}`"
        + (f" -> `{report['durable_uri']}`" if report.get("durable_uri") else ""),
    ]
    if report.get("claim_decision"):
        lines.append(f"- claim decision: `{report['claim_decision']}`")
    lines.extend(
        [
            f"- claim boundary: {report['claim_boundary']}",
            "",
            "| Artifact | Required | Status | SHA256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for artifact in report["artifacts"]:
        status = "present" if artifact["exists"] else "missing"
        digest = artifact["sha256"] or "n/a"
        lines.append(
            f"| `{artifact['path']}` | {str(artifact['required']).lower()} | {status} | `{digest}` |"
        )
    lines.extend(["", f"Next action: {report['next_action']}"])
    return "\n".join(lines)


def ledger_update_markdown(report: dict[str, Any]) -> str:
    """Return one compact ledger row for context notes or issue comments."""
    claim_decision = report.get("claim_decision") or "n/a"
    return (
        f"| #{report['issue_number']} | `{report['job_id']}` | "
        f"`{report['classification']}` | `{report['artifact_status']}` | "
        f"`{claim_decision}` | {report['next_action']} |"
    )


def _next_action(classification: str) -> str:
    """Return conservative next-action text for a classification."""
    return {
        "success": (
            "Review checksums, promote artifacts to a durable store if downstream work depends on "
            "them, and cite only this compact manifest until durable URIs exist."
        ),
        "missing_artifacts": (
            "Do not close as successful; locate or rerun the job so all required artifacts exist."
        ),
        "failed": "Record the failure reason, keep artifacts caveated, and decide rerun versus revise.",
        "incomplete": "Wait for completion before classifying artifacts or closing the issue.",
        "not_available": "Record the unavailable job state and avoid treating local outputs as evidence.",
        "manual_decision_required": (
            "A maintainer or follow-up issue must decide whether available files are sufficient."
        ),
    }[classification]


def build_finalization_report(  # noqa: PLR0913
    *,
    issue_number: int,
    job_id: str,
    job_state: str,
    expected_artifacts: list[str | Path],
    optional_artifacts: list[str | Path] | None = None,
    repo_root: Path = REPO_ROOT,
    manual_decision: bool = False,
    notes: str = "",
    durable_uri: str | None = None,
    claim_decision: str | None = None,
) -> dict[str, Any]:
    """Build a compact SLURM finalization report."""
    validated_uri = validate_durable_uri(durable_uri)
    normalized_claim_decision = normalize_claim_decision(claim_decision)
    artifacts = [
        artifact_record(path, repo_root=repo_root, role="expected", required=True)
        for path in expected_artifacts
    ]
    artifacts.extend(
        artifact_record(path, repo_root=repo_root, role="optional", required=False)
        for path in (optional_artifacts or [])
    )
    classification = classify_finalization(
        job_state=job_state,
        artifacts=artifacts,
        manual_decision=manual_decision,
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "issue_number": int(issue_number),
        "job_id": str(job_id),
        "job_state": normalize_job_state(job_state),
        "classification": classification,
        "artifact_status": _artifact_status(artifacts),
        "durable_uri": validated_uri,
        "durable_status": classify_durable_status(classification, validated_uri),
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "claim_boundary": (
            "compact local finalization manifest only; not durable benchmark evidence until "
            "artifacts have durable retrieval URIs and policy-specific validation"
        ),
        "next_action": _next_action(classification),
        "notes": notes,
    }
    if normalized_claim_decision is not None:
        report["claim_decision"] = normalized_claim_decision
    report["issue_update_markdown"] = issue_update_markdown(report)
    report["ledger_update_markdown"] = ledger_update_markdown(report)
    return report


def build_control_plane_finalization_report(  # noqa: PLR0913
    *,
    issue_number: int,
    job_id: str,
    job_state: str,
    run_root: str | Path,
    optional_artifacts: list[str | Path] | None = None,
    repo_root: Path = REPO_ROOT,
    manual_decision: bool = False,
    notes: str = "",
    durable_uri: str | None = None,
    claim_decision: str | None = None,
) -> dict[str, Any]:
    """Build a report using the July 2026 research-control-plane run contract."""
    root = Path(run_root)
    expected_artifacts = [root / name for name in CONTROL_PLANE_RUN_ARTIFACTS]
    return build_finalization_report(
        issue_number=issue_number,
        job_id=job_id,
        job_state=job_state,
        expected_artifacts=expected_artifacts,
        optional_artifacts=optional_artifacts,
        repo_root=repo_root,
        manual_decision=manual_decision,
        notes=notes,
        durable_uri=durable_uri,
        claim_decision=claim_decision,
    )


def write_report(
    report: dict[str, Any], output: Path, *, markdown_output: Path | None = None
) -> None:
    """Write JSON and optional Markdown finalization reports."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(report["issue_update_markdown"] + "\n", encoding="utf-8")


def _durable_uri_cli(value: str) -> str:
    """argparse adapter that rejects non-durable URI schemes with a clean CLI error."""
    try:
        validated = validate_durable_uri(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if validated is None:
        raise argparse.ArgumentTypeError("durable URI must be non-empty")
    return validated


def _claim_decision_cli(value: str) -> str:
    """argparse type for bounded claim-decision labels."""
    try:
        normalized = normalize_claim_decision(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if normalized is None:
        raise argparse.ArgumentTypeError("claim decision must be non-empty")
    return normalized


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Finalize a SLURM job into compact evidence.")
    parser.add_argument("--issue", type=int, required=True, help="GitHub issue number.")
    parser.add_argument("--job-id", required=True, help="SLURM job id or external run id.")
    parser.add_argument("--job-state", required=True, help="Observed SLURM job state.")
    artifact_source = parser.add_mutually_exclusive_group()
    artifact_source.add_argument(
        "--expected-artifact",
        action="append",
        default=[],
        help="Required artifact path, relative to repo root unless absolute. May be repeated.",
    )
    artifact_source.add_argument(
        "--control-plane-run-root",
        type=Path,
        help=(
            "Use the research-control-plane required artifact set under this run root "
            "instead of passing each expected artifact manually."
        ),
    )
    parser.add_argument(
        "--optional-artifact",
        action="append",
        default=[],
        help="Optional artifact path, relative to repo root unless absolute. May be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON report output path.")
    parser.add_argument("--markdown-output", type=Path, help="Optional issue-update Markdown path.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root.")
    parser.add_argument(
        "--manual-decision",
        action="store_true",
        help="Force manual_decision_required even if artifacts are present.",
    )
    parser.add_argument("--notes", default="", help="Optional short note copied into the report.")
    parser.add_argument(
        "--claim-decision",
        type=_claim_decision_cli,
        metavar="{promote,keep-diagnostic,keep_diagnostic,block,stop}",
        help=(
            "Bounded final disposition for the SLURM-to-claim slice. "
            "Accepted labels normalize to promote, keep_diagnostic, block, or stop."
        ),
    )
    parser.add_argument(
        "--durable-uri",
        type=_durable_uri_cli,
        help=(
            "Durable artifact-store URI for this run (approved backend: Weights & Biases, e.g. "
            "wandb://entity/project/artifact:version). Recorded in the manifest so "
            "reconcile_slurm_evidence can replace pending aliases; must use a recognized "
            "durable scheme."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    if args.control_plane_run_root is not None:
        report = build_control_plane_finalization_report(
            issue_number=args.issue,
            job_id=args.job_id,
            job_state=args.job_state,
            run_root=args.control_plane_run_root,
            optional_artifacts=args.optional_artifact,
            repo_root=args.repo_root,
            manual_decision=args.manual_decision,
            notes=args.notes,
            durable_uri=args.durable_uri,
            claim_decision=args.claim_decision,
        )
    else:
        report = build_finalization_report(
            issue_number=args.issue,
            job_id=args.job_id,
            job_state=args.job_state,
            expected_artifacts=args.expected_artifact,
            optional_artifacts=args.optional_artifact,
            repo_root=args.repo_root,
            manual_decision=args.manual_decision,
            notes=args.notes,
            durable_uri=args.durable_uri,
            claim_decision=args.claim_decision,
        )
    write_report(report, args.output, markdown_output=args.markdown_output)
    print(report["issue_update_markdown"])
    return 0 if report["classification"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
