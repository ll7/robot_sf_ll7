#!/usr/bin/env python3
"""CPU-only readiness gate for the SLURM-to-claim finalizer vertical slice.

Issue #3425 asks for one complete vertical slice through the existing SLURM
finalizer bridge: ``submission -> finalizer -> durable evidence -> summary ->
claim decision``. That slice can only run on a SLURM-capable host, and it must
**fail closed** when durable pointers or manifest linkage are missing rather than
simulating success. This script is the shared-PC-safe *pre-execution* half of
that contract: it inspects the inputs the bridge will consume and reports whether
the slice is ready to run, **without** submitting SLURM work, executing the
reconciler end-to-end, or promoting any claim.

It is a thin orchestrator. It reuses the canonical loaders and finalizer fields
from :mod:`scripts.tools.reconcile_slurm_evidence` (the finalizer-bridge owner
added in PR #3120) so the readiness gate evaluates exactly the inputs the real
reconciliation consumes; it does not re-implement queue/manifest/finalizer
parsing or durable-pointer extraction.

What it checks (each maps to an acceptance criterion of #3425):

* ``queue_present`` — a submission queue YAML is provided, parses, and is
  non-empty (the slice has a submission record to reconcile against).
* ``submission_manifests_present`` — at least one submission manifest is
  provided, parses, and contributes at least one job (manifest linkage source).
* ``finalizer_manifests_present`` — at least one finalizer manifest is provided
  and carries the expected finalization schema and a job id.
* ``finalizer_manifest_linkage`` — every finalizer ``job_id`` resolves to a
  submission-manifest job (the "manifest linkage missing" fail-closed case).
* ``durable_pointer_present`` — every *successful* finalizer carries a durable
  pointer (the "durable pointers missing" fail-closed case).
* ``claim_boundary_present`` — every finalizer carries a claim boundary so the
  downstream summary/claim decision stays inside an explicit boundary.
* ``evidence_root_present`` — advisory: the compact evidence root exists so the
  reconciler can confirm preservation.

Exit codes follow the repo preflight convention used by
``preflight_evidence_contract.py``:

* ``0`` — ready: every required check passed; the slice may proceed on a
  SLURM-capable host.
* ``1`` — blocked: at least one required prerequisite is missing. Each blocker
  carries the smallest external action needed to unblock it.
* ``2`` — input error: an input path could not be read or parsed.

CPU-only: no GPU, no SLURM submission, no network, no claim promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Make the repo root importable when run as a bare script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Canonical finalizer-bridge owner: loaders and finalizer fields live here, not
# in this script. We only orchestrate them into a go/no-go readiness verdict.
from scripts.tools.reconcile_slurm_evidence import (  # noqa: E402
    FinalizerReport,
    ManifestJob,
    QueueEntry,
    _load_finalizer_report,
    load_queue,
    load_submission_manifests,
)

SCHEMA_VERSION = "slurm-finalizer-preflight.v1"

# Finalizer classifications that mean the job produced final artifacts and is
# therefore expected to carry a durable pointer before a claim can be made.
_SUCCESS_CLASSIFICATIONS = {"success"}

# The readiness gate makes no research claim; it only reports whether the inputs
# for the vertical slice are present and provenance-complete.
CLAIM_BOUNDARY = (
    "Readiness gate only: confirms the SLURM-to-claim finalizer slice inputs are "
    "present and provenance-complete. It does not submit SLURM work, run the "
    "reconciler end-to-end, or promote any benchmark/research claim."
)

# Severities.
_REQUIRED = "required"
_ADVISORY = "advisory"

# Statuses.
_READY = "ready"
_BLOCKED = "blocked"
_SKIPPED = "skipped"


@dataclass(frozen=True)
class PreflightCheck:
    """One readiness check outcome.

    Attributes:
        name: Stable identifier for the check.
        status: ``ready``, ``blocked``, or ``skipped``.
        severity: ``required`` (a blocked check blocks the slice) or ``advisory``.
        detail: Human-readable description of what was observed.
        remediation: Smallest external action to clear a blocked check, or
            ``None`` when the check is ready or only advisory.
    """

    name: str
    status: str
    severity: str
    detail: str
    remediation: str | None = None


@dataclass(frozen=True)
class LoadedInputs:
    """Parsed preflight inputs plus per-input load errors.

    Load errors are kept separate from readiness blockers: a malformed input is
    an *input error* (exit 2), while a missing-but-well-formed prerequisite is a
    readiness *blocker* (exit 1).
    """

    queue: list[QueueEntry]
    manifest_jobs: list[ManifestJob]
    finalizers: list[FinalizerReport]
    manifest_load_warnings: list[str]
    load_errors: list[str]


def _load_inputs(
    *,
    queue_path: Path | None,
    submission_manifests: list[Path],
    finalizer_manifests: list[Path],
) -> LoadedInputs:
    """Load every provided input with the canonical loaders, collecting errors.

    Missing optional paths are not errors here; absence is evaluated later as a
    readiness blocker. Only genuinely malformed/unreadable inputs become
    ``load_errors`` (which the caller maps to exit code 2).
    """
    load_errors: list[str] = []

    queue: list[QueueEntry] = []
    if queue_path is not None and queue_path.exists():
        try:
            queue = load_queue(queue_path)
        except RuntimeError as exc:
            load_errors.append(f"queue: {exc}")

    manifest_jobs: list[ManifestJob] = []
    manifest_load_warnings: list[str] = []
    present_manifests = [path for path in submission_manifests if path.exists()]
    if present_manifests:
        manifest_jobs, manifest_errors, manifest_load_warnings = load_submission_manifests(
            present_manifests
        )
        load_errors.extend(f"submission_manifest: {err}" for err in manifest_errors)

    finalizers: list[FinalizerReport] = []
    for path in finalizer_manifests:
        if not path.exists():
            continue
        try:
            finalizers.extend(_load_finalizer_report(path))
        except RuntimeError as exc:
            load_errors.append(f"finalizer_manifest: {exc}")

    return LoadedInputs(
        queue=queue,
        manifest_jobs=manifest_jobs,
        finalizers=finalizers,
        manifest_load_warnings=manifest_load_warnings,
        load_errors=load_errors,
    )


def _check_queue(queue_path: Path | None, inputs: LoadedInputs) -> PreflightCheck:
    """Require a parseable, non-empty submission queue."""
    if queue_path is None:
        return PreflightCheck(
            name="queue_present",
            status=_BLOCKED,
            severity=_REQUIRED,
            detail="no submission queue provided",
            remediation="pass --queue pointing at the submission queue YAML for this slice",
        )
    if not queue_path.exists():
        return PreflightCheck(
            name="queue_present",
            status=_BLOCKED,
            severity=_REQUIRED,
            detail=f"submission queue not found: {queue_path}",
            remediation=f"create or point --queue at an existing queue YAML (got {queue_path})",
        )
    if not inputs.queue:
        return PreflightCheck(
            name="queue_present",
            status=_BLOCKED,
            severity=_REQUIRED,
            detail=f"submission queue has no entries: {queue_path}",
            remediation="add at least one queue entry for the slice before running the finalizer",
        )
    return PreflightCheck(
        name="queue_present",
        status=_READY,
        severity=_REQUIRED,
        detail=f"{len(inputs.queue)} queue entr(y/ies) loaded from {queue_path}",
    )


def _check_submission_manifests(
    submission_manifests: list[Path], inputs: LoadedInputs
) -> PreflightCheck:
    """Require at least one submission manifest contributing at least one job."""
    if not submission_manifests:
        return PreflightCheck(
            name="submission_manifests_present",
            status=_BLOCKED,
            severity=_REQUIRED,
            detail="no submission manifest provided",
            remediation="pass --submission-manifest with the SLURM submission record manifest",
        )
    missing = [str(path) for path in submission_manifests if not path.exists()]
    if missing:
        return PreflightCheck(
            name="submission_manifests_present",
            status=_BLOCKED,
            severity=_REQUIRED,
            detail=f"submission manifest(s) not found: {', '.join(sorted(missing))}",
            remediation="ensure each --submission-manifest path exists before running the finalizer",
        )
    if not inputs.manifest_jobs:
        return PreflightCheck(
            name="submission_manifests_present",
            status=_BLOCKED,
            severity=_REQUIRED,
            detail="submission manifest(s) contributed no jobs",
            remediation="record at least one submitted job (with slurm_job_id) in the manifest",
        )
    return PreflightCheck(
        name="submission_manifests_present",
        status=_READY,
        severity=_REQUIRED,
        detail=f"{len(inputs.manifest_jobs)} manifest job(s) loaded",
    )


def _check_finalizer_present(
    finalizer_manifests: list[Path], inputs: LoadedInputs
) -> PreflightCheck:
    """Require at least one finalizer manifest with a valid schema and job id."""
    if not finalizer_manifests:
        return PreflightCheck(
            name="finalizer_manifests_present",
            status=_BLOCKED,
            severity=_REQUIRED,
            detail="no finalizer manifest provided",
            remediation=(
                "run scripts/tools/slurm_job_finalize.py on the completed job and pass "
                "--finalizer-manifest with its output"
            ),
        )
    missing = [str(path) for path in finalizer_manifests if not path.exists()]
    if missing:
        return PreflightCheck(
            name="finalizer_manifests_present",
            status=_BLOCKED,
            severity=_REQUIRED,
            detail=f"finalizer manifest(s) not found: {', '.join(sorted(missing))}",
            remediation="ensure each --finalizer-manifest path exists",
        )
    if not inputs.finalizers:
        # Paths existed but loaded zero reports; a load_error will have been
        # recorded and surfaced as an input error (exit 2).
        return PreflightCheck(
            name="finalizer_manifests_present",
            status=_BLOCKED,
            severity=_REQUIRED,
            detail="finalizer manifest(s) yielded no valid finalization records",
            remediation="regenerate the finalizer manifest with scripts/tools/slurm_job_finalize.py",
        )
    return PreflightCheck(
        name="finalizer_manifests_present",
        status=_READY,
        severity=_REQUIRED,
        detail=f"{len(inputs.finalizers)} finalizer record(s) loaded",
    )


def _check_finalizer_linkage(inputs: LoadedInputs) -> PreflightCheck:
    """Require every finalizer job to resolve to a submission-manifest job."""
    if not inputs.finalizers:
        return PreflightCheck(
            name="finalizer_manifest_linkage",
            status=_SKIPPED,
            severity=_REQUIRED,
            detail="no finalizer records to link",
            remediation="resolve finalizer_manifests_present first",
        )
    manifest_job_ids = {job.slurm_job_id for job in inputs.manifest_jobs if job.slurm_job_id}
    unlinked = sorted(
        finalizer.job_id
        for finalizer in inputs.finalizers
        if finalizer.job_id not in manifest_job_ids
    )
    if unlinked:
        return PreflightCheck(
            name="finalizer_manifest_linkage",
            status=_BLOCKED,
            severity=_REQUIRED,
            detail=f"finalizer job(s) with no submission-manifest linkage: {', '.join(unlinked)}",
            remediation=(
                "add the matching submission-manifest job (matching slurm_job_id) for each "
                "finalizer, or pass the manifest that already records it"
            ),
        )
    return PreflightCheck(
        name="finalizer_manifest_linkage",
        status=_READY,
        severity=_REQUIRED,
        detail="every finalizer job links to a submission-manifest job",
    )


def _check_durable_pointer(inputs: LoadedInputs) -> PreflightCheck:
    """Require every successful finalizer to carry a durable pointer."""
    successful = [
        finalizer
        for finalizer in inputs.finalizers
        if finalizer.classification in _SUCCESS_CLASSIFICATIONS
    ]
    if not inputs.finalizers:
        return PreflightCheck(
            name="durable_pointer_present",
            status=_SKIPPED,
            severity=_REQUIRED,
            detail="no finalizer records to inspect",
            remediation="resolve finalizer_manifests_present first",
        )
    if not successful:
        # No successful finalizers means there is no claim to gate yet; the slice
        # would legitimately resolve to keep-diagnostic/block, not a blocker for
        # the preflight itself.
        return PreflightCheck(
            name="durable_pointer_present",
            status=_READY,
            severity=_REQUIRED,
            detail="no successful finalizers require a durable pointer yet",
        )
    missing = sorted(finalizer.job_id for finalizer in successful if not finalizer.durable_pointer)
    if missing:
        return PreflightCheck(
            name="durable_pointer_present",
            status=_BLOCKED,
            severity=_REQUIRED,
            detail=f"successful finalizer(s) missing a durable pointer: {', '.join(missing)}",
            remediation=(
                "promote each job's artifacts to a durable store (wandb/s3/gs/dvc/https) and "
                "record the durable URI in the finalizer manifest before claiming"
            ),
        )
    return PreflightCheck(
        name="durable_pointer_present",
        status=_READY,
        severity=_REQUIRED,
        detail=f"{len(successful)} successful finalizer(s) carry a durable pointer",
    )


def _check_claim_boundary(inputs: LoadedInputs) -> PreflightCheck:
    """Require every finalizer to carry a claim boundary."""
    if not inputs.finalizers:
        return PreflightCheck(
            name="claim_boundary_present",
            status=_SKIPPED,
            severity=_REQUIRED,
            detail="no finalizer records to inspect",
            remediation="resolve finalizer_manifests_present first",
        )
    missing = sorted(
        finalizer.job_id for finalizer in inputs.finalizers if not finalizer.claim_boundary
    )
    if missing:
        return PreflightCheck(
            name="claim_boundary_present",
            status=_BLOCKED,
            severity=_REQUIRED,
            detail=f"finalizer(s) missing a claim boundary: {', '.join(missing)}",
            remediation=(
                "regenerate the finalizer manifest so each record carries claim_boundary "
                "(scripts/tools/slurm_job_finalize.py emits this)"
            ),
        )
    return PreflightCheck(
        name="claim_boundary_present",
        status=_READY,
        severity=_REQUIRED,
        detail="every finalizer carries a claim boundary",
    )


def _check_evidence_root(evidence_root: Path | None) -> PreflightCheck:
    """Advisory: the compact evidence root should exist for preservation checks."""
    if evidence_root is None:
        return PreflightCheck(
            name="evidence_root_present",
            status=_READY,
            severity=_ADVISORY,
            detail="no evidence root configured; reconciler will skip preservation checks",
        )
    if not evidence_root.exists():
        return PreflightCheck(
            name="evidence_root_present",
            status=_BLOCKED,
            severity=_ADVISORY,
            detail=f"evidence root does not exist: {evidence_root}",
            remediation=(
                "create the compact evidence root or pass --evidence-root at the directory the "
                "reconciler should scan"
            ),
        )
    return PreflightCheck(
        name="evidence_root_present",
        status=_READY,
        severity=_ADVISORY,
        detail=f"evidence root present: {evidence_root}",
    )


def preflight(
    *,
    queue_path: Path | None,
    submission_manifests: list[Path],
    finalizer_manifests: list[Path],
    evidence_root: Path | None = None,
    generated_at: str | None = None,
) -> dict:
    """Evaluate readiness of the SLURM-to-claim finalizer slice inputs.

    Returns a deterministic report mapping; ``ready`` is true only when every
    required check passed. ``input_errors`` (malformed/unreadable inputs) are
    reported separately and force exit code 2 in :func:`main`.
    """
    inputs = _load_inputs(
        queue_path=queue_path,
        submission_manifests=submission_manifests,
        finalizer_manifests=finalizer_manifests,
    )

    checks = [
        _check_queue(queue_path, inputs),
        _check_submission_manifests(submission_manifests, inputs),
        _check_finalizer_present(finalizer_manifests, inputs),
        _check_finalizer_linkage(inputs),
        _check_durable_pointer(inputs),
        _check_claim_boundary(inputs),
        _check_evidence_root(evidence_root),
    ]

    blockers = [
        {"check": check.name, "detail": check.detail, "remediation": check.remediation}
        for check in checks
        if check.severity == _REQUIRED and check.status == _BLOCKED
    ]
    advisories = [
        {"check": check.name, "detail": check.detail, "remediation": check.remediation}
        for check in checks
        if check.severity == _ADVISORY and check.status == _BLOCKED
    ]
    ready = not blockers and not inputs.load_errors

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "ready": ready,
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {
            "queue": str(queue_path) if queue_path is not None else None,
            "submission_manifests": sorted(str(path) for path in submission_manifests),
            "finalizer_manifests": sorted(str(path) for path in finalizer_manifests),
            "evidence_root": str(evidence_root) if evidence_root is not None else None,
        },
        "checks": [asdict(check) for check in checks],
        "blockers": blockers,
        "advisories": advisories,
        "input_errors": sorted(inputs.load_errors),
        "warnings": sorted(set(inputs.manifest_load_warnings)),
    }


def _render_human(report: dict) -> str:
    """Render a compact human-readable readiness report."""
    lines = [
        f"schema_version: {report['schema_version']}",
        f"result: {'READY' if report['ready'] else 'BLOCKED'}",
        "",
        "checks:",
    ]
    for check in report["checks"]:
        marker = {
            _READY: "ok",
            _BLOCKED: "BLOCK" if check["severity"] == _REQUIRED else "warn",
            _SKIPPED: "skip",
        }.get(check["status"], check["status"])
        lines.append(f"- [{marker}] {check['name']}: {check['detail']}")
    if report["input_errors"]:
        lines.append("")
        lines.append("input errors (exit 2):")
        lines.extend(f"- {err}" for err in report["input_errors"])
    if report["blockers"]:
        lines.append("")
        lines.append("blockers — smallest external action to unblock:")
        for blocker in report["blockers"]:
            lines.append(f"- {blocker['check']}: {blocker['remediation']}")
    if report["advisories"]:
        lines.append("")
        lines.append("advisories (non-blocking):")
        for advisory in report["advisories"]:
            lines.append(f"- {advisory['check']}: {advisory['remediation']}")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="preflight_slurm_finalizer.py",
        description=(
            "CPU-only readiness gate for the SLURM-to-claim finalizer slice (#3425). "
            "Exit 0 = ready; 1 = blocked (missing prerequisite); 2 = input error. "
            "Does not submit SLURM work or promote claims."
        ),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=None,
        help="Submission queue YAML the slice reconciles against.",
    )
    parser.add_argument(
        "--submission-manifest",
        action="append",
        default=[],
        type=Path,
        help="Submission manifest path (SLURM submission record). May be repeated.",
    )
    parser.add_argument(
        "--finalizer-manifest",
        action="append",
        default=[],
        type=Path,
        help="Finalizer manifest path (scripts/tools/slurm_job_finalize.py output). May be repeated.",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="Optional compact evidence root directory (advisory check).",
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Optional stable generated_at timestamp for reproducible machine output.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the readiness gate. Returns the process exit code."""
    args = _parse_args(argv)
    report = preflight(
        queue_path=args.queue,
        submission_manifests=list(args.submission_manifest),
        finalizer_manifests=list(args.finalizer_manifest),
        evidence_root=args.evidence_root,
        generated_at=args.generated_at,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_human(report), end="")

    if report["input_errors"]:
        return 2
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
