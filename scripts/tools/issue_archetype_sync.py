"""Report and apply safe issue-body archetype metadata mirror candidates.

The issue body remains authoritative as a full metadata block. The ``report``
subcommand is read-only: it fetches issue metadata with ``gh api``, parses the
existing ``## Archetype Metadata`` block, and prints a dry-run JSON report.

Any archetype-metadata audit finding fails closed for label mirroring. If the
body block is malformed, missing required keys, or otherwise incomplete, this
tool does not propose typed-label mirrors from that block even when individual
fields like ``archetype`` look valid.

The ``apply`` subcommand builds the same pre-apply JSON report, checks the
GitHub API rate limit, and applies only the proposed typed labels when
``--confirm-apply-labels`` is passed. It never creates labels, keeps
``evidence_tier`` body-only by default, and never writes Project fields.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from scripts.tools.issue_template_audit import audit_archetype_metadata

SAFE_ARCHETYPE_LABEL_MAP: dict[str, str] = {
    "workflow": "type:workflow",
    "docs": "type:docs",
    "synthesis": "type:synthesis",
    "benchmark-campaign": "type:benchmark",
    "analysis": "type:analysis",
    "training-campaign": "type:training",
}

TYPED_LABEL_PREFIX = "type:"
RATE_LIMIT_FLOOR = 10
DEFAULT_REPO = "ll7/robot_sf_ll7"
APPLY_RATE_LIMIT_GUIDANCE = (
    "Uses REST for issue label writes after a core rate-limit preflight; Project v2 writes "
    "remain out of scope and should be batched separately per "
    "docs/context/issue_713_batch_first_issue_workflow.md."
)


@dataclass(frozen=True, slots=True)
class LabelCandidate:
    """One proposed or skipped typed-label mirror candidate."""

    source_field: str
    source_value: str
    label: str
    status: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ArchetypeSyncReport:
    """Dry-run report for one issue archetype metadata sync check."""

    issue_number: int
    dry_run: bool
    body_metadata: dict[str, Any] | None
    metadata_findings: list[str] = field(default_factory=list)
    existing_labels: list[str] = field(default_factory=list)
    proposed_label_additions: list[str] = field(default_factory=list)
    skipped_label_candidates: list[LabelCandidate] = field(default_factory=list)
    project_sync_mode: str = "report-only"
    project_field_candidates: list[dict[str, str]] = field(default_factory=list)
    mutation_plan: list[dict[str, Any]] = field(default_factory=list)
    rate_limit_guidance: str = (
        "Uses read-only gh REST calls; keep Project v2 writes out of this default path "
        "and batch GraphQL separately per docs/context/issue_713_batch_first_issue_workflow.md."
    )


def _run_gh_json(args: list[str]) -> Any:
    """Run a read-only gh command and decode JSON output.

    Returns:
        Decoded JSON payload.
    """
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def check_rate_limit() -> dict[str, Any]:
    """Fetch the current GitHub API rate-limit status.

    Returns:
        Decoded rate-limit payload.

    Raises:
        RuntimeError: When core remaining is below ``RATE_LIMIT_FLOOR``.
    """
    payload = _run_gh_json(["gh", "api", "rate_limit"])
    if not isinstance(payload, dict):
        raise RuntimeError("Rate-limit payload did not decode to an object")
    resources = payload.get("resources")
    if not isinstance(resources, dict):
        raise RuntimeError("Rate-limit resources block is missing or invalid")
    core = resources.get("core")
    if not isinstance(core, dict):
        raise RuntimeError("Rate-limit core block is missing or invalid")
    remaining = core.get("remaining")
    if not isinstance(remaining, int):
        raise RuntimeError("Rate-limit core.remaining is missing or not an int")
    if remaining < RATE_LIMIT_FLOOR:
        raise RuntimeError(
            f"Core rate-limit remaining ({remaining}) is below the safe floor "
            f"({RATE_LIMIT_FLOOR}); refusing to mutate."
        )
    return payload


def apply_labels(repo: str, issue_number: int, labels: list[str]) -> list[dict[str, Any]]:
    """Apply a list of typed labels to a GitHub issue via REST POST.

    This must only be called after the rate-limit check passes.

    Note:
        Calls ``gh api -X POST repos/{repo}/issues/{issue_number}/labels``
        with a JSON body of ``{"labels": [...]}``.  Existing labels are
        preserved --- the endpoint adds labels rather than replacing them.
    """
    completed = subprocess.run(
        [
            "gh",
            "api",
            "-X",
            "POST",
            f"repos/{repo}/issues/{issue_number}/labels",
            "--input",
            "-",
        ],
        input=json.dumps({"labels": labels}),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("GitHub label mutation payload did not decode to a list")
    return payload


def _apply_mutation_plan(report: ArchetypeSyncReport, repo: str) -> list[dict[str, Any]]:
    """Return the ordered label mutation plan for an apply report."""
    if not report.proposed_label_additions:
        return []
    return [
        {
            "operation": "add_labels",
            "repo": repo,
            "issue_number": report.issue_number,
            "labels": list(report.proposed_label_additions),
            "api": f"POST repos/{repo}/issues/{report.issue_number}/labels",
        }
    ]


def fetch_issue_payload(repo: str, issue_number: int) -> dict[str, Any]:
    """Fetch issue body and labels through the GitHub REST API.

    Returns:
        Issue payload returned by ``gh api``.
    """
    payload = _run_gh_json(["gh", "api", f"repos/{repo}/issues/{issue_number}"])
    if not isinstance(payload, dict):
        raise RuntimeError("GitHub issue payload did not decode to an object")
    return payload


def fetch_repo_label_names(repo: str) -> set[str]:
    """Fetch existing repository label names using read-only REST pagination.

    Returns:
        Set of label names currently present in the repository.
    """
    payload = _run_gh_json(["gh", "api", f"repos/{repo}/labels", "--paginate"])
    if not isinstance(payload, list):
        raise RuntimeError("GitHub labels payload did not decode to a list")
    labels: set[str] = set()
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            labels.add(item["name"])
    return labels


def _issue_label_names(issue_payload: dict[str, Any]) -> list[str]:
    """Extract sorted issue label names from a GitHub issue payload.

    Returns:
        Sorted label names.
    """
    labels = issue_payload.get("labels")
    if not isinstance(labels, list):
        return []
    names = [
        item["name"]
        for item in labels
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    return sorted(names)


def _failed_metadata_skips(metadata: Any) -> list[LabelCandidate]:
    """Return skipped candidates for incomplete or malformed metadata blocks."""
    parsed = metadata.parsed_metadata
    archetype_value = ""
    target_label = ""
    evidence_tier_value = ""
    if isinstance(parsed, dict):
        archetype = parsed.get("archetype")
        if isinstance(archetype, str):
            archetype_value = archetype
            target_label = SAFE_ARCHETYPE_LABEL_MAP.get(archetype, "")
        evidence_tier = parsed.get("evidence_tier")
        if evidence_tier is not None:
            evidence_tier_value = str(evidence_tier)

    skipped = [
        LabelCandidate(
            source_field="archetype",
            source_value=archetype_value,
            label=target_label,
            status="skipped",
            reason="schema_missing" if metadata.parse_error is None else "malformed",
        )
    ]
    if evidence_tier_value:
        skipped.append(
            LabelCandidate(
                source_field="evidence_tier",
                source_value=evidence_tier_value,
                label="",
                status="skipped",
                reason="not_low_risk",
            )
        )
    return skipped


def build_sync_report(
    *,
    issue_number: int,
    issue_body: str,
    existing_labels: list[str],
    available_labels: set[str] | None = None,
    dry_run: bool = True,
) -> ArchetypeSyncReport:
    """Build a no-mutation issue archetype sync report.

    The issue-body metadata block is authoritative as a whole. If the metadata
    audit reports any finding, the report fails closed and proposes no typed
    label mirrors from that block, even when a single field parses cleanly.

    Returns:
        Structured dry-run report.
    """
    metadata = audit_archetype_metadata(issue_body)
    parsed = metadata.parsed_metadata
    findings = list(metadata.findings)
    proposed: list[str] = []
    skipped: list[LabelCandidate] = []

    if parsed is None or findings:
        skipped.extend(_failed_metadata_skips(metadata))
    else:
        archetype = parsed.get("archetype")
        if not isinstance(archetype, str):
            skipped.append(
                LabelCandidate(
                    source_field="archetype",
                    source_value=repr(archetype),
                    label="",
                    status="skipped",
                    reason="schema_missing",
                )
            )
        else:
            target_label = SAFE_ARCHETYPE_LABEL_MAP.get(archetype)
            existing_typed = sorted(
                label for label in existing_labels if label.startswith(TYPED_LABEL_PREFIX)
            )
            if target_label is None:
                skipped.append(
                    LabelCandidate(
                        source_field="archetype",
                        source_value=archetype,
                        label="",
                        status="skipped",
                        reason="not_low_risk",
                    )
                )
            elif available_labels is not None and target_label not in available_labels:
                skipped.append(
                    LabelCandidate(
                        source_field="archetype",
                        source_value=archetype,
                        label=target_label,
                        status="skipped",
                        reason="label_missing",
                    )
                )
            elif target_label in existing_labels:
                skipped.append(
                    LabelCandidate(
                        source_field="archetype",
                        source_value=archetype,
                        label=target_label,
                        status="skipped",
                        reason="already_present",
                    )
                )
            elif existing_typed:
                skipped.append(
                    LabelCandidate(
                        source_field="archetype",
                        source_value=archetype,
                        label=target_label,
                        status="skipped",
                        reason="ambiguous",
                    )
                )
            else:
                proposed.append(target_label)

        evidence_tier = parsed.get("evidence_tier")
        skipped.append(
            LabelCandidate(
                source_field="evidence_tier",
                source_value=str(evidence_tier),
                label="",
                status="skipped",
                reason="not_low_risk",
            )
        )

    return ArchetypeSyncReport(
        issue_number=issue_number,
        dry_run=dry_run,
        body_metadata=dict(parsed) if isinstance(parsed, dict) else None,
        metadata_findings=findings,
        existing_labels=sorted(existing_labels),
        proposed_label_additions=proposed,
        skipped_label_candidates=skipped,
        project_field_candidates=[
            {"field": "archetype", "mode": "report-only"},
            {"field": "evidence_tier", "mode": "report-only"},
        ],
        mutation_plan=[],
    )


def build_report_from_github(
    repo: str, issue_number: int, *, dry_run: bool = True
) -> ArchetypeSyncReport:
    """Fetch issue context and build a read-only sync report.

    Returns:
        Structured dry-run report.
    """
    issue_payload = fetch_issue_payload(repo, issue_number)
    body = str(issue_payload.get("body") or "")
    labels = _issue_label_names(issue_payload)
    available_labels = fetch_repo_label_names(repo)
    return build_sync_report(
        issue_number=issue_number,
        issue_body=body,
        existing_labels=labels,
        available_labels=available_labels,
        dry_run=dry_run,
    )


def _report_to_json(report: ArchetypeSyncReport) -> str:
    """Serialize a report to stable JSON.

    Returns:
        Pretty JSON string.
    """
    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report = subparsers.add_parser("report", help="Print a read-only sync report.")
    report.add_argument("--issue-number", type=int, required=True)
    report.add_argument("--repo", default=DEFAULT_REPO)
    report.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Kept for explicitness; report mode is always read-only.",
    )
    report.add_argument(
        "--body-file",
        type=Path,
        help="Optional local issue body for parser tests or offline inspection.",
    )
    report.add_argument(
        "--labels-json",
        type=Path,
        help="Optional JSON list of existing issue labels for --body-file mode.",
    )

    apply_parser = subparsers.add_parser(
        "apply", help="Build a report then apply proposed typed labels when confirmed."
    )
    apply_parser.add_argument("--issue-number", type=int, required=True)
    apply_parser.add_argument("--repo", default=DEFAULT_REPO)
    apply_parser.add_argument(
        "--confirm-apply-labels",
        action="store_true",
        default=False,
        help="Required gate before any label mutation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)

    if args.command not in ("report", "apply"):
        raise AssertionError(f"Unhandled command: {args.command}")

    if getattr(args, "body_file", None) is not None:
        body = args.body_file.read_text(encoding="utf-8")
        labels: list[str] = []
        if args.labels_json is not None:
            raw_labels = json.loads(args.labels_json.read_text(encoding="utf-8"))
            if not isinstance(raw_labels, list) or not all(
                isinstance(item, str) for item in raw_labels
            ):
                raise ValueError("--labels-json must contain a JSON list of strings")
            labels = raw_labels
        report = build_sync_report(
            issue_number=args.issue_number,
            issue_body=body,
            existing_labels=labels,
            available_labels=set(SAFE_ARCHETYPE_LABEL_MAP.values()),
            dry_run=True,
        )
    else:
        report = build_report_from_github(args.repo, args.issue_number, dry_run=True)

    if args.command != "apply":
        print(_report_to_json(report), end="")
        return 0

    if not args.confirm_apply_labels:
        print(_report_to_json(report), end="")
        print(
            "Re-run with --confirm-apply-labels to apply the proposed typed labels.",
            flush=True,
        )
        return 0

    report = replace(
        report,
        dry_run=False,
        mutation_plan=_apply_mutation_plan(report, args.repo),
        rate_limit_guidance=APPLY_RATE_LIMIT_GUIDANCE,
    )
    print(_report_to_json(report), end="")

    proposed = report.proposed_label_additions
    if not proposed:
        print("No proposed label additions; nothing to apply.")
        return 0

    check_rate_limit()
    apply_labels(args.repo, args.issue_number, proposed)
    print(f"Applied labels to issue #{args.issue_number}: {proposed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
