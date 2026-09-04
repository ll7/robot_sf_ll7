#!/usr/bin/env python3
"""Merge-queue status-check gate enforcing the gh-pr-merger fail-closed preflight.

This module backs the ``Merge Queue Gate`` status check (issue #6274). Once a
maintainer makes that check required for GitHub's native merge queue, the queue
must not auto-merge a PR until this gate passes. The gate enforces the same
fail-closed preflight as ``gh-pr-merger``:

  - non-draft state,
  - current ``merge-ready`` label,
  - a current exact-head ``gate-verdict: accepted @ <head_sha>`` trailer
    (reuses ``scripts.dev.pr_loop_policy.has_current_accepted_gate_verdict``),
  - a current ``pr-metadata: reconciled @ <digest>`` trailer binding the
    final PR title/body to the review evidence,
  - a successful ``changed-coverage-gate`` check on the exact live PR head, or
    a proven docs-only changed-file set covered by CI's ``paths-ignore`` rules,
  - no unresolved actionable review threads,
  - no outstanding explicitly requested reviewers,
  - no unexpired trusted exact-head ``review-claim`` held by an active review
    worker,
  - a current closing-discipline recheck over the PR body and commit messages,
  - the merge queue's ``ALLGREEN`` strategy, so every constituent entry must
    pass its own required gate check,
  - staleness-free base (fresh by construction inside the merge queue, where the
    base SHA equals current ``main``; evaluated against ``main`` in ``--pr`` mode).

Issue #7515: the gate also rejects stacked ancestry.  When the evaluated PR
snapshot carries an ``ancestry`` block from ``scripts.dev.stack_ancestry`` whose
state is undeclared, mismatched, invalidated, or a declared stack (anything but
``clean``), the gate fails closed with ``stacked_ancestry_not_independently_mergeable``
because a stacked PR must never be merged independently before its parent merges.

It emits a ``merge_queue_gate.v1`` audit record with the evaluated head SHA,
queue merging strategy, base SHA, label set, metadata digest and trailer
statuses, exact-head changed-coverage status, staleness verdict, CI conclusion,
reviewer-thread resolution, requested-reviewer status, and closing-discipline
status, and review-claim status so the merge decision is inspectable and
reproducible.

The pure function ``evaluate_merge_gate`` is deterministic and exercised by
``--self-test`` (the validation contract for issue #6274). The CLI resolves a
live PR (``--pr`` or ``--from-event`` for a ``merge_group`` payload), evaluates,
prints the audit JSON, and appends a ``GITHUB_STEP_SUMMARY`` block. Native
``merge_group`` evaluation always exits 0 on pass / 1 on fail (fail closed).
Source-PR diagnostics may opt into ``--advisory`` so a truthfully failed audit
does not present ordinary implementation CI as red.

Why a separate gate instead of relying on labels alone: issue #6274 observed an
external/parallel auto-merge path merging PRs without ``merge-ready`` or without
a current exact-head gate verdict. This gate covers only native ``merge_group``
events after the required-check configuration is active. It does not locate,
alter, or prove coverage of a direct merge dispatcher; that remaining #6274
work needs separate evidence before the issue can close.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

# Make the sibling ``scripts.dev`` package importable when this file is run as a
# standalone script (``python scripts/dev/merge_queue_gate.py``). Under pytest or
# ``uv run`` the project root is already on ``sys.path``; this insert is a no-op
# there and only matters for direct script invocation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci.pr_contract_check import (  # noqa: E402
    check_closes_discipline,
    get_pr_commit_messages,
)
from scripts.dev.check_pr_ci_status import (  # noqa: E402
    _enrich_rest_check_runs,
    _latest_check_runs,
    _rest_check_runs_to_rollup,
    _rollup_conclusion,
    _rollup_status,
)
from scripts.dev.github_graphql_retry import GraphQLRetryOutcome, run_with_retry  # noqa: E402
from scripts.dev.github_quota import quota_reset_handoff  # noqa: E402
from scripts.dev.pr_loop_policy import (  # noqa: E402
    active_review_claim,
    has_any_pr_metadata_verdict,
    has_current_accepted_gate_verdict,
    has_current_pr_metadata_verdict,
)
from scripts.dev.pr_metadata import (  # noqa: E402
    extract_metadata_digests,
    find_not_ready_body_sentinels,
    metadata_digest,
    metadata_trailer,
)
from scripts.dev.snapshot_pr_queue import (  # noqa: E402
    _extract_base_policies,
    _extract_gate_verdicts,
    _extract_metadata_verdicts,
)

AUDIT_SCHEMA = "merge_queue_gate.v1"
RECEIPT_REVIEW_CHECK_NAMES = frozenset({"pr-contract-check"})
NON_REQUIRED_RECEIPT_CHECK_NAMES = frozenset({"coderabbit"})

# CI rollup classification constants (mirror scripts/dev/check_pr_ci_status.py
# so the gate does not couple to that module's private helpers).
FAILURE_CONCLUSIONS = {
    "failure",
    "error",
    "cancelled",
    "stale",
    "timed_out",
    "action_required",
    "startup_failure",
}
SUCCESS_CONCLUSIONS = {"neutral", "skipped", "success"}
PENDING_STATUSES = {"expected", "in_progress", "pending", "queued", "requested", "waiting"}
COMPLETED_STATUS = "completed"
GATE_WORKFLOW_NAME = "Merge Queue Gate"
GATE_JOB_NAME = "merge-queue-gate"
CHANGED_COVERAGE_CHECK_NAME = "changed-coverage-gate"
SNAPSHOT_PROVENANCE_SCHEMA = "single_account_merge_evidence_provenance.v1"
# Keep this list in lockstep with the top-level ``paths-ignore`` filters in
# ``.github/workflows/ci.yml``.  The merge gate may need to explain why that
# workflow did not create an exact-head changed-coverage check for a PR.
CI_PATHS_IGNORE_PATTERNS = ("**/*.md", "docs/**")
CHANGED_COVERAGE_NOT_REQUIRED = "not_required"
_CHANGED_FILES_PAGE_SIZE = 100
_MAX_CHANGED_FILES_PAGES = 100
_REST_EVIDENCE_PAGE_SIZE = 100
_MAX_REST_EVIDENCE_PAGES = 10
_REST_CHECK_STATUSES = frozenset(
    {"queued", "in_progress", "completed", "requested", "waiting", "pending"}
)
_REST_COMMIT_STATUS_STATES = frozenset({"error", "failure", "pending", "success"})
_CHANGED_FILE_STATUSES = frozenset(
    {"added", "changed", "copied", "deleted", "modified", "renamed", "removed"}
)
_MAX_CHANGED_TEST_CONTENT_FILES = 200
_MAX_GITHUB_PR_FILES = 3000

# GitHub's native merge-queue ref is exposed as either the full
# ``refs/heads/gh-readonly-queue/<base>/pr-<number>-<source-sha>`` ref or its
# branch-name form in event payload surfaces. The ``pr-<number>`` component
# identifies the source pull request; it is not the source branch name. Keep
# this strict so a changed queue-ref format fails closed rather than selecting
# an unrelated PR.
_MERGE_QUEUE_REF_RE = re.compile(
    r"^(?:refs/heads/)?gh-readonly-queue/(?P<base>.+)/pr-"
    r"(?P<number>[1-9][0-9]*)-(?P<head_sha>[0-9a-f]{7,40})$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MergeGateAudit:
    """Inspectable, reproducible merge-queue gate verdict for one PR evaluation."""

    schema: str
    pr: int | None
    head_sha: str
    merge_group_head_sha: str
    merge_group_head_binding: str
    queue_merging_strategy: str
    base_sha: str
    main_sha: str
    labels: list[str]
    draft: bool
    ci_overall: str
    changed_coverage_status: str
    changed_coverage_head_sha: str
    gate_verdict_status: str
    metadata_digest: str
    metadata_verdict_status: str
    staleness_verdict: str
    thread_resolution: str
    reviewer_request_status: str
    review_claim_status: str
    ancestry_state: str
    merge_ready: bool
    passed: bool
    body_narrative_status: str = "clean"
    body_not_ready_sentinels: list[str] = field(default_factory=list)
    closing_discipline_status: str = "not_evaluated"
    closing_discipline_blockers: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit as a plain JSON-able dict."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MergeGroupPR:
    """Source PR identity encoded by a canonical GitHub merge-queue ref."""

    number: int
    head_sha: str


def _label_names(pr: dict[str, Any]) -> list[str]:
    """Return compact label-name strings from a PR dict.

    Accepts both the compact-snapshot form (``["merge-ready", ...]`` strings) and
    the raw ``gh`` form (``[{"name": "merge-ready", ...}, ...]`` objects).
    """
    labels = pr.get("labels") or []
    if not isinstance(labels, list):
        return []
    names: list[str] = []
    for label in labels:
        if isinstance(label, str):
            names.append(label)
        elif isinstance(label, dict):
            name = label.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def _gate_verdict_status(pr: dict[str, Any], head_sha: str) -> str:
    """Classify the exact-head gate-verdict trailer state.

    Returns ``accepted`` when a current exact-head ``gate-verdict: accepted``
    trailer exists, ``missing`` otherwise (including an empty head SHA, a missing
    trailer, or a trailer whose SHA does not identify the exact head). Mirrors
    the fail-closed contract in ``pr_loop_policy.has_current_accepted_gate_verdict``.
    """
    if not head_sha:
        return "missing"
    return "accepted" if has_current_accepted_gate_verdict(pr, head_sha) else "missing"


def _metadata_verdict_status(pr: dict[str, Any], digest: str) -> str:
    """Classify the final title/body reconciliation trailer state."""
    if not digest:
        return "missing"
    if has_current_pr_metadata_verdict(pr, digest):
        return "accepted"
    return "stale" if has_any_pr_metadata_verdict(pr) else "missing"


def _reviewers_requested_value(pr: dict[str, Any], reviewers_requested: bool | None) -> bool | None:
    """Use an explicit reviewer state or a validated snapshot value."""
    if reviewers_requested is not None:
        return reviewers_requested
    snapshot_value = pr.get("reviewers_requested")
    return snapshot_value if type(snapshot_value) is bool else None


def _metadata_verdict_reason(status: str) -> str | None:
    """Return fail-closed reason for metadata verdict status, or None when accepted."""
    if status == "accepted":
        return None
    return "stale_pr_metadata_verdict" if status == "stale" else "missing_pr_metadata_verdict"


def _thread_resolution_reason(status: str) -> str | None:
    """Return fail-closed reason for thread resolution status, or None when resolved."""
    if status == "resolved":
        return None
    return "unresolved_review_threads" if status == "unresolved" else "review_threads_not_evaluated"


def _reviewer_request_reason(status: str) -> str | None:
    """Return fail-closed reason for reviewer request status, or None when clear."""
    if status == "clear":
        return None
    return (
        "outstanding_requested_reviewers"
        if status == "requested"
        else "requested_reviewers_not_evaluated"
    )


def _review_claim_status(pr: dict[str, Any], *, head_sha: str, now: datetime | None) -> str:
    """Classify whether a trusted review worker still claims the live head.

    Live snapshots carry the result explicitly so the queue audit records the
    exact read used for admission. Pure callers may instead provide raw
    ``comments``/``reviews`` collections; those are evaluated with the shared
    ``active_review_claim`` parser. Older fixtures without either source remain
    ``not_evaluated`` for compatibility, while an explicitly malformed live
    result fails closed.
    """
    explicit = pr.get("review_claim")
    if explicit is not None:
        if not isinstance(explicit, dict):
            return "unknown"
        status = explicit.get("status")
        return status if status in {"active", "clear", "unavailable"} else "unknown"

    if "comments" not in pr and "reviews" not in pr:
        return "not_evaluated"
    comments = pr.get("comments", [])
    reviews = pr.get("reviews", [])
    if not isinstance(comments, list) or not isinstance(reviews, list):
        return "unavailable"
    claim = active_review_claim(
        {"comments": comments, "reviews": reviews},
        head_sha,
        now,
    )
    return "active" if claim is not None else "clear"


def _closing_discipline_state(pr: dict[str, Any]) -> tuple[str, list[str]]:
    """Validate the live closing-discipline result carried by a PR snapshot.

    Older pure-evaluator fixtures do not carry this optional field and retain the
    ``not_evaluated`` status.  A live snapshot must carry one of the explicit
    results so a missing or malformed merge-time recheck cannot be mistaken for a
    successful check.
    """
    value = pr.get("closing_discipline")
    if value is None:
        return "not_evaluated", []
    if not isinstance(value, dict):
        return "unknown", []
    status = value.get("status")
    blockers = value.get("blockers", [])
    if status not in {"passed", "blocked", "unavailable"}:
        return "unknown", []
    if not isinstance(blockers, list) or not all(isinstance(item, str) for item in blockers):
        return "unknown", []
    if status == "passed" and blockers:
        return "unknown", blockers
    return status, blockers


def _core_preflight_reasons(
    *,
    draft: bool,
    merge_ready: bool,
    ci_overall: str,
    changed_coverage_status: str,
    staleness_verdict: str,
    gate_verdict_status: str,
) -> list[str]:
    """Collect core preflight gate failure reasons."""
    reasons: list[str] = []
    if draft:
        reasons.append("pr_is_draft")
    if not merge_ready:
        reasons.append("missing_merge_ready_label")
    if ci_overall != "success":
        reasons.append(f"ci_not_green:{ci_overall}")
    if changed_coverage_status not in {"success", CHANGED_COVERAGE_NOT_REQUIRED}:
        reasons.append(
            {
                "missing": "changed_coverage_proof_missing",
                "pending": "changed_coverage_proof_pending",
                "failure": "changed_coverage_proof_failed",
                "stale": "changed_coverage_proof_stale",
                "malformed": "changed_coverage_proof_malformed",
            }.get(changed_coverage_status, "changed_coverage_proof_unknown")
        )
    if staleness_verdict == "stale":
        reasons.append("stale_merge_base")
    if gate_verdict_status != "accepted":
        reasons.append("missing_exact_head_gate_verdict")
    return reasons


def _fail_closed_reasons(  # noqa: PLR0913
    *,
    draft: bool,
    merge_ready: bool,
    ci_overall: str,
    changed_coverage_status: str,
    staleness_verdict: str,
    gate_verdict_status: str,
    metadata_verdict_status: str,
    thread_resolution: str,
    reviewer_request_status: str,
    merge_group_head_binding: str,
    review_claim_status: str = "not_evaluated",
    body_not_ready_sentinels: list[str] | None = None,
    ancestry_state: str = "",
    closing_discipline_status: str = "not_evaluated",
) -> list[str]:
    """Collect fail-closed reasons for one gate evaluation.

    Any non-empty reason list means the gate must fail closed. The order is
    stable so audit records are comparable across runs and machines.
    """
    reasons = _core_preflight_reasons(
        draft=draft,
        merge_ready=merge_ready,
        ci_overall=ci_overall,
        changed_coverage_status=changed_coverage_status,
        staleness_verdict=staleness_verdict,
        gate_verdict_status=gate_verdict_status,
    )

    meta_reason = _metadata_verdict_reason(metadata_verdict_status)
    if meta_reason:
        reasons.append(meta_reason)

    if merge_ready and body_not_ready_sentinels:
        reasons.append("stale_not_ready_body_narrative")
    if merge_group_head_binding == "mismatch":
        reasons.append("merge_group_head_sha_mismatch")

    thread_reason = _thread_resolution_reason(thread_resolution)
    if thread_reason:
        reasons.append(thread_reason)

    rev_reason = _reviewer_request_reason(reviewer_request_status)
    if rev_reason:
        reasons.append(rev_reason)

    if review_claim_status == "active":
        reasons.append("active_review_claim")
    elif review_claim_status not in {"clear", "not_evaluated"}:
        reasons.append("review_claim_not_verified")

    if ancestry_state and ancestry_state != "clean":
        reasons.append("stacked_ancestry_not_independently_mergeable")

    if closing_discipline_status not in {"passed", "not_evaluated"}:
        reasons.append(
            {
                "blocked": "closing_discipline_blocked",
                "unavailable": "closing_discipline_unavailable",
            }.get(closing_discipline_status, "closing_discipline_not_verified")
        )
    return reasons


def _is_ci_paths_ignored(path: str) -> bool:
    """Return whether ``path`` matches the CI workflow's ignored path set.

    GitHub's ``**/*.md`` filter covers Markdown at any repository depth,
    including a root-level README or changelog.  The explicit checks below
    mirror that contract without making the admission gate depend on a local
    glob implementation with subtly different ``**`` semantics.
    """
    normalized = path.strip()
    if (
        not normalized
        or normalized != path
        or normalized.startswith(("/", "./", "../"))
        or "\\" in normalized
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        return False
    markdown_pattern, docs_pattern = CI_PATHS_IGNORE_PATTERNS
    return bool(normalized) and (
        (markdown_pattern == "**/*.md" and normalized.endswith(".md"))
        or (docs_pattern == "docs/**" and (normalized == "docs" or normalized.startswith("docs/")))
    )


def _docs_only_changed_files(changed_files: Any, *, complete: bool) -> bool:
    """Prove that a complete, non-empty changed-file set is CI-ignored."""
    if not complete or not isinstance(changed_files, list) or not changed_files:
        return False
    if any(not isinstance(path, str) or not path.strip() for path in changed_files):
        return False
    return all(_is_ci_paths_ignored(path) for path in changed_files)


def _proven_docs_only_scope(changed_coverage: Any) -> bool:
    """Return whether a changed-coverage payload carries the required proof."""
    if not isinstance(changed_coverage, dict):
        return False
    return _docs_only_changed_files(
        changed_coverage.get("changed_files"),
        complete=changed_coverage.get("changed_files_complete") is True,
    )


def _resolve_narrative_status(body_text: str, sentinels: list[str]) -> str:
    """Return narrative status classification for PR body."""
    if not body_text:
        return "empty"
    return "stale" if sentinels else "clean"


def _resolve_staleness_verdict(base_sha: str, main_sha: str) -> str:
    """Return base staleness verdict against current main."""
    if base_sha and main_sha:
        return "fresh" if base_sha == main_sha else "stale"
    return "not_applicable"


def _resolve_bool_status(value: bool | None, true_name: str, false_name: str) -> str:
    """Map a tri-state bool/None to standard status string."""
    if value is True:
        return true_name
    if value is False:
        return false_name
    return "not_evaluated"


def _resolve_merge_group_binding(
    merge_group_head_sha: str, head_sha: str, queue_merging_strategy: str
) -> tuple[str, str]:
    """Resolve merge group head binding and normalized strategy."""
    if not merge_group_head_sha:
        return "not_applicable", "not_applicable"
    binding = "match" if _merge_group_head_matches(merge_group_head_sha, head_sha) else "mismatch"
    return binding, str(queue_merging_strategy or "unknown").upper()


def evaluate_merge_gate(  # noqa: C901, PLR0913 - explicit fail-closed admission dimensions.
    pr: dict[str, Any],
    *,
    main_sha: str = "",
    ci_overall: str | None = None,
    changed_coverage_status: str | None = None,
    changed_coverage_head_sha: str = "",
    threads_resolved: bool | None = None,
    reviewers_requested: bool | None = None,
    merge_group_head_sha: str = "",
    queue_merging_strategy: str = "",
    now: datetime | None = None,
) -> MergeGateAudit:
    """Evaluate the merge-queue gate for one PR snapshot.

    Pure function: no side effects, no GitHub calls. Fail-closed by design.

    Inputs:
      pr: compact PR snapshot. Recognized fields: ``head_sha`` (required for a
        pass), ``labels``, ``draft``, ``base_sha``, ``checks.overall``,
        ``changed_coverage`` (which must bind a success result to ``head_sha``), plus any
        gate-verdict carrier fields understood by
        ``has_current_accepted_gate_verdict`` (``gate_verdict`` /
        ``gate_verdicts`` / ``comments`` / ``reviews`` body excerpts),
        ``metadata_digest`` and trusted ``metadata_verdicts``, and
        ``reviewers_requested`` when supplied by the live snapshot, plus the
        optional live ``closing_discipline`` and ``review_claim`` results.
      main_sha: current ``main`` HEAD SHA. When both ``base_sha`` and
        ``main_sha`` are present and differ, the gate fails closed as stale. When
        either is absent, staleness is reported as ``not_applicable`` (the merge
        queue constructs a fresh base, so this is the normal queue-time path).
      ci_overall: authoritative CI conclusion (``success`` / ``failure`` /
        ``pending`` / ``unknown``). When ``None``, falls back to
        ``pr["checks"]["overall"]``; when still empty, the CI dimension is
        treated as unknown and fails closed.
      threads_resolved: ``True`` when all actionable review threads are resolved,
        ``False`` when at least one remains unresolved, ``None`` when not
        evaluated (fails closed; the runtime CLI supplies a definitive value and
        fails closed on a query error).
      reviewers_requested: ``True`` when one or more explicitly requested
        reviewers remain, ``False`` when no reviewer request remains, ``None``
        when not evaluated (fails closed; the runtime CLI always supplies a
        definitive value).
      merge_group_head_sha: source-head SHA encoded in a canonical
        ``merge_group.head_ref``. When provided, it must prefix-match the live
        PR head SHA; any mismatch fails closed so a queue ref cannot be rebound
        to a newer or unrelated PR head.
      now: explicit UTC evaluation instant for raw review-claim sources. Live
        snapshots already carry a claim result; ``None`` uses the current UTC
        time through ``active_review_claim``.

    Returns a ``MergeGateAudit`` with ``passed`` and a list of fail-closed
    ``reasons``. The audit always records the evaluated head SHA, base SHA, label
    set, gate-verdict, metadata-verdict, and review-claim statuses, staleness
    verdict, CI conclusion, changed-coverage status, and thread resolution so
    the decision is inspectable.
    """
    head_sha = str(pr.get("head_sha", "") or "")
    labels = _label_names(pr)
    draft_value = pr.get("draft")
    draft_state_valid = type(draft_value) is bool
    draft = draft_value is True
    merge_ready = "merge-ready" in labels
    base_sha = str(pr.get("base_sha", "") or "")

    body_text = str(pr.get("body") or "")
    body_not_ready_sentinels = find_not_ready_body_sentinels(body_text)
    body_narrative_status = _resolve_narrative_status(body_text, body_not_ready_sentinels)
    closing_discipline_status, closing_discipline_blockers = _closing_discipline_state(pr)

    if ci_overall is None:
        ci_overall = str((pr.get("checks") or {}).get("overall", "") or "")
    ci_overall = str(ci_overall).lower() or "unknown"

    changed_coverage = pr.get("changed_coverage")
    if isinstance(changed_coverage, dict):
        if changed_coverage_status is None:
            changed_coverage_status = str(changed_coverage.get("status") or "")
        if not changed_coverage_head_sha:
            changed_coverage_head_sha = str(changed_coverage.get("head_sha") or "")
    changed_coverage_status = str(changed_coverage_status or "").lower() or "unknown"
    docs_only_scope_proven = _proven_docs_only_scope(changed_coverage)
    if changed_coverage_status == "missing" and docs_only_scope_proven:
        # The workflow is intentionally skipped for exactly this path set;
        # accepting it is safe only after the live API proves the complete
        # changed-file list.  No source-changing PR receives this bypass.
        changed_coverage_status = CHANGED_COVERAGE_NOT_REQUIRED
        changed_coverage_head_sha = ""
    elif changed_coverage_status == CHANGED_COVERAGE_NOT_REQUIRED and not docs_only_scope_proven:
        # Do not trust a caller-provided status without the same proof used by
        # the live snapshot path.
        changed_coverage_status = "unknown"
        changed_coverage_head_sha = ""
    if changed_coverage_status == "success" and (
        not changed_coverage_head_sha or changed_coverage_head_sha.lower() != head_sha.lower()
    ):
        changed_coverage_status = "stale"

    reviewers_requested = _reviewers_requested_value(pr, reviewers_requested)

    gate_verdict_status = _gate_verdict_status(pr, head_sha)
    metadata_digest_value = str(pr.get("metadata_digest", "") or "")
    metadata_verdict_status = _metadata_verdict_status(pr, metadata_digest_value)
    review_claim_status = _review_claim_status(pr, head_sha=head_sha, now=now)

    merge_group_head_sha = str(merge_group_head_sha or "").lower()
    merge_group_head_binding, queue_strategy = _resolve_merge_group_binding(
        merge_group_head_sha, head_sha, queue_merging_strategy
    )

    staleness_verdict = _resolve_staleness_verdict(base_sha, main_sha)
    thread_resolution = _resolve_bool_status(threads_resolved, "resolved", "unresolved")
    reviewer_request_status = _resolve_bool_status(reviewers_requested, "requested", "clear")

    ancestry_block = pr.get("ancestry")
    if isinstance(ancestry_block, dict):
        ancestry_state = str(ancestry_block.get("state") or "")
    else:
        ancestry_state = ""

    reasons = ["draft_state_unavailable"] if not draft_state_valid else []
    reasons.extend(
        _fail_closed_reasons(
            draft=draft,
            merge_ready=merge_ready,
            ci_overall=ci_overall,
            changed_coverage_status=changed_coverage_status,
            staleness_verdict=staleness_verdict,
            gate_verdict_status=gate_verdict_status,
            metadata_verdict_status=metadata_verdict_status,
            thread_resolution=thread_resolution,
            reviewer_request_status=reviewer_request_status,
            merge_group_head_binding=merge_group_head_binding,
            review_claim_status=review_claim_status,
            body_not_ready_sentinels=body_not_ready_sentinels,
            ancestry_state=ancestry_state,
            closing_discipline_status=closing_discipline_status,
        )
    )
    if not head_sha:
        reasons.insert(0, "missing_head_sha")
    if queue_strategy not in {"not_applicable", "ALLGREEN"}:
        reasons.append(f"unsafe_merge_queue_strategy:{queue_strategy}")

    passed = not reasons

    return MergeGateAudit(
        schema=AUDIT_SCHEMA,
        pr=_safe_int(pr.get("number")),
        head_sha=head_sha,
        merge_group_head_sha=merge_group_head_sha,
        merge_group_head_binding=merge_group_head_binding,
        queue_merging_strategy=queue_strategy,
        base_sha=base_sha,
        main_sha=str(main_sha or ""),
        labels=labels,
        draft=draft,
        ci_overall=ci_overall or "unknown",
        changed_coverage_status=changed_coverage_status,
        changed_coverage_head_sha=changed_coverage_head_sha,
        gate_verdict_status=gate_verdict_status,
        metadata_digest=metadata_digest_value,
        metadata_verdict_status=metadata_verdict_status,
        staleness_verdict=staleness_verdict,
        thread_resolution=thread_resolution,
        reviewer_request_status=reviewer_request_status,
        review_claim_status=review_claim_status,
        ancestry_state=ancestry_state,
        merge_ready=merge_ready,
        passed=passed,
        body_narrative_status=body_narrative_status,
        body_not_ready_sentinels=body_not_ready_sentinels,
        closing_discipline_status=closing_discipline_status,
        closing_discipline_blockers=closing_discipline_blockers,
        reasons=reasons,
    )


def _safe_int(value: Any) -> int | None:
    """Coerce a PR number-like value to int, returning None when not parseable."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _gh(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a gh command and return the completed process."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _fetch_live_closing_discipline(
    pr_number: str | int, *, repo: str, body: str
) -> tuple[str, list[str]]:
    """Recheck semantic closing references against current PR and issue metadata.

    The PR-contract status check runs on pull-request events, so a later incident
    label or marker change could otherwise make its earlier green result stale.
    Native merge admission repeats the check against the current commit list and
    current issue metadata.  An unavailable commit list is an explicit
    fail-closed result.
    """
    commit_messages = get_pr_commit_messages(str(pr_number), repo)
    blockers = check_closes_discipline(
        body,
        repo,
        commit_messages=commit_messages,
        commit_messages_checked=True,
    )
    status = "unavailable" if commit_messages is None else ("blocked" if blockers else "passed")
    return status, blockers


def _parse_json(stdout: str) -> tuple[Any, str | None]:
    """Parse JSON stdout into a Python object or return an error string."""
    try:
        return json.loads(stdout), None
    except json.JSONDecodeError as exc:
        return None, f"Failed to parse JSON: {exc}"


def _rollup_overall(rollup: list[dict[str, Any]]) -> str:  # noqa: C901
    """Classify a PR ``statusCheckRollup`` into an overall CI conclusion.

    Returns ``failure`` if any check failed, ``pending`` if any check is
    in-progress/queued (or the rollup is empty), otherwise ``success``. Mirrors
    the classification in ``scripts/dev/check_pr_ci_status.py``.
    """
    if not isinstance(rollup, list) or not rollup:
        return "pending"
    if any(not isinstance(check, dict) for check in rollup):
        return "pending"
    if any(
        check.get("name") == GATE_JOB_NAME and check.get("workflowName") != GATE_WORKFLOW_NAME
        for check in rollup
    ):
        # A job name without its workflow identity cannot prove whether this is
        # the required Merge Queue Gate context or an unrelated check.
        return "unknown"
    effective_rollup, _superseded_count = _latest_check_runs(rollup)
    without_current_gate = [
        check
        for check in effective_rollup
        if not (
            check.get("workflowName") == GATE_WORKFLOW_NAME and check.get("name") == GATE_JOB_NAME
        )
    ]
    if not without_current_gate:
        # A source-head gate run must not be allowed to establish its own CI
        # prerequisite.  The exact-head verdict is evidence about a prior
        # review, not a substitute for at least one current non-gate check.
        return "pending"
    for check in without_current_gate:
        conclusion = _rollup_conclusion(check)
        status = _rollup_status(check)
        if conclusion in FAILURE_CONCLUSIONS:
            return "failure"
        if status in PENDING_STATUSES:
            return "pending"
        if conclusion in PENDING_STATUSES:
            return "pending"
        if status != COMPLETED_STATUS:
            return "unknown"
        if conclusion not in SUCCESS_CONCLUSIONS:
            return "unknown"
    return "success"


def _classify_changed_coverage_checks(
    check_runs: list[dict[str, Any]], *, head_sha: str
) -> dict[str, Any]:
    """Classify the newest changed-coverage check for one exact commit."""
    candidates = [
        check
        for check in check_runs
        if str(check.get("name") or "").strip() == CHANGED_COVERAGE_CHECK_NAME
    ]
    if not candidates:
        return {
            "status": "missing",
            "head_sha": head_sha,
            "name": CHANGED_COVERAGE_CHECK_NAME,
        }

    def sort_key(check: dict[str, Any]) -> tuple[str, int]:
        timestamp = str(
            check.get("completed_at") or check.get("started_at") or check.get("created_at") or ""
        )
        try:
            identifier = int(check.get("id", 0) or 0)
        except (TypeError, ValueError):
            identifier = 0
        return timestamp, identifier

    latest = max(candidates, key=sort_key)
    reported_head = str(latest.get("head_sha") or "")
    if not reported_head:
        status = "malformed"
    elif reported_head.lower() != head_sha.lower():
        status = "stale"
    elif str(latest.get("status") or "").lower() != COMPLETED_STATUS:
        status = "pending"
    elif str(latest.get("conclusion") or "").lower() != "success":
        status = "failure"
    else:
        status = "success"
    return {
        "status": status,
        "head_sha": reported_head or head_sha,
        "name": str(latest.get("name") or CHANGED_COVERAGE_CHECK_NAME),
        "check_run_id": latest.get("id"),
        "started_at": latest.get("started_at"),
        "completed_at": latest.get("completed_at"),
        "conclusion": latest.get("conclusion"),
        "details_url": latest.get("html_url") or latest.get("details_url"),
    }


def _fetch_exact_head_changed_coverage(
    head_sha: str, *, repo: str
) -> tuple[dict[str, Any], str | None]:
    """Fetch changed-coverage evidence from the exact PR-head check-run list."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", head_sha):
        return {}, "PR head SHA is missing or malformed"
    result = _gh(["api", f"repos/{repo}/commits/{head_sha}/check-runs?per_page=100"])
    if result.returncode != 0:
        return {}, result.stderr.strip() or "exact-head check-run query failed"
    payload, err = _parse_json(result.stdout)
    if err or not isinstance(payload, dict):
        return {}, err or "exact-head check-run response is not a JSON object"
    check_runs = payload.get("check_runs")
    if not isinstance(check_runs, list) or any(not isinstance(item, dict) for item in check_runs):
        return {}, "exact-head check-run response is missing a valid check_runs list"
    total_count = payload.get("total_count")
    if type(total_count) is not int or total_count < len(check_runs):
        return {}, "exact-head check-run response has an invalid or incomplete total_count"
    if total_count > len(check_runs):
        return {}, "exact-head check-run response is incomplete; refusing stale-proof bypass"
    return _classify_changed_coverage_checks(check_runs, head_sha=head_sha), None


def _graphql_error(payload: dict[str, Any]) -> str | None:
    """Return a normalized GraphQL error, including partial-data errors."""
    errors = payload.get("errors")
    if errors is None:
        return None
    if not isinstance(errors, list):
        return "GraphQL errors field is malformed"
    if not errors:
        return None
    messages = [
        str(error.get("message") or error) if isinstance(error, dict) else str(error)
        for error in errors
    ]
    return "; ".join(messages) or "GraphQL returned errors"


def _graphql_pull_request(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Return the pull-request object from a validated GraphQL response."""
    data = payload.get("data")
    if not isinstance(data, dict):
        return None, "GraphQL data is missing or malformed"
    repository = data.get("repository")
    if not isinstance(repository, dict):
        return None, "GraphQL repository data is missing or malformed"
    pull_request = repository.get("pullRequest")
    if not isinstance(pull_request, dict):
        return None, "GraphQL pull-request data is missing or malformed"
    return pull_request, None


def _failed_audit(audit: MergeGateAudit, reason: str) -> MergeGateAudit:
    """Return an audit forced to fail with one additional machine-readable reason."""
    return replace(audit, passed=False, reasons=[*audit.reasons, reason])


def _resolve_owner_repo(explicit: str) -> str | None:
    """Resolve the ``owner/repo`` identifier, auto-detecting from gh when empty."""
    if explicit:
        return explicit
    result = _gh(["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"])
    if result.returncode != 0:
        return None
    repo = result.stdout.strip()
    return repo if repo else None


def _merge_group_head_matches(encoded_sha: str, current_head_sha: str) -> bool:
    """Return whether an encoded queue-ref SHA identifies the current PR head."""
    if not encoded_sha or not current_head_sha:
        return False
    return current_head_sha.lower().startswith(encoded_sha.lower())


def resolve_pr_from_merge_group(event: dict[str, Any]) -> MergeGroupPR | None:
    """Resolve the PR identity encoded by a canonical ``merge_group`` event.

    GitHub uses ``pr-<number>-<source-sha>`` in its readonly queue ref. Parse
    the PR number directly rather than querying a fictitious ``pr-<number>``
    source branch. The caller binds ``head_sha`` to the current PR head before
    evaluating the gate, and fails closed on a mismatch.
    """
    if not isinstance(event, dict):
        return None
    merge_group = event.get("merge_group")
    if not isinstance(merge_group, dict):
        return None
    head_ref = str(merge_group.get("head_ref") or "")
    match = _MERGE_QUEUE_REF_RE.match(head_ref)
    if not match:
        return None
    number = _safe_int(match.group("number"))
    head_sha = str(match.group("head_sha") or "").lower()
    if number is None or not head_sha:
        return None
    return MergeGroupPR(number=number, head_sha=head_sha)


def _normalize_labels(raw: Any) -> list[str]:
    """Normalize ``gh pr view --json labels`` objects into label-name strings."""
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for label in raw:
        if isinstance(label, dict):
            name = label.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        elif isinstance(label, str) and label:
            names.append(label)
    return names


def _to_body_snapshot(items: Any, *, limit: int = 180) -> dict[str, Any]:
    """Convert raw ``gh`` comment/review objects into the compact snapshot shape.

    The compact excerpts are audit context only. ``fetch_pr_snapshot`` extracts
    accepted gate-verdict trailers from the full raw bodies before truncation,
    so a valid trailer after the excerpt limit cannot be discarded.
    """
    if not isinstance(items, list):
        return {"latest": []}
    latest: list[dict[str, str]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        body = str(entry.get("body") or "")
        if body:
            latest.append(
                {
                    "author_association": str(entry.get("authorAssociation", "")),
                    "body_excerpt": body[:limit],
                }
            )
    return {"latest": latest}


def _to_review_claim_snapshot(comments: Any, reviews: Any, *, head_sha: str) -> dict[str, Any]:
    """Project the live review-claim result onto the evaluated head."""
    if not isinstance(comments, list) or not isinstance(reviews, list):
        return {"status": "unavailable"}
    claim = active_review_claim(
        {"comments": comments, "reviews": reviews},
        head_sha,
        None,
    )
    if claim is None:
        return {"status": "clear"}
    return {
        "status": "active",
        "lane": claim.lane,
        "head_sha": claim.sha,
        "expires_at": claim.expires_at.isoformat() if claim.expires_at is not None else None,
    }


def _to_receipt_check_runs(
    items: Any, *, head_sha: str, expected_metadata_digest: str = ""
) -> list[dict[str, Any]]:
    """Project the live status rollup onto exact-head receipt check evidence.

    ``gh pr view`` scopes ``statusCheckRollup`` to the current PR head.  The
    projection records that binding explicitly even when the GraphQL rollup
    omits a separate ``head_sha`` field, so receipt verification never has to
    guess which head the snapshot described.
    """
    if not isinstance(items, list):
        return []
    mapping_items = [item for item in items if isinstance(item, dict)]
    effective_items, _superseded_count = _latest_check_runs(mapping_items)
    checks: list[dict[str, Any]] = []
    for item in effective_items:
        if not isinstance(item, dict):
            continue
        app_value = item.get("app")
        app: dict[str, Any] = app_value if isinstance(app_value, dict) else {}
        name = str(item.get("name") or item.get("context") or "")
        if name.strip().lower() in NON_REQUIRED_RECEIPT_CHECK_NAMES:
            continue
        is_receipt_review = name in RECEIPT_REVIEW_CHECK_NAMES
        checks.append(
            {
                "name": name,
                "head_sha": str(item.get("head_sha") or item.get("headSha") or head_sha),
                "status": str(item.get("status") or "").lower(),
                "conclusion": str(item.get("conclusion") or "").lower(),
                "started_at": item.get("startedAt") or item.get("started_at"),
                "completed_at": item.get("completedAt") or item.get("completed_at"),
                "details_url": item.get("detailsUrl")
                or item.get("targetUrl")
                or item.get("html_url"),
                "identity": str(app.get("slug") or app.get("name") or item.get("name") or ""),
                "app": {
                    "slug": str(app.get("slug") or ""),
                    "name": str(app.get("name") or ""),
                },
                "approved_reviewer": item.get("approved_reviewer") is True,
                "approved_source": is_receipt_review,
                "metadata_digest": expected_metadata_digest if is_receipt_review else None,
            }
        )
    return checks


def _to_receipt_review_evidence(
    items: Any, *, head_sha: str, expected_metadata_digest: str
) -> list[dict[str, Any]]:
    """Keep the raw fields needed to classify independent carriers.

    Review comments can carry the canonical metadata trailer and an exact-head
    marker even though GitHub does not expose either as a typed field.  Preserve
    those declarations so the receipt classifier can reject stale or missing
    bindings rather than stamping the current snapshot onto old prose.
    """
    if not isinstance(items, list):
        return []
    evidence: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        author_value = item.get("author")
        author: dict[str, Any] = author_value if isinstance(author_value, dict) else {}
        user_value = item.get("user")
        user: dict[str, Any] = user_value if isinstance(user_value, dict) else {}
        body = str(item.get("body") or "")
        declared_shas = re.findall(r"(?<![0-9a-fA-F])([0-9a-fA-F]{40})(?![0-9a-fA-F])", body)
        observed_head = item.get("head_sha") or item.get("commit_id") or item.get("commitId")
        if not observed_head:
            observed_head = next(
                (value for value in declared_shas if value.lower() == head_sha.lower()),
                declared_shas[-1] if declared_shas else None,
            )
        observed_metadata = item.get("metadata_digest")
        metadata_values = extract_metadata_digests(body)
        if not observed_metadata:
            observed_metadata = next(
                (
                    value
                    for value in metadata_values
                    if value.lower() == expected_metadata_digest.lower()
                ),
                metadata_values[-1] if metadata_values else None,
            )
        evidence.append(
            {
                "id": item.get("id"),
                "identity": str(
                    item.get("identity") or author.get("login") or user.get("login") or ""
                ),
                "authorAssociation": str(
                    item.get("authorAssociation") or item.get("author_association") or ""
                ).upper(),
                "state": str(item.get("state") or ""),
                "commit_id": item.get("commit_id") or item.get("commitId"),
                "head_sha": observed_head,
                "metadata_digest": observed_metadata,
                "evidence_digest": item.get("evidence_digest") or item.get("digest"),
                "approved_reviewer": item.get("approved_reviewer") is True,
                "approved_source": item.get("approved_source") is True,
                "dismissed": item.get("dismissed") is True,
                "withdrawn": item.get("withdrawn") is True,
                "superseded": item.get("superseded") is True,
                "body": body,
            }
        )
    return evidence


def _requested_review_identities(items: Any) -> tuple[list[str], list[str]]:
    """Separate requested user and team identities from GitHub's union list."""
    reviewers: list[str] = []
    teams: list[str] = []
    if not isinstance(items, list):
        return reviewers, teams
    for item in items:
        if not isinstance(item, dict):
            continue
        user_value = item.get("user")
        user: dict[str, Any] = user_value if isinstance(user_value, dict) else {}
        team_value = item.get("team")
        team: dict[str, Any] = team_value if isinstance(team_value, dict) else {}
        typename = str(item.get("__typename") or "").lower()
        if team or "team" in typename:
            identity = str(team.get("name") or team.get("slug") or item.get("name") or "")
            if identity:
                teams.append(identity)
        else:
            identity = str(user.get("login") or item.get("login") or item.get("name") or "")
            if identity:
                reviewers.append(identity)
    return sorted(set(reviewers)), sorted(set(teams))


def _fetch_pr_base_sha(pr_number: str | int, *, repo: str) -> tuple[str | None, str | None]:
    """Return the PR base SHA through the gh-compatible REST pull endpoint."""
    # ``baseRefOid`` is not available in the repository's supported gh 2.45.0
    # JSON field set. The REST pull endpoint is stable across supported gh
    # versions and exposes the same base commit as ``base.sha``.
    result = _gh(["api", f"repos/{repo}/pulls/{pr_number}"])
    if result.returncode != 0:
        return None, result.stderr.strip() or "gh api pull request failed"
    payload, err = _parse_json(result.stdout)
    if err or not isinstance(payload, dict):
        return None, err or "gh API pull-request response is not a JSON object"
    base = payload.get("base")
    if not isinstance(base, dict):
        return None, "PR base metadata is missing"
    sha = base.get("sha")
    if not isinstance(sha, str) or not sha:
        return None, "PR base SHA is empty"
    return sha, None


def _fetch_pr_changed_file_records(
    pr_number: str | int, *, repo: str
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Fetch complete paginated changed-file records for exact-ref classification.

    The REST files endpoint does not expose a total count in the response, so
    page until a short page (including an empty terminal page) proves that no
    file was omitted.
    """
    changed_records: list[dict[str, Any]] = []
    for page in range(1, _MAX_CHANGED_FILES_PAGES + 1):
        result = _gh(
            [
                "api",
                f"repos/{repo}/pulls/{pr_number}/files?per_page={_CHANGED_FILES_PAGE_SIZE}&page={page}",
            ]
        )
        if result.returncode != 0:
            return None, result.stderr.strip() or "changed-file query failed"
        payload, err = _parse_json(result.stdout)
        if err or not isinstance(payload, list):
            return None, err or "changed-file response is not a JSON array"

        page_records: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                return None, "changed-file response contains a malformed entry"
            filename = item.get("filename")
            if not isinstance(filename, str) or not filename.strip():
                return None, "changed-file response contains an invalid filename"
            page_records.append(dict(item))

        changed_records.extend(page_records)
        if len(changed_records) >= _MAX_GITHUB_PR_FILES:
            return (
                None,
                f"changed-file inventory reached GitHub's {_MAX_GITHUB_PR_FILES}-file cap",
            )
        if len(page_records) < _CHANGED_FILES_PAGE_SIZE:
            if not changed_records:
                return None, "changed-file response is empty"
            return changed_records, None

    return None, "changed-file response exceeded the bounded pagination limit"


def _fetch_pr_changed_files(
    pr_number: str | int, *, repo: str
) -> tuple[list[str] | None, str | None]:
    """Fetch the complete changed-file names needed for a missing-proof ruling."""
    records, error = _fetch_pr_changed_file_records(pr_number, repo=repo)
    if error or records is None:
        return None, error
    return [str(record["filename"]) for record in records], None


def _fetch_text_at_ref(
    *, repo: str, path: str, ref: str, allow_absent: bool = False
) -> tuple[str | None, str | None]:
    """Read one UTF-8 repository file through an immutable full-SHA REST ref."""
    encoded_path = quote(path, safe="/")
    result = _gh(["api", f"repos/{repo}/contents/{encoded_path}?ref={ref}"])
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or f"content query failed for {path} at {ref}"
        if allow_absent and "HTTP 404" in diagnostic:
            return None, None
        return None, diagnostic
    payload, error = _parse_json(result.stdout)
    if error or not isinstance(payload, dict):
        return None, error or f"content response is malformed for {path} at {ref}"
    if payload.get("encoding") != "base64" or not isinstance(payload.get("content"), str):
        return None, f"content response is not inline base64 for {path} at {ref}"
    try:
        raw = base64.b64decode("".join(payload["content"].split()), validate=True)
        return raw.decode("utf-8"), None
    except (binascii.Error, UnicodeDecodeError) as exc:
        return None, f"content decode failed for {path} at {ref}: {exc}"


def _verify_git_commit_ref(*, repo: str, ref: str) -> tuple[bool, str | None]:
    """Verify that a full immutable commit ref resolves before optional path reads."""
    result = _gh(["api", f"repos/{repo}/git/commits/{ref}"])
    if result.returncode != 0:
        return False, result.stderr.strip() or f"commit ref query failed for {ref}"
    payload, error = _parse_json(result.stdout)
    if error or not isinstance(payload, dict) or str(payload.get("sha") or "") != ref:
        return False, error or f"commit ref response did not bind {ref}"
    return True, None


def fetch_pr_changed_file_marker_inventory(  # noqa: C901, PLR0912 - statuses have distinct refs.
    pr_number: str | int,
    *,
    repo: str,
    base_sha: str,
    head_sha: str,
    current_main_sha: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Classify changed test files using content bound to exact base/head SHAs."""
    records, error = _fetch_pr_changed_file_records(pr_number, repo=repo)
    if error or records is None:
        return None, error or "changed-file inventory unavailable"
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        filename = str(record["filename"])
        status = str(record.get("status") or "").lower()
        previous_filename = record.get("previous_filename")
        if not status:
            return None, f"changed file lacks status: {filename}"
        if status not in _CHANGED_FILE_STATUSES:
            return None, f"changed file has unsupported status {status}: {filename}"
        if previous_filename is not None and (
            not isinstance(previous_filename, str) or not previous_filename.strip()
        ):
            return None, f"changed file has invalid previous_filename: {filename}"
        if status == "renamed" and previous_filename is None:
            return None, f"renamed file lacks previous_filename: {filename}"
        if status != "renamed" and previous_filename is not None:
            return None, f"non-renamed file has previous_filename: {filename}"
        normalized_records.append(
            {
                "filename": filename,
                "previous_filename": previous_filename,
                "status": status,
            }
        )
    records = sorted(normalized_records, key=lambda record: str(record["filename"]))

    current_main_verified, current_main_error = _verify_git_commit_ref(
        repo=repo, ref=current_main_sha
    )
    if not current_main_verified:
        return None, current_main_error or "current-main commit ref unavailable"

    candidates = [
        record
        for record in records
        if (
            Path(str(record["filename"])).name.startswith("test_")
            and str(record["filename"]).endswith(".py")
        )
        or (
            isinstance(record.get("previous_filename"), str)
            and Path(str(record["previous_filename"])).name.startswith("test_")
            and str(record["previous_filename"]).endswith(".py")
        )
    ]
    if len(candidates) > _MAX_CHANGED_TEST_CONTENT_FILES:
        return None, "changed test-file inventory exceeds exact-content bound"

    sensitive: list[str] = []
    provenance: list[dict[str, Any]] = []
    for record in candidates:
        filename = str(record["filename"])
        status = str(record.get("status") or "").lower()
        refs: list[tuple[str, str, str]] = []
        if status in {"added", "copied"}:
            refs.append(("head", filename, head_sha))
        elif status in {"deleted", "removed"}:
            refs.append(("base", filename, base_sha))
        elif status in {"changed", "modified"}:
            refs.extend((("base", filename, base_sha), ("head", filename, head_sha)))
        elif status == "renamed":
            previous = record.get("previous_filename")
            if not isinstance(previous, str) or not previous.strip():
                return None, f"renamed test file lacks previous_filename: {filename}"
            refs.extend((("base", previous, base_sha), ("head", filename, head_sha)))
        else:
            return (
                None,
                f"changed test file has unsupported status {status or 'missing'}: {filename}",
            )

        proof: dict[str, Any] = {
            "base": None,
            "current_main": [],
            "filename": filename,
            "head": None,
            "previous_filename": record.get("previous_filename"),
            "status": status,
        }
        contains_marker = False
        for side, path, ref in refs:
            text, text_error = _fetch_text_at_ref(repo=repo, path=path, ref=ref)
            if text_error or text is None:
                return None, text_error or f"content unavailable for {path} at {ref}"
            marker_present = "base_sensitive" in text
            proof[side] = {
                "contains_marker": marker_present,
                "path": path,
                "ref": ref,
            }
            contains_marker = contains_marker or marker_present
        current_paths = {filename}
        if status == "renamed":
            current_paths.add(str(record["previous_filename"]))
        for path in sorted(current_paths):
            text, text_error = _fetch_text_at_ref(
                repo=repo,
                path=path,
                ref=current_main_sha,
                allow_absent=True,
            )
            if text_error:
                return None, text_error
            current_marker = text is not None and "base_sensitive" in text
            proof["current_main"].append(
                {
                    "contains_marker": current_marker,
                    "exists": text is not None,
                    "path": path,
                    "ref": current_main_sha,
                }
            )
            contains_marker = contains_marker or current_marker
        if contains_marker:
            sensitive.append(filename)
        provenance.append(proof)

    return {
        "base_sha": base_sha,
        "candidate_files": [str(record["filename"]) for record in candidates],
        "changed_file_records": records,
        "changed_files": [str(record["filename"]) for record in records],
        "changed_sensitive_files": sorted(sensitive),
        "complete": True,
        "content_provenance": provenance,
        "current_main_sha": current_main_sha,
        "current_main_ref_verified": True,
        "head_sha": head_sha,
    }, None


def _attach_missing_coverage_scope(
    changed_coverage: dict[str, Any], pr_number: str | int, *, repo: str
) -> tuple[dict[str, Any], str | None]:
    """Attach a complete changed-file proof when the exact coverage check is absent."""
    if changed_coverage.get("status") != "missing":
        return changed_coverage, None
    changed_files, changed_files_err = _fetch_pr_changed_files(pr_number, repo=repo)
    if changed_files_err:
        return {}, f"failed to prove changed-file scope: {changed_files_err}"
    return {
        **changed_coverage,
        "changed_files": changed_files,
        "changed_files_complete": True,
    }, None


def _rest_labels(raw: Any) -> list[dict[str, Any]]:
    """Normalize REST PR label objects into the ``gh pr view`` label shape."""
    if not isinstance(raw, list):
        return []
    return [
        {"name": str(label.get("name") or "")}
        for label in raw
        if isinstance(label, dict) and label.get("name")
    ]


def _rest_json_list(
    *, owner: str, name: str, path: str, fail_message: str
) -> tuple[list[Any], str | None]:
    """Fetch a bounded REST JSON-list endpoint with fail-closed pagination."""
    rows: list[Any] = []
    for page in range(1, _MAX_REST_EVIDENCE_PAGES + 1):
        result = _gh(
            [
                "api",
                f"repos/{owner}/{name}/{path}?per_page={_REST_EVIDENCE_PAGE_SIZE}&page={page}",
            ],
            timeout=45,
        )
        if result.returncode != 0:
            return [], result.stderr.strip() or fail_message
        payload, err = _parse_json(result.stdout)
        if err or not isinstance(payload, list):
            return [], err or fail_message
        if any(not isinstance(item, dict) for item in payload):
            return [], f"{fail_message}: response contains a malformed entry"
        rows.extend(payload)
        if len(payload) < _REST_EVIDENCE_PAGE_SIZE:
            return rows, None
    return [], f"{fail_message}: response exceeded the bounded pagination limit"


def _rest_comments(
    *, owner: str, name: str, pr_number: str | int
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch issue comments in the ``gh pr view`` comment shape."""
    comments, err = _rest_json_list(
        owner=owner,
        name=name,
        path=f"issues/{pr_number}/comments",
        fail_message="REST comment response is not a JSON list",
    )
    if err:
        return [], err
    normalized: list[dict[str, Any]] = []
    for index, comment in enumerate(comments):
        user = comment.get("user")
        body = comment.get("body")
        association = comment.get("author_association")
        if (
            not isinstance(user, dict)
            or not isinstance(user.get("login"), str)
            or not user["login"].strip()
            or not isinstance(body, str)
            or not isinstance(association, str)
            or not association.strip()
        ):
            return [], f"REST comment response contains a malformed entry at index {index}"
        normalized.append(
            {
                "body": body,
                "authorAssociation": association.upper(),
                "author": {"login": user["login"]},
                "user": {"login": user["login"]},
                "createdAt": comment.get("created_at"),
                "updatedAt": comment.get("updated_at"),
            }
        )
    return normalized, None


def _rest_reviews(
    *, owner: str, name: str, pr_number: str | int
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch PR reviews in the ``gh pr view`` review shape."""
    reviews, err = _rest_json_list(
        owner=owner,
        name=name,
        path=f"pulls/{pr_number}/reviews",
        fail_message="REST review response is not a JSON list",
    )
    if err:
        return [], err
    normalized: list[dict[str, Any]] = []
    for index, review in enumerate(reviews):
        user = review.get("user")
        state = review.get("state")
        association = review.get("author_association")
        commit_id = review.get("commit_id")
        if (
            not isinstance(user, dict)
            or not isinstance(user.get("login"), str)
            or not user["login"].strip()
            or not isinstance(state, str)
            or state.upper()
            not in {"APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED", "PENDING"}
            or not isinstance(association, str)
            or not association.strip()
            or not isinstance(commit_id, str)
            or not re.fullmatch(r"[0-9a-fA-F]{40}", commit_id)
            or not isinstance(review.get("body"), (str, type(None)))
        ):
            return [], f"REST review response contains a malformed entry at index {index}"
        normalized.append(
            {
                "id": review.get("id"),
                "body": review.get("body") or "",
                "state": state.upper(),
                "author": {"login": user["login"]},
                "authorAssociation": association.upper(),
                "commit_id": commit_id,
                "submitted_at": review.get("submitted_at"),
                "user": {"login": user["login"]},
            }
        )
    return normalized, None


def _rest_requested_reviewers(
    *, owner: str, name: str, pr_number: str | int
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch requested reviewers in the ``gh pr view`` union shape."""
    requested_result = _gh(
        ["api", f"repos/{owner}/{name}/pulls/{pr_number}/requested_reviewers"],
        timeout=45,
    )
    if requested_result.returncode != 0:
        return [], requested_result.stderr.strip() or "REST requested-reviewer fetch failed"
    requested, err = _parse_json(requested_result.stdout)
    if err or not isinstance(requested, dict):
        return [], err or "REST requested-reviewer response is not a JSON object"
    users = requested.get("users")
    teams = requested.get("teams")
    if not isinstance(users, list) or not isinstance(teams, list):
        return [], "REST requested-reviewer response is missing users or teams lists"
    items: list[dict[str, Any]] = []
    for index, user in enumerate(users):
        if (
            not isinstance(user, dict)
            or not isinstance(user.get("login"), str)
            or not user["login"].strip()
        ):
            return [], f"REST requested-reviewer users contains a malformed entry at index {index}"
        items.append(
            {
                "user": {
                    "__typename": "User",
                    "login": user["login"],
                }
            }
        )
    for index, team in enumerate(teams):
        if (
            not isinstance(team, dict)
            or not isinstance(team.get("slug"), str)
            or not team["slug"].strip()
        ):
            return [], f"REST requested-reviewer teams contains a malformed entry at index {index}"
        items.append(
            {
                "team": {
                    "__typename": "Team",
                    "slug": team["slug"],
                    "name": str(team.get("name") or team["slug"]),
                }
            }
        )
    return items, None


def _rest_check_runs_page(
    *, owner: str, name: str, head_sha: str, page: int
) -> tuple[list[dict[str, Any]], int | None, str | None]:
    """Fetch and validate one exact-head check-run page."""
    result = _gh(
        [
            "api",
            f"repos/{owner}/{name}/commits/{head_sha}/check-runs?"
            f"per_page={_REST_EVIDENCE_PAGE_SIZE}&page={page}",
        ],
        timeout=45,
    )
    if result.returncode != 0:
        return [], None, result.stderr.strip() or "REST check-run fetch failed"
    payload, err = _parse_json(result.stdout)
    if err or not isinstance(payload, dict):
        return [], None, err or "REST check-run response is not a JSON object"
    page_rows = payload.get("check_runs")
    if not isinstance(page_rows, list) or any(not isinstance(item, dict) for item in page_rows):
        return [], None, "REST check-run response contains a malformed check_runs list"
    for index, check in enumerate(page_rows):
        check_error = _validate_rest_check_run(check, head_sha=head_sha, index=index)
        if check_error:
            return [], None, check_error
    raw_total_count = payload.get("total_count")
    if raw_total_count is not None and (type(raw_total_count) is not int or raw_total_count < 0):
        return [], None, "REST check-run response has an invalid total_count"
    return page_rows, raw_total_count, None


def _rest_check_runs(
    *, owner: str, name: str, head_sha: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch every exact-head check-run page and validate its count contract."""
    rows: list[dict[str, Any]] = []
    total_count: int | None = None
    for page in range(1, _MAX_REST_EVIDENCE_PAGES + 1):
        page_rows, page_total_count, page_error = _rest_check_runs_page(
            owner=owner, name=name, head_sha=head_sha, page=page
        )
        if page_error:
            return [], page_error
        if page_total_count is not None and total_count not in {None, page_total_count}:
            return [], "REST check-run response has an inconsistent total_count"
        total_count = page_total_count if page_total_count is not None else total_count
        rows.extend(page_rows)
        if total_count is not None and len(rows) > total_count:
            return [], "REST check-run response contains more rows than total_count"
        if total_count is not None and len(rows) == total_count:
            return rows, None
        if len(page_rows) < _REST_EVIDENCE_PAGE_SIZE:
            if total_count is not None and len(rows) < total_count:
                return [], "REST check-run response is incomplete"
            return rows, None
    return [], "REST check-run response exceeded the bounded pagination limit"


def _validate_rest_check_run(check: dict[str, Any], *, head_sha: str, index: int) -> str | None:
    """Validate one check-run object before it enters the fallback rollup."""
    name_value = check.get("name")
    status_value = check.get("status")
    conclusion_value = check.get("conclusion")
    reported_head = check.get("head_sha")
    malformed = (
        not isinstance(name_value, str)
        or not name_value.strip()
        or not isinstance(status_value, str)
        or status_value.lower() not in _REST_CHECK_STATUSES
        or (conclusion_value is not None and not isinstance(conclusion_value, str))
        or (status_value.lower() == "completed" and not isinstance(conclusion_value, str))
        or (
            reported_head is not None
            and (
                not isinstance(reported_head, str)
                or not re.fullmatch(r"[0-9a-fA-F]{40}", reported_head)
                or reported_head.lower() != head_sha.lower()
            )
        )
    )
    if malformed:
        return f"REST check-run response contains a malformed entry at index {index}"
    return None


def _rest_status_timestamp(status: dict[str, Any]) -> datetime | None:
    """Parse the newest available REST status timestamp, or return ``None``."""
    raw_timestamp = status.get("updated_at") or status.get("created_at")
    if not isinstance(raw_timestamp, str) or not raw_timestamp.strip():
        return None
    try:
        timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return timestamp if timestamp.tzinfo is not None else None


def _rest_commit_statuses(
    *, owner: str, name: str, head_sha: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch and validate the newest legacy status for each exact-head context.

    The REST ``statuses`` endpoint returns the status history, while the GraphQL
    rollup exposes the current context. Keeping historical entries would allow
    an older pending or failed status to block a newer successful one.
    """
    statuses, err = _rest_json_list(
        owner=owner,
        name=name,
        path=f"commits/{head_sha}/statuses",
        fail_message="REST commit-status response is invalid",
    )
    if err:
        return [], err
    latest_by_context: dict[str, tuple[datetime | None, dict[str, Any]]] = {}
    for index, status in enumerate(statuses):
        context = status.get("context")
        state = status.get("state")
        if (
            not isinstance(context, str)
            or not context.strip()
            or not isinstance(state, str)
            or state.lower() not in _REST_COMMIT_STATUS_STATES
        ):
            return [], f"REST commit-status response contains a malformed entry at index {index}"
        state_upper = state.upper()
        normalized = {
            "__typename": "StatusContext",
            "name": context,
            "context": context,
            "status": "COMPLETED" if state.lower() != "pending" else "PENDING",
            "state": state_upper,
            "conclusion": state_upper,
            "targetUrl": status.get("target_url"),
            "createdAt": status.get("created_at"),
            "updatedAt": status.get("updated_at"),
            "head_sha": head_sha,
        }
        timestamp = _rest_status_timestamp(status)
        previous = latest_by_context.get(context)
        if previous is None:
            latest_by_context[context] = (timestamp, normalized)
        elif timestamp is not None and previous[0] is not None and timestamp > previous[0]:
            latest_by_context[context] = (timestamp, normalized)
        # When either timestamp is unavailable or malformed, retain the first
        # entry. GitHub returns this history newest-first, so this avoids letting
        # a timestamped older entry replace an untimestamped current entry.
    return [entry for _timestamp, entry in latest_by_context.values()], None


def _rest_check_rollup(
    *, owner: str, name: str, head_sha: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch check-runs for the exact head in the ``statusCheckRollup`` shape."""
    check_runs, check_err = _rest_check_runs(owner=owner, name=name, head_sha=head_sha)
    if check_err:
        return [], check_err
    statuses, status_err = _rest_commit_statuses(owner=owner, name=name, head_sha=head_sha)
    if status_err:
        return [], status_err
    enriched_check_runs = _enrich_rest_check_runs(check_runs)
    rollup = _rest_check_runs_to_rollup(enriched_check_runs)
    for check, projected in zip(enriched_check_runs, rollup, strict=True):
        projected["head_sha"] = check.get("head_sha") or head_sha
        projected["html_url"] = check.get("html_url")
        projected["app"] = check.get("app") if isinstance(check.get("app"), dict) else {}
        projected["context"] = projected["name"]
    return [*rollup, *statuses], None


def _rest_pull_core(
    *, owner: str, name: str, pr_number: str | int
) -> tuple[dict[str, Any], str | None]:
    """Fetch the REST pull payload and validate its gate-critical fields.

    Returns ``(normalized_core, error)`` where ``normalized_core`` carries the
    pull number, title, body, lifecycle state, merge timestamp, draft state,
    exact head SHA, and labels in the ``gh pr view`` shape. Fails closed on any
    missing gate-critical field.
    """
    pull_result = _gh(["api", f"repos/{owner}/{name}/pulls/{pr_number}"], timeout=45)
    if pull_result.returncode != 0:
        return {}, pull_result.stderr.strip() or f"REST pull fetch failed for #{pr_number}"
    pull, err = _parse_json(pull_result.stdout)
    if err or not isinstance(pull, dict):
        return {}, err or "REST pull response is not a JSON object"

    head_sha = pull.get("head", {}).get("sha") if isinstance(pull.get("head"), dict) else None
    if not isinstance(head_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", head_sha):
        return {}, "REST pull head.sha is missing or malformed"
    draft_value = pull.get("draft")
    if type(draft_value) is not bool:
        return {}, "REST pull draft field is missing or malformed"
    title = pull.get("title")
    body = pull.get("body")
    if not isinstance(title, str) or body is None or not isinstance(body, str):
        return {}, "REST pull title or body is missing or malformed"
    state = pull.get("state")
    if not isinstance(state, str) or not state.strip():
        return {}, "REST pull state is missing or malformed"
    labels = pull.get("labels")
    if not isinstance(labels, list) or any(
        not isinstance(label, dict)
        or not isinstance(label.get("name"), str)
        or not label["name"].strip()
        for label in labels
    ):
        return {}, "REST pull labels are missing or malformed"
    merged_at = pull.get("merged_at")
    if merged_at is not None and (not isinstance(merged_at, str) or not merged_at.strip()):
        return {}, "REST pull merged_at is malformed"

    return {
        "number": pull.get("number"),
        "title": title,
        "body": body,
        "state": state,
        "mergedAt": merged_at,
        "isDraft": draft_value,
        "headRefOid": head_sha,
        "labels": _rest_labels(labels),
        "merge_commit_sha": pull.get("merge_commit_sha"),
        "mergeCommit": (
            {"oid": pull.get("merge_commit_sha")} if pull.get("merge_commit_sha") else None
        ),
    }, None


def _rest_pr_view_payload(pr_number: str | int, *, repo: str) -> tuple[dict[str, Any], str | None]:
    """Reconstruct the ``gh pr view`` payload shape from REST endpoints.

    Used when GraphQL quota is exhausted (issue #7705): REST reads remain
    available even when the authenticated user's GraphQL budget is spent. Builds
    exactly the same normalized payload keys that ``fetch_pr_snapshot`` passes to
    the shared consumers, so no downstream projection changes.

    Returns ``(payload, error)``; fail-closed: any missing REST-required field
    returns ``({}, error)`` so a partially reconstructed snapshot is never
    treated as a complete gate snapshot.
    """
    owner, _, name = repo.partition("/")
    if not owner or not name:
        return {}, f"invalid repo identifier: {repo!r}"

    core, core_err = _rest_pull_core(owner=owner, name=name, pr_number=pr_number)
    if core_err:
        return {}, core_err

    comments, comments_err = _rest_comments(owner=owner, name=name, pr_number=pr_number)
    if comments_err:
        return {}, comments_err
    reviews, reviews_err = _rest_reviews(owner=owner, name=name, pr_number=pr_number)
    if reviews_err:
        return {}, reviews_err
    reviewers, reviewers_err = _rest_requested_reviewers(
        owner=owner, name=name, pr_number=pr_number
    )
    if reviewers_err:
        return {}, reviewers_err
    rollup, checks_err = _rest_check_rollup(owner=owner, name=name, head_sha=core["headRefOid"])
    if checks_err:
        return {}, checks_err

    return {
        **core,
        "comments": comments,
        "reviews": reviews,
        "reviewRequests": reviewers,
        "statusCheckRollup": rollup,
    }, None


def fetch_pr_snapshot(  # noqa: C901, PLR0912 - validates several independent live API fields fail-closed.
    pr_number: str | int, *, repo: str
) -> tuple[dict[str, Any], str | None]:
    """Fetch a compact PR snapshot via ``gh pr view`` for gate evaluation.

    Returns ``(snapshot, error)``. The snapshot carries the fields consumed by
    ``evaluate_merge_gate`` plus the compact comment/review body excerpts that
    carry gate-verdict trailers (in the shape
    ``has_current_accepted_gate_verdict`` reads).
    """
    retry = run_with_retry(
        _gh,
        [
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "number,title,body,state,mergedAt,isDraft,headRefOid,labels,statusCheckRollup,comments,reviews,reviewRequests,mergeCommit",
        ],
        timeout=30,
    )
    result = retry.result
    snapshot_data_source = "graphql"
    graphql_fallback_diagnostic = ""
    if retry.quota_exhausted:
        # GraphQL quota is spent but REST reads remain available; rebuild the
        # snapshot payload through REST so hosted-check evidence can still be
        # refreshed (issue #7705). Fail closed when REST is also unavailable.
        rest_payload, rest_err = _rest_pr_view_payload(pr_number, repo=repo)
        if rest_err or not isinstance(rest_payload, dict):
            return {}, rest_err or "REST snapshot fallback returned no payload"
        payload = rest_payload
        snapshot_data_source = "rest_fallback_graphql_quota"
        graphql_fallback_diagnostic = retry.terminal_diagnostic
    elif result.returncode != 0:
        diagnostic = retry.terminal_diagnostic if retry.exhausted else result.stderr.strip()
        return {}, diagnostic or f"gh pr view failed (exit {result.returncode})"
    else:
        payload, err = _parse_json(result.stdout)
        if err or not isinstance(payload, dict):
            return {}, err or "gh pr view output is not a JSON object"

    draft_value = payload.get("isDraft")
    if type(draft_value) is not bool:
        return {}, "gh pr view isDraft field is missing or malformed"

    title = payload.get("title")
    body = payload.get("body")
    if not isinstance(title, str):
        return {}, "gh pr view title field is missing or malformed"
    if body is None:
        body = ""
    if not isinstance(body, str):
        return {}, "gh pr view body field is malformed"

    raw_state = payload.get("state")
    if not isinstance(raw_state, str) or not raw_state.strip():
        return {}, "gh pr view state field is missing or malformed"
    merged_at = payload.get("mergedAt")
    if merged_at is not None and (not isinstance(merged_at, str) or not merged_at.strip()):
        return {}, "gh pr view mergedAt field is malformed"
    pr_state = raw_state.upper()
    if merged_at:
        pr_state = "MERGED"
    elif pr_state not in {"OPEN", "CLOSED"}:
        return {}, f"gh pr view state field is unsupported: {pr_state}"

    raw_merge_commit = payload.get("mergeCommit")
    merge_commit_sha = (
        raw_merge_commit.get("oid")
        if isinstance(raw_merge_commit, dict)
        else payload.get("merge_commit_sha")
    )
    if not isinstance(merge_commit_sha, str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", merge_commit_sha
    ):
        merge_commit_sha = None

    review_requests = payload.get("reviewRequests")
    if not isinstance(review_requests, list):
        return {}, "gh pr view reviewRequests field is missing or malformed"

    base_sha, base_err = _fetch_pr_base_sha(pr_number, repo=repo)
    if base_err:
        return {}, f"failed to fetch PR base SHA: {base_err}"

    head_sha = str(payload.get("headRefOid") or "")
    changed_coverage, changed_coverage_err = _fetch_exact_head_changed_coverage(head_sha, repo=repo)
    if changed_coverage_err:
        return {}, f"failed to fetch exact-head changed coverage: {changed_coverage_err}"
    changed_coverage, changed_scope_err = _attach_missing_coverage_scope(
        changed_coverage, pr_number, repo=repo
    )
    if changed_scope_err:
        return {}, changed_scope_err

    current_metadata_digest = metadata_digest(title, body)
    requested_reviewers, requested_teams = _requested_review_identities(review_requests)
    required_checks = _to_receipt_check_runs(
        payload.get("statusCheckRollup"),
        head_sha=head_sha,
        expected_metadata_digest=current_metadata_digest,
    )
    review_evidence = {
        "check_runs": required_checks,
        "reviews": _to_receipt_review_evidence(
            payload.get("reviews"),
            head_sha=head_sha,
            expected_metadata_digest=current_metadata_digest,
        ),
        "comments": _to_receipt_review_evidence(
            payload.get("comments"),
            head_sha=head_sha,
            expected_metadata_digest=current_metadata_digest,
        ),
    }
    review_claim = _to_review_claim_snapshot(
        payload.get("comments"),
        payload.get("reviews"),
        head_sha=head_sha,
    )

    snapshot: dict[str, Any] = {
        "number": payload.get("number"),
        "pr_state": pr_state,
        "pr_merged_at": merged_at,
        "merge_commit_sha": merge_commit_sha,
        "draft": draft_value,
        "head_sha": head_sha,
        "metadata_digest": current_metadata_digest,
        "body": body,
        "base_sha": base_sha,
        "labels": _normalize_labels(payload.get("labels")),
        "checks": {"overall": _rollup_overall(payload.get("statusCheckRollup") or [])},
        "changed_coverage": changed_coverage,
        # Canonical extraction rejects trailers from untrusted author associations.
        "gate_verdicts": _extract_gate_verdicts(payload),
        "base_policy": _extract_base_policies(payload),
        "metadata_verdicts": _extract_metadata_verdicts(payload),
        "review_snapshot": _to_body_snapshot(payload.get("reviews")),
        "comment_snapshot": _to_body_snapshot(payload.get("comments")),
        "review_claim": review_claim,
        "required_checks": required_checks,
        "review_evidence": review_evidence,
        "requested_reviewers": requested_reviewers,
        "requested_teams": requested_teams,
        # Each GitHub review request is an explicit request for review. Treat any
        # outstanding request as protective rather than guessing whether a user,
        # team, or bot is "external"; this is conservative parity with the
        # gh-pr-merger preflight.
        "reviewers_requested": bool(review_requests),
        "data_source": snapshot_data_source,
        "evidence_provenance": {
            "schema": SNAPSHOT_PROVENANCE_SCHEMA,
            "data_source": snapshot_data_source,
            "ordinary_facts": {
                "pull_request": "rest" if snapshot_data_source.startswith("rest_") else "graphql",
                "labels": "rest" if snapshot_data_source.startswith("rest_") else "graphql",
                "comments": "rest" if snapshot_data_source.startswith("rest_") else "graphql",
                "reviews": "rest" if snapshot_data_source.startswith("rest_") else "graphql",
                "requested_reviewers": (
                    "rest" if snapshot_data_source.startswith("rest_") else "graphql"
                ),
                "check_rollup": "rest" if snapshot_data_source.startswith("rest_") else "graphql",
                "base_sha": "rest",
                "changed_coverage": "rest",
            },
            "review_threads": {
                "source": "graphql",
                "status": "separate_query",
            },
            "fallback_diagnostic": graphql_fallback_diagnostic or None,
        },
    }
    return snapshot, None


def fetch_main_sha(*, repo: str) -> str:
    """Return current ``main`` HEAD SHA, or an empty string on failure."""
    result = _gh(["api", f"repos/{repo}/git/refs/heads/main", "--jq", ".object.sha"])
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _complete_review_thread_nodes(threads: Any) -> tuple[list[Any] | None, str | None]:
    """Return nodes only when a review-thread connection is complete and well-formed."""
    if not isinstance(threads, dict):
        return None, "reviewThreads is missing from graphql response"
    nodes = threads.get("nodes")
    if not isinstance(nodes, list):
        return None, "reviewThreads.nodes missing from graphql response"
    total_count = threads.get("totalCount")
    if type(total_count) is not int or total_count < 0:
        return None, "reviewThreads.totalCount missing from graphql response"
    page_info = threads.get("pageInfo")
    if not isinstance(page_info, dict):
        return None, "reviewThreads.pageInfo missing from graphql response"
    has_next_page = page_info.get("hasNextPage")
    if type(has_next_page) is not bool:
        return None, "reviewThreads.pageInfo.hasNextPage missing from graphql response"
    if has_next_page:
        return None, "reviewThreads connection is incomplete; refusing an unresolved-thread bypass"
    if total_count != len(nodes):
        return None, "reviewThreads totalCount does not match the complete node list"
    return nodes, None


def _review_thread_state(node: Any) -> tuple[bool, bool] | None:
    """Return ``(is_resolved, is_outdated)`` only for a complete thread node."""
    if not isinstance(node, dict):
        return None
    is_resolved = node.get("isResolved")
    is_outdated = node.get("isOutdated")
    if type(is_resolved) is not bool or type(is_outdated) is not bool:
        return None
    return is_resolved, is_outdated


def fetch_merge_queue_strategy(pr_number: str | int, *, repo: str) -> tuple[str | None, str | None]:
    """Return the live merge queue's grouping strategy for an enqueued PR.

    ``HEADGREEN`` permits an earlier failing queue entry to merge with a passing
    tail entry. This gate validates the PR encoded in the merge-group ref, so it
    requires ``ALLGREEN`` to ensure every earlier constituent entry also passed
    its own required gate check.
    """
    owner, _, name = repo.partition("/")
    if not owner or not name:
        return None, f"invalid repo identifier: {repo!r}"
    query = (
        "query($owner:String!,$name:String!,$number:Int!){"
        "repository(owner:$owner,name:$name){pullRequest(number:$number){"
        "mergeQueueEntry{mergeQueue{configuration{mergingStrategy}}}}}}"
    )
    retry = run_with_retry(
        _gh,
        [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"number={pr_number}",
        ],
        timeout=45,
    )
    result = retry.result
    if retry.quota_exhausted:
        return None, retry.terminal_diagnostic or "graphql mergeQueue configuration query failed"
    if result.returncode != 0:
        diagnostic = retry.terminal_diagnostic if retry.exhausted else result.stderr.strip()
        return None, diagnostic or "graphql mergeQueue configuration query failed"
    payload, err = _parse_json(result.stdout)
    if err or not isinstance(payload, dict):
        return None, err or "graphql response is not JSON"
    graphql_err = _graphql_error(payload)
    if graphql_err:
        return None, graphql_err
    pull_request, pull_request_error = _graphql_pull_request(payload)
    if pull_request_error or pull_request is None:
        return None, pull_request_error or "GraphQL pull-request data is missing"
    entry = pull_request.get("mergeQueueEntry")
    queue = entry.get("mergeQueue") if isinstance(entry, dict) else None
    configuration = queue.get("configuration") if isinstance(queue, dict) else None
    strategy = configuration.get("mergingStrategy") if isinstance(configuration, dict) else None
    if strategy not in {"ALLGREEN", "HEADGREEN"}:
        return None, "merge queue strategy missing or unsupported in graphql response"
    return str(strategy), None


def _quota_exhausted_thread_diagnostic(pr_number: str | int, *, repo: str) -> str:
    """Build the reset-aware handoff for quota-blocked review-thread reads (issue #8282)."""
    handoff = quota_reset_handoff(
        retry_command=shlex.join(
            [
                "uv",
                "run",
                "python",
                "scripts/dev/single_account_merge_receipt.py",
                "--repo",
                repo,
                "--pr",
                str(pr_number),
                "--mode",
                "report-only",
                "--output",
                f"output/validation/pr-{pr_number}-merge-receipt.json",
            ]
        ),
    )
    return handoff["handoff"]


def _thread_query_failure_diagnostic(
    retry: GraphQLRetryOutcome,
    result: subprocess.CompletedProcess[str],
    pr_number: str | int,
    *,
    repo: str,
) -> str:
    """Render the fail-closed diagnostic for a failed review-thread query."""
    diagnostic = retry.terminal_diagnostic if retry.exhausted else result.stderr.strip()
    if retry.quota_exhausted:
        handoff_text = _quota_exhausted_thread_diagnostic(pr_number, repo=repo)
        return f"{diagnostic} {handoff_text}" if diagnostic else handoff_text
    return diagnostic or "graphql reviewThreads query failed"


def fetch_threads_resolved(pr_number: str | int, *, repo: str) -> tuple[bool | None, str | None]:
    """Return ``(all_resolved, error)`` for a PR's review threads.

    Queries the first review-thread page via GraphQL. Returns ``True`` only when
    that connection is complete and there are no unresolved, non-outdated
    (actionable) threads; ``False`` when at least one remains; ``(None, error)``
    when the query fails or is incomplete (caller fails closed).
    """
    owner, _, name = repo.partition("/")
    if not owner or not name:
        return None, f"invalid repo identifier: {repo!r}"
    query = (
        "query($owner:String!,$name:String!,$number:Int!){"
        "repository(owner:$owner,name:$name){pullRequest(number:$number){"
        "reviewThreads(first:100){totalCount pageInfo{hasNextPage}"
        "nodes{isResolved isOutdated}}}}}"
    )
    retry = run_with_retry(
        _gh,
        [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"number={pr_number}",
        ],
        timeout=45,
    )
    result = retry.result
    if retry.quota_exhausted or result.returncode != 0:
        return None, _thread_query_failure_diagnostic(retry, result, pr_number, repo=repo)
    payload, err = _parse_json(result.stdout)
    if err or not isinstance(payload, dict):
        return None, err or "graphql response is not JSON"
    graphql_err = _graphql_error(payload)
    if graphql_err:
        return None, graphql_err
    pull_request, pull_request_error = _graphql_pull_request(payload)
    if pull_request_error or pull_request is None:
        return None, pull_request_error or "GraphQL pull-request data is missing"
    threads = pull_request.get("reviewThreads")
    nodes, connection_error = _complete_review_thread_nodes(threads)
    if connection_error or nodes is None:
        return None, connection_error or "reviewThreads connection is incomplete"
    for node in nodes:
        state = _review_thread_state(node)
        if state is None:
            return None, "reviewThreads.nodes contains incomplete thread state"
        is_resolved, is_outdated = state
        if not is_resolved and not is_outdated:
            return False, None
    return True, None


def _format_summary(audit: MergeGateAudit) -> str:
    """Format the audit as a compact GitHub step-summary block."""
    verdict = "PASS" if audit.passed else "FAIL"
    lines = [
        f"### Merge Queue Gate: {verdict}",
        "",
        f"- PR: #{audit.pr if audit.pr is not None else '?'}",
        f"- evaluated head SHA: `{audit.head_sha or '?'}`",
        f"- merge-group source head SHA: `{audit.merge_group_head_sha or 'n/a'}`",
        f"- merge-group source-head binding: `{audit.merge_group_head_binding}`",
        f"- merge queue merging strategy: `{audit.queue_merging_strategy}`",
        f"- base SHA: `{audit.base_sha or '?'}`",
        f"- main SHA: `{audit.main_sha or 'n/a'}`",
        f"- labels: `{', '.join(audit.labels) if audit.labels else '(none)'}`",
        f"- draft: `{audit.draft}`",
        f"- merge-ready: `{audit.merge_ready}`",
        f"- exact-head changed-coverage status: `{audit.changed_coverage_status}`",
        f"- changed-coverage head SHA: `{audit.changed_coverage_head_sha or '?'}`",
        f"- gate-verdict status: `{audit.gate_verdict_status}`",
        f"- exact-head review-claim status: `{audit.review_claim_status}`",
        f"- PR metadata digest: `{audit.metadata_digest or '?'}`",
        f"- PR metadata verdict status: `{audit.metadata_verdict_status}`",
        f"- closing-discipline status: `{audit.closing_discipline_status}`",
        f"- body narrative status: `{audit.body_narrative_status}`",
        f"- staleness verdict: `{audit.staleness_verdict}`",
        f"- CI conclusion: `{audit.ci_overall}`",
        f"- thread resolution: `{audit.thread_resolution}`",
        f"- requested-reviewer status: `{audit.reviewer_request_status}`",
        f"- ancestry state: `{audit.ancestry_state or 'not_evaluated'}`",
    ]
    if audit.body_not_ready_sentinels:
        lines.append(
            f"- stale body sentinels: `{', '.join(audit.body_not_ready_sentinels)}` "
            "(run `uv run python scripts/dev/gh_pr_body_rest.py --reconcile` to update)"
        )
    if audit.reasons:
        lines.append(f"- fail-closed reasons: `{', '.join(audit.reasons)}`")
    lines.append("")
    lines.append(
        "Gate contract: non-draft + `merge-ready` + current exact-head "
        "`gate-verdict: accepted` trailer + current `pr-metadata: reconciled` "
        "trailer + exact-head `changed-coverage-gate` proof (or a complete CI-ignored docs-only "
        "file-set proof) + resolved threads + no outstanding reviewer requests + "
        "no active exact-head review claim + `ALLGREEN` queue strategy; fail-closed on any "
        "missing dimension. "
        "See `docs/dev_guide.md` and "
        "`.agents/skills/gh-pr-merger/SKILL.md`."
    )
    return "\n".join(lines)


def _append_step_summary(text: str) -> None:
    """Append the summary block to ``GITHUB_STEP_SUMMARY`` when running in CI."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
    except OSError:
        # Summary is best-effort; never fail the gate on a summary write error.
        pass


def _evaluate_live(
    pr_number: str | int,
    *,
    repo: str,
    merge_group_base_sha: str = "",
    merge_group_head_sha: str = "",
) -> tuple[MergeGateAudit, str | None]:
    """Fetch live PR state, evaluate the gate, and return ``(audit, error)``."""
    snapshot, err = fetch_pr_snapshot(pr_number, repo=repo)
    if err:
        audit = evaluate_merge_gate(
            {"number": pr_number, "head_sha": ""},
            main_sha="",
            ci_overall="unknown",
            threads_resolved=None,
            reviewers_requested=None,
            merge_group_head_sha=merge_group_head_sha,
        )
        return _failed_audit(audit, "pr_snapshot_unavailable"), err

    closing_discipline_status, closing_discipline_blockers = _fetch_live_closing_discipline(
        pr_number, repo=repo, body=str(snapshot.get("body") or "")
    )
    snapshot["closing_discipline"] = {
        "status": closing_discipline_status,
        "blockers": closing_discipline_blockers,
    }

    if merge_group_base_sha:
        # Inside the merge queue the base SHA is the prospective current main, so
        # staleness is fresh by construction; record the queue base as both base
        # and main so the audit reflects that guarantee.
        snapshot["base_sha"] = merge_group_base_sha
        main_sha = merge_group_base_sha
        queue_merging_strategy, strategy_err = fetch_merge_queue_strategy(pr_number, repo=repo)
        if strategy_err:
            return (
                evaluate_merge_gate(
                    snapshot,
                    main_sha=main_sha,
                    threads_resolved=None,
                    reviewers_requested=bool(snapshot["reviewers_requested"]),
                    merge_group_head_sha=merge_group_head_sha,
                    queue_merging_strategy="unknown",
                ),
                f"merge queue strategy query failed: {strategy_err}",
            )
    else:
        main_sha = fetch_main_sha(repo=repo)
        if not main_sha:
            audit = evaluate_merge_gate(
                snapshot,
                main_sha="",
                threads_resolved=None,
                reviewers_requested=bool(snapshot["reviewers_requested"]),
            )
            return _failed_audit(audit, "main_sha_unavailable"), (
                "failed to fetch current main SHA; refusing a source-head gate pass"
            )
        queue_merging_strategy = ""

    threads_resolved, thread_err = fetch_threads_resolved(pr_number, repo=repo)
    if thread_err:
        # Fail closed when the thread query fails: the audit records the reason.
        return (
            evaluate_merge_gate(
                snapshot,
                main_sha=main_sha,
                threads_resolved=None,
                reviewers_requested=bool(snapshot["reviewers_requested"]),
                merge_group_head_sha=merge_group_head_sha,
                queue_merging_strategy=queue_merging_strategy or "",
            ),
            f"thread resolution query failed: {thread_err}",
        )

    audit = evaluate_merge_gate(
        snapshot,
        main_sha=main_sha,
        threads_resolved=threads_resolved,
        reviewers_requested=bool(snapshot["reviewers_requested"]),
        merge_group_head_sha=merge_group_head_sha,
        queue_merging_strategy=queue_merging_strategy or "",
    )
    return audit, None


def _self_test() -> int:
    """Run deterministic assertions covering the issue #6274 gate contract.

    Exercises the three validation scenarios plus the additional fail-closed
    dimensions (CI, staleness, threads, draft, stale head). Exits 0 when all
    assertions hold and 1 otherwise. This is the executable proof that the
    workflow fails/passes CI for each scenario.
    """
    full_sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
    other_sha = "deadbeefcafebabe111122223333444455556666"

    def _pr(
        *,
        labels: list[str],
        gate_verdict_sha: str = "",
        head_sha: str = full_sha,
        draft: bool = False,
        base_sha: str = "",
        ci_overall: str | None = None,
        body: str = "final body",
    ) -> dict[str, Any]:
        pr: dict[str, Any] = {
            "number": 6274,
            "head_sha": head_sha,
            "labels": list(labels),
            "draft": draft,
            "base_sha": base_sha,
            "body": body,
            "changed_coverage": {"status": "success", "head_sha": head_sha},
        }
        digest = metadata_digest("merge queue gate self-test", body)
        pr["metadata_digest"] = digest
        if ci_overall is not None:
            pr["checks"] = {"overall": ci_overall}
        if gate_verdict_sha:
            pr["gate_verdict"] = {"verdict": "accepted", "sha": gate_verdict_sha}
            pr["metadata_verdicts"] = [metadata_trailer(digest)]
        return pr

    failures: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    # Scenario 1: PR lacks merge-ready label -> fail.
    audit = evaluate_merge_gate(_pr(labels=[]))
    expect(
        not audit.passed and "missing_merge_ready_label" in audit.reasons,
        "scenario1: PR without merge-ready must fail with missing_merge_ready_label",
    )
    expect(
        audit.gate_verdict_status == "missing",
        "scenario1: gate-verdict status must be missing without a trailer",
    )

    # Scenario 2: merge-ready but no current exact-head gate-verdict -> fail.
    audit = evaluate_merge_gate(_pr(labels=["merge-ready"]))
    expect(
        not audit.passed and "missing_exact_head_gate_verdict" in audit.reasons,
        "scenario2: merge-ready without gate-verdict must fail with "
        "missing_exact_head_gate_verdict",
    )

    # Scenario 3: all live preflight dimensions green -> pass.
    audit = evaluate_merge_gate(
        _pr(labels=["merge-ready"], gate_verdict_sha=full_sha, ci_overall="success"),
        threads_resolved=True,
        reviewers_requested=False,
    )
    expect(audit.passed, "scenario3: all live preflight dimensions must pass")
    expect(
        audit.gate_verdict_status == "accepted",
        "scenario3: gate-verdict status must be accepted",
    )

    # Stale body narrative: merge-ready PR carrying unapproved/not-ready narrative -> fail.
    unapproved_body = (
        "The PR remains unapproved and not merge-ready pending independent "
        "exact-head review and current hosted checks."
    )
    audit = evaluate_merge_gate(
        _pr(
            labels=["merge-ready"],
            gate_verdict_sha=full_sha,
            body=unapproved_body,
            ci_overall="success",
        ),
        threads_resolved=True,
        reviewers_requested=False,
    )
    expect(
        not audit.passed
        and "stale_not_ready_body_narrative" in audit.reasons
        and audit.body_narrative_status == "stale"
        and len(audit.body_not_ready_sentinels) > 0,
        "stale-narrative: PR with 'not merge-ready' / 'remains unapproved' body must fail closed",
    )

    # Stale head: gate-verdict for a different SHA -> fail (exact-head contract).
    audit = evaluate_merge_gate(
        _pr(labels=["merge-ready"], gate_verdict_sha=other_sha, head_sha=full_sha)
    )
    expect(
        not audit.passed and "missing_exact_head_gate_verdict" in audit.reasons,
        "stale-head: gate-verdict for a different SHA must fail closed",
    )

    # CI failure -> fail.
    audit = evaluate_merge_gate(
        _pr(labels=["merge-ready"], gate_verdict_sha=full_sha, ci_overall="failure")
    )
    expect(
        not audit.passed and any(r.startswith("ci_not_green") for r in audit.reasons),
        "ci-failure: red CI must fail closed",
    )

    # Stale base -> fail.
    audit = evaluate_merge_gate(
        _pr(labels=["merge-ready"], gate_verdict_sha=full_sha, base_sha=other_sha),
        main_sha=full_sha,
    )
    expect(
        not audit.passed and "stale_merge_base" in audit.reasons,
        "stale-base: base != main must fail closed",
    )
    expect(
        audit.staleness_verdict == "stale",
        "stale-base: staleness verdict must be stale",
    )

    # Unresolved threads -> fail.
    audit = evaluate_merge_gate(
        _pr(labels=["merge-ready"], gate_verdict_sha=full_sha),
        threads_resolved=False,
    )
    expect(
        not audit.passed and "unresolved_review_threads" in audit.reasons,
        "unresolved-threads: open actionable threads must fail closed",
    )
    expect(
        audit.thread_resolution == "unresolved",
        "unresolved-threads: thread resolution must be unresolved",
    )

    # Explicitly requested reviewers must be allowed to complete their review.
    audit = evaluate_merge_gate(
        _pr(labels=["merge-ready"], gate_verdict_sha=full_sha),
        threads_resolved=True,
        reviewers_requested=True,
    )
    expect(
        not audit.passed and "outstanding_requested_reviewers" in audit.reasons,
        "requested-reviewer: an outstanding review request must fail closed",
    )
    expect(
        audit.reviewer_request_status == "requested",
        "requested-reviewer: audit must record the outstanding request",
    )

    # Draft -> fail (merge queue must never merge a draft).
    audit = evaluate_merge_gate(_pr(labels=["merge-ready"], gate_verdict_sha=full_sha, draft=True))
    expect(
        not audit.passed and "pr_is_draft" in audit.reasons,
        "draft: draft PR must fail closed",
    )

    # HEADGREEN permits an earlier failing entry to hitchhike with a passing tail.
    audit = evaluate_merge_gate(
        _pr(labels=["merge-ready"], gate_verdict_sha=full_sha),
        threads_resolved=True,
        merge_group_head_sha=full_sha,
        queue_merging_strategy="HEADGREEN",
    )
    expect(
        not audit.passed and "unsafe_merge_queue_strategy:HEADGREEN" in audit.reasons,
        "queue-strategy: HEADGREEN must fail closed",
    )

    # Full pass: all dimensions satisfied and explicitly authoritative.
    audit = evaluate_merge_gate(
        _pr(labels=["merge-ready"], gate_verdict_sha=full_sha, base_sha=full_sha),
        main_sha=full_sha,
        ci_overall="success",
        threads_resolved=True,
        reviewers_requested=False,
    )
    expect(audit.passed and audit.reasons == [], "full-pass: all dimensions green must pass")

    # Audit record must carry every required inspectable field.
    audit = evaluate_merge_gate(_pr(labels=["merge-ready"], gate_verdict_sha=full_sha))
    required_fields = {
        "schema",
        "pr",
        "head_sha",
        "merge_group_head_sha",
        "merge_group_head_binding",
        "queue_merging_strategy",
        "base_sha",
        "main_sha",
        "labels",
        "draft",
        "ci_overall",
        "changed_coverage_status",
        "changed_coverage_head_sha",
        "gate_verdict_status",
        "staleness_verdict",
        "thread_resolution",
        "reviewer_request_status",
        "review_claim_status",
        "merge_ready",
        "passed",
        "reasons",
    }
    expect(required_fields <= set(audit.to_dict()), "audit: all required fields present")
    expect(audit.schema == AUDIT_SCHEMA, "audit: schema tag is merge_queue_gate.v1")
    expect(audit.head_sha == full_sha, "audit: evaluated head SHA recorded")
    expect(
        "merge-ready" in audit.labels,
        "audit: label set recorded and includes merge-ready",
    )

    # Abbreviated gate-verdict SHA (>=7 hex) must match the full head (parity with
    # pr_loop_policy.GATE_VERDICT_MIN_SHA_OVERLAP).
    audit = evaluate_merge_gate(
        _pr(labels=["merge-ready"], gate_verdict_sha=full_sha[:12], ci_overall="success"),
        threads_resolved=True,
        reviewers_requested=False,
    )
    expect(audit.passed, "abbreviated-sha: 12-char gate-verdict prefix must match head")

    # Trailer carried in a comment body must satisfy the gate. The compact
    # snapshot shape (``comment_snapshot.latest[].body_excerpt``) is what
    # ``has_current_accepted_gate_verdict`` reads, so the runtime CLI feeds real
    # comment bodies through the same shape.
    audit = evaluate_merge_gate(
        {
            "number": 6274,
            "head_sha": full_sha,
            "labels": ["merge-ready"],
            "draft": False,
            "checks": {"overall": "success"},
            "changed_coverage": {"status": "success", "head_sha": full_sha},
            "reviewers_requested": False,
            "metadata_digest": metadata_digest("merge queue gate self-test", "final body"),
            "metadata_verdicts": [
                metadata_trailer(metadata_digest("merge queue gate self-test", "final body"))
            ],
            "comment_snapshot": {
                "latest": [
                    {
                        "author_association": "OWNER",
                        "body_excerpt": f"lgtm\n\ngate-verdict: accepted @ {full_sha}",
                    }
                ]
            },
        },
        threads_resolved=True,
    )
    expect(audit.passed, "comment-carrier: gate-verdict trailer in a comment must satisfy gate")

    if failures:
        for message in failures:
            print(f"FAIL: {message}", file=sys.stderr)
        return 1
    print("merge_queue_gate self-test: all assertions passed")
    return 0


def _load_merge_group_event(path: str) -> tuple[dict[str, Any] | None, str | None]:
    """Load and validate the native merge-group event envelope."""
    try:
        event = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Failed to read merge_group event payload: {exc}"
    if not isinstance(event, dict):
        return None, "Event payload is not a JSON object; failing closed."
    event_name = event.get("event_name")
    if "merge_group" not in event or (event_name is not None and event_name != "merge_group"):
        return (
            None,
            "Event payload is not a merge_group event; "
            "merge_queue_gate only gates the native merge queue; failing closed.",
        )
    merge_group = event.get("merge_group")
    if not isinstance(merge_group, dict) or not merge_group.get("base_sha"):
        return None, "merge_group.base_sha is missing; failing closed."
    return event, None


def _audit_exit_code(audit: MergeGateAudit, *, advisory: bool) -> int:
    """Return the CLI status while preserving a truthful advisory audit."""
    if advisory and not audit.passed:
        print(
            "Source-PR admission is advisory; merge_group remains fail-closed.",
            file=sys.stderr,
        )
    return 0 if audit.passed or advisory else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point: evaluate the merge-queue gate and print the audit JSON."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pr", help="GitHub PR number to evaluate (live mode)")
    source.add_argument(
        "--from-event",
        metavar="PATH",
        help="resolve the source PR from a merge_group event JSON payload",
    )
    source.add_argument(
        "--self-test",
        action="store_true",
        help="run deterministic gate-contract assertions and exit",
    )
    parser.add_argument("--repo", default="", help="owner/repo (default: detect from gh)")
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="emit the audit record as JSON (default in CI; implied unless --summary-only)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        default=False,
        help="write only the step summary; suppress stdout audit JSON",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        default=False,
        help=("preserve a failed source-PR audit but exit zero; valid only with --pr"),
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if args.advisory and not args.pr:
        parser.error("--advisory is valid only with --pr; merge_group remains fail-closed")

    repo = _resolve_owner_repo(args.repo)
    if not repo:
        print("Failed to detect repository. Pass --repo owner/repo.", file=sys.stderr)
        return 1

    if args.from_event:
        event, event_error = _load_merge_group_event(args.from_event)
        if event_error:
            print(event_error, file=sys.stderr)
            return 1
        assert event is not None
        merge_group_pr = resolve_pr_from_merge_group(event)
        if merge_group_pr is None:
            print(
                "Could not parse a canonical source PR from the merge_group payload; "
                "failing closed.",
                file=sys.stderr,
            )
            return 1
        pr_number = merge_group_pr.number
        merge_group_head_sha = merge_group_pr.head_sha
        merge_group_base_sha = str(event["merge_group"]["base_sha"])
    else:
        parsed_pr_number = _safe_int(args.pr)
        if parsed_pr_number is None:
            print(f"Invalid PR number: {args.pr!r}", file=sys.stderr)
            return 1
        pr_number = parsed_pr_number
        merge_group_base_sha = ""
        merge_group_head_sha = ""

    audit, error = _evaluate_live(
        pr_number,
        repo=repo,
        merge_group_base_sha=merge_group_base_sha,
        merge_group_head_sha=merge_group_head_sha,
    )

    _append_step_summary(_format_summary(audit))
    if not args.summary_only:
        print(json.dumps(audit.to_dict()))
    if error:
        print(f"gate evaluation warning: {error}", file=sys.stderr)

    return _audit_exit_code(audit, advisory=args.advisory)


if __name__ == "__main__":
    raise SystemExit(main())
