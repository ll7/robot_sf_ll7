#!/usr/bin/env python3
"""Merge-queue status-check gate enforcing the gh-pr-merger fail-closed preflight.

This module backs the ``Merge Queue Gate`` required status check (issue #6274).
The GitHub native merge queue must not auto-merge a PR until this gate passes.
The gate enforces the same fail-closed preflight as ``gh-pr-merger``:

  - non-draft state,
  - current ``merge-ready`` label,
  - a current exact-head ``gate-verdict: accepted @ <head_sha>`` trailer
    (reuses ``scripts.dev.pr_loop_policy.has_current_accepted_gate_verdict``),
  - no unresolved actionable review threads,
  - staleness-free base (fresh by construction inside the merge queue, where the
    base SHA equals current ``main``; evaluated against ``main`` in ``--pr`` mode).

It emits a ``merge_queue_gate.v1`` audit record with the evaluated head SHA,
base SHA, label set, gate-verdict status, staleness verdict, CI conclusion, and
reviewer-thread resolution so the merge decision is inspectable and reproducible.

The pure function ``evaluate_merge_gate`` is deterministic and exercised by
``--self-test`` (the validation contract for issue #6274). The CLI resolves a
live PR (``--pr`` or ``--from-event`` for a ``merge_group`` payload), evaluates,
prints the audit JSON, appends a ``GITHUB_STEP_SUMMARY`` block, and exits 0 on
pass / 1 on fail (fail closed).

Why a separate gate instead of relying on labels alone: issue #6274 observed an
external/parallel auto-merge path merging PRs without ``merge-ready`` or without
a current exact-head gate verdict. A required status check that runs inside the
merge queue is the only GitHub-native surface that can fail-closed the queue
itself, because branch-protection required checks gate the merge event
regardless of which dispatcher requested the merge.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Make the sibling ``scripts.dev`` package importable when this file is run as a
# standalone script (``python scripts/dev/merge_queue_gate.py``). Under pytest or
# ``uv run`` the project root is already on ``sys.path``; this insert is a no-op
# there and only matters for direct script invocation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dev.pr_loop_policy import (  # noqa: E402
    has_current_accepted_gate_verdict,
)

AUDIT_SCHEMA = "merge_queue_gate.v1"

# CI rollup classification constants (mirror scripts/dev/check_pr_ci_status.py
# so the gate does not couple to that module's private helpers).
FAILURE_CONCLUSIONS = {
    "failure",
    "error",
    "cancelled",
    "timed_out",
    "action_required",
    "startup_failure",
}
PENDING_STATUSES = {"expected", "in_progress", "pending", "queued", "requested", "waiting"}

# GitHub merge-queue readonly ref: ``gh-readonly-queue/<base>/<srcbranch>-<sha>``.
_MERGE_QUEUE_REF_RE = re.compile(r"^refs/heads/gh-readonly-queue/(?P<base>[^/]+)/(?P<rest>.+)$")
# Trailing abbreviated commit SHA appended to the source branch in the queue ref.
_QUEUE_REF_SHA_SUFFIX_RE = re.compile(r"-(?P<sha>[0-9a-f]{7,40})$")


@dataclass(frozen=True, slots=True)
class MergeGateAudit:
    """Inspectable, reproducible merge-queue gate verdict for one PR evaluation."""

    schema: str
    pr: int | None
    head_sha: str
    base_sha: str
    main_sha: str
    labels: list[str]
    draft: bool
    ci_overall: str
    gate_verdict_status: str
    staleness_verdict: str
    thread_resolution: str
    merge_ready: bool
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit as a plain JSON-able dict."""
        return asdict(self)


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


def _fail_closed_reasons(
    *,
    head_sha: str,
    draft: bool,
    merge_ready: bool,
    ci_overall: str,
    staleness_verdict: str,
    gate_verdict_status: str,
    thread_resolution: str,
) -> list[str]:
    """Collect fail-closed reasons for one gate evaluation.

    Any non-empty reason list means the gate must fail closed. The order is
    stable so audit records are comparable across runs and machines.
    """
    reasons: list[str] = []
    if not head_sha:
        reasons.append("missing_head_sha")
    if draft:
        reasons.append("pr_is_draft")
    if not merge_ready:
        reasons.append("missing_merge_ready_label")
    if ci_overall and ci_overall != "success":
        reasons.append(f"ci_not_green:{ci_overall}")
    if staleness_verdict == "stale":
        reasons.append("stale_merge_base")
    if gate_verdict_status != "accepted":
        reasons.append("missing_exact_head_gate_verdict")
    if thread_resolution == "unresolved":
        reasons.append("unresolved_review_threads")
    return reasons


def evaluate_merge_gate(
    pr: dict[str, Any],
    *,
    main_sha: str = "",
    ci_overall: str | None = None,
    threads_resolved: bool | None = None,
) -> MergeGateAudit:
    """Evaluate the merge-queue gate for one PR snapshot.

    Pure function: no side effects, no GitHub calls. Fail-closed by design.

    Inputs:
      pr: compact PR snapshot. Recognized fields: ``head_sha`` (required for a
        pass), ``labels``, ``draft``, ``base_sha``, ``checks.overall``, plus any
        gate-verdict carrier fields understood by
        ``has_current_accepted_gate_verdict`` (``gate_verdict`` /
        ``gate_verdicts`` / ``comments`` / ``reviews`` body excerpts).
      main_sha: current ``main`` HEAD SHA. When both ``base_sha`` and
        ``main_sha`` are present and differ, the gate fails closed as stale. When
        either is absent, staleness is reported as ``not_applicable`` (the merge
        queue constructs a fresh base, so this is the normal queue-time path).
      ci_overall: authoritative CI conclusion (``success`` / ``failure`` /
        ``pending`` / ``unknown``). When ``None``, falls back to
        ``pr["checks"]["overall"]``; when still empty, the CI dimension is
        treated as not-evaluated and does not block (the exact-head gate-verdict
        trailer is only posted after CI went green on that head, so its presence
        subsumes the CI-green-on-head requirement; the merge queue's required
        checks supply the CI authority for the queued merge).
      threads_resolved: ``True`` when all actionable review threads are resolved,
        ``False`` when at least one remains unresolved, ``None`` when not
        evaluated (does not block; the runtime CLI always supplies a definitive
        value and fails closed on a query error).

    Returns a ``MergeGateAudit`` with ``passed`` and a list of fail-closed
    ``reasons``. The audit always records the evaluated head SHA, base SHA, label
    set, gate-verdict status, staleness verdict, CI conclusion, and thread
    resolution so the decision is inspectable.
    """
    head_sha = str(pr.get("head_sha", "") or "")
    labels = _label_names(pr)
    draft = bool(pr.get("draft", False))
    merge_ready = "merge-ready" in labels
    base_sha = str(pr.get("base_sha", "") or "")

    if ci_overall is None:
        ci_overall = str((pr.get("checks") or {}).get("overall", "") or "")
    ci_overall = str(ci_overall).lower()

    gate_verdict_status = _gate_verdict_status(pr, head_sha)

    if base_sha and main_sha:
        staleness_verdict = "fresh" if base_sha == main_sha else "stale"
    else:
        staleness_verdict = "not_applicable"

    if threads_resolved is True:
        thread_resolution = "resolved"
    elif threads_resolved is False:
        thread_resolution = "unresolved"
    else:
        thread_resolution = "not_applicable"

    reasons = _fail_closed_reasons(
        head_sha=head_sha,
        draft=draft,
        merge_ready=merge_ready,
        ci_overall=ci_overall,
        staleness_verdict=staleness_verdict,
        gate_verdict_status=gate_verdict_status,
        thread_resolution=thread_resolution,
    )

    passed = not reasons

    return MergeGateAudit(
        schema=AUDIT_SCHEMA,
        pr=_safe_int(pr.get("number")),
        head_sha=head_sha,
        base_sha=base_sha,
        main_sha=str(main_sha or ""),
        labels=labels,
        draft=draft,
        ci_overall=ci_overall or "unknown",
        gate_verdict_status=gate_verdict_status,
        staleness_verdict=staleness_verdict,
        thread_resolution=thread_resolution,
        merge_ready=merge_ready,
        passed=passed,
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


def _parse_json(stdout: str) -> tuple[Any, str | None]:
    """Parse JSON stdout into a Python object or return an error string."""
    try:
        return json.loads(stdout), None
    except json.JSONDecodeError as exc:
        return None, f"Failed to parse JSON: {exc}"


def _rollup_overall(rollup: list[dict[str, Any]]) -> str:
    """Classify a PR ``statusCheckRollup`` into an overall CI conclusion.

    Returns ``failure`` if any check failed, ``pending`` if any check is
    in-progress/queued (or the rollup is empty), otherwise ``success``. Mirrors
    the classification in ``scripts/dev/check_pr_ci_status.py``.
    """
    if not isinstance(rollup, list) or not rollup:
        return "pending"
    for check in rollup:
        if not isinstance(check, dict):
            continue
        conclusion = str(check.get("conclusion") or check.get("state") or "").lower()
        status = str(check.get("status") or check.get("state") or "").lower()
        if conclusion in FAILURE_CONCLUSIONS:
            return "failure"
        if status in PENDING_STATUSES:
            return "pending"
        if conclusion in PENDING_STATUSES:
            return "pending"
    return "success"


def _resolve_owner_repo(explicit: str) -> str | None:
    """Resolve the ``owner/repo`` identifier, auto-detecting from gh when empty."""
    if explicit:
        return explicit
    result = _gh(["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"])
    if result.returncode != 0:
        return None
    repo = result.stdout.strip()
    return repo if repo else None


def _source_branch_from_merge_group(head_ref: str) -> str | None:
    """Extract the source PR branch from a merge-queue readonly ref.

    The native merge-queue ref has the shape
    ``refs/heads/gh-readonly-queue/<base>/<srcbranch>-<abbrevsha>``. The source
    branch may itself contain hyphens, so only a trailing 7-40 hex suffix is
    stripped. Returns ``None`` when the ref does not match the queue shape.
    """
    if not head_ref:
        return None
    match = _MERGE_QUEUE_REF_RE.match(head_ref)
    if not match:
        return None
    rest = match.group("rest")
    return _QUEUE_REF_SHA_SUFFIX_RE.sub("", rest, count=1)


def resolve_pr_from_merge_group(event: dict[str, Any], *, repo: str) -> int | None:
    """Resolve the source PR number for a ``merge_group`` event.

    Uses the readonly-queue ref to recover the source branch, then lists open PRs
    on that head targeting ``main``. Returns the PR number, or ``None`` when no
    unique source PR is found (the caller fails closed).
    """
    merge_group = event.get("merge_group") or {}
    head_ref = str(merge_group.get("head_ref") or "")
    source_branch = _source_branch_from_merge_group(head_ref)
    if not source_branch:
        return None
    result = _gh(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--head",
            source_branch,
            "--state",
            "open",
            "--json",
            "number",
            "--limit",
            "5",
        ]
    )
    if result.returncode != 0:
        return None
    payload, err = _parse_json(result.stdout)
    if err or not isinstance(payload, list) or not payload:
        return None
    numbers = [p.get("number") for p in payload if isinstance(p, dict) and p.get("number")]
    if len(numbers) != 1:
        return None
    return _safe_int(numbers[0])


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

    ``has_current_accepted_gate_verdict`` reads trailers from
    ``comment_snapshot.latest[].body_excerpt`` / ``review_snapshot.latest[]``.
    The raw ``gh pr view`` objects expose ``body``; we truncate to the same
    ``COMMENT_BODY_LIMIT`` (180) used by ``snapshot_pr_queue`` so a full
    ``gate-verdict: accepted @ <40-char-sha>`` trailer always survives.
    """
    if not isinstance(items, list):
        return {"latest": []}
    latest: list[dict[str, str]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        body = str(entry.get("body") or "")
        if body:
            latest.append({"body_excerpt": body[:limit]})
    return {"latest": latest}


def fetch_pr_snapshot(pr_number: str | int, *, repo: str) -> tuple[dict[str, Any], str | None]:
    """Fetch a compact PR snapshot via ``gh pr view`` for gate evaluation.

    Returns ``(snapshot, error)``. The snapshot carries the fields consumed by
    ``evaluate_merge_gate`` plus the compact comment/review body excerpts that
    carry gate-verdict trailers (in the shape
    ``has_current_accepted_gate_verdict`` reads).
    """
    result = _gh(
        [
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "number,isDraft,headRefOid,baseRefOid,labels,statusCheckRollup,comments,reviews",
        ]
    )
    if result.returncode != 0:
        return {}, result.stderr.strip() or f"gh pr view failed (exit {result.returncode})"
    payload, err = _parse_json(result.stdout)
    if err or not isinstance(payload, dict):
        return {}, err or "gh pr view output is not a JSON object"

    snapshot: dict[str, Any] = {
        "number": payload.get("number"),
        "draft": bool(payload.get("isDraft")),
        "head_sha": str(payload.get("headRefOid") or ""),
        "base_sha": str(payload.get("baseRefOid") or ""),
        "labels": _normalize_labels(payload.get("labels")),
        "checks": {"overall": _rollup_overall(payload.get("statusCheckRollup") or [])},
        "review_snapshot": _to_body_snapshot(payload.get("reviews")),
        "comment_snapshot": _to_body_snapshot(payload.get("comments")),
    }
    return snapshot, None


def fetch_main_sha(*, repo: str) -> str:
    """Return current ``main`` HEAD SHA, or an empty string on failure."""
    result = _gh(["api", f"repos/{repo}/git/refs/heads/main", "--jq", ".object.sha"])
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def fetch_threads_resolved(pr_number: str | int, *, repo: str) -> tuple[bool | None, str | None]:
    """Return ``(all_resolved, error)`` for a PR's review threads.

    Queries review threads via GraphQL. Returns ``True`` when there are no
    unresolved, non-outdated (actionable) threads; ``False`` when at least one
    remains; ``(None, error)`` when the query fails (caller fails closed).
    """
    owner, _, name = repo.partition("/")
    if not owner or not name:
        return None, f"invalid repo identifier: {repo!r}"
    query = (
        "query($owner:String!,$name:String!,$number:Int!){"
        "repository(owner:$owner,name:$name){pullRequest(number:$number){"
        "reviewThreads(first:100){nodes{isResolved isOutdated}}}}}"
    )
    result = _gh(
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
    if result.returncode != 0:
        return None, result.stderr.strip() or "graphql reviewThreads query failed"
    payload, err = _parse_json(result.stdout)
    if err or not isinstance(payload, dict):
        return None, err or "graphql response is not JSON"
    threads = (
        payload.get("data", {})
        .get("repository", {})
        .get("pullRequest", {})
        .get("reviewThreads", {})
    )
    nodes = threads.get("nodes") if isinstance(threads, dict) else None
    if not isinstance(nodes, list):
        return None, "reviewThreads.nodes missing from graphql response"
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if not node.get("isResolved") and not node.get("isOutdated"):
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
        f"- base SHA: `{audit.base_sha or '?'}`",
        f"- main SHA: `{audit.main_sha or 'n/a'}`",
        f"- labels: `{', '.join(audit.labels) if audit.labels else '(none)'}`",
        f"- draft: `{audit.draft}`",
        f"- merge-ready: `{audit.merge_ready}`",
        f"- gate-verdict status: `{audit.gate_verdict_status}`",
        f"- staleness verdict: `{audit.staleness_verdict}`",
        f"- CI conclusion: `{audit.ci_overall}`",
        f"- thread resolution: `{audit.thread_resolution}`",
    ]
    if audit.reasons:
        lines.append(f"- fail-closed reasons: `{', '.join(audit.reasons)}`")
    lines.append("")
    lines.append(
        "Gate contract: non-draft + `merge-ready` + current exact-head "
        "`gate-verdict: accepted` trailer + resolved threads; fail-closed on any "
        "missing dimension. See `docs/dev_guide.md` and "
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
) -> tuple[MergeGateAudit, str | None]:
    """Fetch live PR state, evaluate the gate, and return ``(audit, error)``."""
    snapshot, err = fetch_pr_snapshot(pr_number, repo=repo)
    if err:
        return (
            evaluate_merge_gate(
                {"number": pr_number, "head_sha": ""},
                main_sha="",
                ci_overall="unknown",
                threads_resolved=None,
            ),
            err,
        )

    if merge_group_base_sha:
        # Inside the merge queue the base SHA is the prospective current main, so
        # staleness is fresh by construction; record the queue base as both base
        # and main so the audit reflects that guarantee.
        snapshot["base_sha"] = merge_group_base_sha
        main_sha = merge_group_base_sha
    else:
        main_sha = fetch_main_sha(repo=repo)

    threads_resolved, thread_err = fetch_threads_resolved(pr_number, repo=repo)
    if thread_err:
        # Fail closed when the thread query fails: the audit records the reason.
        return (
            evaluate_merge_gate(
                snapshot,
                main_sha=main_sha,
                threads_resolved=False,
            ),
            f"thread resolution query failed: {thread_err}",
        )

    audit = evaluate_merge_gate(
        snapshot,
        main_sha=main_sha,
        threads_resolved=threads_resolved,
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
    ) -> dict[str, Any]:
        pr: dict[str, Any] = {
            "number": 6274,
            "head_sha": head_sha,
            "labels": list(labels),
            "draft": draft,
            "base_sha": base_sha,
        }
        if ci_overall is not None:
            pr["checks"] = {"overall": ci_overall}
        if gate_verdict_sha:
            pr["gate_verdict"] = {"verdict": "accepted", "sha": gate_verdict_sha}
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

    # Scenario 3: merge-ready + current exact-head gate-verdict -> pass.
    audit = evaluate_merge_gate(_pr(labels=["merge-ready"], gate_verdict_sha=full_sha))
    expect(audit.passed, "scenario3: merge-ready + current gate-verdict must pass")
    expect(
        audit.gate_verdict_status == "accepted",
        "scenario3: gate-verdict status must be accepted",
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

    # Draft -> fail (merge queue must never merge a draft).
    audit = evaluate_merge_gate(_pr(labels=["merge-ready"], gate_verdict_sha=full_sha, draft=True))
    expect(
        not audit.passed and "pr_is_draft" in audit.reasons,
        "draft: draft PR must fail closed",
    )

    # Full pass: all dimensions satisfied and explicitly authoritative.
    audit = evaluate_merge_gate(
        _pr(labels=["merge-ready"], gate_verdict_sha=full_sha, base_sha=full_sha),
        main_sha=full_sha,
        ci_overall="success",
        threads_resolved=True,
    )
    expect(audit.passed and audit.reasons == [], "full-pass: all dimensions green must pass")

    # Audit record must carry every required inspectable field.
    audit = evaluate_merge_gate(_pr(labels=["merge-ready"], gate_verdict_sha=full_sha))
    required_fields = {
        "schema",
        "pr",
        "head_sha",
        "base_sha",
        "main_sha",
        "labels",
        "draft",
        "ci_overall",
        "gate_verdict_status",
        "staleness_verdict",
        "thread_resolution",
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
    audit = evaluate_merge_gate(_pr(labels=["merge-ready"], gate_verdict_sha=full_sha[:12]))
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
            "comment_snapshot": {
                "latest": [{"body_excerpt": f"lgtm\n\ngate-verdict: accepted @ {full_sha}"}]
            },
        }
    )
    expect(audit.passed, "comment-carrier: gate-verdict trailer in a comment must satisfy gate")

    if failures:
        for message in failures:
            print(f"FAIL: {message}", file=sys.stderr)
        return 1
    print("merge_queue_gate self-test: all assertions passed")
    return 0


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
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    repo = _resolve_owner_repo(args.repo)
    if not repo:
        print("Failed to detect repository. Pass --repo owner/repo.", file=sys.stderr)
        return 1

    if args.from_event:
        try:
            event = json.loads(Path(args.from_event).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to read merge_group event payload: {exc}", file=sys.stderr)
            return 1
        if event.get("event_name") != "merge_group" and "merge_group" not in event:
            print(
                "Event payload is not a merge_group event; "
                "merge_queue_gate only gates the native merge queue.",
                file=sys.stderr,
            )
            return 1
        pr_number = resolve_pr_from_merge_group(event, repo=repo)
        if pr_number is None:
            print(
                "Could not resolve a unique source PR from the merge_group payload; "
                "failing closed.",
                file=sys.stderr,
            )
            return 1
        merge_group_base_sha = str((event.get("merge_group") or {}).get("base_sha") or "")
    else:
        pr_number = _safe_int(args.pr)
        if pr_number is None:
            print(f"Invalid PR number: {args.pr!r}", file=sys.stderr)
            return 1
        merge_group_base_sha = ""

    audit, error = _evaluate_live(
        pr_number,
        repo=repo,
        merge_group_base_sha=merge_group_base_sha,
    )

    _append_step_summary(_format_summary(audit))
    if not args.summary_only:
        print(json.dumps(audit.to_dict()))
    if error:
        print(f"gate evaluation warning: {error}", file=sys.stderr)

    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
