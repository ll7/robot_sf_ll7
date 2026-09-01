#!/usr/bin/env python3
"""Plan, render, apply, and verify goal-autopilot preparation packets.

This helper consumes the report-only ``open_issue_contract_audit.v1`` output from
:mod:`scripts.dev.audit_open_issue_contracts` and emits per-issue
``goal-autopilot`` preparation packets for every open issue. It is the apply
successor for issue #7929: it never reimplements canonical classification,
claim ownership, dependency resolution, blocker transitions, terminal-label
policy, or scientific admission.

Modes
=====

- plan (default, report-only): read an audit JSON, emit ``open_issue_preparation_plan.v1``
  with per-issue packets and aggregate counts. Zero writes.
- render: print the rendered ``goal-autopilot-preparation:v1`` marker block for one
  issue number (read from the audit).
- verify: check that every prepared issue body contains exactly one
  ``goal-autopilot-preparation:v1`` marker and that bytes outside the marker
  region are unchanged.
- apply (``--apply``): bounded, exact-item body and label mutations with a
  label-set compare-and-swap guard and a credential-free receipt. Requires an
  explicit reviewed plan digest and issue list; aborts the whole batch on any
  relevant drift.

The tool defaults to no-write mode and never creates arbitrary labels, never
adds PR runner labels to issues, and never mutates issue state, assignments,
milestones, projects, comments, parent relations, PRs, or merges.
The only readiness mutation is an explicit call to the canonical
``issue_readiness_gate.gate_issue`` operation; generic label writes cannot add
``state:ready``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.dev import issue_implementability, issue_readiness_gate

MARKER_START = "<!-- goal-autopilot-preparation:v1:start -->"
MARKER_END = "<!-- goal-autopilot-preparation:v1:end -->"
PACKET_SCHEMA = "goal_autopilot_preparation.v1"
PLAN_SCHEMA = "open_issue_preparation_plan.v1"
RECEIPT_SCHEMA = "open_issue_preparation_receipt.v1"

DEFAULT_MUTATION_CEILING = 10
HARD_BODY_CEILING = 25
HARD_LABEL_CEILING = 50
DEFAULT_REPOSITORY = "ll7/robot_sf_ll7"
_RUNNER_LABELS = frozenset({"runner:luna", "runner:max"})
_READINESS_GATE_ACTION = "gate_readiness"
_READINESS_GATE_SUCCESS_OUTCOMES = frozenset({"ready", "already_ready"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# LunaRunner: bounded docs/tests/config/CLI/adapter work with no planner,
# metric, model, safety, or evidence semantics. MaxRunner: anything touching
# planner/metric/model/evidence semantics or cross-module scope.
_LUNA_CLASSIFICATIONS = frozenset({"ready", "needs_ready_label", "needs_spec"})
_MAX_CLASSIFICATIONS = frozenset(
    {
        "parent",
        "human_decision",
        "needs_dependency",
        "needs_compute",
        "blocked",
        "wrong_owner_repo",
        "state_conflict",
        "stale_running",
        "assigned",
        "already_claimed",
        "working",
        "review",
        "closed",
        "error",
    }
)

_EXECUTION_MODE = {
    "ready": "implementation",
    "needs_ready_label": "implementation",
    "needs_spec": "formalization",
    "parent": "decomposition",
    "human_decision": "decision",
    "needs_dependency": "dependency",
    "needs_compute": "compute",
    "blocked": "blocker",
    "wrong_owner_repo": "ownership",
    "state_conflict": "lifecycle",
    "stale_running": "lifecycle",
    "assigned": "active-handoff",
    "already_claimed": "active-handoff",
    "working": "active-handoff",
    "review": "active-handoff",
    "closed": "stale-closure",
    "error": "error-repair",
}

# Labels that carry an authority that must never be overwritten by body prose.
_AUTHORITY_LABELS = frozenset(
    {
        "state:blocked",
        "state:parked",
        "state:hold",
        "state:blocked-external-input",
        "state:blocked-no-code-slice",
        "ruled",
        "needs-triage",
        "domain-review-required",
        "needs-campaign",
        "needs-research",
        "parent",
        "epic",
    }
)

# Marker region replacement is the only permitted body mutation. This regex
# captures a body with an existing marker block so apply can replace it.
_MARKER_BLOCK_RE = re.compile(re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END), re.DOTALL)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(payload: object) -> str:
    return _sha256_text(_stable_json(payload))


def _content_digest(plan: Mapping[str, Any]) -> str:
    """Compute the deterministic digest of a plan without its digest field."""
    digest_input = {key: value for key, value in plan.items() if key != "content_sha256"}
    return _sha256_json(digest_input)


def _normalize_label_names(value: Any) -> list[str]:
    """Normalize REST or fixture label values to sorted unique names."""
    if not isinstance(value, list):
        raise ValueError("labels must be a list")
    names: list[str] = []
    for row in value:
        name = row.get("name") if isinstance(row, Mapping) else row
        if not isinstance(name, str) or not name.strip():
            raise ValueError("labels must contain non-empty names")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("labels must not contain duplicate names")
    return sorted(names)


def _normalize_assignee_names(value: Any) -> list[str]:
    """Normalize REST or fixture assignee values to sorted unique logins."""
    if not isinstance(value, list):
        raise ValueError("assignees must be a list")
    names: list[str] = []
    for row in value:
        name = row.get("login") if isinstance(row, Mapping) else row
        if not isinstance(name, str) or not name.strip():
            raise ValueError("assignees must contain non-empty names")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("assignees must not contain duplicate names")
    return sorted(names)


def _live_repository_label_names(
    *,
    repo: str = DEFAULT_REPOSITORY,
    page_size: int = 100,
    page_ceiling: int = 10,
) -> set[str]:
    """Read the repository label catalog through the shared REST transport."""
    from scripts.dev import _gh_rest

    names: set[str] = set()
    for page in range(1, page_ceiling + 1):
        path = f"repos/{repo}/labels?per_page={page_size}&page={page}"
        result = _gh_rest.run_gh_api(path)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "REST read failed"
            raise RuntimeError(f"repository label catalog read failed on page {page}: {detail}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"repository label catalog page {page} was not valid JSON") from exc
        if not isinstance(payload, list):
            raise RuntimeError(
                f"repository label catalog page {page} was not a list: {type(payload).__name__}"
            )
        try:
            names.update(_normalize_label_names(payload))
        except ValueError as exc:
            raise RuntimeError(
                f"repository label catalog page {page} was malformed: {exc}"
            ) from exc
        if len(payload) < page_size:
            return names
    raise RuntimeError(f"repository label catalog exceeded the page ceiling of {page_ceiling}")


def _live_issue_reader(issue: int, *, repo: str = DEFAULT_REPOSITORY) -> dict[str, Any]:
    """Read and strictly normalize one issue before a bounded mutation."""
    from scripts.dev import _gh_rest

    result = _gh_rest.run_gh_api(f"repos/{repo}/issues/{issue}")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "REST read failed"
        raise RuntimeError(f"issue {issue} read failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"issue {issue} read was not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"issue {issue} read was not an object")
    try:
        labels = _normalize_label_names(payload.get("labels", []))
    except ValueError as exc:
        raise RuntimeError(f"issue {issue} labels were malformed: {exc}") from exc
    try:
        assignee_names = _normalize_assignee_names(payload.get("assignees", []))
    except ValueError as exc:
        raise RuntimeError(f"issue {issue} assignees were malformed: {exc}") from exc
    body = payload.get("body")
    if not isinstance(body, str):
        raise RuntimeError(f"issue {issue} body was not a string")
    return {
        "number": payload.get("number"),
        "state": str(payload.get("state") or "").lower(),
        "body": body,
        "body_sha256": _sha256_text(body),
        "labels": labels,
        "assignees": assignee_names,
        "updated_at": str(payload.get("updated_at") or ""),
    }


def _readiness_gate_candidate(  # noqa: C901, PLR0912 - explicit authority and leaf checks
    item: Mapping[str, Any],
) -> bool:
    """Return whether a complete local leaf may enter the canonical readiness gate.

    ``issue_implementability`` intentionally classifies an issue with no
    execution-state label as ``state_conflict``.  Preparation must recognize
    only the narrow, safe subset of that classification that is actually a
    complete local leaf; it must not turn missing facts, authority labels, or
    active claims into readiness.
    """
    if item.get("classification") != "state_conflict":
        return False
    if item.get("admission_reason") != "state_label_conflict":
        return False
    if item.get("applicable") is not True or item.get("dispatch_eligible") is True:
        return False
    if str(item.get("state") or "open").lower() != "open":
        return False
    raw_labels = item.get("labels") or []
    if not isinstance(raw_labels, list) or any(not isinstance(label, str) for label in raw_labels):
        return False
    labels = set(raw_labels)
    if "state:ready" in labels or any(label.startswith("state:") for label in labels):
        return False
    if labels & (
        _AUTHORITY_LABELS
        | issue_implementability.PARENT_LABELS
        | issue_implementability.HUMAN_DECISION_LABELS
        | issue_implementability.COMPUTE_LABELS
        | issue_implementability.EXTERNAL_LABELS
        | issue_implementability.BLOCKING_LABELS
    ) or any(label.startswith("blocked:") for label in labels):
        return False
    title = str(item.get("title") or "").strip().lower()
    if title.startswith("[parent]") or title.startswith("[epic]"):
        return False
    missing_fields = item.get("missing_fields")
    if not isinstance(missing_fields, list) or missing_fields:
        return False
    assignees = item.get("assignees")
    if not isinstance(assignees, list) or assignees:
        return False
    claim = item.get("claim")
    if not isinstance(claim, Mapping) or claim.get("ok") is not True:
        return False
    if claim.get("claimed") is not False:
        return False
    execution_contract = item.get("execution_contract")
    if not isinstance(execution_contract, Mapping):
        return False
    if execution_contract.get("valid") is not True:
        return False
    if execution_contract.get("route_required", "local") != "local":
        return False
    if execution_contract.get("external_inputs"):
        return False
    if execution_contract.get("owning_repo") != DEFAULT_REPOSITORY:
        return False
    mutation_repos = execution_contract.get("mutation_repos")
    if not isinstance(mutation_repos, list) or any(
        not isinstance(repo, str) for repo in mutation_repos
    ):
        return False
    if set(mutation_repos) != {DEFAULT_REPOSITORY}:
        return False
    body_sha256 = item.get("body_sha256")
    return isinstance(body_sha256, str) and _SHA256_RE.fullmatch(body_sha256) is not None


def _readiness_gate_operation(item: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build the explicit CAS inputs for one canonical readiness-gate call."""
    if not _readiness_gate_candidate(item):
        return None
    labels = _normalize_label_names(list(item.get("labels") or []))
    return {
        "issue": str(item.get("number")),
        "action": _READINESS_GATE_ACTION,
        "expected_body_sha256": item.get("body_sha256"),
        "expected_labels": labels,
    }


def _authority_preparation_action(item: Mapping[str, Any]) -> str | None:
    """Return a preparation action owned by an explicit issue authority."""
    raw_labels = item.get("labels")
    labels = (
        {label for label in raw_labels if isinstance(label, str)}
        if isinstance(raw_labels, list)
        else set()
    )
    title = str(item.get("title") or "").strip().lower()
    if labels & issue_implementability.PARENT_LABELS or title.startswith(("[parent]", "[epic]")):
        return "decompose_issue"
    if labels & issue_implementability.HUMAN_DECISION_LABELS:
        return "prepare_decision"
    if labels & issue_implementability.EXTERNAL_LABELS:
        return "reconcile_blockers"
    if labels & issue_implementability.COMPUTE_LABELS:
        return "route_to_compute"
    if labels & issue_implementability.BLOCKING_LABELS or any(
        label.startswith("blocked:") for label in labels
    ):
        return "reconcile_blockers"
    return None


def _preparation_action(item: Mapping[str, Any]) -> str:
    """Return the deterministic preparation action for one audit item."""
    if _readiness_gate_candidate(item):
        return _READINESS_GATE_ACTION
    classification = item.get("classification")
    if classification in {"error", "closed"}:
        return str(item.get("next_action") or "")
    authority_action = _authority_preparation_action(item)
    if authority_action is not None:
        return authority_action
    if classification in {"assigned", "already_claimed", "working", "review", "stale_running"}:
        return "active_handoff"
    missing_fields = item.get("missing_fields")
    if isinstance(missing_fields, list) and missing_fields:
        return "formalize_issue"
    action_by_classification = {
        "needs_spec": "formalize_issue",
        "parent": "decompose_issue",
        "human_decision": "prepare_decision",
        "blocked": "reconcile_blockers",
    }
    if classification in action_by_classification:
        return action_by_classification[classification]
    return str(item.get("next_action") or "")


def _execution_mode(item: Mapping[str, Any]) -> str:
    """Return the execution mode, including the explicit readiness-gate lane."""
    action = _preparation_action(item)
    mode_by_action = {
        _READINESS_GATE_ACTION: _READINESS_GATE_ACTION,
        "formalize_issue": "formalization",
        "decompose_issue": "decomposition",
        "prepare_decision": "decision",
        "reconcile_blockers": "blocker",
        "route_to_compute": "compute",
        "active_handoff": "active-handoff",
    }
    if action in mode_by_action:
        return mode_by_action[action]
    return _EXECUTION_MODE.get(item.get("classification", "error"), "error-repair")


def _worker_route(item: Mapping[str, Any]) -> str:
    """Return the worker route for one audit item."""
    if _preparation_action(item) in {_READINESS_GATE_ACTION, "formalize_issue"}:
        return "LunaRunner"
    classification = item.get("classification")
    if classification in _LUNA_CLASSIFICATIONS:
        return "LunaRunner"
    if classification in _MAX_CLASSIFICATIONS:
        return "MaxRunner"
    return "none"


def _render_envelope(
    item: Mapping[str, Any], *, audit_digest: str, batch_id: str
) -> dict[str, Any]:
    """Build the machine-readable envelope for one issue packet."""
    classification = item.get("classification", "error")
    source_body_sha = item.get("body_sha256")
    return {
        "schema": PACKET_SCHEMA,
        "repository": "ll7/robot_sf_ll7",
        "issue": item.get("number"),
        "source_body_sha256": source_body_sha or "",
        "source_comments_sha256": "",
        "audit_schema": "open_issue_contract_audit.v1",
        "audit_digest": audit_digest,
        "audit_classification": classification,
        "next_action": _preparation_action(item),
        "authority": item.get("authority", ""),
        "execution_mode": _execution_mode(item),
        "preferred_worker": _worker_route(item),
        "expected_pr_runner_label": _pr_runner_label(classification),
        "implementation_admitted": bool(item.get("dispatch_eligible")),
        "state_ready_change_proposed": _state_ready_proposed(item),
        "readiness_gate": _readiness_gate_operation(item),
        "mutation_batch": batch_id,
    }


def _pr_runner_label(classification: str) -> str:
    """Map one classification to its expected PR runner label (issues never get it)."""
    return "runner:max" if classification in _MAX_CLASSIFICATIONS else "runner:luna"


def _state_ready_proposed(item: Mapping[str, Any]) -> bool:
    """Return whether the packet proposes a reviewed state:ready transition."""
    return _readiness_gate_operation(item) is not None


def _render_marker_block(item: Mapping[str, Any], *, audit_digest: str, batch_id: str) -> str:
    """Render the full packet marker block for one issue."""
    envelope = _render_envelope(item, audit_digest=audit_digest, batch_id=batch_id)
    body = ["<!-- goal-autopilot-preparation:v1:start -->", ""]
    body.append("```yaml")
    for key, value in envelope.items():
        body.append(f"{key}: {value}")
    body.append("```")
    body.append("")
    body.append(
        "This packet is preparation evidence only. It never overrides live labels, "
        "exact claim state, branch state, typed dependencies, domain gates, compute "
        "authority, release authority, or scientific evidence rules."
    )
    body.append("<!-- goal-autopilot-preparation:v1:end -->")
    return "\n".join(body) + "\n"


def _label_plan(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact reviewed label plan, including canonical gate operations."""
    labels = set(item.get("labels") or [])
    plan: list[dict[str, Any]] = []
    gate_operation = _readiness_gate_operation(item)
    if gate_operation is not None:
        plan.append(gate_operation)
    if "state:ready" in labels and item.get("classification") not in ("ready", "needs_ready_label"):
        plan.append({"issue": str(item.get("number")), "action": "remove", "label": "state:ready"})
    return plan


def _body_patch_proposal(item: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the proposed body patch for one item (no actual body here)."""
    number = item.get("number")
    return {
        "issue": number,
        "proposed": bool(item.get("applicable")),
        "reason": "add goal-autopilot-preparation packet marker",
        "expected_digest_after": None,
        "marker_count_after": 1,
    }


def build_plan(audit: Mapping[str, Any], *, batch_id: str) -> dict[str, Any]:
    """Build an ``open_issue_preparation_plan.v1`` from a complete audit report."""
    items = audit.get("items")
    if not isinstance(items, list):
        raise ValueError("audit report has no items list")
    audit_digest = audit.get("content_sha256") or _sha256_json(audit)
    entries: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("audit item is not an object")
        number = item.get("number")
        entries.append(
            {
                "issue": number,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "labels": item.get("labels", []),
                "assignees": item.get("assignees", []),
                "state": str(item.get("state") or "open").lower(),
                "body_sha256": item.get("body_sha256"),
                "claim_state": item.get("claim"),
                "classification_before": item.get("observed_classification"),
                "classification_after": item.get("classification"),
                "admission_reason": item.get("admission_reason"),
                "preparation_action": _preparation_action(item),
                "execution_mode": _execution_mode(item),
                "worker_route": _worker_route(item),
                "next_action": _preparation_action(item),
                "authority": item.get("authority", ""),
                "dispatch_eligible": bool(item.get("dispatch_eligible")),
                "state_ready_change_proposed": _state_ready_proposed(item),
                "readiness_gate": _readiness_gate_operation(item),
                "body_patch": _body_patch_proposal(item),
                "label_plan": _label_plan(item),
                "skip_reason": _skip_reason(item),
            }
        )
    counts = Counter(str(entry.get("classification_before") or "error") for entry in entries)
    admission_reasons = Counter(
        str(entry.get("admission_reason") or "unknown") for entry in entries
    )
    route_counts = Counter(str(entry.get("worker_route") or "none") for entry in entries)
    plan = {
        "schema": PLAN_SCHEMA,
        "repository": audit.get("repository", "ll7/robot_sf_ll7"),
        "base_sha": audit.get("base_sha"),
        "audit_schema": audit.get("schema", "open_issue_contract_audit.v1"),
        "audit_digest": audit_digest,
        "listing_complete": bool(audit.get("complete")),
        "pagination": audit.get("pagination"),
        "batch_id": batch_id,
        "mutation_authorized": False,
        "item_count": len(entries),
        "entries": entries,
        "summary": {
            "by_classification_before": dict(counts),
            "admission_reason_histogram": dict(sorted(admission_reasons.items())),
            "not_admitted": dict(
                sorted(
                    (reason, count)
                    for reason, count in admission_reasons.items()
                    if reason != "claimable"
                )
            ),
            "by_worker_route": dict(route_counts),
            "ready_items": sum(1 for e in entries if e["classification_before"] == "ready"),
            "dispatch_eligible": sum(1 for e in entries if e["dispatch_eligible"]),
            "promotable_count": sum(1 for e in entries if e["readiness_gate"] is not None),
            "formalizable_count": sum(
                1 for e in entries if e["preparation_action"] == "formalize_issue"
            ),
            "label_operations": sum(len(e["label_plan"]) for e in entries),
        },
    }
    plan["content_sha256"] = _content_digest(plan)
    return plan


def _skip_reason(item: Mapping[str, Any]) -> str:
    """Return the skip reason for items that must not be mutated."""
    classification = item.get("classification")
    if item.get("listing_drift"):
        return "listing_drift"
    if classification in ("assigned", "already_claimed", "working", "review", "stale_running"):
        return "active_owner"
    if classification in ("closed",):
        return "closed"
    if classification == "error":
        return "error_row"
    raw_labels = item.get("labels")
    labels = (
        {label for label in raw_labels if isinstance(label, str)}
        if isinstance(raw_labels, list)
        else set()
    )
    title = str(item.get("title") or "").strip().lower()
    if (
        classification in ("parent", "human_decision")
        or labels & issue_implementability.PARENT_LABELS
        or labels & issue_implementability.HUMAN_DECISION_LABELS
        or title.startswith(("[parent]", "[epic]"))
    ):
        return "authority_held"
    return ""


def _load_audit(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audit JSON must be an object")
    if payload.get("schema") != "open_issue_contract_audit.v1":
        raise ValueError(f"unexpected audit schema: {payload.get('schema')!r}")
    if payload.get("mutation_authorized") not in (False, None):
        raise ValueError("audit must be report-only (mutation_authorized false)")
    return payload


def _render_plan_markdown(plan: Mapping[str, Any]) -> str:
    """Render a compact markdown summary of the plan."""
    summary = plan.get("summary", {})
    lines = [
        f"# Open-issue preparation plan ({plan.get('schema')})",
        "",
        f"- Repository: `{plan.get('repository')}`",
        f"- Audit digest: `{plan.get('audit_digest')}`",
        f"- Items: {plan.get('item_count')}",
        f"- Dispatch-eligible: {summary.get('dispatch_eligible')}",
        f"- Label operations: {summary.get('label_operations')}",
        f"- Admission reasons: {summary.get('admission_reason_histogram')}",
        f"- By worker: {summary.get('by_worker_route')}",
        "",
        "## Per-issue packets",
        "",
    ]
    for entry in plan.get("entries", []):
        lines.append(
            f"- #{entry['issue']} [{entry['classification_before']} -> "
            f"{entry['classification_after']}] {entry['execution_mode']} / "
            f"{entry['worker_route']} | {entry['admission_reason']} | {entry['next_action']}"
        )
    return "\n".join(lines) + "\n"


def _verify_batch(plan: Mapping[str, Any], bodies: Mapping[str, str]) -> list[dict[str, Any]]:
    """Verify marker uniqueness and byte preservation for a batch of bodies."""
    findings: list[dict[str, Any]] = []
    for entry in plan.get("entries", []):
        issue = str(entry.get("issue"))
        original_sha = entry.get("body_sha256")
        body = bodies.get(issue)
        if body is None:
            continue
        markers = len(_MARKER_BLOCK_RE.findall(body))
        if markers > 1:
            findings.append({"issue": issue, "ok": False, "reason": "duplicate marker"})
            continue
        stripped = _MARKER_BLOCK_RE.sub("", body)
        # The apply path concatenates the original body with "\n\n" before the
        # marker block; normalize the resulting boundary blank lines before
        # comparing against the source digest.
        normalized = re.sub(r"\n{3,}", "\n\n", stripped).strip("\n") + "\n"
        candidates = (stripped, stripped.rstrip("\n"), normalized)
        if original_sha and not any(
            _sha256_text(candidate) == original_sha for candidate in candidates
        ):
            findings.append({"issue": issue, "ok": False, "reason": "content drift outside marker"})
            continue
        findings.append({"issue": issue, "ok": True, "reason": ""})
    return findings


def _compose_body(body: str, block: str) -> str:
    """Return the body with exactly one marker block and no other edits."""
    if _MARKER_BLOCK_RE.search(body):
        return _MARKER_BLOCK_RE.sub(block.rstrip("\n"), body)
    if body.endswith("\n"):
        return body + "\n" + block.rstrip("\n")
    return body + block.rstrip("\n")


def _normalize_issue_snapshot(snapshot: Mapping[str, Any], issue: int) -> dict[str, Any]:
    """Normalize an injected or REST issue snapshot for CAS comparisons."""
    raw_number = snapshot.get("number")
    if raw_number is None:
        raise ValueError(f"issue identity missing for {issue}")
    try:
        if isinstance(raw_number, bool) or int(raw_number) != issue:
            raise ValueError(f"issue identity mismatch: expected {issue}, got {raw_number}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"issue identity mismatch for {issue}: {raw_number!r}") from exc
    body = snapshot.get("body")
    if not isinstance(body, str):
        raise ValueError(f"issue {issue} body was not a string")
    try:
        labels = _normalize_label_names(snapshot.get("labels", []))
    except ValueError as exc:
        raise ValueError(f"issue {issue} labels were malformed: {exc}") from exc
    try:
        assignees = _normalize_assignee_names(snapshot.get("assignees", []))
    except ValueError as exc:
        raise ValueError(f"issue {issue} assignees were malformed: {exc}") from exc
    return {
        "number": issue,
        "state": str(snapshot.get("state") or "open").lower(),
        "body": body,
        "body_sha256": _sha256_text(body),
        "labels": labels,
        "assignees": assignees,
        "updated_at": str(snapshot.get("updated_at") or ""),
    }


def _validate_one_label_operation(  # noqa: C901 - validate every mutation branch
    operation: object,
    *,
    issue: object,
    issue_number: int | None,
    label_catalog: set[str] | None,
    seen: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Return all validation errors for one reviewed label operation."""
    if not isinstance(operation, Mapping):
        return [{"issue": issue, "operation": "failed", "reason": "label_plan_item_not_object"}]
    errors: list[dict[str, Any]] = []
    operation_issue = operation.get("issue")
    try:
        if (
            issue_number is None
            or isinstance(operation_issue, bool)
            or int(operation_issue) != issue_number
        ):
            raise ValueError(f"operation issue {operation_issue!r} does not match entry {issue!r}")
    except (TypeError, ValueError) as exc:
        errors.append(
            {
                "issue": issue,
                "operation": "failed",
                "reason": f"label_operation_issue_mismatch:{exc}",
            }
        )
    action = operation.get("action")
    if action == _READINESS_GATE_ACTION:
        expected_body_sha256 = operation.get("expected_body_sha256")
        if (
            not isinstance(expected_body_sha256, str)
            or _SHA256_RE.fullmatch(expected_body_sha256) is None
        ):
            errors.append(
                {
                    "issue": issue,
                    "operation": "failed",
                    "action": action,
                    "reason": "expected_body_sha256_invalid",
                }
            )
        expected_labels = operation.get("expected_labels")
        try:
            normalized_expected_labels = _normalize_label_names(expected_labels)
        except ValueError as exc:
            errors.append(
                {
                    "issue": issue,
                    "operation": "failed",
                    "action": action,
                    "reason": f"expected_labels_invalid:{exc}",
                }
            )
        else:
            if "state:ready" in normalized_expected_labels:
                errors.append(
                    {
                        "issue": issue,
                        "operation": "failed",
                        "action": action,
                        "reason": "readiness_gate_expected_labels_already_ready",
                    }
                )
        key = (str(action), str(expected_body_sha256))
        if key in seen:
            errors.append(
                {
                    "issue": issue,
                    "operation": "failed",
                    "action": action,
                    "reason": "duplicate_label_operation",
                }
            )
        seen.add(key)
        return errors
    label = operation.get("label")
    if action not in {"add", "remove"}:
        errors.append(
            {
                "issue": issue,
                "operation": "failed",
                "reason": f"unsupported_label_action:{action}",
            }
        )
        return errors
    if not isinstance(label, str) or not label.strip():
        errors.append({"issue": issue, "operation": "failed", "reason": "label_name_not_nonempty"})
        return errors
    if label != label.strip():
        errors.append(
            {
                "issue": issue,
                "operation": "failed",
                "action": action,
                "label": label,
                "reason": "label_name_has_surrounding_whitespace",
            }
        )
        return errors
    if label in _RUNNER_LABELS:
        errors.append(
            {
                "issue": issue,
                "operation": "failed",
                "action": action,
                "label": label,
                "reason": "runner_label_forbidden",
            }
        )
    if label_catalog is not None and label not in label_catalog:
        errors.append(
            {
                "issue": issue,
                "operation": "failed",
                "action": action,
                "label": label,
                "reason": "label_not_in_repository_catalog",
            }
        )
    key = (str(action), label)
    if key in seen:
        errors.append(
            {
                "issue": issue,
                "operation": "failed",
                "action": action,
                "label": label,
                "reason": "duplicate_label_operation",
            }
        )
    seen.add(key)
    return errors


def _validate_label_entry(  # noqa: C901 - aggregate exact plan errors before writes
    entry: Mapping[str, Any],
    *,
    label_catalog: set[str] | None,
) -> tuple[int, list[dict[str, Any]]]:
    """Validate one plan entry's expected labels and label operations."""
    issue = entry.get("issue")
    try:
        if isinstance(issue, bool):
            raise ValueError("boolean issue number")
        issue_number = int(issue)
        if issue_number < 1:
            raise ValueError("issue number must be positive")
    except (TypeError, ValueError) as exc:
        issue_number = None
        issue_error = {
            "issue": issue,
            "operation": "failed",
            "reason": f"invalid_issue:{exc}",
        }
    else:
        issue_error = None
    errors: list[dict[str, Any]] = []
    if issue_error is not None:
        errors.append(issue_error)
    try:
        expected_labels = _normalize_label_names(entry.get("labels", []))
    except ValueError as exc:
        expected_labels = []
        errors.append(
            {
                "issue": issue,
                "operation": "failed",
                "reason": f"expected_labels_invalid:{exc}",
            }
        )
    operations = entry.get("label_plan", [])
    if not isinstance(operations, list):
        errors.append({"issue": issue, "operation": "failed", "reason": "label_plan_not_list"})
        return 0, errors
    seen: set[tuple[str, str]] = set()
    for operation in operations:
        errors.extend(
            _validate_one_label_operation(
                operation,
                issue=issue,
                issue_number=issue_number,
                label_catalog=label_catalog,
                seen=seen,
            )
        )
        if isinstance(operation, Mapping) and operation.get("action") == _READINESS_GATE_ACTION:
            if operation.get("expected_body_sha256") != entry.get("body_sha256"):
                errors.append(
                    {
                        "issue": issue,
                        "operation": "failed",
                        "action": _READINESS_GATE_ACTION,
                        "reason": "readiness_gate_body_digest_does_not_match_entry",
                    }
                )
            try:
                operation_labels = _normalize_label_names(operation.get("expected_labels"))
            except ValueError:
                operation_labels = None
            if operation_labels is not None and operation_labels != expected_labels:
                errors.append(
                    {
                        "issue": issue,
                        "operation": "failed",
                        "action": _READINESS_GATE_ACTION,
                        "reason": "readiness_gate_labels_do_not_match_entry",
                    }
                )
    return len(operations), errors


def _validate_label_operations(
    entries: Sequence[Mapping[str, Any]],
    *,
    label_catalog: set[str] | None,
) -> tuple[int, list[dict[str, Any]]]:
    """Validate exact reviewed label plans before any body or label write."""
    planned = 0
    errors: list[dict[str, Any]] = []
    for entry in entries:
        entry_planned, entry_errors = _validate_label_entry(
            entry,
            label_catalog=label_catalog,
        )
        planned += entry_planned
        errors.extend(entry_errors)
    return planned, errors


def _base_receipt(
    *,
    audit: Mapping[str, Any],
    plan: Mapping[str, Any],
    mutation_ceiling: int,
    batch_id: str,
    dry_run: bool,
    issues: Sequence[int],
) -> dict[str, Any]:
    """Build the stable receipt fields shared by successful and failed batches."""
    return {
        "schema": RECEIPT_SCHEMA,
        "repository": plan.get("repository", audit.get("repository", DEFAULT_REPOSITORY)),
        "base_sha": audit.get("base_sha"),
        "audit_digest": plan.get("audit_digest", ""),
        "plan_digest": plan.get("content_sha256", ""),
        "issues": sorted(issues),
        "batch_id": batch_id,
        "mutation_ceiling": mutation_ceiling,
        "hard_body_ceiling": HARD_BODY_CEILING,
        "hard_label_ceiling": HARD_LABEL_CEILING,
        "dry_run": dry_run,
        "safe_order": "body_then_labels",
        "transactional": False,
        "partial_failure": False,
        "aborted": False,
        "abort_reason": None,
        "unauthorized_mutations": False,
        "operations": [],
        "written": 0,
        "would_write": 0,
        "idempotent": 0,
        "skipped": 0,
        "drifted": 0,
        "failed": 0,
        "label_operations_planned": 0,
        "label_operations_attempted": 0,
    }


def _entry_is_actionable(entry: Mapping[str, Any]) -> bool:
    """Return whether a plan entry requests a body or label operation."""
    return not entry.get("skip_reason") and (
        bool(entry.get("body_patch", {}).get("proposed")) or bool(entry.get("label_plan"))
    )


def _is_readiness_gate_operation(operation: object) -> bool:
    """Return whether a reviewed operation delegates readiness to the canonical gate."""
    return isinstance(operation, Mapping) and operation.get("action") == _READINESS_GATE_ACTION


def _has_generic_label_operations(entries: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether the batch needs the repository label catalog."""
    return any(
        any(
            not _is_readiness_gate_operation(operation) for operation in entry.get("label_plan", [])
        )
        for entry in entries
    )


def _apply_bodies(  # noqa: C901, PLR0912, PLR0913, PLR0915 - explicit fail-closed batch state machine
    audit: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    mutation_ceiling: int,
    batch_id: str,
    dry_run: bool,
    body_writer: Callable[[int, str], object],
    issue_reader: Callable[[int], Mapping[str, Any]] | None = None,
    label_catalog: set[str] | None = None,
    label_writer: Callable[[int, str, str], Mapping[str, Any]] | None = None,
    readiness_gater: Callable[[int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply a reviewed body/label batch with fail-closed CAS and readiness guards.

    ``body_writer``, ``issue_reader``, and ``label_writer`` are injectable so
    the complete mutation protocol can be tested offline.  Readiness operations
    use ``readiness_gater`` and never call the generic label writer.  The live
    path uses the shared REST transport and :mod:`gh_pr_label_rest`.
    """
    entries = [entry for entry in plan.get("entries", []) if isinstance(entry, Mapping)]
    issues = [int(entry["issue"]) for entry in entries if str(entry.get("issue", "")).isdigit()]
    receipt = _base_receipt(
        audit=audit,
        plan=plan,
        mutation_ceiling=mutation_ceiling,
        batch_id=batch_id,
        dry_run=dry_run,
        issues=issues,
    )
    if any(
        _is_readiness_gate_operation(operation)
        for entry in entries
        for operation in entry.get("label_plan", [])
    ):
        receipt["safe_order"] = "readiness_gate_then_body_then_labels"
    operations: list[dict[str, Any]] = receipt["operations"]

    def add_operation(operation: dict[str, Any]) -> None:
        """Append an operation and update the stable status counters."""
        operations.append(operation)
        status = operation.get("operation")
        if status == "written":
            receipt["written"] += 1
        elif status == "would_write":
            receipt["would_write"] += 1
        elif status == "idempotent":
            receipt["idempotent"] += 1
        elif status in {"skip", "skipped"}:
            receipt["skipped"] += 1
        elif status == "drifted":
            receipt["drifted"] += 1
        elif status == "failed":
            receipt["failed"] += 1

    if len(issues) != len(set(issues)):
        receipt["aborted"] = True
        receipt["abort_reason"] = "duplicate_issue_entries"
        add_operation(
            {
                "kind": "batch",
                "operation": "failed",
                "reason": "plan contains duplicate issue entries",
            }
        )
        return receipt
    if mutation_ceiling < 0 or mutation_ceiling > HARD_BODY_CEILING:
        receipt["aborted"] = True
        receipt["abort_reason"] = "body_mutation_ceiling_out_of_range"
        add_operation(
            {
                "kind": "batch",
                "operation": "failed",
                "reason": f"mutation ceiling must be between 0 and {HARD_BODY_CEILING}",
            }
        )
        return receipt

    planned_labels, label_errors = _validate_label_operations(
        entries,
        label_catalog=label_catalog,
    )
    receipt["label_operations_planned"] = planned_labels
    if _has_generic_label_operations(entries) and label_catalog is None:
        receipt["aborted"] = True
        receipt["abort_reason"] = "label_catalog_unavailable"
        add_operation(
            {
                "kind": "batch",
                "operation": "failed",
                "reason": "label catalog is required for label mutations",
            }
        )
        return receipt
    if planned_labels > HARD_LABEL_CEILING:
        receipt["aborted"] = True
        receipt["abort_reason"] = "label_operation_ceiling_exceeded"
        add_operation(
            {
                "kind": "batch",
                "operation": "failed",
                "reason": f"planned label operations exceed hard max {HARD_LABEL_CEILING}",
                "planned": planned_labels,
            }
        )
        return receipt
    if label_errors:
        receipt["aborted"] = True
        receipt["abort_reason"] = "invalid_label_plan"
        for error in label_errors:
            add_operation(dict(error))
        return receipt

    actionable_entries = [
        entry
        for entry in entries
        if not entry.get("skip_reason")
        and (bool(entry.get("body_patch", {}).get("proposed")) or bool(entry.get("label_plan")))
    ]

    snapshots: dict[int, dict[str, Any]] = {}
    if issue_reader is not None and not dry_run:
        preflight_errors: list[dict[str, Any]] = []
        for entry in actionable_entries:
            issue = int(entry["issue"])
            try:
                raw_snapshot = issue_reader(issue)
                if not isinstance(raw_snapshot, Mapping):
                    raise ValueError("issue snapshot was not an object")
                snapshot = _normalize_issue_snapshot(raw_snapshot, issue)
                expected_labels = _normalize_label_names(entry.get("labels", []))
                expected_assignees = _normalize_assignee_names(entry.get("assignees", []))
                expected_state = str(entry.get("state") or "open").lower()
                mismatches: dict[str, Any] = {}
                if snapshot["state"] != expected_state:
                    mismatches["state"] = {
                        "expected": expected_state,
                        "observed": snapshot["state"],
                    }
                if snapshot["labels"] != expected_labels:
                    mismatches["labels"] = {
                        "expected": expected_labels,
                        "observed": snapshot["labels"],
                    }
                if snapshot["assignees"] != expected_assignees:
                    mismatches["assignees"] = {
                        "expected": expected_assignees,
                        "observed": snapshot["assignees"],
                    }
                for operation in entry.get("label_plan", []):
                    if not _is_readiness_gate_operation(operation):
                        continue
                    if operation.get("expected_body_sha256") != snapshot["body_sha256"]:
                        mismatches["body_sha256"] = {
                            "expected": operation.get("expected_body_sha256"),
                            "observed": snapshot["body_sha256"],
                        }
                    try:
                        gate_labels = _normalize_label_names(operation.get("expected_labels"))
                    except ValueError:
                        gate_labels = []
                    if gate_labels != snapshot["labels"]:
                        mismatches["readiness_gate_labels"] = {
                            "expected": gate_labels,
                            "observed": snapshot["labels"],
                        }
                if mismatches:
                    preflight_errors.append(
                        {
                            "issue": issue,
                            "kind": "preflight",
                            "operation": "drifted",
                            "reason": "expected/current issue fields drifted",
                            "mismatches": mismatches,
                        }
                    )
                else:
                    snapshots[issue] = snapshot
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                preflight_errors.append(
                    {
                        "issue": issue,
                        "kind": "preflight",
                        "operation": "failed",
                        "reason": str(exc),
                    }
                )
        if preflight_errors:
            receipt["aborted"] = True
            receipt["abort_reason"] = "preflight_drift_or_unavailable"
            for error in preflight_errors:
                add_operation(error)
            failed_issues = {error.get("issue") for error in preflight_errors}
            for entry in entries:
                issue = entry.get("issue")
                if issue in failed_issues:
                    continue
                reason = (
                    "preflight_aborted"
                    if _entry_is_actionable(entry)
                    else entry.get("skip_reason") or "no_mutations"
                )
                add_operation(
                    {
                        "issue": issue,
                        "operation": "skip",
                        "status": "skipped",
                        "reason": reason,
                    }
                )
            return receipt

    body_count = 0
    label_count = 0
    aborted = False
    abort_reason: str | None = None
    for index, entry in enumerate(entries):
        issue = entry.get("issue")
        skip = entry.get("skip_reason")
        label_plan = entry.get("label_plan", [])
        body_proposed = bool(entry.get("body_patch", {}).get("proposed"))
        if skip:
            add_operation(
                {"issue": issue, "operation": "skip", "status": "skipped", "reason": skip}
            )
            continue
        if not body_proposed and not label_plan:
            add_operation(
                {"issue": issue, "operation": "skip", "status": "skipped", "reason": "no_mutations"}
            )
            continue
        if aborted:
            add_operation(
                {
                    "issue": issue,
                    "operation": "skip",
                    "status": "skipped",
                    "reason": abort_reason or "aborted_after_failure",
                }
            )
            continue

        issue_number = int(issue)
        block = _render_marker_block(
            {
                "number": issue,
                "classification": entry.get("classification_after"),
                "next_action": entry.get("next_action"),
                "authority": entry.get("authority"),
                "dispatch_eligible": entry.get("dispatch_eligible"),
                "labels": entry.get("labels", []),
                "body_sha256": entry.get("body_sha256"),
            },
            audit_digest=plan.get("audit_digest", ""),
            batch_id=batch_id,
        )
        if issue_number in snapshots:
            local_labels = set(snapshots[issue_number]["labels"])
        else:
            local_labels = set(_normalize_label_names(entry.get("labels", [])))

        gate_operations = [
            operation for operation in label_plan if _is_readiness_gate_operation(operation)
        ]
        generic_label_operations = [
            operation for operation in label_plan if not _is_readiness_gate_operation(operation)
        ]
        for gate_operation in gate_operations:
            label_count += 1
            receipt["label_operations_attempted"] = label_count
            expected_labels = set(_normalize_label_names(gate_operation.get("expected_labels")))
            expected_body_sha256 = gate_operation.get("expected_body_sha256")
            if issue_reader is not None and not dry_run:
                try:
                    gate_before = _normalize_issue_snapshot(
                        issue_reader(issue_number), issue_number
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    aborted = True
                    abort_reason = "readiness_gate_prewrite_read_failed"
                    add_operation(
                        {
                            "issue": issue,
                            "kind": "readiness_gate",
                            "action": _READINESS_GATE_ACTION,
                            "operation": "failed",
                            "reason": str(exc),
                        }
                    )
                    break
                if (
                    gate_before["body_sha256"] != expected_body_sha256
                    or set(gate_before["labels"]) != expected_labels
                ):
                    aborted = True
                    abort_reason = "readiness_gate_prewrite_drift"
                    add_operation(
                        {
                            "issue": issue,
                            "kind": "readiness_gate",
                            "action": _READINESS_GATE_ACTION,
                            "operation": "drifted",
                            "expected_body_sha256": expected_body_sha256,
                            "observed_body_sha256": gate_before["body_sha256"],
                            "expected_labels": sorted(expected_labels),
                            "observed_labels": gate_before["labels"],
                        }
                    )
                    break
                snapshots[issue_number] = gate_before
                local_labels = set(gate_before["labels"])
            if local_labels != expected_labels:
                aborted = True
                abort_reason = "readiness_gate_prewrite_drift"
                add_operation(
                    {
                        "issue": issue,
                        "kind": "readiness_gate",
                        "action": _READINESS_GATE_ACTION,
                        "operation": "drifted",
                        "expected_labels": sorted(expected_labels),
                        "observed_labels": sorted(local_labels),
                    }
                )
                break
            if dry_run:
                local_labels.add("state:ready")
                add_operation(
                    {
                        "issue": issue,
                        "kind": "readiness_gate",
                        "action": _READINESS_GATE_ACTION,
                        "operation": "would_write",
                        "expected_body_sha256": gate_operation.get("expected_body_sha256"),
                        "expected_labels": sorted(expected_labels),
                        "expected_labels_after": sorted(local_labels),
                    }
                )
                continue
            if readiness_gater is None:
                aborted = True
                abort_reason = "readiness_gate_unavailable"
                add_operation(
                    {
                        "issue": issue,
                        "kind": "readiness_gate",
                        "action": _READINESS_GATE_ACTION,
                        "operation": "failed",
                        "reason": "canonical issue_readiness_gate.gate_issue is unavailable",
                    }
                )
                break
            try:
                gate_result = readiness_gater(issue_number)
                if not isinstance(gate_result, Mapping):
                    raise RuntimeError("readiness gate returned a non-object result")
                outcome = gate_result.get("outcome")
                if (
                    outcome not in _READINESS_GATE_SUCCESS_OUTCOMES
                    or gate_result.get("verified") is not True
                ):
                    raise RuntimeError(
                        f"readiness gate outcome {outcome!r} was not a verified ready result"
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                aborted = True
                abort_reason = "readiness_gate_failed"
                add_operation(
                    {
                        "issue": issue,
                        "kind": "readiness_gate",
                        "action": _READINESS_GATE_ACTION,
                        "operation": "failed",
                        "reason": str(exc),
                    }
                )
                break
            else:
                local_labels.add("state:ready")
                if issue_number in snapshots:
                    snapshots[issue_number]["labels"] = sorted(local_labels)
                add_operation(
                    {
                        "issue": issue,
                        "kind": "readiness_gate",
                        "action": _READINESS_GATE_ACTION,
                        "operation": "written" if outcome == "ready" else "idempotent",
                        "outcome": outcome,
                        "verified": True,
                        "expected_body_sha256": gate_operation.get("expected_body_sha256"),
                        "expected_labels": sorted(expected_labels),
                        "observed_labels_after": sorted(local_labels),
                    }
                )
                if issue_reader is not None:
                    try:
                        gate_after = _normalize_issue_snapshot(
                            issue_reader(issue_number), issue_number
                        )
                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                        aborted = True
                        abort_reason = "readiness_gate_readback_failed"
                        add_operation(
                            {
                                "issue": issue,
                                "kind": "readiness_gate",
                                "action": _READINESS_GATE_ACTION,
                                "operation": "failed",
                                "reason": str(exc),
                            }
                        )
                        break
                    expected_labels_after = expected_labels | {"state:ready"}
                    if (
                        gate_after["body_sha256"] != expected_body_sha256
                        or set(gate_after["labels"]) != expected_labels_after
                    ):
                        aborted = True
                        abort_reason = "readiness_gate_readback_drift"
                        add_operation(
                            {
                                "issue": issue,
                                "kind": "readiness_gate",
                                "action": _READINESS_GATE_ACTION,
                                "operation": "drifted",
                                "expected_body_sha256": expected_body_sha256,
                                "observed_body_sha256": gate_after["body_sha256"],
                                "expected_labels": sorted(expected_labels_after),
                                "observed_labels": gate_after["labels"],
                            }
                        )
                        break
                    snapshots[issue_number] = gate_after
                    local_labels = set(gate_after["labels"])

        if aborted:
            if body_proposed:
                add_operation(
                    {
                        "issue": issue,
                        "kind": "body",
                        "operation": "skip",
                        "status": "skipped",
                        "reason": abort_reason or "aborted_after_failure",
                    }
                )
            for remaining in entries[index + 1 :]:
                add_operation(
                    {
                        "issue": remaining.get("issue"),
                        "operation": "skip",
                        "status": "skipped",
                        "reason": abort_reason or "aborted_after_failure",
                    }
                )
            break

        if body_proposed:
            if body_count >= mutation_ceiling:
                aborted = True
                abort_reason = "body_mutation_ceiling_reached"
                add_operation(
                    {
                        "issue": issue,
                        "kind": "body",
                        "operation": "skip",
                        "status": "skipped",
                        "reason": abort_reason,
                    }
                )
            else:
                body_count += 1
                current_snapshot = snapshots.get(issue_number)
                current_body = current_snapshot.get("body") if current_snapshot else None
                if dry_run:
                    add_operation(
                        {
                            "issue": issue,
                            "kind": "body",
                            "operation": "would_write",
                            "marker": block,
                            "expected_digest_before": entry.get("body_sha256"),
                        }
                    )
                elif (
                    current_body is not None and _compose_body(current_body, block) == current_body
                ):
                    add_operation(
                        {
                            "issue": issue,
                            "kind": "body",
                            "operation": "idempotent",
                            "expected_digest_before": entry.get("body_sha256"),
                            "observed_digest_after": _sha256_text(current_body),
                        }
                    )
                else:
                    try:
                        body_result = body_writer(issue_number, block)
                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                        aborted = True
                        abort_reason = "body_write_failed"
                        add_operation(
                            {
                                "issue": issue,
                                "kind": "body",
                                "operation": "failed",
                                "reason": str(exc),
                            }
                        )
                    else:
                        body_operation: dict[str, Any] = {
                            "issue": issue,
                            "kind": "body",
                            "operation": "written",
                            "expected_digest_before": entry.get("body_sha256"),
                        }
                        if isinstance(body_result, Mapping):
                            body_operation["api_status"] = body_result.get("status")
                            if body_result.get("body_sha256"):
                                body_operation["observed_digest_after"] = body_result["body_sha256"]
                        add_operation(body_operation)

        if aborted:
            if generic_label_operations:
                add_operation(
                    {
                        "issue": issue,
                        "kind": "labels",
                        "operation": "skip",
                        "status": "skipped",
                        "reason": abort_reason or "aborted_after_failure",
                    }
                )
            continue

        for label_operation in label_plan:
            if _is_readiness_gate_operation(label_operation):
                continue
            label_count += 1
            receipt["label_operations_attempted"] = label_count
            action = str(label_operation["action"])
            label = str(label_operation["label"]).strip()
            target_labels = set(local_labels)
            already_applied = (action == "add" and label in local_labels) or (
                action == "remove" and label not in local_labels
            )
            if already_applied:
                add_operation(
                    {
                        "issue": issue,
                        "kind": "label",
                        "action": action,
                        "label": label,
                        "operation": "idempotent",
                        "expected_labels_before": sorted(local_labels),
                        "observed_labels_after": sorted(local_labels),
                    }
                )
                continue
            if action == "add":
                target_labels.add(label)
            else:
                target_labels.discard(label)
            if dry_run:
                add_operation(
                    {
                        "issue": issue,
                        "kind": "label",
                        "action": action,
                        "label": label,
                        "operation": "would_write",
                        "expected_labels_before": sorted(local_labels),
                        "expected_labels_after": sorted(target_labels),
                    }
                )
                local_labels = target_labels
                continue
            try:
                if issue_reader is not None:
                    live_before = _normalize_issue_snapshot(
                        issue_reader(issue_number), issue_number
                    )
                    if set(live_before["labels"]) != local_labels:
                        aborted = True
                        abort_reason = "label_prewrite_drift"
                        add_operation(
                            {
                                "issue": issue,
                                "kind": "label",
                                "action": action,
                                "label": label,
                                "operation": "drifted",
                                "expected_labels_before": sorted(local_labels),
                                "observed_labels_before": live_before["labels"],
                            }
                        )
                        break
                if label_writer is None:
                    raise RuntimeError("label writer is unavailable")
                label_result = label_writer(issue_number, action, label)
                if issue_reader is not None:
                    live_after = _normalize_issue_snapshot(issue_reader(issue_number), issue_number)
                    if set(live_after["labels"]) != target_labels:
                        aborted = True
                        abort_reason = "label_readback_mismatch"
                        add_operation(
                            {
                                "issue": issue,
                                "kind": "label",
                                "action": action,
                                "label": label,
                                "operation": "drifted",
                                "expected_labels_after": sorted(target_labels),
                                "observed_labels_after": live_after["labels"],
                            }
                        )
                        break
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                aborted = True
                abort_reason = "label_write_failed"
                add_operation(
                    {
                        "issue": issue,
                        "kind": "label",
                        "action": action,
                        "label": label,
                        "operation": "failed",
                        "reason": str(exc),
                    }
                )
                break
            else:
                label_operation_result: dict[str, Any] = {
                    "issue": issue,
                    "kind": "label",
                    "action": action,
                    "label": label,
                    "operation": "written",
                    "expected_labels_before": sorted(local_labels),
                    "observed_labels_after": sorted(target_labels),
                }
                if isinstance(label_result, Mapping):
                    label_operation_result["api_status"] = label_result.get("status")
                add_operation(label_operation_result)
                local_labels = target_labels
        if aborted:
            # The remaining entries are summarized below as skipped operations.
            for remaining in entries[index + 1 :]:
                add_operation(
                    {
                        "issue": remaining.get("issue"),
                        "operation": "skip",
                        "status": "skipped",
                        "reason": abort_reason or "aborted_after_failure",
                    }
                )
            break
        if body_count >= mutation_ceiling >= 0:
            # Stop before starting the next body/label item once the body budget is consumed.
            aborted = True
            abort_reason = "body_mutation_ceiling_reached"
        if label_count >= HARD_LABEL_CEILING:
            aborted = True
            abort_reason = "label_operation_ceiling_reached"

    receipt["aborted"] = aborted
    receipt["abort_reason"] = abort_reason
    receipt["partial_failure"] = bool(receipt["failed"] or receipt["drifted"]) and bool(
        receipt["written"]
    )
    return receipt


def _select_entries(plan: Mapping[str, Any], numbers: Sequence[int]) -> list[dict[str, Any]]:
    if not numbers:
        return list(plan.get("entries", []))
    wanted = {int(n) for n in numbers}
    return [e for e in plan.get("entries", []) if int(e.get("issue", -1)) in wanted]


def _restrict_plan(plan: Mapping[str, Any], numbers: Sequence[int]) -> dict[str, Any]:
    """Return a digestable plan containing exactly the requested entries."""
    entries = _select_entries(plan, numbers)
    counts = Counter(str(entry.get("classification_before") or "error") for entry in entries)
    admission_reasons = Counter(
        str(entry.get("admission_reason") or "unknown") for entry in entries
    )
    route_counts = Counter(str(entry.get("worker_route") or "none") for entry in entries)
    restricted = {
        **plan,
        "item_count": len(entries),
        "entries": entries,
        "summary": {
            "by_classification_before": dict(counts),
            "admission_reason_histogram": dict(sorted(admission_reasons.items())),
            "not_admitted": dict(
                sorted(
                    (reason, count)
                    for reason, count in admission_reasons.items()
                    if reason != "claimable"
                )
            ),
            "by_worker_route": dict(route_counts),
            "ready_items": sum(1 for e in entries if e["classification_before"] == "ready"),
            "dispatch_eligible": sum(1 for e in entries if e["dispatch_eligible"]),
            "promotable_count": sum(1 for e in entries if e["readiness_gate"] is not None),
            "formalizable_count": sum(
                1 for e in entries if e["preparation_action"] == "formalize_issue"
            ),
            "label_operations": sum(len(e["label_plan"]) for e in entries),
        },
    }
    restricted["content_sha256"] = _content_digest(restricted)
    return restricted


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912 - explicit CLI gate branches
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-json", required=True, help="Path to open_issue_contract_audit.v1 JSON"
    )
    parser.add_argument(
        "--plan-json", default=None, help="Path to write open_issue_preparation_plan.v1 JSON"
    )
    parser.add_argument(
        "--plan-markdown", default=None, help="Path to write a compact plan markdown summary"
    )
    parser.add_argument("--batch-id", default="local", help="Stable batch identifier")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPOSITORY,
        help="GitHub owner/repository for apply mode",
    )
    parser.add_argument(
        "--source-ref",
        default="origin/main",
        help="Fresh source ref used by the canonical readiness gate",
    )
    parser.add_argument(
        "--issues",
        nargs="*",
        type=int,
        default=[],
        help="Restrict plan entries to these issue numbers (default: all)",
    )
    parser.add_argument(
        "--mode",
        choices=("plan", "render", "verify", "apply"),
        default="plan",
        help="plan (default) is report-only; apply requires --apply",
    )
    parser.add_argument("--issue", type=int, default=None, help="Issue number for render mode")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Enable mutation mode (plan/render/verify remain report-only)",
    )
    parser.add_argument(
        "--reviewed-plan-digest",
        default=None,
        help="Exact content_sha256 of the reviewed plan required for real apply",
    )
    parser.add_argument(
        "--mutation-ceiling",
        type=int,
        default=DEFAULT_MUTATION_CEILING,
        help="Max body writes per batch",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Render apply operations without writing"
    )
    parser.add_argument("--bodies-json", default=None, help="Mapping issue->body for verify mode")
    args = parser.parse_args(argv)

    if args.mode == "apply" and not args.apply and not args.dry_run:
        print("ERROR: apply mode requires --apply (or --dry-run)", file=sys.stderr)
        return 2

    try:
        audit = _load_audit(args.audit_json)
        plan = build_plan(audit, batch_id=args.batch_id)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.issues:
        if len(args.issues) != len(set(args.issues)):
            print("ERROR: --issues must not contain duplicate issue numbers", file=sys.stderr)
            return 2
        selected = _select_entries(plan, args.issues)
        selected_numbers = {int(entry.get("issue", -1)) for entry in selected}
        unknown = sorted(set(args.issues) - selected_numbers)
        if unknown:
            print(f"ERROR: issue(s) not present in audit plan: {unknown}", file=sys.stderr)
            return 2
        plan = _restrict_plan(plan, args.issues)

    real_apply = args.mode == "apply" and args.apply and not args.dry_run
    if real_apply:
        if not args.issues:
            print("ERROR: real apply requires an explicit --issues list", file=sys.stderr)
            return 2
        if args.repo != plan.get("repository", DEFAULT_REPOSITORY):
            print(
                "ERROR: --repo does not match the repository frozen in the reviewed plan",
                file=sys.stderr,
            )
            return 2
        audit_errors = audit.get("errors", [])
        truncation_errors = audit.get("truncation_or_errors", [])
        if audit.get("complete") is not True or audit_errors or truncation_errors:
            print("ERROR: real apply requires a complete audit with no errors", file=sys.stderr)
            return 2
        reviewed_digest = args.reviewed_plan_digest
        if not reviewed_digest:
            print("ERROR: real apply requires --reviewed-plan-digest", file=sys.stderr)
            return 2
        if reviewed_digest != plan.get("content_sha256"):
            print(
                "ERROR: reviewed plan digest does not match the selected plan",
                file=sys.stderr,
            )
            return 2

    if args.plan_json:
        Path(args.plan_json).write_text(_stable_json(plan) + "\n", encoding="utf-8")
    if args.plan_markdown:
        Path(args.plan_markdown).write_text(_render_plan_markdown(plan), encoding="utf-8")

    if args.mode == "render":
        return _render_mode(plan, issue=args.issue, batch_id=args.batch_id)

    if args.mode == "verify":
        return _verify_mode(plan, bodies_json=args.bodies_json)

    if args.mode == "apply":
        return _apply_mode(
            audit,
            plan,
            mutation_ceiling=args.mutation_ceiling,
            batch_id=args.batch_id,
            dry_run=args.dry_run,
            repo=args.repo,
            source_ref=args.source_ref,
        )

    # Plan mode output
    sys.stdout.write(_stable_json(plan) + "\n")
    return 0


def _render_mode(plan: Mapping[str, Any], *, issue: int | None, batch_id: str) -> int:
    """Render mode: print exactly one packet marker block for one issue."""
    if issue is None:
        print("ERROR: render mode requires --issue", file=sys.stderr)
        return 2
    entry = next((e for e in plan["entries"] if e.get("issue") == issue), None)
    if entry is None:
        print(f"ERROR: issue {issue} not in plan", file=sys.stderr)
        return 2
    block = _render_marker_block(
        {
            "number": entry.get("issue"),
            "classification": entry.get("classification_after"),
            "next_action": entry.get("next_action"),
            "authority": entry.get("authority"),
            "dispatch_eligible": entry.get("dispatch_eligible"),
            "labels": entry.get("labels", []),
            "body_sha256": entry.get("body_sha256"),
        },
        audit_digest=plan.get("audit_digest", ""),
        batch_id=batch_id,
    )
    sys.stdout.write(block)
    return 0


def _verify_mode(plan: Mapping[str, Any], *, bodies_json: str | None) -> int:
    """Verify mode: check marker uniqueness and byte preservation."""
    if not bodies_json:
        print("ERROR: verify mode requires --bodies-json", file=sys.stderr)
        return 2
    bodies = json.loads(Path(bodies_json).read_text(encoding="utf-8"))
    findings = _verify_batch(plan, bodies)
    bad = [f for f in findings if not f["ok"]]
    for finding in findings:
        status = "OK" if finding["ok"] else "FAIL"
        print(f"[{status}] #{finding['issue']} {finding['reason']}")
    return 1 if bad else 0


def _live_body_writer(  # noqa: C901, PLR0912 - read/write/readback guard branches
    issue: int,
    block: str,
    *,
    expected_labels: Sequence[str] | None = None,
    repo: str = DEFAULT_REPOSITORY,
) -> dict[str, Any]:
    """Write one issue body through REST with an immediate label-state guard."""
    from scripts.dev import _gh_rest

    endpoint = f"repos/{repo}/issues/{issue}"
    current_result = _gh_rest.run_gh_api(endpoint)
    if current_result.returncode != 0:
        detail = (
            current_result.stderr.strip() or current_result.stdout.strip() or "REST read failed"
        )
        raise RuntimeError(f"issue {issue} body read failed: {detail}")
    try:
        current_payload = json.loads(current_result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"issue {issue} body read was not valid JSON") from exc
    if not isinstance(current_payload, dict):
        raise RuntimeError(f"issue {issue} body read was not an object")
    current = current_payload.get("body")
    if not isinstance(current, str):
        raise RuntimeError(f"issue {issue} body read was not a string")
    expected: list[str] | None = None
    observed_labels: list[str] | None = None
    if expected_labels is not None:
        try:
            expected = _normalize_label_names(list(expected_labels))
            observed_labels = _normalize_label_names(current_payload.get("labels", []))
        except ValueError as exc:
            raise RuntimeError(f"issue {issue} labels were malformed: {exc}") from exc
        if observed_labels != expected:
            raise RuntimeError(
                f"issue {issue} labels drifted: expected {expected}, observed {observed_labels}"
            )
    body = _compose_body(current, block)
    if body == current:
        return {
            "status": "idempotent",
            "body_sha256": _sha256_text(current),
            "labels": observed_labels,
        }
    write_result = _gh_rest.run_gh_api(endpoint, {"body": body}, method="PATCH")
    if write_result.returncode != 0:
        detail = write_result.stderr.strip() or write_result.stdout.strip() or "REST write failed"
        raise RuntimeError(f"issue {issue} body write failed: {detail}")
    readback_result = _gh_rest.run_gh_api(endpoint)
    if readback_result.returncode != 0:
        detail = (
            readback_result.stderr.strip()
            or readback_result.stdout.strip()
            or "REST readback failed"
        )
        raise RuntimeError(f"issue {issue} body readback failed: {detail}")
    try:
        readback_payload = json.loads(readback_result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"issue {issue} body readback was not valid JSON") from exc
    if not isinstance(readback_payload, dict):
        raise RuntimeError(f"issue {issue} body readback was not an object")
    readback = readback_payload.get("body")
    if readback != body:
        raise RuntimeError(f"issue {issue} body readback mismatch")
    readback_labels: list[str] | None = None
    if expected is not None:
        try:
            readback_labels = _normalize_label_names(readback_payload.get("labels", []))
        except ValueError as exc:
            raise RuntimeError(f"issue {issue} body readback labels were malformed: {exc}") from exc
        if readback_labels != expected:
            raise RuntimeError(
                f"issue {issue} body readback changed labels: expected {expected}, "
                f"observed {readback_labels}"
            )
    return {
        "status": "written",
        "body_sha256": _sha256_text(body),
        "labels": readback_labels,
    }


def _live_label_writer(issue: int, action: str, label: str, *, repo: str) -> dict[str, Any]:
    """Apply one reviewed label operation through the shared REST label helper."""
    from scripts.dev import gh_pr_label_rest

    if action == "add":
        result = gh_pr_label_rest.add_label(issue, label, repo=repo)
    elif action == "remove":
        result = gh_pr_label_rest.remove_label(issue, label, repo=repo)
    else:
        raise RuntimeError(f"unsupported label action: {action}")
    if result.get("status") != "ok":
        raise RuntimeError(
            f"issue {issue} label {action} {label!r} failed: "
            f"{result.get('error', 'unknown REST label error')}"
        )
    return result


def _apply_mode(
    audit: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    mutation_ceiling: int,
    batch_id: str,
    dry_run: bool,
    repo: str = DEFAULT_REPOSITORY,
    source_ref: str = "origin/main",
) -> int:
    """Apply mode: bounded, CAS-guarded body and label writes with a receipt."""
    if mutation_ceiling < 0 or mutation_ceiling > HARD_BODY_CEILING:
        print(
            f"ERROR: mutation ceiling must be between 0 and {HARD_BODY_CEILING}",
            file=sys.stderr,
        )
        return 2
    entries = [entry for entry in plan.get("entries", []) if isinstance(entry, Mapping)]
    has_label_operations = _has_generic_label_operations(entries)
    label_catalog: set[str] | None = None
    if has_label_operations:
        try:
            label_catalog = _live_repository_label_names(repo=repo)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            receipt = _base_receipt(
                audit={**audit, "repository": repo},
                plan=plan,
                mutation_ceiling=mutation_ceiling,
                batch_id=batch_id,
                dry_run=dry_run,
                issues=[int(entry["issue"]) for entry in entries],
            )
            receipt["aborted"] = True
            receipt["abort_reason"] = "label_catalog_unavailable"
            receipt["failed"] = 1
            receipt["operations"].append(
                {"kind": "batch", "operation": "failed", "reason": str(exc)}
            )
            print(_stable_json(receipt))
            return 1

    entries_by_issue = {int(entry["issue"]): entry for entry in entries}

    def expected_body_labels(entry: Mapping[str, Any]) -> list[str]:
        """Return labels expected by the body writer after a readiness gate."""
        labels = set(_normalize_label_names(entry.get("labels", [])))
        if any(
            _is_readiness_gate_operation(operation) for operation in entry.get("label_plan", [])
        ):
            labels.add("state:ready")
        return sorted(labels)

    def body_writer(issue: int, block: str) -> Mapping[str, Any]:
        entry = entries_by_issue[issue]
        return _live_body_writer(
            issue,
            block,
            expected_labels=expected_body_labels(entry),
            repo=repo,
        )

    receipt = _apply_bodies(
        audit,
        plan,
        mutation_ceiling=mutation_ceiling,
        batch_id=batch_id,
        dry_run=dry_run,
        body_writer=body_writer,
        issue_reader=None if dry_run else lambda issue: _live_issue_reader(issue, repo=repo),
        label_catalog=label_catalog,
        label_writer=None
        if dry_run
        else lambda issue, action, label: _live_label_writer(issue, action, label, repo=repo),
        readiness_gater=None
        if dry_run
        else lambda issue: issue_readiness_gate.gate_issue(
            issue,
            repo=repo,
            source_ref=source_ref,
        ),
    )
    print(_stable_json(receipt))
    return 0 if not receipt["failed"] and not receipt["drifted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
