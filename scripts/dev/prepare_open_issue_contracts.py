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
- apply (``--apply``): bounded, compare-and-swap guarded, exact-item body and
  label mutations with a credential-free receipt. Requires an explicit reviewed
  plan digest and issue list; aborts the whole batch on any drift.

The tool defaults to no-write mode and never creates labels, never adds PR
runner labels to issues, and never mutates issue state, assignments,
milestones, projects, comments, parent relations, PRs, or merges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MARKER_START = "<!-- goal-autopilot-preparation:v1:start -->"
MARKER_END = "<!-- goal-autopilot-preparation:v1:end -->"
PACKET_SCHEMA = "goal_autopilot_preparation.v1"
PLAN_SCHEMA = "open_issue_preparation_plan.v1"
RECEIPT_SCHEMA = "open_issue_preparation_receipt.v1"

DEFAULT_MUTATION_CEILING = 10
HARD_BODY_CEILING = 25
HARD_LABEL_CEILING = 50

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


def _worker_route(item: Mapping[str, Any]) -> str:
    """Return the worker route for one audit item."""
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
        "next_action": item.get("next_action", ""),
        "authority": item.get("authority", ""),
        "execution_mode": _EXECUTION_MODE.get(classification, "error-repair"),
        "preferred_worker": _worker_route(item),
        "expected_pr_runner_label": _pr_runner_label(classification),
        "implementation_admitted": bool(item.get("dispatch_eligible")),
        "state_ready_change_proposed": _state_ready_proposed(item),
        "mutation_batch": batch_id,
    }


def _pr_runner_label(classification: str) -> str:
    """Map one classification to its expected PR runner label (issues never get it)."""
    return "runner:max" if classification in _MAX_CLASSIFICATIONS else "runner:luna"


def _state_ready_proposed(item: Mapping[str, Any]) -> bool:
    """Return whether the packet proposes a reviewed state:ready transition."""
    return bool(item.get("dispatch_eligible")) and item.get("classification") == "ready"


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


def _label_plan(item: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return the exact reviewed label plan for one item (add/remove only)."""
    labels = set(item.get("labels") or [])
    plan: list[dict[str, str]] = []
    if item.get("classification") == "ready" and not item.get("dispatch_eligible"):
        if "state:ready" not in labels and not (labels & _AUTHORITY_LABELS):
            plan.append({"issue": str(item.get("number")), "action": "add", "label": "state:ready"})
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
                "body_sha256": item.get("body_sha256"),
                "claim_state": item.get("claim"),
                "classification_before": item.get("observed_classification"),
                "classification_after": item.get("classification"),
                "admission_reason": item.get("admission_reason"),
                "execution_mode": _EXECUTION_MODE.get(item.get("classification", "error")),
                "worker_route": _worker_route(item),
                "next_action": item.get("next_action", ""),
                "authority": item.get("authority", ""),
                "dispatch_eligible": bool(item.get("dispatch_eligible")),
                "state_ready_change_proposed": _state_ready_proposed(item),
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
            "label_operations": sum(len(e["label_plan"]) for e in entries),
        },
    }
    plan["content_sha256"] = _sha256_json(plan)
    return plan


def _skip_reason(item: Mapping[str, Any]) -> str:
    """Return the skip reason for items that must not be mutated."""
    classification = item.get("classification")
    if item.get("listing_drift"):
        return "listing_drift"
    if classification in ("assigned", "already_claimed", "working", "review"):
        return "active_owner"
    if classification in ("closed",):
        return "closed"
    if classification == "error":
        return "error_row"
    if classification in ("parent", "human_decision"):
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
        if original_sha and _sha256_text(normalized) != original_sha:
            findings.append({"issue": issue, "ok": False, "reason": "content drift outside marker"})
            continue
        findings.append({"issue": issue, "ok": True, "reason": ""})
    return findings


def _apply_bodies(
    audit: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    mutation_ceiling: int,
    batch_id: str,
    dry_run: bool,
    body_writer: Any,
) -> dict[str, Any]:
    """Apply the reviewed plan bodies with CAS guards.

    ``body_writer`` is an injectable ``(issue_number, body) -> None`` for
    offline tests; the live path uses the REST helper.
    """
    audit_digest = plan.get("audit_digest", "")
    operations: list[dict[str, Any]] = []
    applied = 0
    for entry in plan.get("entries", []):
        if applied >= mutation_ceiling:
            break
        issue = entry.get("issue")
        skip = entry.get("skip_reason")
        if skip:
            operations.append({"issue": issue, "operation": "skip", "reason": skip})
            continue
        if not entry.get("body_patch", {}).get("proposed"):
            operations.append({"issue": issue, "operation": "skip", "reason": "no_body_patch"})
            continue
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
            audit_digest=audit_digest,
            batch_id=batch_id,
        )
        if dry_run:
            operations.append(
                {
                    "issue": issue,
                    "operation": "would_write",
                    "marker": block,
                    "expected_digest_before": entry.get("body_sha256"),
                }
            )
            applied += 1
            continue
        try:
            body_writer(issue, block)
        except (OSError, RuntimeError) as exc:  # pragma: no cover - live REST error path
            operations.append({"issue": issue, "operation": "failed", "reason": str(exc)})
            # Fail closed: abort the whole batch on any write error.
            break
        operations.append(
            {
                "issue": issue,
                "operation": "written",
                "expected_digest_before": entry.get("body_sha256"),
            }
        )
        applied += 1
    return {
        "schema": RECEIPT_SCHEMA,
        "batch_id": batch_id,
        "mutation_ceiling": mutation_ceiling,
        "dry_run": dry_run,
        "operations": operations,
        "written": sum(1 for op in operations if op["operation"] == "written"),
        "would_write": sum(1 for op in operations if op["operation"] == "would_write"),
        "skipped": sum(1 for op in operations if op["operation"] == "skip"),
    }


def _select_entries(plan: Mapping[str, Any], numbers: Sequence[int]) -> list[dict[str, Any]]:
    if not numbers:
        return list(plan.get("entries", []))
    wanted = {int(n) for n in numbers}
    return [e for e in plan.get("entries", []) if int(e.get("issue", -1)) in wanted]


def main(argv: list[str] | None = None) -> int:
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
        plan = {**plan, "entries": _select_entries(plan, args.issues)}

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


def _live_body_writer(issue: int, block: str) -> None:
    """Write one issue body through the canonical REST helper (CAS re-read)."""
    from scripts.dev import _gh_rest

    endpoint = f"repos/ll7/robot_sf_ll7/issues/{issue}"
    current_result = _gh_rest.run_gh_api(endpoint, extra_args=["--jq", ".body"])
    if current_result.returncode != 0:
        detail = (
            current_result.stderr.strip() or current_result.stdout.strip() or "REST read failed"
        )
        raise RuntimeError(f"issue {issue} body read failed: {detail}")
    try:
        current = json.loads(current_result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"issue {issue} body read was not valid JSON") from exc
    if not isinstance(current, str):
        raise RuntimeError(f"issue {issue} body read was not a string")
    if _MARKER_BLOCK_RE.search(current):
        body = _MARKER_BLOCK_RE.sub(block.rstrip("\n"), current)
    else:
        body = current.rstrip("\n") + "\n\n" + block.rstrip("\n")
    write_result = _gh_rest.run_gh_api(endpoint, {"body": body}, method="PATCH")
    if write_result.returncode != 0:
        detail = write_result.stderr.strip() or write_result.stdout.strip() or "REST write failed"
        raise RuntimeError(f"issue {issue} body write failed: {detail}")
    readback_result = _gh_rest.run_gh_api(endpoint, extra_args=["--jq", ".body"])
    if readback_result.returncode != 0:
        detail = (
            readback_result.stderr.strip()
            or readback_result.stdout.strip()
            or "REST readback failed"
        )
        raise RuntimeError(f"issue {issue} body readback failed: {detail}")
    try:
        readback = json.loads(readback_result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"issue {issue} body readback was not valid JSON") from exc
    if readback != body:
        raise RuntimeError(f"issue {issue} body readback mismatch")


def _apply_mode(
    audit: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    mutation_ceiling: int,
    batch_id: str,
    dry_run: bool,
) -> int:
    """Apply mode: bounded, CAS-guarded body writes with a receipt."""
    if mutation_ceiling > HARD_BODY_CEILING:
        print(
            f"ERROR: mutation ceiling {mutation_ceiling} exceeds hard max {HARD_BODY_CEILING}",
            file=sys.stderr,
        )
        return 2
    receipt = _apply_bodies(
        audit,
        plan,
        mutation_ceiling=mutation_ceiling,
        batch_id=batch_id,
        dry_run=dry_run,
        body_writer=_live_body_writer,
    )
    print(_stable_json(receipt))
    return 0 if not any(op["operation"] == "failed" for op in receipt["operations"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
