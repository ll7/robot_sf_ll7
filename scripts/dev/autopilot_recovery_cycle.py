#!/usr/bin/env python3
"""Execute one deterministic goal-autopilot empty-queue recovery cycle.

Issue #9534: the documented reconcile -> prepare -> admit -> review ->
discover -> arbitrate sequence is prompt-only convention.  This driver makes
it a machine-executable control-plane operation by composing the canonical
owners instead of reimplementing them:

- ready-candidate snapshot: ``snapshot_issue_batch --claimable``
- lifecycle reconciliation evidence: ``open_state_label_hygiene``
- contract preparation: ``audit_open_issue_contracts`` + ``prepare_open_issue_contracts --mode plan``
- live admission rerun: second ``snapshot_issue_batch --claimable``
- PR queue: ``snapshot_pr_queue --active``
- bounded discovery decision: first discovery lane without a head-bound saturated verdict
- arbitration: ``goal_autopilot_controller.arbitrate_controller``

Report-only: the driver performs zero GitHub writes.  Every lane runs through
an injectable runner so offline fixtures provide deterministic regression
coverage without GitHub or Git access.

Receipt schema: ``autopilot_recovery_cycle_receipt.v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "autopilot_recovery_cycle_receipt.v1"

DISCOVERY_LANES = (
    "workflow_and_api_friction",
    "failing_skipped_disabled_tests",
    "current_main_todo_fixme",
    "stale_parent_and_merged_pr_residuals",
    "documentation_and_ci_contract_drift",
    "measured_test_or_runtime_regressions",
    "context_note_indexing",
)

Runner = Callable[[Sequence[str]], "LaneResult"]


class LaneResult:
    """Outcome of one composed canonical owner invocation."""

    def __init__(
        self,
        *,
        ok: bool,
        payload: Mapping[str, Any] | None = None,
        error: str = "",
    ) -> None:
        """Store a lane outcome with its evidence payload or failure reason."""
        self.ok = ok
        self.payload = dict(payload or {})
        self.error = error


def _subprocess_runner(command: Sequence[str], *, timeout: int = 300) -> LaneResult:
    """Run one canonical CLI and parse its JSON stdout payload.

    Stdout is parsed on any exit code: several canonical owners emit a valid
    evidence payload while reporting findings via nonzero exit (for example
    the lifecycle hygiene guard exits 1 when candidates exist).  A Mapping
    payload is self-describing evidence; only unreadable output fails the
    lane.
    """
    try:
        completed = subprocess.run(
            [sys.executable, *command],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return LaneResult(ok=False, error=f"runner_failed: {exc}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return LaneResult(
            ok=False,
            error=f"exit_{completed.returncode}_payload_not_json:"
            f" {completed.stderr.strip()[:200] or exc}",
        )
    if not isinstance(payload, Mapping):
        return LaneResult(ok=False, error=f"exit_{completed.returncode}_payload_not_object")
    # Exit code is deliberately ignored once a valid payload exists: several
    # canonical owners report findings via nonzero exit while the payload's
    # own ok/error fields carry the verdict (e.g. lifecycle hygiene exits 1
    # when candidates exist).  The payload is returned pristine so downstream
    # digests and file staging stay clean.
    return LaneResult(ok=True, payload=payload)


def _digest(payload: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 digest of one lane evidence payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _lane(
    results: dict[str, LaneResult],
    errors: list[str],
    skipped: list[dict[str, str]],
    name: str,
    result: LaneResult,
    *,
    skip_reason: str = "",
) -> Mapping[str, Any]:
    """Record one lane outcome and return its payload (empty on failure)."""
    results[name] = result
    if not result.ok:
        errors.append(f"{name}_lane_unavailable: {result.error}")
        if skip_reason:
            skipped.append({"lane": name, "reason": skip_reason})
        return {}
    return result.payload


def _pr_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Map PR snapshot rows to controller lane counts with documented rules.

    - merge_ready: carries the ``merge-ready`` label.
    - review_eligible: open, non-draft, no ``state:blocked`` label, no
      merge-ready/merge-if-ci-green label.
    - recoverable_active: open with a ``state:blocked`` label.
    - open: every open row.
    """
    counts = {"merge_ready": 0, "review_eligible": 0, "recoverable_active": 0, "open": 0}
    conditional = {"merge-ready", "merge-if-ci-green"}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        labels = {str(label) for label in row.get("labels", []) if isinstance(label, str)}
        if row.get("state", "OPEN") != "OPEN":
            continue
        counts["open"] += 1
        if "merge-ready" in labels:
            counts["merge_ready"] += 1
        elif "state:blocked" in labels:
            counts["recoverable_active"] += 1
        elif not row.get("draft", False) and not labels.intersection(conditional):
            counts["review_eligible"] += 1
    return counts


def _stale_lifecycle_rows(queue_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flag candidate rows whose lifecycle state contradicts dispatch truth.

    A row is stale when it carries ``state:running`` with no atomic claim and
    no covering PR, or when it carries ``state:ready`` together with a hold
    qualifier (``state:parked``, ``state:blocked``, ``decision-required``)
    that already blocks dispatch.  Flagging is read-only; repair belongs to
    the canonical reconciler.
    """
    hold_qualifiers = {"state:parked", "state:blocked", "decision-required"}
    stale: list[dict[str, Any]] = []
    issues = queue_payload.get("issues", [])
    if not isinstance(issues, list):
        return stale
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        labels = {str(label) for label in issue.get("labels", []) if isinstance(label, str)}
        claim = issue.get("claim", {})
        claimed = isinstance(claim, Mapping) and bool(claim.get("claimed"))
        linked = issue.get("linked_prs", [])
        has_pr = isinstance(linked, list) and len(linked) > 0
        number = issue.get("number")
        if "state:running" in labels and not claimed and not has_pr:
            stale.append({"issue": number, "reason": "stale_running_without_claim_or_pr"})
        elif "state:ready" in labels and labels.intersection(hold_qualifiers):
            stale.append({"issue": number, "reason": "ready_with_hold_qualifier"})
    return stale


class LaneBundle:
    """Collected lane payloads with their error and skip bookkeeping."""

    def __init__(self) -> None:
        """Start with empty lanes, errors, and skip records."""
        self.results: dict[str, LaneResult] = {}
        self.errors: list[str] = []
        self.skipped: list[dict[str, str]] = []
        self.queue: Mapping[str, Any] = {}
        self.audit: Mapping[str, Any] = {}
        self.preparation: Mapping[str, Any] = {}
        self.rerun: Mapping[str, Any] = {}
        self.pr_queue: Mapping[str, Any] = {}

    def collect(
        self,
        run: Runner,
        name: str,
        command: Sequence[str],
        *,
        skip_reason: str,
    ) -> Mapping[str, Any]:
        """Run one lane, record bookkeeping, and return its payload."""
        return _lane(
            self.results, self.errors, self.skipped, name, run(command), skip_reason=skip_reason
        )

    def stage_preparation(
        self, run: Runner, audit: Mapping[str, Any], work_dir: Path | None
    ) -> None:
        """Run the preparation plan from the staged audit payload, if any."""
        if not audit:
            self.skipped.append(
                {"lane": "preparation", "reason": "skipped without contract audit payload"}
            )
            return
        try:
            if work_dir is not None:
                self._stage_preparation_from(run, audit, Path(work_dir))
                return
            with tempfile.TemporaryDirectory(prefix="robot_sf_recovery_cycle_") as scratch:
                self._stage_preparation_from(run, audit, Path(scratch))
        except OSError as exc:
            self.results["preparation"] = LaneResult(ok=False, error=f"audit_staging_failed: {exc}")
            self.errors.append(f"preparation_audit_staging_failed: {exc}")
            self.skipped.append(
                {"lane": "preparation", "reason": "cannot stage audit payload locally"}
            )

    def _stage_preparation_from(self, run: Runner, audit: Mapping[str, Any], scratch: Path) -> None:
        """Invoke preparation while its staged audit remains available to the runner."""
        audit_path = scratch / "recovery_cycle_audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        self.preparation = self.collect(
            run,
            "preparation",
            [
                "scripts/dev/prepare_open_issue_contracts.py",
                "--audit-json",
                str(audit_path),
                "--mode",
                "plan",
            ],
            skip_reason="preparation plan unavailable; promotable/formalizable unknown",
        )


def _collect_lanes(
    run: Runner,
    *,
    repo: str,
    issue_limit: int,
    pr_limit: int,
    audit_item_limit: int,
    audit_max_pages: int,
    audit_page_size: int,
    work_dir: Path | None,
) -> LaneBundle:
    """Run every canonical owner once and bundle the lane payloads."""
    bundle = LaneBundle()
    scripts = "scripts.dev."
    bundle.queue = bundle.collect(
        run,
        "issue_queue",
        [
            "-m",
            f"{scripts}snapshot_issue_batch",
            "--claimable",
            "--limit",
            str(issue_limit),
            "--json",
        ],
        skip_reason="queue snapshot unavailable; cannot prove 0 claimable",
    )
    bundle.collect(
        run,
        "lifecycle",
        [
            "scripts/dev/open_state_label_hygiene.py",
            "--repo",
            repo,
            "--max-issue-pages",
            "3",
            "--max-timeline-pages",
            "2",
            "--max-pr-lookups",
            "100",
        ],
        skip_reason="lifecycle reconciliation evidence unavailable",
    )
    bundle.audit = bundle.collect(
        run,
        "contract_audit",
        [
            "scripts/dev/audit_open_issue_contracts.py",
            "--repo",
            repo,
            "--format",
            "json",
            "--item-limit",
            str(audit_item_limit),
            "--max-pages",
            str(audit_max_pages),
            "--page-size",
            str(audit_page_size),
        ],
        skip_reason="contract audit unavailable; preparation counts unknown",
    )
    audit = bundle.audit
    bundle.stage_preparation(run, audit, work_dir)
    bundle.rerun = bundle.collect(
        run,
        "admission_rerun",
        [
            "-m",
            f"{scripts}snapshot_issue_batch",
            "--claimable",
            "--limit",
            str(issue_limit),
            "--json",
        ],
        skip_reason="live admission rerun unavailable",
    )
    bundle.pr_queue = bundle.collect(
        run,
        "pr_queue",
        ["-m", f"{scripts}snapshot_pr_queue", "--active", "--limit", str(pr_limit), "--json"],
        skip_reason="PR queue snapshot unavailable",
    )
    return bundle


def _build_controller_snapshot(bundle: LaneBundle, *, origin_main_sha: str) -> dict[str, Any]:
    """Assemble arbiter evidence from collected lane payloads."""
    queue, preparation = bundle.queue, bundle.preparation
    pr_queue = bundle.pr_queue
    stale_rows = _stale_lifecycle_rows(queue) if queue else []
    summary = preparation.get("summary", {}) if isinstance(preparation, Mapping) else {}
    if not isinstance(summary, Mapping):
        summary = {}
    queue_histogram: Mapping[str, Any] = {}
    if isinstance(queue, Mapping):
        histogram = queue.get("admission_reason_histogram", {})
        queue_histogram = histogram if isinstance(histogram, Mapping) else {}
    audit_digest = _digest({"audit": dict(bundle.audit)})
    return {
        "origin_main_sha": origin_main_sha,
        "freshness": {
            "issue_state_digest": _digest({"queue": queue.get("issues", [])}),
            "claim_state_digest": _digest({"queue_claims": queue.get("issues", [])}),
            "pr_head_digest": _digest({"prs": pr_queue.get("prs", [])}),
            "preparation_audit_digest": audit_digest,
            "discovery_relevant_paths_digest": _digest({"lanes": list(DISCOVERY_LANES)}),
        },
        "implementation": {
            "candidate_scope": "state:ready",
            "queue_completeness": queue.get("queue_completeness", "incomplete"),
            "zero_work_authoritative": queue.get("zero_work_authoritative", False),
            "claimable_count": len(queue.get("claimable_issues", []))
            if isinstance(queue.get("claimable_issues"), list)
            else 0,
            "admission_reason_histogram": dict(queue_histogram),
        },
        "pull_requests": {
            "open_count": 0,
            "recoverable_active_count": 0,
            "review_eligible_count": 0,
            "merge_ready_count": 0,
            **_pr_counts(pr_queue.get("prs", []) if isinstance(pr_queue, Mapping) else []),
        },
        "preparation": {
            "audit_digest": audit_digest,
            "audit_base_sha": origin_main_sha,
            "reconciliation_base_sha": origin_main_sha,
            "stale_state_count": len(stale_rows),
            "promotable_count": int(summary.get("promotable_count", 0)),
            "formalizable_count": int(summary.get("formalizable_count", 0)),
            "blocker_reconciliation_count": 0,
            "blocker_reconciliation_complete": True,
            "decision_count": 0,
            "blocker_count": 0,
            "decomposition_count": 0,
            "active_handoff_count": 0,
            "stale_rows": stale_rows,
        },
        "discovery": {
            "lane": DISCOVERY_LANES[0],
            "relevant_head_sha": origin_main_sha,
            "status": "pending",
            "created_issue_numbers": [],
            "readiness_outcomes": [],
            "next_lane": DISCOVERY_LANES[0],
            "next_lane_rationale": "no head-bound saturated verdict recorded; first cursor lane due",
        },
    }


def run_cycle(  # noqa: PLR0913 - parameters map 1:1 to CLI lane budgets
    *,
    repo: str,
    origin_main_sha: str,
    runner: Runner | None = None,
    issue_limit: int = 100,
    pr_limit: int = 50,
    audit_item_limit: int = 100,
    audit_max_pages: int = 5,
    audit_page_size: int = 100,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one report-only recovery cycle and return the versioned receipt."""
    run = runner or _subprocess_runner
    bundle = _collect_lanes(
        run,
        repo=repo,
        issue_limit=issue_limit,
        pr_limit=pr_limit,
        audit_item_limit=audit_item_limit,
        audit_max_pages=audit_max_pages,
        audit_page_size=audit_page_size,
        work_dir=work_dir,
    )
    stale_rows = _stale_lifecycle_rows(bundle.queue) if bundle.queue else []
    snapshot_payload = _build_controller_snapshot(bundle, origin_main_sha=origin_main_sha)
    try:
        from scripts.dev import goal_autopilot_controller as controller

        decision = controller.arbitrate_controller(snapshot_payload)
    except (ImportError, ValueError) as exc:
        bundle.errors.append(f"controller_arbitration_failed: {exc}")
        decision = {
            "schema": "goal_autopilot_controller_decision.v1",
            "global_zero_work": False,
            "next_action": "refresh_controller_evidence",
            "stop_reason": None,
            "counts": {},
            "lane_status": {},
            "reasons": [f"controller_arbitration_failed: {exc}"],
        }

    return _finalize_receipt(bundle, decision, stale_rows, origin_main_sha=origin_main_sha)


_LANE_ORDER = (
    "issue_queue",
    "lifecycle",
    "contract_audit",
    "preparation",
    "admission_rerun",
    "pr_queue",
)
_LANE_RECOVERY = {
    "issue_queue": "refresh_issue_queue",
    "lifecycle": "reconcile_lifecycle",
    "contract_audit": "run_preparation",
    "preparation": "run_preparation",
    "admission_rerun": "refresh_issue_queue",
    "pr_queue": "refresh_controller_evidence",
}


def _fallback_counts(bundle: LaneBundle, stale_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Derive receipt counts from lane payloads when the arbiter omits them."""
    rerun_rows = bundle.rerun.get("claimable_issues", [])
    summary = bundle.preparation.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    pr_rows = bundle.pr_queue.get("prs", []) if isinstance(bundle.pr_queue, Mapping) else []
    pr_derived = _pr_counts(pr_rows if isinstance(pr_rows, list) else [])
    return {
        "claimable_count": len(rerun_rows) if isinstance(rerun_rows, list) else 0,
        "promotable_count": int(summary.get("promotable_count", 0) or 0),
        "formalizable_count": int(summary.get("formalizable_count", 0) or 0),
        "stale_state_count": len(stale_rows),
        "blocker_reconciliation_count": 0,
        "merge_ready_count": pr_derived["merge_ready"],
        "review_eligible_count": pr_derived["review_eligible"],
        "recoverable_active_count": pr_derived["recoverable_active"],
        "open_count": pr_derived["open"],
    }


def _lifecycle_notes(bundle: LaneBundle) -> tuple[int | None, list[str]]:
    """Summarize the hygiene payload without failing its lane.

    The guard exits nonzero exactly when candidates exist, so a payload with
    ``ok: false`` and rows is healthy evidence.  Surface the candidate count
    and any tool-level error as notes instead.
    """
    notes: list[str] = []
    payload = bundle.results.get("lifecycle")
    if payload is None or not payload.ok:
        return None, notes
    body = payload.payload
    count = body.get("candidate_count")
    candidates = count if isinstance(count, int) and count >= 0 else None
    if body.get("ok") is False:
        if isinstance(body.get("error"), str) and body["error"]:
            notes.append(f"lifecycle_hygiene_tool_error: {body['error'][:200]}")
        elif candidates:
            notes.append(f"lifecycle_hygiene_reports_candidates: {candidates}")
    if body.get("complete_for_open_issues") is False:
        notes.append("lifecycle_hygiene_coverage_incomplete")
    return candidates, notes


def _finalize_receipt(
    bundle: LaneBundle,
    decision: Mapping[str, Any],
    stale_rows: Sequence[Mapping[str, Any]],
    *,
    origin_main_sha: str,
) -> dict[str, Any]:
    """Apply driver-side recovery routing and build the versioned receipt."""
    errors, skipped, lanes = bundle.errors, bundle.skipped, bundle.results
    failed = [name for name in _LANE_ORDER if name in lanes and not lanes[name].ok]
    next_action = decision.get("next_action")
    stop_reason = decision.get("stop_reason")
    if failed:
        next_action = _LANE_RECOVERY[failed[0]]
        stop_reason = None
        skipped.append(
            {"lane": failed[0], "reason": f"recovery routed after lane failure: {failed[0]}"}
        )
    elif stale_rows:
        next_action = "reconcile_lifecycle"
        stop_reason = None
    terminal_refused = True
    if decision.get("global_zero_work") is True:
        errors.append("arbiter_terminal_contradicts_report_only_cycle")
    decision_counts = decision.get("counts")
    if not isinstance(decision_counts, Mapping) or not decision_counts:
        decision_counts = _fallback_counts(bundle, stale_rows)
        skipped.append(
            {
                "lane": "controller_counts",
                "reason": "arbiter omitted counts; lane-derived fallback used",
            }
        )
    lifecycle_candidates, notes = _lifecycle_notes(bundle)
    return {
        "schema": RECEIPT_SCHEMA,
        "origin_main_sha": origin_main_sha,
        "lanes_evaluated": sorted(lanes.keys()),
        "lane_errors": list(errors),
        "skipped": skipped,
        "notes": notes,
        "lifecycle_candidates": lifecycle_candidates,
        "counts": dict(decision_counts),
        "next_action": next_action,
        "stop_reason": stop_reason,
        "terminal_refused": terminal_refused,
        "stale_lifecycle_rows": list(stale_rows),
        "discovery_decision": {
            "next_lane": DISCOVERY_LANES[0],
            "executed": False,
            "rationale": "report-only cycle records the lane decision; scouts run separately",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the recovery-cycle CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="ll7/robot_sf_ll7")
    parser.add_argument("--origin-main-sha", default=None)
    parser.add_argument("--issue-limit", type=int, default=100)
    parser.add_argument("--pr-limit", type=int, default=50)
    parser.add_argument("--audit-item-limit", type=int, default=100)
    parser.add_argument("--audit-max-pages", type=int, default=5)
    parser.add_argument("--audit-page-size", type=int, default=100)
    parser.add_argument("--receipt-out", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON (the default)")
    return parser


def _resolve_origin_main_sha(explicit: str | None) -> str:
    """Return the explicit SHA or resolve the current origin/main tip."""
    if explicit:
        return explicit
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "origin/main^{commit}"],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    sha = completed.stdout.strip()
    if completed.returncode != 0 or len(sha) != 40:
        raise ValueError("cannot resolve origin/main tip")
    return sha


def main(argv: Sequence[str] | None = None) -> int:
    """Run one report-only recovery cycle and emit the receipt."""
    args = _build_parser().parse_args(argv)
    try:
        receipt = run_cycle(
            repo=args.repo,
            origin_main_sha=_resolve_origin_main_sha(args.origin_main_sha),
            issue_limit=args.issue_limit,
            pr_limit=args.pr_limit,
            audit_item_limit=args.audit_item_limit,
            audit_max_pages=args.audit_max_pages,
            audit_page_size=args.audit_page_size,
            work_dir=args.work_dir,
        )
    except (OSError, ValueError) as exc:
        print(f"recovery cycle failed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    if args.receipt_out is not None:
        try:
            args.receipt_out.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"recovery cycle failed: {exc}", file=sys.stderr)
            return 2
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
