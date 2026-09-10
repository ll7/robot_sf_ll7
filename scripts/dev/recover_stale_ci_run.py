#!/usr/bin/env python3
"""Guarded operator recovery for a stale GitHub Actions setup-starvation run.

This is the repository-owned recovery path advertised by
``scripts/dev/check_pr_ci_status.py`` as
``checks.recovery.stale_runs[*].guarded_recovery_command``. It never mutates
GitHub unless ``--apply`` is passed together with an explicit ``--reason``, and
even then it re-reads the live PR CI state under the host-local PR write lock
immediately before requesting exactly one ``gh run rerun``.

A successful rerun request is route evidence only: it is not implementation
evidence, it does not weaken required checks, and it does not authorize a
merge. Merge admission stays blocked until a fresh exact-head CI success.

Exit codes: 0 plan/apply request success; 1 refused/blocked; 2 usage/error.

Example (report-only plan, the default):

    uv run python scripts/dev/recover_stale_ci_run.py \\
        --pr <pr-number> --run-id <run-id> \\
        --expected-head-sha <40-hex-sha> --json

Example (explicit operator-authorized rerun request):

    uv run python scripts/dev/recover_stale_ci_run.py \\
        --pr <pr-number> --run-id <run-id> \\
        --expected-head-sha <40-hex-sha> --apply \\
        --reason "operator confirmed the setup step is stalled"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dev.check_pr_ci_status import (  # noqa: E402
    DEFAULT_ACTIONS_STALE_AFTER_SECONDS,
    DEFAULT_QUEUE_STARVATION_SECONDS,
    _fetch_ci_status,
    _gh,
    _non_negative_float,
    _non_negative_int,
)
from scripts.dev.pr_write_guard import pr_write_lock  # noqa: E402

RECEIPT_SCHEMA = "ci_setup_recovery_receipt.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
ACTIVE_RUN_STATUSES = {"queued", "in_progress"}
STDERR_EXCERPT_LIMIT = 200
FIELD_EXCERPT_LIMIT = 120


def _bounded_excerpt(value: Any, *, limit: int = FIELD_EXCERPT_LIMIT) -> str | None:
    """Return a bounded string excerpt, or ``None`` for non-string fields."""
    if not isinstance(value, str):
        return None
    return value[:limit]


def _observed_at() -> str:
    """Return the current UTC timestamp in ISO 8601 form."""
    return datetime.now(UTC).isoformat()


def _none_mutation() -> dict[str, Any]:
    """Return the report-only mutation record."""
    return {"status": "none", "command": None, "exit_code": None, "stderr_excerpt": ""}


def _failed_mutation(
    command: list[str],
    exit_code: int | None,
    stderr: Any,
) -> dict[str, Any]:
    """Return a failed mutation record with a bounded stderr excerpt."""
    return {
        "status": "failed",
        "command": command,
        "exit_code": exit_code,
        "stderr_excerpt": _bounded_excerpt(stderr, limit=STDERR_EXCERPT_LIMIT) or "",
    }


def _run_status_active(item: dict[str, Any]) -> bool:
    """Return whether the run/job status is still queued or in progress."""
    for key in ("run_status", "job_status"):
        status = item.get(key)
        if isinstance(status, str) and status.strip():
            return status.strip().lower() in ACTIVE_RUN_STATUSES
    return False


def _find_starvation_item(checks: dict[str, Any], run_id: int) -> dict[str, Any] | None:
    """Find the lifecycle item for ``run_id`` in the CI payload, if present."""
    candidates: list[Any] = []
    lifecycle = checks.get("actions_lifecycle")
    if isinstance(lifecycle, dict) and isinstance(lifecycle.get("items"), list):
        candidates.extend(lifecycle["items"])
    starvation_items = checks.get("setup_starvation_items")
    if isinstance(starvation_items, list):
        candidates.extend(starvation_items)
    for item in candidates:
        if not isinstance(item, dict):
            continue
        item_run_id = item.get("run_id")
        try:
            matches = int(item_run_id) == run_id
        except (TypeError, ValueError):
            continue
        if matches:
            return item
    return None


def _setup_starvation_evidence(
    checks: dict[str, Any],
    item: dict[str, Any] | None,
    *,
    stale_after_seconds: int,
) -> dict[str, Any]:
    """Build a bounded setup-starvation evidence excerpt."""
    evidence: dict[str, Any] = {
        "matched": item is not None,
        "pending_reason": _bounded_excerpt(checks.get("pending_reason")),
        "diagnostic": _bounded_excerpt(checks.get("diagnostic")),
        "stale_after_seconds": stale_after_seconds,
    }
    if item is None:
        return evidence
    evidence.update(
        {
            "run_id": item.get("run_id"),
            "job_id": item.get("job_id"),
            "phase": _bounded_excerpt(item.get("phase")),
            "run_status": _bounded_excerpt(item.get("run_status")),
            "job_status": _bounded_excerpt(item.get("job_status")),
            "step_name": _bounded_excerpt(item.get("step_name")),
            "age_seconds": item.get("age_seconds"),
            "age_source": _bounded_excerpt(item.get("age_source")),
            "setup_starvation": item.get("setup_starvation") is True,
            "exact_head_sha_matches": item.get("exact_head_sha_matches"),
            "run_head_sha": _bounded_excerpt(item.get("run_head_sha")),
        }
    )
    return evidence


def _identity_guard_codes(
    data: dict[str, Any],
    *,
    pr: int,
    expected_head_sha: str,
) -> list[str]:
    """Return fail-closed codes for the live PR identity and head guard."""
    codes: list[str] = []
    payload_pr = data.get("pr")
    try:
        pr_matches = payload_pr is None or int(payload_pr) == pr
    except (TypeError, ValueError):
        pr_matches = False
    if not pr_matches:
        codes.append("pr_payload_mismatch")
    observed_head_sha = str(data.get("head_sha") or "")
    if observed_head_sha.lower() != expected_head_sha.lower():
        codes.append("pr_head_mismatch")
    return codes


def _item_guard_codes(item: dict[str, Any]) -> list[str]:
    """Return fail-closed codes for the matched setup-starvation lifecycle item."""
    codes: list[str] = []
    if item.get("setup_starvation") is not True:
        codes.append("run_setup_starvation_missing")
    if item.get("exact_head_sha_matches") is not True:
        codes.append("run_exact_head_sha_mismatch")
    if str(item.get("phase") or "") != "setup":
        codes.append("run_phase_not_setup")
    if not _run_status_active(item):
        codes.append("run_status_not_active")
    return codes


def _evaluate_guards(
    data: dict[str, Any],
    *,
    pr: int,
    run_id: int,
    expected_head_sha: str,
) -> tuple[list[str], dict[str, Any] | None]:
    """Return fail-closed refusal codes and the matched starvation item."""
    reason_codes: list[str] = []
    if not FULL_SHA_RE.fullmatch(expected_head_sha):
        reason_codes.append("invalid_expected_head_sha")
    if data.get("status") != "ok":
        reason_codes.append("ci_status_not_ok")
        return reason_codes, None

    reason_codes.extend(_identity_guard_codes(data, pr=pr, expected_head_sha=expected_head_sha))

    checks = data.get("checks")
    if not isinstance(checks, dict):
        checks = {}
    if checks.get("setup_starvation") is not True:
        reason_codes.append("setup_starvation_not_reported")

    item = _find_starvation_item(checks, run_id)
    if item is None:
        reason_codes.append("run_not_in_setup_starvation_items")
        return reason_codes, None
    reason_codes.extend(_item_guard_codes(item))
    return reason_codes, item


def _build_receipt(
    args: argparse.Namespace,
    *,
    expected_head_sha: str,
    observed_head_sha: str,
    decision: str,
    reason_codes: list[str],
    mutation: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build a ``ci_setup_recovery_receipt.v1`` document from parsed CLI args."""
    return {
        "schema": RECEIPT_SCHEMA,
        "pr": args.pr,
        "run_id": args.run_id,
        "repo": args.repo,
        "expected_head_sha": expected_head_sha,
        "observed_head_sha": observed_head_sha,
        "decision": decision,
        "reason_codes": reason_codes,
        "mutation": mutation,
        "setup_starvation_evidence": evidence,
        "route_evidence_only": True,
        "implementation_evidence": False,
        "observed_at": _observed_at(),
    }


def _format_human_receipt(receipt: dict[str, Any]) -> str:
    """Format the receipt as a compact human-readable summary."""
    mutation = receipt["mutation"]
    mutation_line = f"  mutation: {mutation['status']}"
    if mutation["status"] != "none":
        command = mutation.get("command") or []
        mutation_line += f" ({' '.join(command)} -> exit {mutation.get('exit_code')})"
    lines = [
        f"CI setup recovery receipt ({receipt['schema']})",
        f"  pr: {receipt['pr']}  run: {receipt['run_id']}  repo: {receipt['repo']}",
        f"  expected head: {receipt['expected_head_sha']}",
        f"  observed head: {receipt['observed_head_sha'] or 'unavailable'}",
        f"  decision: {receipt['decision']}",
        f"  reason_codes: {', '.join(receipt['reason_codes']) or 'none'}",
        mutation_line,
        "  route evidence only: true  |  implementation evidence: false",
    ]
    return "\n".join(lines)


def _emit_receipt(receipt: dict[str, Any], args: argparse.Namespace, exit_code: int) -> int:
    """Write the receipt to ``--output`` when requested and emit it on stdout."""
    text = json.dumps(receipt, indent=2)
    if args.output is not None:
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: could not write receipt to {args.output}: {exc}", file=sys.stderr)
            return 2
    if args.json:
        print(text)
    else:
        print(_format_human_receipt(receipt))
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    """Return the recovery command argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pr", type=int, required=True, help="GitHub PR number")
    parser.add_argument("--run-id", type=int, required=True, help="Actions run ID to recover")
    parser.add_argument(
        "--expected-head-sha",
        required=True,
        help="exact 40-hex PR head SHA the stale run must match",
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help=f"owner/name (default: {DEFAULT_REPO})"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="request exactly one `gh run rerun` after the guarded live re-read",
    )
    parser.add_argument(
        "--reason",
        default="",
        help="operator authorization reason; required with --apply",
    )
    parser.add_argument("--json", action="store_true", default=False, help="emit receipt JSON")
    parser.add_argument("--output", type=Path, default=None, help="also write receipt JSON to PATH")
    parser.add_argument(
        "--actions-stale-after-seconds",
        type=_non_negative_int,
        default=DEFAULT_ACTIONS_STALE_AFTER_SECONDS,
        help="setup-starvation age threshold used by the CI read",
    )
    parser.add_argument(
        "--starvation-seconds",
        type=_non_negative_float,
        default=DEFAULT_QUEUE_STARVATION_SECONDS,
        help="runner queue-starvation threshold used by the CI read",
    )
    return parser


def _read_ci_status(args: argparse.Namespace) -> dict[str, Any]:
    """Read live PR CI status with the configured thresholds."""
    return _fetch_ci_status(
        str(args.pr),
        repo=args.repo,
        actions_stale_after_seconds=args.actions_stale_after_seconds,
        starvation_seconds=args.starvation_seconds,
    )


def _read_checks(data: dict[str, Any]) -> dict[str, Any]:
    """Return the CI payload ``checks`` mapping, or an empty mapping."""
    checks = data.get("checks")
    return checks if isinstance(checks, dict) else {}


def _guarded_rerun_state(
    args: argparse.Namespace,
    *,
    expected_head_sha: str,
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    """Re-read live CI under the caller's lock and evaluate the fail-closed guards.

    Returns ``(refusal_receipt_or_none, observed_head_sha, evidence)``.
    """
    data = _read_ci_status(args)
    checks = _read_checks(data)
    reason_codes, item = _evaluate_guards(
        data,
        pr=args.pr,
        run_id=args.run_id,
        expected_head_sha=expected_head_sha,
    )
    observed_head_sha = str(data.get("head_sha") or "")
    evidence = _setup_starvation_evidence(
        checks,
        item,
        stale_after_seconds=args.actions_stale_after_seconds,
    )
    if not reason_codes:
        return None, observed_head_sha, evidence
    refusal = _build_receipt(
        args,
        expected_head_sha=expected_head_sha,
        observed_head_sha=observed_head_sha,
        decision="refuse",
        reason_codes=reason_codes,
        mutation=_none_mutation(),
        evidence=evidence,
    )
    return refusal, observed_head_sha, evidence


def main(argv: list[str] | None = None) -> int:
    """Run the guarded setup-starvation recovery path."""
    args = build_parser().parse_args(argv)
    expected_head_sha = args.expected_head_sha.strip()

    if args.apply and not args.reason.strip():
        receipt = _build_receipt(
            args,
            expected_head_sha=expected_head_sha,
            observed_head_sha="",
            decision="refuse",
            reason_codes=["apply_reason_missing"],
            mutation=_none_mutation(),
            evidence={"matched": False},
        )
        return _emit_receipt(receipt, args, 1)

    if not FULL_SHA_RE.fullmatch(expected_head_sha):
        receipt = _build_receipt(
            args,
            expected_head_sha=expected_head_sha,
            observed_head_sha="",
            decision="refuse",
            reason_codes=["invalid_expected_head_sha"],
            mutation=_none_mutation(),
            evidence={"matched": False},
        )
        return _emit_receipt(receipt, args, 1)

    try:
        data = _read_ci_status(args)
    except FileNotFoundError:
        print("gh CLI not found. Install GitHub CLI: https://cli.github.com/", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired:
        print("gh CLI command timed out while reading CI status.", file=sys.stderr)
        return 2

    checks = _read_checks(data)
    reason_codes, item = _evaluate_guards(
        data,
        pr=args.pr,
        run_id=args.run_id,
        expected_head_sha=expected_head_sha,
    )
    observed_head_sha = str(data.get("head_sha") or "")
    evidence = _setup_starvation_evidence(
        checks,
        item,
        stale_after_seconds=args.actions_stale_after_seconds,
    )
    if reason_codes:
        receipt = _build_receipt(
            args,
            expected_head_sha=expected_head_sha,
            observed_head_sha=observed_head_sha,
            decision="refuse",
            reason_codes=reason_codes,
            mutation=_none_mutation(),
            evidence=evidence,
        )
        return _emit_receipt(receipt, args, 1)

    if not args.apply:
        receipt = _build_receipt(
            args,
            expected_head_sha=expected_head_sha,
            observed_head_sha=observed_head_sha,
            decision="plan",
            reason_codes=[],
            mutation=_none_mutation(),
            evidence=evidence,
        )
        return _emit_receipt(receipt, args, 0)

    command = ["gh", "run", "rerun", str(args.run_id)]
    try:
        with pr_write_lock(args.repo, args.pr):
            # Re-read under the host-local write lock immediately before mutating.
            refusal, observed_head_sha, evidence = _guarded_rerun_state(
                args,
                expected_head_sha=expected_head_sha,
            )
            if refusal is not None:
                return _emit_receipt(refusal, args, 1)
            try:
                result = _gh(["run", "rerun", str(args.run_id)], timeout=60)
            except (OSError, subprocess.TimeoutExpired) as exc:
                receipt = _build_receipt(
                    args,
                    expected_head_sha=expected_head_sha,
                    observed_head_sha=observed_head_sha,
                    decision="apply",
                    reason_codes=[],
                    mutation=_failed_mutation(command, None, str(exc)),
                    evidence=evidence,
                )
                return _emit_receipt(receipt, args, 2)
            exit_code = int(result.returncode)
            mutation = {
                "status": "rerun_requested" if exit_code == 0 else "failed",
                "command": command,
                "exit_code": exit_code,
                "stderr_excerpt": _bounded_excerpt(result.stderr, limit=STDERR_EXCERPT_LIMIT) or "",
            }
            receipt = _build_receipt(
                args,
                expected_head_sha=expected_head_sha,
                observed_head_sha=observed_head_sha,
                decision="apply",
                reason_codes=[],
                mutation=mutation,
                evidence=evidence,
            )
            return _emit_receipt(receipt, args, 0 if exit_code == 0 else 1)
    except RuntimeError as exc:
        receipt = _build_receipt(
            args,
            expected_head_sha=expected_head_sha,
            observed_head_sha=observed_head_sha,
            decision="refuse",
            reason_codes=["write_lock_unavailable"],
            mutation=_none_mutation(),
            evidence=evidence,
        )
        message = _bounded_excerpt(str(exc), limit=STDERR_EXCERPT_LIMIT)
        receipt["mutation"]["stderr_excerpt"] = message or ""
        return _emit_receipt(receipt, args, 1)


if __name__ == "__main__":
    raise SystemExit(main())
