#!/usr/bin/env python3
"""Exact-diff implementation self-review receipts for autonomous PR handoff.

An implementation worker can finish code and tests without a structured
check that the final diff satisfies the issue, avoided scope creep,
preserved fail-closed behavior, and reported validation honestly. This
module makes that check a mandatory exact-diff gate before autonomous PR
opening or review handoff.

A receipt binds the issue contract digest, one exact base/head pair, the
changed-file inventory with a diff digest, per-question check verdicts,
executed validation with exit statuses, blocking/non-blocking findings, and
the producer identity. ``validate_receipt`` is the offline fixture path;
``verify_receipt_against_git`` re-derives head, paths, and diff digest from
the task worktree. ``handoff_decision`` is the gate consumed by
``goal-issue-implementation`` and ``gh-pr-opener``: it refuses when the
receipt is missing, stale, malformed, bound to another head, claims passing
validation that never ran, or carries unresolved blocking findings.

Self-review is implementation-quality proof only. It never counts as
independent merge-review authority (see issue #8677), merge approval,
benchmark evidence, or a scientific claim. The post-handoff delivery
contract stays on :mod:`scripts.dev.issue_completion_receipt`.

Design note on :mod:`scripts.dev.issue_completion_receipt`: the small
type/shape validators below deliberately mirror that module's fail-closed
style without importing its privates. The two schemas share no field
semantics beyond trivial type checks (different required fields, different
authority boundaries); the one genuinely shared semantic — canonical JSON
digests — is reused via the public
:func:`scripts.dev.goal_autopilot_controller.sha256_json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from scripts.dev.goal_autopilot_controller import sha256_json

SCHEMA = "implementation_self_review.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_FIELDS = (
    "schema",
    "repository",
    "issue",
    "contract",
    "delivery",
    "diff",
    "checks",
    "validation",
    "findings",
    "producer",
    "claim_boundary",
    "receipt_digest",
)

CHECK_IDS = (
    "dod_coverage",
    "scope_discipline",
    "semantic_preservation",
    "fail_closed",
    "unavailable_fail_closed",
    "idempotency",
    "concurrency_cas",
    "base_movement",
    "read_write_race",
    "no_duplicate_owner",
    "no_second_state_machine",
    "hygiene",
    "secrets",
    "validation_executed",
    "todos",
    "claim_boundary",
)

CHECK_VERDICTS = frozenset({"pass", "fail", "na"})
VALIDATION_RESULTS = frozenset({"passed", "failed"})
FINDING_SEVERITIES = frozenset({"blocking", "non_blocking"})

GitRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _is_int(value: object) -> bool:
    """Return whether ``value`` is an integer but not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _mapping(value: object, *, field: str, errors: list[str]) -> Mapping[str, Any] | None:
    """Return ``value`` as a mapping or record a fail-closed error."""
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be an object")
        return None
    return value


def _string(value: object, *, field: str, errors: list[str]) -> str | None:
    """Return ``value`` as a string or record a fail-closed error."""
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _string_list(value: object, *, field: str, errors: list[str]) -> list[str] | None:
    """Return ``value`` as a list of non-empty strings or record an error."""
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        errors.append(f"{field} must be a list of non-empty strings")
        return None
    return list(value)


def compute_receipt_digest(receipt: Mapping[str, Any]) -> str:
    """Return the canonical digest over the receipt payload minus its digest."""
    payload = {key: receipt[key] for key in receipt if key != "receipt_digest"}
    return sha256_json(payload)


def _require_mapping(
    receipt: Mapping[str, Any], field: str, errors: list[str]
) -> Mapping[str, Any] | None:
    """Return a required object field, recording a fail-closed type error.

    A present-but-malformed section must refuse, never silently skip its
    checks: skipping would let a malformed receipt authorize handoff.
    """
    value = receipt.get(field)
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be an object")
        return None
    return value


def _validate_contract(contract: Mapping[str, Any] | None, errors: list[str]) -> str | None:
    """Validate the issue-contract binding and return its digest."""
    if contract is None:
        return None
    digest = contract.get("digest")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        errors.append("contract.digest must be a lowercase SHA-256 digest")
        return None
    _string(contract.get("source"), field="contract.source", errors=errors)
    return digest


def _validate_delivery(
    delivery: Mapping[str, Any] | None,
    *,
    expected_base_sha: str | None,
    expected_head_sha: str | None,
    expected_branch: str | None,
    errors: list[str],
) -> tuple[str | None, str | None]:
    """Validate the exact base/head binding and return both SHAs."""
    if delivery is None:
        return None, None
    _string(delivery.get("base_ref"), field="delivery.base_ref", errors=errors)
    branch = _string(delivery.get("branch"), field="delivery.branch", errors=errors)
    base_sha = delivery.get("base_sha")
    head_sha = delivery.get("head_sha")
    if not isinstance(base_sha, str) or SHA_RE.fullmatch(base_sha) is None:
        errors.append("delivery.base_sha must be a full Git SHA")
        base_sha = None
    if not isinstance(head_sha, str) or SHA_RE.fullmatch(head_sha) is None:
        errors.append("delivery.head_sha must be a full Git SHA")
        head_sha = None
    if expected_base_sha is not None and base_sha != expected_base_sha:
        errors.append("delivery.base_sha does not match the expected base")
    if expected_head_sha is not None and head_sha != expected_head_sha:
        errors.append("delivery.head_sha does not match the expected head")
    if expected_branch is not None and branch != expected_branch:
        errors.append("delivery.branch does not match the expected branch")
    worktree = delivery.get("worktree")
    if worktree is not None and (not isinstance(worktree, str) or not worktree.strip()):
        errors.append("delivery.worktree must be a non-empty string when present")
    return base_sha, head_sha


def _validate_diff(diff: Mapping[str, Any] | None, errors: list[str]) -> str | None:
    """Validate the changed-file inventory and return the diff digest."""
    if diff is None:
        return None
    paths = diff.get("changed_paths")
    if not isinstance(paths, list) or not paths or any(_diff_path_error(path) for path in paths):
        errors.append("diff.changed_paths must be a non-empty list of repository-relative paths")
    elif len(set(paths)) != len(paths):
        errors.append("diff.changed_paths must not contain duplicates")
    digest = diff.get("diff_digest")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        errors.append("diff.diff_digest must be a lowercase SHA-256 digest")
        return None
    stat = diff.get("stat")
    if stat is not None:
        if not isinstance(stat, Mapping):
            errors.append("diff.stat must be an object when present")
        else:
            for field in ("files", "additions", "deletions"):
                if field in stat and (not _is_int(stat[field]) or int(stat[field]) < 0):
                    errors.append(f"diff.stat.{field} must be a non-negative integer")
    return digest


def _diff_path_error(path: object) -> bool:
    """Return whether a diff path entry is invalid."""
    return (
        not isinstance(path, str)
        or not path.strip()
        or path.startswith("/")
        or ".." in path.split("/")
        or "\x00" in path
    )


def _validate_checks(checks: object, errors: list[str]) -> tuple[list[dict[str, Any]], bool]:
    """Validate per-question verdicts; returns rows and whether any failed."""
    if not isinstance(checks, list) or not checks:
        errors.append("checks must be a non-empty list")
        return [], False
    seen: set[str] = set()
    failed = False
    for index, check in enumerate(checks):
        where = f"checks[{index}]"
        if not isinstance(check, Mapping):
            errors.append(f"{where} must be an object")
            continue
        check_id = check.get("id")
        if check_id not in CHECK_IDS:
            errors.append(f"{where}.id must be one of {sorted(CHECK_IDS)}")
            continue
        if check_id in seen:
            errors.append(f"{where}.id duplicates {check_id!r}")
            continue
        seen.add(str(check_id))
        verdict = check.get("verdict")
        if verdict not in CHECK_VERDICTS:
            errors.append(f"{where}.verdict must be one of {sorted(CHECK_VERDICTS)}")
            continue
        if verdict == "fail":
            failed = True
        evidence = check.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append(f"{where}.evidence must be a non-empty string")
    missing = [check_id for check_id in CHECK_IDS if check_id not in seen]
    if missing:
        errors.append(f"checks missing required ids: {sorted(missing)}")
    return [check for check in checks if isinstance(check, Mapping)], failed


def _validate_validation(validation: object, errors: list[str]) -> bool:
    """Validate executed validation; passing claims require exit zero."""
    if not isinstance(validation, list) or not validation:
        errors.append("validation must be a non-empty list")
        return False
    claimed_pass = False
    for index, record in enumerate(validation):
        where = f"validation[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{where} must be an object")
            continue
        _string(record.get("command"), field=f"{where}.command", errors=errors)
        exit_code = record.get("exit_code")
        if not _is_int(exit_code):
            errors.append(f"{where}.exit_code must be an integer")
            continue
        result = record.get("result")
        if result not in VALIDATION_RESULTS:
            errors.append(f"{where}.result must be one of {sorted(VALIDATION_RESULTS)}")
            continue
        if result == "passed" and exit_code != 0:
            errors.append(f"{where} claims passed with non-zero exit {exit_code}")
            continue
        if result == "passed":
            claimed_pass = True
    return claimed_pass


def _validate_findings(findings: Mapping[str, Any] | None, errors: list[str]) -> bool:
    """Validate finding lists; returns True when blocking findings exist."""
    if findings is None:
        return False
    blocking = findings.get("blocking")
    non_blocking = findings.get("non_blocking")
    has_blocking = False
    for field, values in (("blocking", blocking), ("non_blocking", non_blocking)):
        items = _string_list(values, field=f"findings.{field}", errors=errors)
        if items is None:
            continue
        if field == "blocking" and items:
            has_blocking = True
    return has_blocking


def validate_receipt(  # noqa: C901, PLR0912 - schema gate validates every field fail-closed
    receipt: Mapping[str, Any] | object,
    *,
    expected_repository: str | None = None,
    expected_issue: int | None = None,
    expected_base_sha: str | None = None,
    expected_head_sha: str | None = None,
    expected_branch: str | None = None,
    issue_contract: str | None = None,
) -> dict[str, Any]:
    """Validate a self-review receipt offline and report fail-closed errors."""
    if not isinstance(receipt, Mapping):
        return {"schema": SCHEMA, "ok": False, "errors": ["receipt must be an object"]}
    errors: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in receipt]
    errors.extend(f"missing required field: {field}" for field in missing)
    if receipt.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    repository = receipt.get("repository")
    if not isinstance(repository, str) or not re.fullmatch(r"[^/\s]+/[^/\s]+", repository):
        errors.append("repository must be an OWNER/REPO string")
    elif expected_repository is not None and repository != expected_repository:
        errors.append(f"repository {repository} does not match expected {expected_repository}")
    issue = receipt.get("issue")
    if not _is_int(issue) or int(issue) <= 0:
        errors.append("issue must be a positive integer")
    elif expected_issue is not None and issue != expected_issue:
        errors.append(f"issue {issue} does not match expected issue {expected_issue}")
    contract_digest = _validate_contract(
        _require_mapping(receipt, "contract", errors),
        errors,
    )
    if issue_contract is not None and contract_digest is not None:
        if hashlib.sha256(issue_contract.encode("utf-8")).hexdigest() != contract_digest:
            errors.append("contract digest does not match the supplied issue body text")
    _validate_delivery(
        _require_mapping(receipt, "delivery", errors),
        expected_base_sha=expected_base_sha,
        expected_head_sha=expected_head_sha,
        expected_branch=expected_branch,
        errors=errors,
    )
    _validate_diff(
        _require_mapping(receipt, "diff", errors),
        errors,
    )
    _, checks_failed = _validate_checks(receipt.get("checks"), errors)
    claimed_pass = _validate_validation(receipt.get("validation"), errors)
    has_blocking = _validate_findings(
        _require_mapping(receipt, "findings", errors),
        errors,
    )
    producer = receipt.get("producer")
    if isinstance(producer, Mapping):
        _string(producer.get("identity"), field="producer.identity", errors=errors)
    claim_boundary = receipt.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        errors.append("claim_boundary must be a non-empty string")
    recorded_digest = receipt.get("receipt_digest")
    if not isinstance(recorded_digest, str) or SHA256_RE.fullmatch(recorded_digest) is None:
        errors.append("receipt_digest must be a lowercase SHA-256 digest")
    else:
        try:
            observed_digest = compute_receipt_digest(receipt)
        except (TypeError, ValueError) as exc:
            errors.append(f"receipt digest cannot be recomputed: {exc}")
        else:
            if recorded_digest != observed_digest:
                errors.append("receipt_digest does not match the canonical receipt payload")
    if checks_failed:
        errors.append("one or more checks carry a fail verdict; return to implementation")
    if not claimed_pass:
        errors.append("no validation record claims passed; missing proof cannot be passed")
    if has_blocking:
        errors.append("blocking findings are unresolved; return to implementation")
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "errors": errors,
        "issue": receipt.get("issue"),
        "base_sha": (receipt.get("delivery") or {}).get("base_sha")
        if isinstance(receipt.get("delivery"), Mapping)
        else None,
        "head_sha": (receipt.get("delivery") or {}).get("head_sha")
        if isinstance(receipt.get("delivery"), Mapping)
        else None,
    }


def build_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a self-digested receipt from an offline declaration."""
    result = dict(payload)
    result["schema"] = SCHEMA
    result.pop("receipt_digest", None)
    result["receipt_digest"] = compute_receipt_digest(result)
    validation = validate_receipt(result)
    if not validation["ok"]:
        raise ValueError("invalid receipt declaration: " + "; ".join(validation["errors"]))
    return result


def _default_git_runner(command: list[str], *, worktree: str) -> subprocess.CompletedProcess[str]:
    """Run one Git command inside the task worktree without a shell."""
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            cwd=worktree,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{' '.join(command)}: timed out") from exc


def collect_git_evidence(
    *,
    worktree: str,
    base_ref: str = "origin/main",
    git_runner: GitRunner | None = None,
) -> dict[str, Any]:
    """Collect exact head, changed paths, stat, and diff digest from a worktree.

    The runner is injectable so the acceptance harness can drive the same
    evidence path deterministically. ``git_runner`` receives the full command
    and runs it in the worktree; the default runner executes real Git there.
    """
    runner = git_runner or (lambda command: _default_git_runner(command, worktree=worktree))

    def _run(command: list[str]) -> str:
        result = runner(command)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip() or "command failed"
            raise RuntimeError(f"{' '.join(command)}: {detail}")
        return result.stdout

    head_sha = _run(["git", "rev-parse", "HEAD"]).strip()
    if SHA_RE.fullmatch(head_sha) is None:
        raise RuntimeError("worktree HEAD is not a full Git SHA")
    base_sha = _run(["git", "rev-parse", "--verify", f"{base_ref}^{{commit}}"]).strip()
    if SHA_RE.fullmatch(base_sha) is None:
        raise RuntimeError("base ref did not resolve to a full Git SHA")
    branch = _run(["git", "branch", "--show-current"]).strip()
    raw_paths = _run(["git", "diff", "--name-only", f"{base_sha}...{head_sha}"]).strip()
    changed_paths = sorted(path for path in raw_paths.splitlines() if path.strip())
    if not changed_paths:
        raise RuntimeError("worktree carries no diff against the base ref")
    diff_text = _run(["git", "diff", f"{base_sha}...{head_sha}"])
    diff_digest = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
    return {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "branch": branch,
        "changed_paths": changed_paths,
        "diff_digest": diff_digest,
    }


def verify_receipt_against_git(
    receipt: Mapping[str, Any],
    *,
    worktree: str,
    base_ref: str = "origin/main",
    git_runner: GitRunner | None = None,
) -> dict[str, Any]:
    """Verify a receipt against live worktree Git state, failing closed on drift."""
    errors: list[str] = []
    try:
        evidence = collect_git_evidence(worktree=worktree, base_ref=base_ref, git_runner=git_runner)
    except (RuntimeError, ValueError, OSError) as exc:
        return {"schema": SCHEMA, "ok": False, "errors": [f"git evidence failed: {exc}"]}
    delivery = receipt.get("delivery") if isinstance(receipt.get("delivery"), Mapping) else {}
    diff = receipt.get("diff") if isinstance(receipt.get("diff"), Mapping) else {}
    if not isinstance(delivery, Mapping) or not isinstance(diff, Mapping):
        return {"schema": SCHEMA, "ok": False, "errors": ["receipt delivery/diff unreadable"]}
    for field in ("base_sha", "head_sha", "branch"):
        observed = evidence[field]
        recorded = delivery.get(field)
        if recorded != observed:
            errors.append(f"delivery.{field} drifted: receipt has {recorded!r}")
    recorded_paths = diff.get("changed_paths")
    if not isinstance(recorded_paths, list) or sorted(recorded_paths) != evidence["changed_paths"]:
        errors.append("diff.changed_paths drifted from the live worktree diff")
    if diff.get("diff_digest") != evidence["diff_digest"]:
        errors.append("diff.diff_digest drifted from the live worktree diff")
    validation = validate_receipt(
        receipt,
        expected_base_sha=evidence["base_sha"],
        expected_head_sha=evidence["head_sha"],
    )
    errors.extend(validation["errors"])
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "errors": errors,
        "head_sha": evidence["head_sha"],
    }


def handoff_decision(
    receipt: Mapping[str, Any] | None,
    *,
    expected_issue: int,
    expected_base_sha: str,
    expected_head_sha: str,
    expected_branch: str | None = None,
    issue_contract: str | None = None,
) -> dict[str, Any]:
    """Decide whether an autonomous PR handoff may proceed.

    Returns ``{"ok": True}`` only for a valid receipt bound to the exact
    expected head with passing checks, executed validation, and no blocking
    findings. Every other case refuses with stable reason codes.
    """
    if receipt is None:
        return {
            "ok": False,
            "reasons": ["missing_receipt: no implementation self-review was produced"],
        }
    validation = validate_receipt(
        receipt,
        expected_issue=expected_issue,
        expected_base_sha=expected_base_sha,
        expected_head_sha=expected_head_sha,
        expected_branch=expected_branch,
        issue_contract=issue_contract,
    )
    if validation["ok"]:
        return {"ok": True, "reasons": []}
    return {"ok": False, "reasons": validation["errors"]}


def _load_json_file(path: str) -> Any:
    """Load one JSON document or raise a fail-closed error."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON file {path}: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository as OWNER/REPO.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="digest and validate an offline declaration")
    build.add_argument("--payload-file", required=True, help="JSON receipt declaration.")
    build.add_argument("--output", default=None, help="Write the digested receipt here.")
    validate = subparsers.add_parser("validate", help="validate a receipt file offline")
    validate.add_argument("--receipt-file", required=True, help="JSON receipt to validate.")
    validate.add_argument("--issue-body-file", default=None, help="Issue body for digest check.")
    verify = subparsers.add_parser("verify", help="verify a receipt against worktree Git")
    verify.add_argument("--receipt-file", required=True, help="JSON receipt to verify.")
    verify.add_argument("--worktree", required=True, help="Task worktree directory.")
    verify.add_argument("--base-ref", default="origin/main", help="Base ref for the diff.")
    collect = subparsers.add_parser("collect", help="print Git evidence for receipt construction")
    collect.add_argument("--worktree", required=True, help="Task worktree directory.")
    collect.add_argument("--base-ref", default="origin/main", help="Base ref for the diff.")
    gate = subparsers.add_parser("gate", help="decide whether PR handoff may proceed")
    gate.add_argument("--receipt-file", required=True, help="JSON receipt to gate on.")
    gate.add_argument("--issue", type=int, required=True, help="Expected issue number.")
    gate.add_argument("--expected-head-sha", required=True, help="Expected exact head SHA.")
    gate.add_argument("--expected-base-sha", required=True, help="Expected exact base SHA.")
    gate.add_argument("--expected-branch", default=None, help="Expected branch name.")
    gate.add_argument(
        "--issue-body-file",
        default=None,
        help="Issue body text for contract-digest binding; recommended when at hand.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run receipt build, validation, verification, collection, or gating."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build":
            receipt = build_receipt(_load_json_file(args.payload_file))
            serialized = json.dumps(receipt, indent=2, sort_keys=True)
            if args.output is not None:
                Path(args.output).write_text(serialized + "\n", encoding="utf-8")
            else:
                print(serialized)
            return 0
        if args.command == "validate":
            receipt = _load_json_file(args.receipt_file)
            contract = (
                Path(args.issue_body_file).read_text(encoding="utf-8")
                if args.issue_body_file is not None
                else None
            )
            result = validate_receipt(receipt, issue_contract=contract)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ok"] else 1
        if args.command == "verify":
            receipt = _load_json_file(args.receipt_file)
            result = verify_receipt_against_git(
                receipt, worktree=args.worktree, base_ref=args.base_ref
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ok"] else 1
        if args.command == "collect":
            evidence = collect_git_evidence(worktree=args.worktree, base_ref=args.base_ref)
            print(json.dumps(evidence, indent=2, sort_keys=True))
            return 0
        receipt = _load_json_file(args.receipt_file)
        contract = None
        if getattr(args, "issue_body_file", None) is not None:
            try:
                contract = Path(args.issue_body_file).read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                print(f"implementation self-review blocked: {exc}", file=sys.stderr)
                return 2
        decision = handoff_decision(
            receipt,
            expected_issue=args.issue,
            expected_base_sha=args.expected_base_sha,
            expected_head_sha=args.expected_head_sha,
            expected_branch=args.expected_branch,
            issue_contract=contract,
        )
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["ok"] else 2
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"implementation self-review blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
