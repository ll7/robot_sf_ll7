#!/usr/bin/env python3
"""Build and compare deterministic goal-autopilot blocker receipts.

The receipt is a compact routing contract.  It records why work is blocked and binds the
blocker to the exact issue, repository, base/head, dependency, and required-input values
that were inspected.  It never mutates GitHub state or authorizes a retry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.dev.git_common import resolve_agent_artifact_dir

SCHEMA_VERSION = "goal_blocker_receipt.v1"
FINGERPRINT_SCHEMA_VERSION = "goal_blocker_fingerprint.v1"
BLOCKER_CLASSES = frozenset(
    {
        "needs_spec",
        "human_decision",
        "external_input",
        "compute",
        "environment",
        "dependency",
        "validation",
        "authority",
    }
)
_SHA256_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_UNAVAILABLE_STATE = "$state"
_UNAVAILABLE_VALUE = "unavailable"
_MISSING_VALUE = "missing"
_RECEIPT_DIGEST = "receipt_digest"
DEFAULT_RECEIPT_SUBDIR = "goal-blocker-receipts"


def unavailable(reason: str = "not_observed") -> dict[str, str]:
    """Return an explicit unavailable value for receipt inputs."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("unavailable reason must be a non-empty string")
    return {_UNAVAILABLE_STATE: _UNAVAILABLE_VALUE, "reason": reason.strip()}


def _canonical_value(value: Any) -> Any:
    """Normalize JSON-like values while preserving missing, empty, false, and unavailable."""
    if value is None:
        return {_UNAVAILABLE_STATE: _MISSING_VALUE}
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("receipt mapping keys must be non-empty strings")
            normalized[key] = _canonical_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("receipt floats must be finite")
        return value
    raise ValueError(f"receipt value is not JSON-compatible: {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    """Serialize a normalized value deterministically."""
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def blocker_fingerprint(inputs: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 fingerprint for the inspected blocker inputs."""
    if not isinstance(inputs, Mapping):
        raise ValueError("blocker fingerprint inputs must be a mapping")
    payload = {
        "schema": FINGERPRINT_SCHEMA_VERSION,
        "inputs": _canonical_value(inputs),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    """Hash a receipt without its self-referential digest field."""
    payload = {key: value for key, value in receipt.items() if key != _RECEIPT_DIGEST}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_receipt(  # noqa: PLR0913 - the versioned contract names each receipt field explicitly.
    *,
    repository: str,
    issue: int,
    issue_revision: Any,
    origin_main_sha: Any,
    blocker_class: str,
    required_transition: str,
    evidence: Sequence[Mapping[str, Any]],
    safe_work: Sequence[str],
    next_owner: str,
    recommended_state: str,
    retryable: bool,
    invalidating_fields: Sequence[str],
    fingerprint_inputs: Mapping[str, Any],
    pr_head_sha: Any = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated, self-digesting ``goal_blocker_receipt.v1`` object."""
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "status": "blocked",
        "repository": repository,
        "issue": issue,
        "issue_revision": unavailable("issue revision not observed")
        if issue_revision is None
        else issue_revision,
        "origin_main_sha": unavailable("origin main SHA not observed")
        if origin_main_sha is None
        else origin_main_sha,
        "pr_head_sha": unavailable("no PR head exists") if pr_head_sha is None else pr_head_sha,
        "blocker_class": blocker_class,
        "required_transition": required_transition,
        "evidence": list(evidence),
        "safe_work": list(safe_work),
        "next_owner": next_owner,
        "recommended_state": recommended_state,
        "retryable": retryable,
        "invalidating_fields": list(invalidating_fields),
        "fingerprint_inputs": dict(fingerprint_inputs),
        "fingerprint": blocker_fingerprint(fingerprint_inputs),
        "observed_at": observed_at or _now_utc(),
    }
    report = validate_receipt({**receipt, _RECEIPT_DIGEST: "0" * 64}, check_digest=False)
    if not report["valid"]:
        raise ValueError("invalid blocker receipt: " + "; ".join(report["errors"]))
    receipt[_RECEIPT_DIGEST] = _receipt_digest(receipt)
    return receipt


def receipt_artifact_path(issue: int) -> Path:
    """Return the canonical private active-artifact path for one issue receipt."""
    if not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0:
        raise ValueError("issue must be a positive integer")
    return resolve_agent_artifact_dir(DEFAULT_RECEIPT_SUBDIR) / f"issue-{issue}.json"


def write_receipt(receipt: Mapping[str, Any], path: str | Path | None = None) -> Path:
    """Atomically store a validated receipt outside the repository worktree."""
    report = validate_receipt(receipt)
    if not report["valid"]:
        raise ValueError("invalid blocker receipt: " + "; ".join(report["errors"]))
    target = Path(path) if path is not None else receipt_artifact_path(receipt["issue"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(receipt), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return target


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _is_explicit_unavailable(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get(_UNAVAILABLE_STATE) == _UNAVAILABLE_VALUE
        and isinstance(value.get("reason"), str)
        and bool(value["reason"].strip())
    )


def _validate_sequence(value: Any, *, field: str, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")


def validate_receipt(  # noqa: C901, PLR0912, PLR0915 - ordered fail-closed field validation.
    receipt: Any, *, check_digest: bool = True
) -> dict[str, Any]:
    """Validate a receipt and return structured errors instead of raising."""
    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return {"valid": False, "errors": ["receipt must be an object"]}

    required = {
        "schema",
        "status",
        "repository",
        "issue",
        "issue_revision",
        "origin_main_sha",
        "pr_head_sha",
        "blocker_class",
        "required_transition",
        "evidence",
        "safe_work",
        "next_owner",
        "recommended_state",
        "retryable",
        "invalidating_fields",
        "fingerprint_inputs",
        "fingerprint",
        "observed_at",
        _RECEIPT_DIGEST,
    }
    missing = sorted(required.difference(receipt))
    errors.extend(f"missing field: {field}" for field in missing)
    if errors:
        return {"valid": False, "errors": errors}

    if receipt["schema"] != SCHEMA_VERSION:
        errors.append(f"schema must be {SCHEMA_VERSION!r}")
    if receipt["status"] != "blocked":
        errors.append("status must be 'blocked'")
    if not isinstance(receipt["repository"], str) or not receipt["repository"].strip():
        errors.append("repository must be a non-empty string")
    if (
        not isinstance(receipt["issue"], int)
        or isinstance(receipt["issue"], bool)
        or receipt["issue"] <= 0
    ):
        errors.append("issue must be a positive integer")
    if not (
        isinstance(receipt["issue_revision"], str)
        or _is_explicit_unavailable(receipt["issue_revision"])
    ):
        errors.append("issue_revision must be a string or explicit unavailable object")
    if not isinstance(receipt["origin_main_sha"], (str, Mapping)):
        errors.append("origin_main_sha must be a full SHA or explicit unavailable object")
    if not (
        _is_sha(receipt["origin_main_sha"]) or _is_explicit_unavailable(receipt["origin_main_sha"])
    ):
        errors.append("origin_main_sha must be a full SHA or explicit unavailable object")
    if not (_is_sha(receipt["pr_head_sha"]) or _is_explicit_unavailable(receipt["pr_head_sha"])):
        errors.append("pr_head_sha must be a full SHA or explicit unavailable object")
    if not isinstance(receipt["blocker_class"], str):
        errors.append("blocker_class must be a string")
    elif receipt["blocker_class"] not in BLOCKER_CLASSES:
        errors.append(f"unknown blocker_class: {receipt['blocker_class']!r}")
    for field in ("required_transition", "next_owner", "recommended_state", "observed_at"):
        if not isinstance(receipt[field], str) or not receipt[field].strip():
            errors.append(f"{field} must be a non-empty string")
    if not isinstance(receipt["retryable"], bool):
        errors.append("retryable must be boolean")
    _validate_sequence(receipt["evidence"], field="evidence", errors=errors)
    _validate_sequence(receipt["safe_work"], field="safe_work", errors=errors)
    _validate_sequence(receipt["invalidating_fields"], field="invalidating_fields", errors=errors)
    if isinstance(receipt["safe_work"], list) and not all(
        isinstance(item, str) for item in receipt["safe_work"]
    ):
        errors.append("safe_work entries must be strings")
    if isinstance(receipt["evidence"], list) and not all(
        isinstance(item, Mapping) for item in receipt["evidence"]
    ):
        errors.append("evidence entries must be objects")
    if isinstance(receipt["invalidating_fields"], list) and (
        not receipt["invalidating_fields"]
        or not all(isinstance(item, str) and item for item in receipt["invalidating_fields"])
    ):
        errors.append("invalidating_fields must contain non-empty strings")
    if not isinstance(receipt["fingerprint_inputs"], Mapping):
        errors.append("fingerprint_inputs must be an object")
    if not _is_digest(receipt["fingerprint"]):
        errors.append("fingerprint must be a full SHA-256 digest")
    elif isinstance(receipt["fingerprint_inputs"], Mapping):
        try:
            expected = blocker_fingerprint(receipt["fingerprint_inputs"])
        except ValueError as exc:
            errors.append(f"fingerprint_inputs invalid: {exc}")
        else:
            if receipt["fingerprint"] != expected:
                errors.append("fingerprint does not match fingerprint_inputs")
    if not _is_digest(receipt[_RECEIPT_DIGEST]):
        errors.append("receipt_digest must be a full SHA-256 digest")
    elif check_digest:
        try:
            digest = _receipt_digest(receipt)
        except ValueError as exc:
            errors.append(f"receipt contains invalid JSON value: {exc}")
        else:
            if receipt[_RECEIPT_DIGEST] != digest:
                errors.append("receipt_digest does not match receipt contents")

    return {
        "valid": not errors,
        "errors": errors,
        "schema": receipt.get("schema"),
        "fingerprint": receipt.get("fingerprint"),
        "receipt_digest": receipt.get(_RECEIPT_DIGEST),
    }


def _changed_paths(previous: Any, current: Any, *, prefix: str = "") -> list[str]:
    if isinstance(previous, Mapping) and isinstance(current, Mapping):
        paths: list[str] = []
        for key in sorted(set(previous) | set(current)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in previous or key not in current:
                paths.append(child)
            else:
                paths.extend(_changed_paths(previous[key], current[key], prefix=child))
        return paths
    if isinstance(previous, list) and isinstance(current, list):
        paths = []
        for index in range(max(len(previous), len(current))):
            child = f"{prefix}[{index}]"
            if index >= len(previous) or index >= len(current):
                paths.append(child)
            else:
                paths.extend(_changed_paths(previous[index], current[index], prefix=child))
        return paths
    return [] if previous == current else [prefix or "$"]


def _decision_identity(receipt: Any) -> dict[str, Any]:
    """Carry known receipt identity into a queue-consumable decision artifact."""
    if not isinstance(receipt, Mapping):
        return {}
    return {
        field: receipt[field]
        for field in ("repository", "issue", _RECEIPT_DIGEST)
        if field in receipt
    }


def compare_blocker_inputs(current_inputs: Mapping[str, Any], prior_receipt: Any) -> dict[str, Any]:
    """Compare current blocker inputs with a receipt without performing any writes."""
    validation = validate_receipt(prior_receipt)
    identity = _decision_identity(prior_receipt)
    try:
        current_fingerprint = blocker_fingerprint(current_inputs)
    except ValueError as exc:
        return {
            **identity,
            "status": "re_evaluate",
            "reason": "invalid_current_inputs",
            "errors": [str(exc)],
        }
    if not validation["valid"]:
        return {
            **identity,
            "status": "re_evaluate",
            "reason": "invalid_or_stale_receipt",
            "errors": validation["errors"],
            "current_fingerprint": current_fingerprint,
        }

    previous_inputs = _canonical_value(prior_receipt["fingerprint_inputs"])
    current_inputs_canonical = _canonical_value(current_inputs)
    changed_fields = _changed_paths(previous_inputs, current_inputs_canonical)
    if not changed_fields and current_fingerprint == prior_receipt["fingerprint"]:
        return {
            **identity,
            "status": "blocked_unchanged",
            "reason": "blocker_fingerprint_unchanged",
            "current_fingerprint": current_fingerprint,
            "next_owner": prior_receipt["next_owner"],
            "required_transition": prior_receipt["required_transition"],
            "retryable": prior_receipt["retryable"],
            "blocker_class": prior_receipt["blocker_class"],
            "changed_fields": [],
        }
    invalidating_changes = [
        invalidator
        for invalidator in prior_receipt["invalidating_fields"]
        if any(
            field == invalidator
            or field.startswith(f"{invalidator}.")
            or field.startswith(f"{invalidator}[")
            for field in changed_fields
        )
    ]
    return {
        **identity,
        "status": "blocker_changed",
        "reason": "fingerprint_changed",
        "current_fingerprint": current_fingerprint,
        "previous_fingerprint": prior_receipt["fingerprint"],
        "changed_fields": changed_fields,
        "invalidating_fields_changed": invalidating_changes,
        "re_evaluate": True,
        "blocker_class": prior_receipt["blocker_class"],
    }


def summarize_decisions(decisions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize blocker decisions for the goal-loop state snapshot."""
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    classes: dict[str, int] = {}
    for decision in decisions:
        status = str(decision.get("status", "invalid"))
        counts[status] = counts.get(status, 0) + 1
        reason = str(decision.get("reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
        blocker_class = decision.get("blocker_class")
        if isinstance(blocker_class, str):
            classes[blocker_class] = classes.get(blocker_class, 0) + 1
    return {
        "schema": "goal_blocker_summary.v1",
        "decision_count": sum(counts.values()),
        "suppressed_redispatch_count": counts.get("blocked_unchanged", 0),
        "re_evaluation_count": counts.get("blocker_changed", 0) + counts.get("re_evaluate", 0),
        "by_status": dict(sorted(counts.items())),
        "by_reason": dict(sorted(reasons.items())),
        "by_blocker_class": dict(sorted(classes.items())),
    }


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_receipt(path: str | Path) -> dict[str, Any]:
    """Load and validate a stored receipt, refusing malformed artifacts."""
    payload = _load_json(str(path))
    if not isinstance(payload, dict):
        raise ValueError("blocker receipt must be a JSON object")
    report = validate_receipt(payload)
    if not report["valid"]:
        raise ValueError("invalid blocker receipt: " + "; ".join(report["errors"]))
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fingerprint_parser = subparsers.add_parser("fingerprint")
    fingerprint_parser.add_argument("inputs", type=Path)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("receipt", type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--inputs", required=True, type=Path)
    compare_parser.add_argument("--receipt", required=True, type=Path)
    write_parser = subparsers.add_parser("write")
    write_parser.add_argument("receipt", type=Path)
    write_parser.add_argument(
        "--path",
        type=Path,
        help="explicit private artifact path; defaults to the common-Git active owner",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the side-effect-free receipt CLI."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "fingerprint":
            payload = {"fingerprint": blocker_fingerprint(_load_json(str(args.inputs)))}
        elif args.command == "validate":
            report = validate_receipt(_load_json(str(args.receipt)))
            payload = report
        elif args.command == "write":
            path = write_receipt(_load_json(str(args.receipt)), args.path)
            payload = {"status": "stored", "path": str(path)}
        else:
            payload = compare_blocker_inputs(
                _load_json(str(args.inputs)),
                _load_json(str(args.receipt)),
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    requires_re_evaluation = payload.get("status") == "re_evaluate" or payload.get(
        "re_evaluate", False
    )
    return 0 if payload.get("valid", True) and not requires_re_evaluation else 2


if __name__ == "__main__":
    raise SystemExit(main())
