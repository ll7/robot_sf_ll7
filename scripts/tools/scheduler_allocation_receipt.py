#!/usr/bin/env python3
"""Sanitized scheduler/allocation receipt converter and validator (issue #8916).

Public receipt ``robot_sf.scheduler_allocation_receipt.v1``: versioned, deterministic,
topology-free custody record of one scheduled allocation from submission through
terminal state, projected from an explicit private JSON record (never scraped
``sacct``/``squeue``/spool/launcher text).

States: submission_unacknowledged, pending, running, completed, failed, timeout,
cancelled; legal transitions are validated over ``state_history`` and terminal states
have no successors. Unproven values stay explicit as unavailable/not_observed/
not_applicable/redacted. Only whitelisted fields are copied; public strings are
scanned for paths, URLs, topology, credentials, signed parameters, and private-context
overlap. Scheduler completion is custody metadata only, never artifact or scientific
success. CLI: ``validate --receipt <json> [--json]``; ``project --private-input <json>
--output <json> [--markdown-output <md>]`` (project also writes ``<output>.md``).
Exit codes: 0 success, 2 fail.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "robot_sf.scheduler_allocation_receipt.v1"
PRIVATE_INPUT_SCHEMA = "robot_sf.scheduler_allocation_receipt.private_input.v1"
REPORT_SCHEMA = "robot_sf.scheduler_allocation_receipt.validation.v1"
CLAIM_BOUNDARY = (
    "Scheduler and allocation custody only; scheduler state never implies artifact, "
    "benchmark, or scientific success, and statuses stay separate."
)
CLAIM_NOTE = CLAIM_BOUNDARY
EXPLICIT_STATES = ("unavailable", "not_observed", "not_applicable", "redacted")
SCHEDULER_STATES = tuple(
    "submission_unacknowledged pending running completed failed timeout cancelled".split()
)
TRANSITIONS = {
    "submission_unacknowledged": frozenset(SCHEDULER_STATES[1:]),
    "pending": frozenset(SCHEDULER_STATES[2:]),
    "running": frozenset(SCHEDULER_STATES[3:]),
}
TERMINATIONS_BY_STATE = {
    "completed": frozenset({"normal"}),
    "failed": frozenset({"failed"}),
    "timeout": frozenset({"time_limit"}),
    "cancelled": frozenset({"cancelled", "preempted", "scheduler_terminated"}),
}
TERMINATION_CLASSES = frozenset(
    "normal failed time_limit cancelled preempted scheduler_terminated node_failure".split()
)
RESOURCE_CLASSES = tuple("cpu_class memory_class accelerator_class".split())
NON_TERMINAL_TAIL_FIELDS = tuple(
    "ended_at exit_code derived_exit_code signal termination_class".split()
)
_FORBIDDEN_FIELDS = frozenset(
    "account command command_line cmdline env environment host hostname node node_list nodelist "
    "nodes partition password private_path qos scheduler_account secret secrets slurm_account "
    "token url user user_name username".split()
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_CLASS_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_LOGICAL_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_SECRET_RE = re.compile(r"(?i)(?:api[_-]?key|secret|password|passwd|token|credential|bearer)")
_SIGNED_QUERY_RE = re.compile(r"(?i)[?&](?:sig|signature|token|expires|x-amz-|x-goog-)")


@dataclass(frozen=True, slots=True)
class ReceiptIssue:
    """One sanitized fail-closed issue, free of private values."""

    code: str
    location: str
    message: str


class ReceiptError(ValueError):
    """Fail-closed conversion or validation failure carrying sanitized issues."""

    def __init__(self, issues: Sequence[ReceiptIssue]) -> None:
        """Initialize the error with the sanitized issue sequence."""

        super().__init__("scheduler allocation receipt failed closed")
        self.issues: tuple[ReceiptIssue, ...] = tuple(issues)


def _issue(code: str, location: str, message: str) -> ReceiptIssue:
    return ReceiptIssue(code=code, location=location, message=message)


def _is_explicit(value: Any) -> bool:
    return isinstance(value, str) and value in EXPLICIT_STATES


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or _is_explicit(value):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _has_parent_segment(text: str) -> bool:
    return ".." in text.split("/")


def _matches(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _scan_tree(value: Any, location: str, issues: list[ReceiptIssue]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{location}/{key}"
            if str(key).lower() in _FORBIDDEN_FIELDS:
                issues.append(_issue("forbidden_field", child, "private topology field"))
            _scan_tree(item, child, issues)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_tree(item, f"{location}[{index}]", issues)
    elif isinstance(value, str):
        if "://" in value or _SIGNED_QUERY_RE.search(value) or value.startswith(("/", "~", "\\")):
            issues.append(_issue("forbidden_value", location, "URL or private path"))
        if "@" in value or _has_parent_segment(value) or any(ord(c) < 32 for c in value):
            issues.append(_issue("forbidden_value", location, "private identity or path"))
        if _SECRET_RE.search(value):
            issues.append(_issue("forbidden_value", location, "credential-like content"))


def _unknown_keys(
    payload: Mapping[str, Any], allowed: Collection[str], location: str, issues: list[ReceiptIssue]
) -> None:
    for key in sorted(payload):
        if key not in allowed:
            issues.append(_issue("unknown_field", f"{location}/{key}", "not in the schema"))


_LEAF_CHECKS: dict[str, Any] = {
    "schema": lambda value: value == RECEIPT_SCHEMA,
    "sha": lambda value: _matches(value, _SHA256_RE),
    "digest": lambda value: _is_explicit(value) or _matches(value, _SHA256_RE),
    "timestamp": lambda value: _is_explicit(value) or _parse_timestamp(value) is not None,
    "slug": lambda value: _matches(value, _SLUG_RE),
    "class": lambda value: _is_explicit(value) or _matches(value, _CLASS_RE),
    "logical": lambda value: (
        _is_explicit(value)
        or (_matches(value, _LOGICAL_PATH_RE) and not _has_parent_segment(value))
    ),
    "text": lambda value: isinstance(value, str) and bool(value.strip()),
    "states": lambda value: (
        isinstance(value, list) and bool(value) and all(item in SCHEDULER_STATES for item in value)
    ),
}


def _number(low: int, high: int | None = None):
    def check(value: Any) -> bool:
        number = _int_or_none(value)
        return _is_explicit(value) or (
            number is not None and number >= low and (high is None or number <= high)
        )

    return check


def _enum(options: Collection[str]):
    def check(value: Any) -> bool:
        return _is_explicit(value) or value in options

    return check


_RESOURCES_SPEC = dict.fromkeys(RESOURCE_CLASSES, "class")
_RECEIPT_SPEC: dict[str, Any] = {
    "schema": "schema",
    "job_alias": "slug",
    "campaign_id": "slug",
    "scheduler_state": _enum(SCHEDULER_STATES),
    "state_history": "states",
    "claim_boundary": "text",
    "submission": {
        "intent_digest": "sha",
        "source_config_command_digest": "digest",
        "acknowledgement": _enum(("acknowledged", "lost")),
        "submitted_at": "timestamp",
    },
    "execution": {
        "requested_resources": _RESOURCES_SPEC,
        "allocated_resources": _RESOURCES_SPEC,
        "started_at": "timestamp",
        "ended_at": "timestamp",
        "elapsed_seconds": _number(0),
        "time_limit_seconds": _number(0),
        "exit_code": _number(0, 255),
        "derived_exit_code": _number(0, 255),
        "signal": _number(1, 64),
        "termination_class": _enum(TERMINATION_CLASSES),
        "environment_receipt_digest": "digest",
    },
    "artifacts": {"result_root_identity": "logical", "manifest_digest": "digest"},
}
_PRIVATE_TOP_KEYS = set(_RECEIPT_SPEC) | {"redactions", "private_context"}
_PRIVATE_SECTION_KEYS = {
    "submission": set(_RECEIPT_SPEC["submission"]) | {"campaign_id"},
    "execution": set(_RECEIPT_SPEC["execution"]) | {"submission_intent_digest"},
    "artifacts": set(_RECEIPT_SPEC["artifacts"]),
}


def _check_spec(
    node: Any, spec: Mapping[str, Any], location: str, issues: list[ReceiptIssue]
) -> None:
    if not isinstance(node, Mapping):
        issues.append(_issue("invalid_phase", location, "mapping required"))
        return
    for key, check in spec.items():
        predicate = _LEAF_CHECKS[check] if isinstance(check, str) else check
        if key not in node:
            issues.append(_issue("missing_field", f"{location}/{key}", "required"))
        elif isinstance(check, Mapping):
            _check_spec(node[key], check, f"{location}/{key}", issues)
        elif not predicate(node[key]):
            issues.append(_issue("invalid_field", f"{location}/{key}", "invalid value"))
    for key in node:
        if key not in spec:
            issues.append(_issue("unknown_field", f"{location}/{key}", "not in the schema"))


def _check_receipt_shape(payload: Mapping[str, Any], issues: list[ReceiptIssue]) -> str | None:
    _check_spec(payload, _RECEIPT_SPEC, "", issues)
    state, history = payload.get("scheduler_state"), payload.get("state_history")
    if isinstance(history, list) and history and state in SCHEDULER_STATES:
        for previous, current in pairwise(history):
            if current not in TRANSITIONS.get(str(previous), frozenset()):
                issues.append(_issue("illegal_transition", "/state_history", "illegal change"))
        if history[-1] != state:
            issues.append(_issue("history_state_mismatch", "/state_history", "last must match"))
    return state if state in SCHEDULER_STATES else None


def _check_phase_shape(
    state: str | None, execution: Mapping[str, Any], issues: list[ReceiptIssue]
) -> None:
    pre_start = state in {"submission_unacknowledged", "pending"}
    gated = (
        ("started_at", "elapsed_seconds", *NON_TERMINAL_TAIL_FIELDS)
        if pre_start
        else (NON_TERMINAL_TAIL_FIELDS if state == "running" else ())
    )
    for field in gated:
        if not _is_explicit(execution.get(field)):
            issues.append(_issue("phase_contradiction", f"/execution/{field}", "non-terminal"))
    allocated = execution.get("allocated_resources")
    if pre_start and isinstance(allocated, Mapping):
        if any(not _is_explicit(item) for item in allocated.values()):
            issues.append(
                _issue("phase_contradiction", "/execution/allocated_resources", "not yet allocated")
            )


def _check_execution_consistency(  # noqa: C901 - one cross-phase consistency pass
    state: str | None,
    submission: Mapping[str, Any],
    execution: Mapping[str, Any],
    issues: list[ReceiptIssue],
) -> None:
    code = _int_or_none(execution.get("exit_code"))
    derived = _int_or_none(execution.get("derived_exit_code"))
    signal = _int_or_none(execution.get("signal"))
    if signal is not None:
        if code is not None and code != 128 + signal:
            issues.append(_issue("exit_signal_mismatch", "/execution/exit_code", "128+sig"))
        if derived is not None and derived != 128 + signal:
            issues.append(
                _issue("derived_exit_mismatch", "/execution/derived_exit_code", "128+sig")
            )
    elif derived is not None and code is not None and derived != code:
        issues.append(_issue("derived_exit_mismatch", "/execution/derived_exit_code", "exit code"))
    termination = execution.get("termination_class")
    allowed = TERMINATIONS_BY_STATE.get(str(state))
    if allowed is not None and termination in TERMINATION_CLASSES and termination not in allowed:
        issues.append(
            _issue("termination_state_mismatch", "/execution/termination_class", "conflict")
        )
    if state == "completed" and code not in (None, 0):
        issues.append(_issue("exit_state_mismatch", "/execution/exit_code", "completed exit 0"))
    if state == "failed" and code == 0:
        issues.append(_issue("exit_state_mismatch", "/execution/exit_code", "failed exit 0"))
    submitted, started, ended = map(
        _parse_timestamp,
        (submission.get("submitted_at"), execution.get("started_at"), execution.get("ended_at")),
    )
    if submitted and started and submitted > started:
        issues.append(_issue("timestamp_order", "/execution/started_at", "starts before submit"))
    if started and ended and started > ended:
        issues.append(_issue("timestamp_order", "/execution/ended_at", "ends before start"))
    elapsed = _int_or_none(execution.get("elapsed_seconds"))
    limit = _int_or_none(execution.get("time_limit_seconds"))
    delta = (ended - started).total_seconds() if started and ended else None
    if delta is not None and elapsed is not None and abs(delta - elapsed) > 1:
        issues.append(_issue("elapsed_mismatch", "/execution/elapsed_seconds", "delta conflict"))
    if elapsed is not None and limit is not None and elapsed > limit and state == "completed":
        issues.append(_issue("elapsed_exceeds_limit", "/execution/elapsed_seconds", "past limit"))
    acknowledgment = submission.get("acknowledgement")
    expected_ack = "lost" if state == "submission_unacknowledged" else "acknowledged"
    if acknowledgment in {"acknowledged", "lost"} and acknowledgment != expected_ack:
        issues.append(
            _issue("acknowledgement_state_mismatch", "/submission/acknowledgement", "conflict")
        )


def validate_receipt(payload: Mapping[str, Any]) -> tuple[ReceiptIssue, ...]:
    """Validate a public scheduler/allocation receipt and return sanitized issues."""

    issues: list[ReceiptIssue] = []
    state = _check_receipt_shape(payload, issues)
    submission, execution = payload.get("submission"), payload.get("execution")
    if isinstance(submission, Mapping) and isinstance(execution, Mapping):
        _check_phase_shape(state, execution, issues)
        _check_execution_consistency(state, submission, execution, issues)
    _scan_tree(payload, "", issues)
    return tuple(issues)


def _normalize_text(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) and not _is_explicit(value) else value


def _normalize_timestamp(value: Any) -> Any:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return value
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _normalize_resources(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {field: value.get(field) or "not_observed" for field in RESOURCE_CLASSES}


def _build_receipt(private: Mapping[str, Any]) -> dict[str, Any]:
    def section(name: str) -> Mapping[str, Any]:
        value = private.get(name)
        return value if isinstance(value, Mapping) else {}

    sub, exe, art = (section(name) for name in ("submission", "execution", "artifacts"))
    observed = {key: exe.get(key, "not_observed") for key in _RECEIPT_SPEC["execution"]}
    observed |= {key: _normalize_timestamp(exe.get(key)) for key in ("started_at", "ended_at")}
    observed |= {
        "requested_resources": _normalize_resources(exe.get("requested_resources")),
        "allocated_resources": _normalize_resources(exe.get("allocated_resources")),
        "environment_receipt_digest": _normalize_text(exe.get("environment_receipt_digest")),
    }
    history = private.get("state_history")
    return {
        "schema": RECEIPT_SCHEMA,
        "job_alias": _normalize_text(private.get("job_alias")),
        "campaign_id": _normalize_text(private.get("campaign_id")),
        "scheduler_state": private.get("scheduler_state"),
        "state_history": list(history) if isinstance(history, list) and history else [],
        "submission": {
            "intent_digest": _normalize_text(sub.get("intent_digest")),
            "source_config_command_digest": _normalize_text(
                sub.get("source_config_command_digest")
            ),
            "acknowledgement": sub.get("acknowledgement", "not_observed"),
            "submitted_at": _normalize_timestamp(sub.get("submitted_at")),
        },
        "execution": observed,
        "artifacts": {
            "result_root_identity": art.get("result_root_identity", "not_observed"),
            "manifest_digest": _normalize_text(art.get("manifest_digest")),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _check_private_contract(private: Mapping[str, Any], issues: list[ReceiptIssue]) -> None:
    _unknown_keys(private, _PRIVATE_TOP_KEYS, "", issues)
    if private.get("schema") != PRIVATE_INPUT_SCHEMA:
        issues.append(_issue("invalid_schema", "/schema", f"must equal {PRIVATE_INPUT_SCHEMA}"))
    for name, allowed in _PRIVATE_SECTION_KEYS.items():
        value = private.get(name)
        if not isinstance(value, Mapping):
            issues.append(_issue("missing_private_section", f"/{name}", "mapping required"))
        else:
            _unknown_keys(value, allowed, f"/{name}", issues)
    if private.get("redactions") is not None and not isinstance(private.get("redactions"), list):
        issues.append(_issue("invalid_redactions", "/redactions", "list required"))
    context = private.get("private_context")
    if context is not None and not isinstance(context, Mapping):
        issues.append(_issue("invalid_private_context", "/private_context", "mapping required"))
    submission, execution = private.get("submission"), private.get("execution")
    declared = submission.get("campaign_id") if isinstance(submission, Mapping) else None
    if declared is not None and declared != private.get("campaign_id"):
        issues.append(_issue("campaign_identity_mismatch", "/submission/campaign_id", "differs"))
    echo = execution.get("submission_intent_digest") if isinstance(execution, Mapping) else None
    intent = submission.get("intent_digest") if isinstance(submission, Mapping) else None
    if echo is not None and echo != intent:
        issues.append(
            _issue("source_identity_mismatch", "/execution/submission_intent_digest", "diff")
        )


def _apply_redactions(receipt: dict[str, Any], redactions: Any, issues: list[ReceiptIssue]) -> None:
    for entry in redactions if isinstance(redactions, list) else []:
        parts = entry.split(".") if isinstance(entry, str) else []
        node: Any = receipt
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, Mapping) else None
        if not parts or not isinstance(node, dict) or parts[-1] not in node:
            issues.append(_issue("unknown_redaction", "/redactions", "unknown dotted path"))
        else:
            node[parts[-1]] = "redacted"


def _check_private_leak(
    receipt: Mapping[str, Any], context: Any, issues: list[ReceiptIssue]
) -> None:
    if not isinstance(context, Mapping):
        return
    private_values = re.findall(r'"((?:[^"\\]|\\.)+)"', json.dumps(context))
    if any(v.lower() in json.dumps(receipt).lower() for v in private_values if len(v) >= 4):
        issues.append(_issue("private_value_leak", "/", "public output overlaps private context"))


def project_receipt(private: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an explicit private source record into a validated public receipt."""

    issues: list[ReceiptIssue] = []
    _check_private_contract(private, issues)
    if issues:
        raise ReceiptError(issues)
    receipt = _build_receipt(private)
    _apply_redactions(receipt, private.get("redactions"), issues)
    _check_private_leak(receipt, private.get("private_context"), issues)
    if issues:
        raise ReceiptError(issues)
    validation_issues = validate_receipt(receipt)
    if validation_issues:
        raise ReceiptError(validation_issues)
    return receipt


def render_receipt_json(receipt: Mapping[str, Any]) -> str:
    """Return byte-stable receipt JSON with sorted keys and a trailing newline."""

    return json.dumps(receipt, indent=2, sort_keys=True) + "\n"


def _leaf_lines(value: Any, location: str = "") -> list[str]:
    if isinstance(value, Mapping):
        return [
            line
            for key in sorted(value)
            for line in _leaf_lines(value[key], f"{location}.{key}".strip("."))
        ]
    return [f"- `{location}`: `{value}`"]


def render_markdown(receipt: Mapping[str, Any]) -> str:
    """Return a concise deterministic Markdown projection of the receipt."""

    lines = _leaf_lines(receipt)
    return "\n".join(["# Scheduler Allocation Receipt", "", *lines, "", CLAIM_NOTE, ""])


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the scheduler receipt argument parser."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    actions = parser.add_subparsers(dest="command", required=True)
    validate = actions.add_parser("validate")
    validate.add_argument("--receipt", required=True, type=Path)
    validate.add_argument("--json", action="store_true")
    project = actions.add_parser("project")
    project.add_argument("--private-input", required=True, type=Path)
    project.add_argument("--output", type=Path)
    project.add_argument("--markdown-output", type=Path)
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReceiptError(
            [_issue("unreadable_input", str(path), f"cannot load JSON ({type(exc).__name__})")]
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReceiptError([_issue("unreadable_input", str(path), "expected a JSON mapping")])
    return payload


def _report_issues(issues: Sequence[ReceiptIssue]) -> None:
    for issue in issues:
        sys.stderr.write(f"FAIL {issue.code} at {issue.location}: {issue.message}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested subcommand and return a shell-friendly exit code."""

    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "validate":
            issues = validate_receipt(_load_json(args.receipt))
            report = {
                "schema": REPORT_SCHEMA,
                "ok": not issues,
                "issue_count": len(issues),
                "issues": [asdict(issue) for issue in issues],
            }
            if args.json:
                sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            elif not issues:
                sys.stdout.write(f"OK: {args.receipt} is a valid public scheduler receipt\n")
            _report_issues(issues)
            return 0 if not issues else 2
        receipt = project_receipt(_load_json(args.private_input))
    except ReceiptError as error:
        _report_issues(error.issues)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_receipt_json(receipt), encoding="utf-8")
    markdown_path = args.markdown_output or args.output.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(receipt), encoding="utf-8")
    sys.stdout.write(f"wrote {args.output} and {markdown_path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
