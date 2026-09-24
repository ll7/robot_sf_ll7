#!/usr/bin/env python3
"""Check-only operational failure classifier for scheduler/launcher receipts (#8843).

Structured scheduler/launcher/artifact/job-manifest fields outrank bounded log
fingerprints, which are secondary. Every report records classification, source, confidence,
retryability under the existing (unchanged) policy, owner, required remediation, and harvest
need; insufficient evidence stays ``unknown`` and conflicting evidence ``multiple_causes``.
A clean completed receipt is reported as ``unknown`` with an explicit no-failure note rather
than as success. Operational classes never imply scientific status: scheduler completion is
not success and an application nonzero exit is not an infrastructure retry. State lanes and
the termination vocabulary are reused from ``slurm_job_finalize`` and
``scheduler_allocation_receipt``; sanitization follows the compute-window dashboard
``_scan_private`` approach. CLI: ``--check --receipt <json> [--log-excerpt <file>]
[--format json|text]``; exit 0 report, 2 unreadable or malformed input. The tool never
writes, submits, retries, or cancels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from scripts.tools.scheduler_allocation_receipt import TERMINATION_CLASSES
from scripts.tools.slurm_job_finalize import FAILED_STATES, SUCCESS_STATES, normalize_job_state

REPORT_SCHEMA = "robot_sf.scheduler_failure_classification.v1"
VOCABULARY_VERSION = "robot_sf.scheduler_failure_vocabulary.v1"
RECEIPT_SCHEMA = "robot_sf.scheduler_failure_receipt.v1"
MAX_LOG_BYTES = 16384
MAX_TEXT_CHARS = 280
CLAIM_BOUNDARY = (
    "Operational failure classification only: scheduler completion is not scientific success, "
    "an application nonzero exit is not automatically an infrastructure failure, and this "
    "check-only report changes no retry, resubmit, cancel, or scientific result status."
)
FAILURE_CLASSES = (
    "pending_or_queue_delay queue_timeout preemption node_failure out_of_memory walltime_timeout cancellation missing_command_or_module import_failure invalid_config_or_input output_capacity duplicate_submission launcher_failure application_failure harvest_failure unknown multiple_causes"
).split()
VOCABULARY = FAILURE_CLASSES
EXPLICIT_TOKENS = frozenset(
    ("", "unavailable", "not_observed", "not_applicable", "redacted", "unknown", "none")
)
SUCCESS_TOKENS = ("completed", "ok", "success")
QUEUE_HEALTH_STATES = ("healthy", "constrained", "exhausted", "unavailable", "not_observed")
HARVEST_STATES = ("completed", "failed", "pending", "not_applicable", "unavailable", "not_observed")
LAUNCHER_CLASSES = {
    "missing_command": "missing_command_or_module",
    "command_not_found": "missing_command_or_module",
    "module_not_found": "missing_command_or_module",
    "import_error": "import_failure",
    "dependency_error": "import_failure",
    "invalid_config": "invalid_config_or_input",
    "invalid_input": "invalid_config_or_input",
    "output_capacity": "output_capacity",
    "disk_full": "output_capacity",
    "timeout": "walltime_timeout",
    "outer_timeout": "walltime_timeout",
    "submission_unacknowledged": "launcher_failure",
    "duplicate_submission": "duplicate_submission",
    "launcher_error": "launcher_failure",
    "launch_failed": "launcher_failure",
    "application_failure": "application_failure",
    "application_error": "application_failure",
    "campaign_failed": "application_failure",
}
_PRE_APPLICATION = frozenset(
    "queue_timeout preemption node_failure out_of_memory walltime_timeout cancellation "
    "missing_command_or_module import_failure invalid_config_or_input output_capacity "
    "duplicate_submission".split()
)
_CANCELLATION_FAMILY = ("queue_timeout", "preemption", "walltime_timeout", "cancellation")
_GENERIC_CAUSES = frozenset(("application_failure", "launcher_failure"))
_MIXED = "structured+log_fingerprint"
_HARVEST_RANK = {"not_applicable": 0, "no": 1, "unknown": 2, "possible": 3, "required": 4}


@dataclass(frozen=True, slots=True)
class ClassPolicy:
    """Operational disposition for one failure class under the existing policy."""

    retryability: str
    owner: str
    remediation: str
    harvest: str


# name|retryability|owner|remediation|harvest (one row per versioned class)
_POLICY_ROWS = r"""
pending_or_queue_delay|wait_or_recheck_queue|compute_window_operator|Monitor the queue and recheck capacity before changing the submission.|not_applicable
queue_timeout|resubmit_after_queue_policy_review|compute_window_operator|Review the queue window and resubmit under the existing policy; no automatic retry.|no
preemption|resubmit_per_existing_policy|compute_window_operator|Preserve partial checkpoints and resubmit under the existing priority policy.|possible
node_failure|resubmit_per_existing_policy|cluster_operator|Preserve partial artifacts and resubmit on healthy nodes under the existing policy.|possible
out_of_memory|retry_after_resource_change|experiment_owner|Increase the memory request or reduce the batch before any resubmission.|possible
walltime_timeout|retry_after_limit_or_checkpoint_change|experiment_owner|Adjust the time limit or checkpoint cadence before any resubmission.|possible
cancellation|operator_review_required|experiment_owner|Confirm whether the cancellation was intentional before any resubmission.|possible
missing_command_or_module|retry_after_environment_fix|environment_owner|Install or load the missing command/module and rerun preflight before resubmission.|no
import_failure|retry_after_dependency_fix|environment_owner|Repair the Python dependency environment and rerun the import smoke before resubmission.|no
invalid_config_or_input|not_retryable_without_input_fix|experiment_owner|Fix the config or input and revalidate the packet; an unchanged resubmission is not useful.|no
output_capacity|retry_after_storage_fix|storage_operator|Free or relocate output capacity and verify the output root before resubmission.|possible
duplicate_submission|not_retryable_duplicate|submitter|Reconcile with the existing equivalent job; do not submit another duplicate.|no
launcher_failure|operator_review_required|launcher_owner|Inspect the bounded launcher receipt and rerun launcher preflight before resubmission.|unknown
application_failure|operator_review_required|experiment_owner|Diagnose the application failure separately from infrastructure; no automatic retry.|possible
harvest_failure|harvest_retry_only_no_job_rerun|artifact_owner|Retry or repair the harvest step only; do not rerun the job for a harvest failure.|required
unknown|unknown|operator|Collect scheduler state, exit code, launcher result, or a bounded log excerpt.|unknown
multiple_causes|operator_review_required|operator|Separate the recorded causes and remediate each under the existing policy.|unknown
""".strip()
CLASS_POLICY = {
    parts[0]: ClassPolicy(*parts[1:])
    for row in _POLICY_ROWS.splitlines()
    if (parts := row.split("|"))
}
assert frozenset(CLASS_POLICY) == frozenset(VOCABULARY)

# class;;confidence;;case-insensitive regex fingerprint (specific patterns precede generic ones)
_FINGERPRINT_ROWS = r"""
out_of_memory;;high;;\b(?:oom[-_ ]?kill|out[-_ ]of[-_ ]memory|memory cgroup out of memory)\b
walltime_timeout;;high;;(?:due to time limit|time limit exceeded|exceeded.*time limit)
preemption;;high;;preempt
node_failure;;high;;\bnode[_ -]?fail(?:ure|ed)?\b
output_capacity;;high;;(?:no space left on device|disk quota exceeded|enospc|quota exceeded)
missing_command_or_module;;medium;;(?:command not found|module: command not found|module load.*(?:error|not found))
import_failure;;medium;;(?:modulenotfounderror|importerror|cannot import name)
invalid_config_or_input;;medium;;(?:invalid (?:config|input|argument)|config(?:uration)? (?:error|invalid)|validationerror|keyerror|filenotfounderror|missing required)
harvest_failure;;medium;;harvest[a-z_ ]*(?:failed|failure|error)
duplicate_submission;;medium;;(?:duplicate (?:submission|job)|already submitted)
launcher_failure;;low;;(?:launcher (?:failed|error)|launch failed|sbatch[: ]+error)
application_failure;;low;;(?:traceback \(most recent call last\)|application error|unhandled exception)
""".strip()
_LOG_FINGERPRINTS = tuple(
    (parts[0], parts[1], re.compile(rf"(?i){parts[2]}"))
    for row in _FINGERPRINT_ROWS.splitlines()
    if (parts := row.split(";;"))
)
_URL_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://\S+")
_SIGNED_QUERY_RE = re.compile(
    r"(?i)\?[^\s?]*(?:sig|signature|token|expires|x-amz-[a-z-]+|x-goog-[a-z-]+)=[^\s&]*"
)
_CREDENTIAL_ASSIGN_RE = re.compile(
    r"(?i)\b(?:token|secret|password|passwd|api[_-]?key|credential|bearer|authorization)\s*[=:]\s*\S+"
)
_TOPOLOGY_ASSIGN_RE = re.compile(
    r"(?i)\b(?:account|user|username|uid|slurm_account|partition|qos|host|hostname|node|nodes|"
    r"nodelist)\s*[=:]\s*\S+"
)
_IDENTITY_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\b")
_HOST_RE = re.compile(r"(?i)\b(?:[a-z0-9][a-z0-9-]{0,61}\.){2,}[a-z]{2,24}\b")
_NODE_RE = re.compile(
    r"(?i)\b(?:"
    r"(?:node|host|worker|login|compute|gpu|cpu)[-_]?\d+"
    r"|[a-z][a-z0-9-]*node[a-z0-9-]*\d+"
    r"|[a-z0-9]+\[\d+-\d+\]"
    r"|nid\d+"
    r")\b"
)
_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9_./-])(?:[a-z]:\\|/|~/|\\\\|\.\./)[A-Za-z0-9._/-]*")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


@dataclass(frozen=True, slots=True)
class Cause:
    """One candidate failure cause plus its evidence source and confidence."""

    class_name: str
    field: str
    value: str
    confidence: str
    source: str
    suppressed: bool = False


def sanitize_text(value: Any, *, limit: int = MAX_TEXT_CHARS) -> str:
    """Redact private paths, hosts, identities, topology, credentials, and signed URLs."""
    text = str(value)
    text = _SIGNED_QUERY_RE.sub("<redacted-secret>", text)
    text = _URL_RE.sub("<redacted-url>", text)
    text = _CREDENTIAL_ASSIGN_RE.sub("<redacted-secret>", text)
    text = _TOPOLOGY_ASSIGN_RE.sub("<redacted-topology>", text)
    text = _IDENTITY_RE.sub("<redacted-identity>", text)
    text = _HOST_RE.sub("<redacted-host>", text)
    text = _NODE_RE.sub("<redacted-topology>", text)
    text = _PATH_RE.sub("<redacted-path>", text)
    return " ".join(_CONTROL_RE.sub(" ", text).split())[:limit]


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in EXPLICIT_TOKENS)


def _first_present(*values: Any) -> Any:
    return next((value for value in values if not _is_missing(value)), None)


def _section(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _as_int(value: Any, *, low: int, high: int | None = None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= low and (high is None or value <= high) else None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_enum(value: Any, options: Sequence[str]) -> str | None:
    if _is_missing(value) or not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_")
    return normalized if normalized in options else None


def _safe_slug(value: Any) -> str:
    return value if isinstance(value, str) and _SLUG_RE.fullmatch(value) else "unavailable"


def _extract_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read only whitelisted structured fields; every other value stays private."""
    scheduler, launcher = _section(payload, "scheduler"), _section(payload, "launcher")
    artifacts, manifest = _section(payload, "artifacts"), _section(payload, "job_manifest")
    first = _first_present
    state = first(scheduler.get("state"), payload.get("scheduler_state"))
    term = first(scheduler.get("termination_class"), payload.get("termination_class"))
    ack = first(scheduler.get("acknowledgement"), payload.get("acknowledgement"))
    message = launcher.get("message")
    return {
        "job_alias": _safe_slug(first(payload.get("job_alias"), "unavailable")),
        "campaign_id": _safe_slug(first(payload.get("campaign_id"), "unavailable")),
        "state": normalize_job_state(state) if isinstance(state, str) else "UNAVAILABLE",
        "exit_code": _as_int(
            first(scheduler.get("exit_code"), payload.get("exit_code")), low=0, high=255
        ),
        "termination_class": _as_enum(term, TERMINATION_CLASSES),
        "signal": _as_int(scheduler.get("signal"), low=1, high=64),
        "acknowledgement": _as_enum(ack, ("acknowledged", "lost")),
        "queue_health": _as_enum(scheduler.get("queue_health"), QUEUE_HEALTH_STATES),
        "queue_timeout": _as_bool(
            first(scheduler.get("queue_timeout"), payload.get("queue_timeout"))
        ),
        "duplicate_submission": _as_bool(
            first(scheduler.get("duplicate_submission"), manifest.get("duplicate_submission"))
        ),
        "duplicate_of": first(manifest.get("duplicate_of"), payload.get("duplicate_of")),
        "launcher_result": _as_enum(
            first(launcher.get("result"), payload.get("launcher_result")),
            (*LAUNCHER_CLASSES, *SUCCESS_TOKENS),
        ),
        "outer_timeout": _as_bool(
            first(launcher.get("outer_timeout"), payload.get("outer_timeout"))
        ),
        "launcher_message": message if isinstance(message, str) else "",
        "harvest_status": _as_enum(artifacts.get("harvest_status"), HARVEST_STATES),
        "harvest_required": _as_bool(artifacts.get("harvest_required")),
        "output_capacity_exceeded": _as_bool(artifacts.get("output_capacity_exceeded")),
        "log_excerpt": payload.get("log_excerpt")
        if isinstance(payload.get("log_excerpt"), str)
        else "",
    }


def _cause(
    class_name: str,
    value: Any,
    confidence: str,
    field: str = "scheduler.state",
    source: str = "structured",
    *,
    suppressed: bool = False,
) -> Cause:
    return Cause(class_name, field, str(value), confidence, source, suppressed)


def _scheduler_cause(fields: Mapping[str, Any]) -> tuple[Cause | None, list[str], list[str]]:
    """Resolve one ordered scheduler-state cause, preserving contradictions as conflicts."""
    state, term = fields["state"], fields["termination_class"]
    notes: list[str] = []
    conflicts: list[str] = []
    pending = state in {"PENDING", "CONFIGURING", "REQUEUED"}
    rules = (
        (term == "time_limit" or state == "TIMEOUT", "walltime_timeout", term or state, "high"),
        (term == "preempted" or state == "PREEMPTED", "preemption", state, "high"),
        (term == "node_failure" or state == "NODE_FAIL", "node_failure", state, "high"),
        (state == "OUT_OF_MEMORY", "out_of_memory", state, "high"),
        (fields["queue_timeout"] is True, "queue_timeout", "true", "high"),
        (pending and fields["queue_health"] == "exhausted", "queue_timeout", "exhausted", "high"),
        (pending, "pending_or_queue_delay", state, "high"),
        (
            term in {"cancelled", "scheduler_terminated"} or state == "CANCELLED",
            "cancellation",
            state,
            "high",
        ),
        (
            state == "SUBMISSION_UNACKNOWLEDGED" or fields["acknowledgement"] == "lost",
            "launcher_failure",
            "lost",
            "high",
        ),
    )
    for matched, name, value, confidence in rules:
        if matched:
            return _cause(name, value, confidence), notes, conflicts
    if state in FAILED_STATES:
        code = fields["exit_code"]
        if code == 127:
            return _cause("missing_command_or_module", code, "medium"), notes, conflicts
        if code == 126:
            return _cause("launcher_failure", code, "medium"), notes, conflicts
        if code is not None and code > 0:
            return _cause("application_failure", code, "medium"), notes, conflicts
        if code == 0:
            conflicts.append("scheduler state FAILED conflicts with exit code 0")
        elif fields["signal"] is not None:
            notes.append(f"signal {fields['signal']} recorded without an OOM/timeout state")
    elif state in SUCCESS_STATES and fields["exit_code"] not in (None, 0):
        conflicts.append(f"scheduler state {state} conflicts with exit code {fields['exit_code']}")
    return None, notes, conflicts


def _resolve_structured(fields: Mapping[str, Any]) -> tuple[list[Cause], list[str], list[str]]:
    """Resolve scheduler, launcher, artifact, and duplicate structured causes."""
    scheduler_cause, notes, conflicts = _scheduler_cause(fields)
    causes = [c for c in (scheduler_cause,) if c is not None]
    result = fields["launcher_result"]
    if fields["outer_timeout"]:
        causes.append(_cause("walltime_timeout", "true", "high", "launcher.outer_timeout"))
    elif result in LAUNCHER_CLASSES:
        class_name = LAUNCHER_CLASSES[result]
        causes.append(
            _cause(
                class_name,
                result,
                "medium" if class_name in _GENERIC_CAUSES else "high",
                "launcher.result",
            )
        )
    if fields["output_capacity_exceeded"]:
        causes.append(
            _cause("output_capacity", "true", "high", "artifacts.output_capacity_exceeded")
        )
    if fields["harvest_status"] == "failed":
        causes.append(_cause("harvest_failure", "failed", "high", "artifacts.harvest_status"))
    if fields["duplicate_submission"]:
        causes.append(
            _cause("duplicate_submission", "true", "high", "job_manifest.duplicate_submission")
        )
    elif fields["duplicate_of"]:
        causes.append(
            _cause(
                "duplicate_submission",
                _safe_slug(str(fields["duplicate_of"])),
                "high",
                "job_manifest.duplicate_of",
            )
        )
    return _merge_causes(causes), notes, conflicts


def _merge_causes(causes: list[Cause]) -> list[Cause]:
    """Drop generic causes overshadowed by a specific one and fold cancellation family."""
    classes = {cause.class_name for cause in causes}
    if "application_failure" in classes and classes & _PRE_APPLICATION:
        causes = [
            replace(c, suppressed=True) if c.class_name == "application_failure" else c
            for c in causes
        ]
    classes = {cause.class_name for cause in causes if not cause.suppressed}
    if "launcher_failure" in classes and len(classes) > 1:
        causes = [
            replace(c, suppressed=True) if c.class_name == "launcher_failure" else c for c in causes
        ]
    classes = {cause.class_name for cause in causes if not cause.suppressed}
    if len(classes) > 1 and classes <= set(_CANCELLATION_FAMILY):
        winner = next(name for name in _CANCELLATION_FAMILY if name in classes)
        causes = [c if c.class_name == winner else replace(c, suppressed=True) for c in causes]
    return causes


def _fingerprint_causes(text: str) -> list[Cause]:
    """Resolve bounded log fingerprints, keeping only specific causes over generic ones."""
    if not text.strip():
        return []
    matched = [
        _cause(class_name, match.group(0), confidence, "log_excerpt", "log_fingerprint")
        for class_name, confidence, pattern in _LOG_FINGERPRINTS
        if (match := pattern.search(text)) is not None
    ]
    if len(matched) > 1:
        matched = [c for c in matched if c.class_name not in _GENERIC_CAUSES] or matched
    by_class: dict[str, Cause] = {}
    for cause in matched:
        by_class.setdefault(cause.class_name, cause)
    return list(by_class.values())


def _decide(
    fields: Mapping[str, Any],
    structured: list[Cause],
    fingerprints: list[Cause],
    notes: list[str],
    conflicts: list[str],
) -> tuple[str, list[str], str, str, list[Cause]]:
    """Combine structured and fingerprint evidence into one classification."""
    active = list({c.class_name: c for c in structured if not c.suppressed}.values())
    if conflicts and len(active) <= 1:
        notes.extend([*conflicts, "conflicting structured fields left the class unresolved"])
        return "unknown", [], "structured", "low", structured
    if len(active) > 1:
        classes = sorted({c.class_name for c in active})
        notes.append("distinct structured causes were observed; preserving multiple_causes")
        return "multiple_causes", classes, "structured", "medium", structured
    specific = [c for c in fingerprints if c.class_name not in _GENERIC_CAUSES]
    if active and active[0].class_name == "application_failure" and specific:
        if len(specific) == 1:
            pick = specific[0]
            notes.append("log fingerprint refined the generic application_failure evidence")
            return pick.class_name, [pick.class_name], _MIXED, pick.confidence, [active[0], pick]
        classes = sorted({c.class_name for c in specific})
        notes.append("conflicting log fingerprints refined generic application_failure")
        return "multiple_causes", classes, _MIXED, "low", [active[0], *specific]
    if active:
        if fingerprints:
            notes.append("log fingerprints were secondary evidence; structured evidence won")
        label = active[0].class_name
        secondary = [_suppress(c, label) for c in fingerprints]
        return label, [label], "structured", active[0].confidence, [*structured, *secondary]
    if fingerprints:
        classes = {c.class_name for c in fingerprints}
        if len(classes) == 1:
            conf = fingerprints[0].confidence
            return next(iter(classes)), sorted(classes), "log_fingerprint", conf, list(fingerprints)
        notes.append("distinct log fingerprints were observed; preserving multiple_causes")
        return "multiple_causes", sorted(classes), "log_fingerprint", "low", list(fingerprints)
    if fields["state"] in SUCCESS_STATES and fields["exit_code"] in (None, 0):
        notes.append("completed run without a failure signature; nothing to classify")
        return "unknown", [], "structured", "medium", []
    notes.append("no structured cause or matching log fingerprint was found")
    return "unknown", [], "none", "none", []


def _suppress(cause: Cause, keep: str) -> Cause:
    return cause if cause.class_name == keep else replace(cause, suppressed=True)


def _harvest_disposition(class_name: str, causes: Sequence[str], fields: Mapping[str, Any]) -> str:
    """Return the conservative harvest need for the final classification."""
    if class_name == "multiple_causes":
        base = max(
            (CLASS_POLICY[name].harvest for name in causes or ["unknown"]),
            key=_HARVEST_RANK.__getitem__,
        )
    else:
        base = CLASS_POLICY[class_name].harvest
    return "possible" if fields["harvest_required"] and _HARVEST_RANK[base] < 3 else base


def _evidence(causes: Sequence[Cause], fields: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the sorted sanitized evidence list for the report."""
    entries = [
        {
            "source": cause.source,
            "field": cause.field,
            "value": sanitize_text(cause.value),
            "suppressed": cause.suppressed,
        }
        for cause in causes
    ]
    for field, value in (
        ("scheduler.state", fields["state"]),
        ("scheduler.exit_code", fields["exit_code"]),
        ("scheduler.termination_class", fields["termination_class"]),
        ("scheduler.queue_health", fields["queue_health"]),
        ("launcher.result", fields["launcher_result"]),
        ("artifacts.harvest_status", fields["harvest_status"]),
        ("launcher.message", sanitize_text(fields["launcher_message"], limit=120) or None),
    ):
        if value not in (None, "", "UNAVAILABLE"):
            entries.append(
                {"source": "structured", "field": field, "value": value, "suppressed": False}
            )
    unique = {(e["source"], e["field"], str(e["value"])): e for e in entries}
    return [unique[key] for key in sorted(unique)]


def classify_failure(receipt: Mapping[str, Any], log_excerpt: str = "") -> dict[str, Any]:
    """Classify one sanitized scheduler/launcher receipt into a stable operational class.

    Structured fields are resolved first and always outrank prose fingerprints. When no
    structured cause is found, bounded log fingerprints may name the failure at reduced
    confidence. Contradictory structured sources or fingerprints produce
    ``multiple_causes``; missing evidence produces ``unknown``.
    """
    fields = _extract_fields(receipt)
    text = "\n".join(part for part in (fields["log_excerpt"], log_excerpt) if part)
    structured, notes, conflicts = _resolve_structured(fields)
    class_name, causes, source, confidence, evidence_causes = _decide(
        fields, structured, _fingerprint_causes(text), notes, conflicts
    )
    policy = CLASS_POLICY[class_name]
    return {
        "schema": REPORT_SCHEMA,
        "vocabulary_version": VOCABULARY_VERSION,
        "check_only": True,
        "classification": class_name,
        "causes": sorted(set(causes)),
        "source": source,
        "confidence": confidence,
        "evidence": _evidence(evidence_causes, fields),
        "retryability": {
            "disposition": policy.retryability,
            "automatic_retry": False,
            "policy_changed": False,
        },
        "owner": policy.owner,
        "required_remediation": policy.remediation,
        "outputs_may_require_harvest": _harvest_disposition(class_name, causes, fields),
        "job_alias": fields["job_alias"],
        "campaign_id": fields["campaign_id"],
        "notes": sorted(set(notes)),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def render_report_json(report: Mapping[str, Any]) -> str:
    """Return byte-stable classification JSON with sorted keys and a trailing newline."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_report_text(report: Mapping[str, Any]) -> str:
    """Return the concise deterministic human classification summary."""
    retry = report["retryability"]
    lines = [
        f"classification: {report['classification']}",
        f"source: {report['source']} | confidence: {report['confidence']}",
        f"causes: {', '.join(report['causes']) or 'none'}",
        f"retryability: {retry['disposition']} (automatic_retry={str(retry['automatic_retry']).lower()}, "
        f"policy_changed={str(retry['policy_changed']).lower()})",
        f"owner: {report['owner']} | harvest: {report['outputs_may_require_harvest']}",
        f"remediation: {report['required_remediation']}",
    ]
    if report["notes"]:
        lines.append("notes: " + " | ".join(report["notes"]))
    lines.append(report["claim_boundary"])
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the classifier argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="validate and print only; never mutates jobs"
    )
    parser.add_argument(
        "--receipt", type=Path, required=True, help="sanitized scheduler/launcher receipt JSON"
    )
    parser.add_argument(
        "--log-excerpt",
        type=Path,
        help=f"optional bounded log excerpt file (first {MAX_LOG_BYTES} bytes)",
    )
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser


def _read_log_excerpt(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:MAX_LOG_BYTES]
    except OSError as exc:
        raise ValueError(f"cannot read log excerpt {path}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check-only classification and return a shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)
    try:
        payload = json.loads(args.receipt.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("receipt must be a JSON mapping")
        report = classify_failure(payload, _read_log_excerpt(args.log_excerpt))
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"FAIL invalid_input: {sanitize_text(f'{type(exc).__name__}: {exc}')}\n")
        return 2
    render = render_report_json if args.format == "json" else render_report_text
    sys.stdout.write(render(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
