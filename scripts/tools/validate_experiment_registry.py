"""Validate the question-first experiment registry."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REGISTRY_SCHEMA_VERSION = "experiment-registry.v1"
RECORD_SCHEMA_VERSION = "experiment-record.v1"
RECORD_SCHEMA_VERSION_V2 = "experiment-record.v2"
VALID_RECORD_SCHEMA_VERSIONS = {RECORD_SCHEMA_VERSION, RECORD_SCHEMA_VERSION_V2}
REQUIRED_RECORD_FIELDS = (
    "experiment_id",
    "issue",
    "issue_url",
    "question",
    "hypothesis",
    "config",
    "command",
    "inputs",
    "outputs",
    "expected_artifacts",
    "evidence_grade",
    "paper_relevance",
    "status",
)
V2_REQUIRED_RECORD_FIELDS = tuple(
    field for field in REQUIRED_RECORD_FIELDS if field != "status"
) + ("state",)
VALID_EVIDENCE_GRADES = {"proposal", "inferred", "observed"}
VALID_PAPER_RELEVANCE = {"none", "exploratory", "paper_candidate", "paper_facing"}
ACTIVE_STATES = {
    "idea",
    "protocol_frozen",
    "implementation_ready",
    "preflight_passed",
    "submitted",
    "running",
    "finalized",
    "evidence_promoted",
    "claim_reviewed",
    "released",
}
TERMINAL_STATES = {
    "blocked_external",
    "invalid_execution",
    "negative_result",
    "null_result",
    "superseded",
    "stopped_by_gate",
}
VALID_RECORD_STATES = ACTIVE_STATES | TERMINAL_STATES
LEGACY_TERMINAL_STATUSES = TERMINAL_STATES | {"completed", "released", "closed"}
STATE_LABELS = {"state:ready", "state:running", "state:blocked"}
STATE_TO_LABEL = {
    "idea": "state:ready",
    "protocol_frozen": "state:ready",
    "implementation_ready": "state:ready",
    "preflight_passed": "state:ready",
    "submitted": "state:running",
    "running": "state:running",
    "finalized": "state:running",
    "evidence_promoted": "state:running",
    "claim_reviewed": "state:running",
    "released": None,
    "blocked_external": "state:blocked",
    "invalid_execution": "state:blocked",
    "negative_result": "state:blocked",
    "null_result": "state:blocked",
    "superseded": "state:blocked",
    "stopped_by_gate": "state:blocked",
}
RUNNING_LABEL = "state:running"
PROPOSAL_RECORD_STATES = {"idea", "proposal"}
ANGLE_PLACEHOLDER_RE = re.compile(r"<[A-Za-z0-9][A-Za-z0-9_.:/-]*>")
DEFAULT_STATE_LABEL_AUDIT_DIR = Path("output/issue_state_sync")


def validate_registry(registry_path: Path) -> list[str]:
    """Return validation errors for an experiment registry."""
    errors: list[str] = []
    registry = _load_yaml_mapping(registry_path, errors, display=str(registry_path))
    if registry is None:
        return errors

    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        errors.append(
            f"{registry_path}: expected schema_version {REGISTRY_SCHEMA_VERSION!r}, "
            f"found {registry.get('schema_version')!r}"
        )

    records = registry.get("records")
    if not isinstance(records, list) or not records:
        errors.append(f"{registry_path}: records must be a non-empty list")
        return errors

    seen_ids: set[str] = set()
    for record_entry in records:
        if not isinstance(record_entry, str) or not record_entry:
            errors.append(f"{registry_path}: record entries must be non-empty paths")
            continue
        record_path = (registry_path.parent / record_entry).resolve()
        record = _load_yaml_mapping(record_path, errors, display=record_entry)
        if record is None:
            continue
        _validate_record(record, record_entry, errors, seen_ids=seen_ids)

    return errors


def build_control_plane_report(
    registry_path: Path,
    *,
    issue_states: Mapping[int, str] | None = None,
    issue_labels: Mapping[int, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Build a dry-run report for research-state drift without mutating GitHub."""
    validation_errors = validate_registry(registry_path)
    findings: list[dict[str, Any]] = [
        {
            "kind": "validation_error",
            "severity": "error",
            "record": None,
            "issue": None,
            "message": error,
        }
        for error in validation_errors
    ]
    registry = _load_yaml_mapping(registry_path, [], display=str(registry_path))
    if registry is not None and isinstance(registry.get("records"), list):
        repo_root = _repo_root_for_registry(registry_path)
        for record_entry in registry["records"]:
            if not isinstance(record_entry, str):
                continue
            record_path = (registry_path.parent / record_entry).resolve()
            record = _load_yaml_mapping(record_path, [], display=record_entry)
            if record is None:
                continue
            findings.extend(
                _control_plane_findings_for_record(
                    record,
                    record_entry,
                    repo_root=repo_root,
                    issue_states=issue_states or {},
                    issue_labels=issue_labels or {},
                )
            )
    return {
        "schema_version": "experiment-control-plane-report.v1",
        "registry": registry_path.as_posix(),
        "finding_count": len(findings),
        "findings": findings,
        "derived_update_count": sum(
            1 for finding in findings if finding.get("kind") == "derived_issue_label_update"
        ),
    }


def apply_derived_issue_label_updates(
    report: Mapping[str, Any],
    *,
    max_writes: int,
    audit_log_path: Path,
    min_rate_limit_remaining: int = 25,
    gh_runner: Any | None = None,
) -> dict[str, Any]:
    """Apply derived `state:*` label updates from a control-plane report.

    The control plane remains dry-run by default. This function is the explicit
    write boundary for `--apply-labels`: it only consumes existing
    `derived_issue_label_update` findings, only touches labels in `STATE_LABELS`,
    checks GitHub API quota before the batch, and writes an audit log.
    """

    runner = gh_runner or _run_gh
    updates = _derived_label_updates_from_report(report)
    if len(updates) > max_writes:
        raise RuntimeError(
            f"refusing to apply {len(updates)} label updates; --max-writes is {max_writes}"
        )
    rate_limit = _read_rate_limit(runner)
    remaining = rate_limit.get("remaining")
    if not isinstance(remaining, int):
        raise RuntimeError(
            "refusing to apply label updates; unable to determine GitHub core rate limit remaining"
        )
    if remaining < min_rate_limit_remaining:
        raise RuntimeError(
            f"refusing to apply label updates; GitHub core rate limit remaining "
            f"{remaining} < {min_rate_limit_remaining}"
        )

    applied: list[dict[str, Any]] = []
    error_to_raise: Exception | None = None
    for update in updates:
        command = _issue_label_edit_command(update)
        try:
            result = runner(command)
            stdout = getattr(result, "stdout", "")
            stderr = getattr(result, "stderr", "")
            returncode = int(getattr(result, "returncode", 1))
        except Exception as exc:
            stdout = ""
            stderr = str(exc)
            returncode = 1
            error_to_raise = exc
        applied.append(
            {
                "issue": update["issue"],
                "record": update["record"],
                "labels_to_add": update["labels_to_add"],
                "labels_to_remove": update["labels_to_remove"],
                "command": ["gh", *command],
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        if returncode != 0 and error_to_raise is None:
            error_to_raise = RuntimeError(
                f"gh issue edit failed for issue #{update['issue']} with {returncode}: "
                f"{(stderr or stdout).strip()}"
            )
        if error_to_raise is not None:
            break

    audit = {
        "schema_version": "experiment-state-label-apply.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "max_writes": max_writes,
        "min_rate_limit_remaining": min_rate_limit_remaining,
        "rate_limit": rate_limit,
        "applied_count": sum(1 for entry in applied if entry["returncode"] == 0),
        "applied": applied,
    }
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    audit_log_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if error_to_raise is not None:
        raise error_to_raise
    return audit


def _validate_record(
    record: Mapping[str, Any],
    display_path: str,
    errors: list[str],
    *,
    seen_ids: set[str],
) -> None:
    """Append validation errors for one experiment record."""
    schema_version = record.get("schema_version")
    if schema_version not in VALID_RECORD_SCHEMA_VERSIONS:
        errors.append(
            f"{display_path}: expected schema_version one of {sorted(VALID_RECORD_SCHEMA_VERSIONS)}, "
            f"found {record.get('schema_version')!r}"
        )

    required_fields = (
        V2_REQUIRED_RECORD_FIELDS
        if schema_version == RECORD_SCHEMA_VERSION_V2
        else REQUIRED_RECORD_FIELDS
    )
    for field in required_fields:
        if _is_missing(record.get(field)):
            errors.append(f"{display_path}: missing required field {field!r}")

    experiment_id = record.get("experiment_id")
    if isinstance(experiment_id, str):
        if experiment_id in seen_ids:
            errors.append(f"{display_path}: duplicate experiment_id {experiment_id!r}")
        seen_ids.add(experiment_id)

    _validate_vocab(
        record.get("evidence_grade"),
        "evidence_grade",
        VALID_EVIDENCE_GRADES,
        display_path,
        errors,
    )
    paper_relevance = record.get("paper_relevance")
    _validate_vocab(
        paper_relevance,
        "paper_relevance",
        VALID_PAPER_RELEVANCE,
        display_path,
        errors,
    )
    if schema_version == RECORD_SCHEMA_VERSION_V2:
        _validate_vocab(record.get("state"), "state", VALID_RECORD_STATES, display_path, errors)

    if paper_relevance == "paper_facing":
        for artifact in _iter_artifact_items(record.get("outputs")):
            _validate_paper_facing_artifact(artifact, display_path, errors)
        for artifact in _iter_artifact_items(record.get("expected_artifacts")):
            _validate_paper_facing_artifact(artifact, display_path, errors)
    _validate_actionable_artifact_policy(record, display_path, errors)


def _validate_paper_facing_artifact(
    artifact: Mapping[str, Any],
    display_path: str,
    errors: list[str],
) -> None:
    """Reject paper-facing output artifacts that only point at local output paths."""
    artifact_path = artifact.get("path")
    if not isinstance(artifact_path, str) or not _is_local_output_path(artifact_path):
        return
    durable_reference = artifact.get("durable_reference")
    if _is_missing(durable_reference):
        errors.append(
            f"{display_path}: paper-facing record references local-only output/ artifact "
            f"without durable_reference: {artifact_path}"
        )


def _validate_actionable_artifact_policy(
    record: Mapping[str, Any],
    display_path: str,
    errors: list[str],
) -> None:
    """Reject actionable records that still point at unresolved artifact placeholders."""
    if not _is_actionable_record(record):
        return

    for artifact in _iter_artifact_items(record.get("outputs")):
        _validate_actionable_required_durable_reference(artifact, display_path, errors)
    for artifact in _iter_artifact_items(record.get("expected_artifacts")):
        _validate_actionable_required_durable_reference(artifact, display_path, errors)

    for field_name, value in _iter_actionable_placeholder_values(record):
        if _has_unresolved_artifact_placeholder(value):
            errors.append(
                f"{display_path}: actionable record contains unresolved placeholder in "
                f"{field_name}: {value}"
            )


def _validate_actionable_required_durable_reference(
    artifact: Mapping[str, Any],
    display_path: str,
    errors: list[str],
) -> None:
    """Reject required durable artifacts that do not name the durable reference."""
    if artifact.get("durable_reference_required") is not True:
        return
    if not _is_missing(artifact.get("durable_reference")):
        return
    artifact_name = artifact.get("name") or artifact.get("path") or "<unnamed>"
    errors.append(
        f"{display_path}: actionable record requires durable_reference for artifact {artifact_name}"
    )


def _control_plane_findings_for_record(
    record: Mapping[str, Any],
    display_path: str,
    *,
    repo_root: Path,
    issue_states: Mapping[int, str],
    issue_labels: Mapping[int, Iterable[str]],
) -> list[dict[str, Any]]:
    """Return drift findings for one experiment record."""
    findings: list[dict[str, Any]] = []
    issue = _issue_number(record.get("issue"))
    record_state = _record_state(record)
    findings.extend(
        _issue_state_findings(
            record,
            display_path,
            issue=issue,
            record_state=record_state,
            issue_states=issue_states,
            issue_labels=issue_labels,
        )
    )
    findings.extend(_missing_path_findings(record, display_path, issue=issue, repo_root=repo_root))
    findings.extend(_artifact_alias_findings(record, display_path, issue=issue))
    findings.extend(_durable_reference_findings(record, display_path, issue=issue))
    return findings


def _derived_label_updates_from_report(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return applicable label-update findings from a control-plane report."""

    updates: list[dict[str, Any]] = []
    findings = report.get("findings", [])
    if not isinstance(findings, list):
        return updates
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        if finding.get("kind") != "derived_issue_label_update":
            continue
        labels_to_add = _validate_state_label_list(finding.get("labels_to_add"), "labels_to_add")
        labels_to_remove = _validate_state_label_list(
            finding.get("labels_to_remove"), "labels_to_remove"
        )
        if not labels_to_add and not labels_to_remove:
            continue
        issue = _issue_number(finding.get("issue"))
        if issue is None:
            raise RuntimeError(f"derived label update has invalid issue: {finding!r}")
        updates.append(
            {
                "issue": issue,
                "record": finding.get("record"),
                "labels_to_add": labels_to_add,
                "labels_to_remove": labels_to_remove,
            }
        )
    return updates


def _validate_state_label_list(value: Any, field_name: str) -> list[str]:
    """Return validated `state:*` labels from a report field."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError(f"{field_name} must be a list, got {value!r}")
    labels = [str(label) for label in value]
    invalid = sorted(set(labels) - STATE_LABELS)
    if invalid:
        raise RuntimeError(f"{field_name} contains non-state labels: {invalid}")
    return sorted(labels)


def _issue_label_edit_command(update: Mapping[str, Any]) -> list[str]:
    """Build a `gh issue edit` command for one derived label update."""

    command = ["issue", "edit", str(update["issue"])]
    for label in update["labels_to_add"]:
        command.extend(["--add-label", label])
    for label in update["labels_to_remove"]:
        command.extend(["--remove-label", label])
    return command


def _run_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a GitHub CLI command for label application."""

    return subprocess.run(["gh", *args], capture_output=True, text=True, check=False)


def _read_rate_limit(gh_runner: Any) -> dict[str, Any]:
    """Return compact GitHub API rate-limit state for apply guards."""

    result = gh_runner(["api", "rate_limit"])
    stdout = getattr(result, "stdout", "")
    stderr = getattr(result, "stderr", "")
    if int(getattr(result, "returncode", 1)) != 0:
        raise RuntimeError(f"gh api rate_limit failed: {(stderr or stdout).strip()}")
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gh api rate_limit returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        payload = {}
    resources = payload.get("resources")
    if not isinstance(resources, dict):
        resources = {}
    core = resources.get("core")
    if not isinstance(core, dict):
        core = {}
    return {
        "remaining": core.get("remaining"),
        "limit": core.get("limit"),
        "reset": core.get("reset"),
    }


def _issue_state_findings(
    record: Mapping[str, Any],
    display_path: str,
    *,
    issue: int | None,
    record_state: str,
    issue_states: Mapping[int, str],
    issue_labels: Mapping[int, Iterable[str]],
) -> list[dict[str, Any]]:
    """Return findings that compare card state to live issue state."""
    findings: list[dict[str, Any]] = []
    issue_state = issue_states.get(issue) if issue is not None else None
    if (
        issue is not None
        and _is_closed_issue_state(issue_state)
        and not _is_terminal_record_state(record)
    ):
        findings.append(
            _finding(
                "closed_issue_with_nonterminal_record",
                display_path,
                issue,
                f"issue #{issue} is closed but record state is {record_state!r}",
            )
        )
    for blocker in _iter_blocker_issue_numbers(record):
        blocker_state = issue_states.get(blocker)
        if _is_closed_issue_state(blocker_state) and _looks_blocked(record_state):
            findings.append(
                _finding(
                    "closed_blocker_with_blocked_record",
                    display_path,
                    issue,
                    f"record remains blocked on closed issue #{blocker}",
                )
            )
    labels = set(issue_labels.get(issue, ())) if issue is not None else set()
    if record_state not in STATE_TO_LABEL:
        return findings
    expected_label = STATE_TO_LABEL[record_state]
    current_state_labels = labels & STATE_LABELS
    if (
        issue is not None
        and issue in issue_labels
        and current_state_labels != _label_set(expected_label)
    ):
        findings.append(
            _derived_label_update(
                display_path,
                issue,
                record_state=record_state,
                current_labels=sorted(current_state_labels),
                expected_label=expected_label,
            )
        )
    return findings


def _missing_path_findings(
    record: Mapping[str, Any],
    display_path: str,
    *,
    issue: int | None,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Return findings for repo-local config/input paths that no longer exist."""
    findings: list[dict[str, Any]] = []
    for path in _iter_config_paths(record):
        if not (repo_root / path).exists():
            findings.append(
                _finding(
                    "missing_config_or_input_path",
                    display_path,
                    issue,
                    f"referenced path does not exist: {path.as_posix()}",
                )
            )
    return findings


def _artifact_alias_findings(
    record: Mapping[str, Any],
    display_path: str,
    *,
    issue: int | None,
) -> list[dict[str, Any]]:
    """Return findings for unresolved pending artifact aliases."""
    findings: list[dict[str, Any]] = []
    for value in _iter_string_values(record):
        if _is_pending_artifact_alias(value):
            findings.append(
                _finding(
                    "pending_artifact_alias",
                    display_path,
                    issue,
                    f"pending artifact alias is not durable evidence: {value}",
                )
            )
    return findings


def _durable_reference_findings(
    record: Mapping[str, Any],
    display_path: str,
    *,
    issue: int | None,
) -> list[dict[str, Any]]:
    """Return findings for artifacts that still need durable references."""
    findings: list[dict[str, Any]] = []
    for artifact in _iter_artifact_items(record.get("expected_artifacts")):
        if artifact.get("durable_reference_required") is True and _is_missing(
            artifact.get("durable_reference")
        ):
            artifact_name = artifact.get("name") or artifact.get("path") or "<unnamed>"
            findings.append(
                _finding(
                    "missing_required_durable_reference",
                    display_path,
                    issue,
                    f"expected artifact requires durable_reference: {artifact_name}",
                )
            )
    return findings


def _finding(kind: str, record: str, issue: int | None, message: str) -> dict[str, Any]:
    """Return a normalized control-plane finding."""
    return {
        "kind": kind,
        "severity": "warning",
        "record": record,
        "issue": issue,
        "message": message,
    }


def _derived_label_update(
    record: str,
    issue: int,
    *,
    record_state: str,
    current_labels: list[str],
    expected_label: str | None,
) -> dict[str, Any]:
    """Return a dry-run label update derived from the authoritative card state."""
    expected_labels = sorted(_label_set(expected_label))
    labels_to_add = sorted(set(expected_labels) - set(current_labels))
    labels_to_remove = sorted(set(current_labels) - set(expected_labels))
    message = (
        f"issue #{issue} state labels {current_labels} should derive from "
        f"record state {record_state!r} as {expected_labels}"
    )
    finding = _finding("derived_issue_label_update", record, issue, message)
    finding["expected_state_label"] = expected_label
    finding["current_state_labels"] = current_labels
    finding["labels_to_add"] = labels_to_add
    finding["labels_to_remove"] = labels_to_remove
    finding["dry_run_only"] = True
    return finding


def _label_set(label: str | None) -> set[str]:
    """Return the normalized singleton label set used for state-label comparisons."""
    return {label} if label else set()


def _record_state(record: Mapping[str, Any]) -> str:
    """Return the v2 state or legacy v1 status token."""
    value = (
        record.get("state") if record.get("schema_version") == RECORD_SCHEMA_VERSION_V2 else None
    )
    if _is_missing(value):
        value = record.get("status")
    return str(value or "").strip()


def _is_terminal_record_state(record: Mapping[str, Any]) -> bool:
    """Return whether a record is explicitly terminal."""
    state = _record_state(record)
    if record.get("schema_version") == RECORD_SCHEMA_VERSION_V2:
        return state in TERMINAL_STATES or state == "released"
    return state in LEGACY_TERMINAL_STATUSES


def _is_actionable_record(record: Mapping[str, Any]) -> bool:
    """Return whether a record claims it is ready for concrete execution or evidence use."""
    state = _record_state(record)
    if _is_terminal_record_state(record):
        return False
    if _looks_blocked(state):
        return False
    return state not in PROPOSAL_RECORD_STATES


def _looks_blocked(state: str) -> bool:
    """Return whether a state/status token is a blocked nonterminal."""
    return "blocked" in state


def _issue_number(value: Any) -> int | None:
    """Coerce an issue number field to int when possible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iter_blocker_issue_numbers(record: Mapping[str, Any]) -> Iterable[int]:
    """Yield issue numbers referenced by blocked_on_issue_N status tokens."""
    for field_name in ("status", "state", "blocked_by", "notes"):
        value = record.get(field_name)
        if not isinstance(value, str):
            continue
        for match in re.finditer(r"(?:blocked_on_issue_|#)(\d+)", value):
            yield int(match.group(1))


def _iter_config_paths(record: Mapping[str, Any]) -> Iterable[Path]:
    """Yield repo-relative config and input paths that should exist."""
    for key in ("config", "inputs"):
        value = record.get(key)
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            continue
        for item in items:
            raw_path: str | None = None
            if isinstance(item, str):
                raw_path = item
            elif isinstance(item, Mapping) and isinstance(item.get("path"), str):
                raw_path = item["path"]
            if raw_path is None or _skip_path_existence_check(raw_path):
                continue
            yield Path(raw_path)


def _iter_actionable_placeholder_values(record: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield command and artifact path values that must be concrete for actionable records."""
    command = record.get("command")
    if isinstance(command, str):
        yield "command", command
    for field_name in ("config", "inputs", "outputs", "expected_artifacts"):
        for path in _iter_path_strings(record.get(field_name)):
            yield field_name, path


def _iter_path_strings(value: Any) -> Iterable[str]:
    """Yield path strings from scalar, list, and mapping path fields."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        path = value.get("path")
        if isinstance(path, str):
            yield path
    elif isinstance(value, list):
        for child in value:
            yield from _iter_path_strings(child)


def _skip_path_existence_check(path: str) -> bool:
    """Return whether a path is intentionally not checked as repo-local input."""
    return (
        "TODO" in path
        or path.startswith(("output/", "wandb-artifact://", "http://", "https://"))
        or Path(path).is_absolute()
    )


def _iter_string_values(value: Any) -> Iterable[str]:
    """Yield all string leaves from nested YAML data."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_string_values(child)


def _is_pending_artifact_alias(value: str) -> bool:
    """Return whether a string looks like an unresolved artifact alias."""
    return "wandb-artifact://" in value and (":pending" in value or "/pending/" in value)


def _has_unresolved_artifact_placeholder(value: str) -> bool:
    """Return whether a command or path still contains a placeholder artifact token."""
    return bool(ANGLE_PLACEHOLDER_RE.search(value)) or _is_pending_artifact_alias(value)


def _is_closed_issue_state(value: str | None) -> bool:
    """Return whether a GitHub issue state token is closed."""
    return str(value or "").upper() == "CLOSED"


def _repo_root_for_registry(registry_path: Path) -> Path:
    """Infer the repository root for a registry path."""
    if registry_path.parent.name == "experiments":
        return registry_path.parent.parent
    return registry_path.parent


def _validate_vocab(
    value: Any,
    field_name: str,
    allowed_values: set[str],
    display_path: str,
    errors: list[str],
) -> None:
    """Append an error when a controlled-vocabulary value is unknown."""
    if value is not None and value not in allowed_values:
        errors.append(f"{display_path}: {field_name} must be one of {sorted(allowed_values)}")


def _iter_artifact_items(value: Any) -> Iterable[Mapping[str, Any]]:
    """Yield artifact mappings from string or mapping lists."""
    if not isinstance(value, list):
        return
    for item in value:
        if isinstance(item, str):
            yield {"path": item}
        elif isinstance(item, Mapping):
            yield item


def _is_local_output_path(path: str) -> bool:
    """Return whether a path points at the disposable worktree output root."""
    parts = Path(path).parts
    if parts and parts[0] == ".":
        parts = parts[1:]
    return bool(parts and parts[0] == "output")


def _is_missing(value: Any) -> bool:
    """Return whether a required registry value is absent or empty."""
    if value is None:
        return True
    return value in ("", [], {})


def _load_yaml_mapping(path: Path, errors: list[str], *, display: str) -> Mapping[str, Any] | None:
    """Load a YAML file and return a mapping, appending errors instead of raising."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        errors.append(f"{display}: cannot read YAML: {exc}")
        return None
    except yaml.YAMLError as exc:
        errors.append(f"{display}: invalid YAML: {exc}")
        return None
    if not isinstance(payload, Mapping):
        errors.append(f"{display}: YAML document must be a mapping")
        return None
    return payload


def _load_issue_state_snapshot(
    path: Path,
    errors: list[str] | None = None,
) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Load a compact issue state snapshot for offline drift checks."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if errors is not None:
            errors.append(f"{path}: cannot read issue-state JSON: {exc}")
        return {}, {}
    if isinstance(payload, Mapping):
        rows = payload.get("issues", payload.get("items", payload))
    else:
        rows = payload
    states: dict[int, str] = {}
    labels: dict[int, list[str]] = {}
    if isinstance(rows, Mapping):
        iterable = rows.values()
    else:
        iterable = rows if isinstance(rows, list) else []
    for row in iterable:
        if not isinstance(row, Mapping):
            continue
        number = _issue_number(row.get("number"))
        if number is None:
            continue
        if isinstance(row.get("state"), str):
            states[number] = row["state"]
        label_values = row.get("labels")
        if isinstance(label_values, list):
            labels[number] = _parse_issue_label_names(label_values)
    return states, labels


def _parse_issue_label_names(label_values: Iterable[Any]) -> list[str]:
    """Return label names from GitHub issue snapshot rows."""
    parsed_labels: list[str] = []
    for label in label_values:
        value = label.get("name") if isinstance(label, Mapping) else label
        if value is None:
            continue
        parsed_labels.append(str(value))
    return parsed_labels


def _apply_labels_from_cli_args(
    args: argparse.Namespace,
    report: dict[str, Any] | None,
    errors: list[str],
) -> None:
    """Apply derived label updates for CLI mode, appending errors instead of raising."""

    if not args.apply_labels:
        return
    if args.control_plane_report_json is None:
        errors.append("--apply-labels requires --control-plane-report-json")
    if args.issue_state_json is None:
        errors.append("--apply-labels requires --issue-state-json with current labels")
    if args.max_writes < 1:
        errors.append("--apply-labels requires --max-writes greater than 0")
    if report is None or errors:
        return

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    audit_log = args.label_audit_log or (DEFAULT_STATE_LABEL_AUDIT_DIR / f"{timestamp}.json")
    try:
        audit = apply_derived_issue_label_updates(
            report,
            max_writes=args.max_writes,
            audit_log_path=audit_log,
            min_rate_limit_remaining=args.min_rate_limit_remaining,
        )
    except RuntimeError as exc:
        errors.append(str(exc))
        return
    print(
        f"applied {audit['applied_count']} derived issue state-label updates; "
        f"audit log: {audit_log}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the experiment registry validator CLI."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  uv run python scripts/tools/validate_experiment_registry.py "
            "[experiments/registry.yaml]"
        ),
    )
    parser.add_argument(
        "registry",
        nargs="?",
        default="experiments/registry.yaml",
        type=Path,
        help="Path to the experiment registry index.",
    )
    parser.add_argument(
        "--issue-state-json",
        type=Path,
        help="Optional compact issue snapshot with number/state/labels for drift reporting.",
    )
    parser.add_argument(
        "--control-plane-report-json",
        type=Path,
        help="Write a dry-run research control-plane drift report.",
    )
    parser.add_argument(
        "--apply-labels",
        action="store_true",
        help="Apply derived state:* label updates from the control-plane report via gh.",
    )
    parser.add_argument(
        "--max-writes",
        type=int,
        default=0,
        help="Maximum issue-label writes allowed with --apply-labels.",
    )
    parser.add_argument(
        "--min-rate-limit-remaining",
        type=int,
        default=25,
        help="Minimum GitHub core API quota required before --apply-labels writes.",
    )
    parser.add_argument(
        "--label-audit-log",
        type=Path,
        help="Optional audit-log path for --apply-labels.",
    )
    args = parser.parse_args(argv)

    errors = validate_registry(args.registry)
    report: dict[str, Any] | None = None
    if args.control_plane_report_json is not None:
        issue_states: dict[int, str] = {}
        issue_labels: dict[int, list[str]] = {}
        if args.issue_state_json is not None:
            issue_states, issue_labels = _load_issue_state_snapshot(
                args.issue_state_json, errors=errors
            )
        report = build_control_plane_report(
            args.registry,
            issue_states=issue_states,
            issue_labels=issue_labels,
        )
        args.control_plane_report_json.parent.mkdir(parents=True, exist_ok=True)
        args.control_plane_report_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"wrote control-plane report: {args.control_plane_report_json} "
            f"({report['finding_count']} findings)"
        )
        if report["derived_update_count"] and not args.apply_labels:
            errors.append(
                f"derived issue state-label updates pending: {report['derived_update_count']} "
                "updates; rerun with --apply-labels and --max-writes to apply"
            )
    _apply_labels_from_cli_args(args, report, errors)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"validated experiment registry: {args.registry}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
