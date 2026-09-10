#!/usr/bin/env python3
# ruff: noqa: C901, PLR0912, PLR0913, PLR0915
"""Shared, fail-closed issue-audit inventory, classification, and mutation core.

The two issue-audit skills deliberately have different authority boundaries, but
they consume the same plan produced here.  This module keeps the policy-bearing
parts deterministic and testable:

* bounded REST inventory of issues, pull requests, labels, and repository state;
* local correlation of issue references with PRs, claims, worktrees, and jobs;
* composable versus mutually-exclusive label classification;
* conservative mutation planning and URI-safe label deletion;
* bounded mutation execution followed by REST readback; and
* explicit closure and decision-gate evidence.

The module never creates labels and never writes Project #5 fields.  Callers may
use the plan as a dry-run artifact or apply only the operations it contains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import yaml

from scripts.dev import prepare_open_issue_contracts
from scripts.dev.github_quota import (
    DEFAULT_CORE_SAFETY_THRESHOLD,
    RateLimitSnapshot,
    core_quota_reset_handoff,
    fetch_rate_limit_snapshot,
)
from scripts.dev.issue_completion_receipt import admit_completion_receipt
from scripts.dev.issue_state_taxonomy import (
    EXECUTION_STATE_LABELS,
    STATE_QUALIFIER_LABELS,
)
from scripts.dev.issue_state_taxonomy import (
    execution_state_labels as shared_execution_state_labels,
)
from scripts.dev.issue_state_taxonomy import (
    state_labels as shared_state_labels,
)
from scripts.tools.issue_archetype_sync import SAFE_ARCHETYPE_LABEL_MAP
from scripts.tools.issue_template_audit import audit_archetype_metadata

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"
PER_PAGE = 100
DEFAULT_MAX_PAGES = 10
DEFAULT_MAX_CLOSED_PR_PAGES = 50
DEFAULT_MAX_COMMENT_PAGES = 3
DEFAULT_MAX_TIMELINE_PAGES = 3
DEFAULT_MAX_MUTATIONS = 250
DEFAULT_GH_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_AUDIT_WALL_SECONDS = 120.0
PLAN_SCHEMA = "issue_audit_plan.v1"
PROVENANCE_SCHEMA = "issue_audit_provenance.v1"
ENVELOPE_SCHEMA = "issue_decision_envelope.v1"
MAX_SOURCE_EXCERPT = 280
ISSUE_SOURCE_STATUS_COMPLETE = "complete"
ISSUE_SOURCE_STATUS_EMPTY = "empty"
ISSUE_SOURCE_STATUS_UNAVAILABLE = "unavailable"
ISSUE_SOURCE_STATUS_ANOMALOUS = "anomalous"
ISSUE_SOURCE_EMPTY_PROOF = "successful_empty_response"
ISSUE_SOURCE_KIND = "canonical_open_issues"
LEGACY_ISSUE_INVENTORY_MARKER = "legacy_issue_inventory"
EXPECTED_GITHUB_HOST = "github.com"
UNCERTAIN_QUOTA_STATUSES = frozenset(
    {"exhausted", "insufficient", "quota_blocked", "failed", "malformed", "unavailable"}
)

_PREPARATION_AUDIT_SCHEMA = "open_issue_contract_audit.v1"
PREPARATION_MARKER_END = prepare_open_issue_contracts.MARKER_END
PREPARATION_MARKER_START = prepare_open_issue_contracts.MARKER_START
PREPARATION_PACKET_SCHEMA = prepare_open_issue_contracts.PACKET_SCHEMA
_PREPARATION_MARKER_RE = re.compile(
    rf"{re.escape(PREPARATION_MARKER_START)}.*?{re.escape(PREPARATION_MARKER_END)}",
    re.DOTALL,
)
_PREPARATION_YAML_RE = re.compile(
    r"```ya?ml[ \t]*\r?\n(?P<content>.*?)\r?\n[ \t]*```",
    re.IGNORECASE | re.DOTALL,
)
_PREPARATION_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PREPARATION_REQUIRED_FIELDS = (
    "schema",
    "repository",
    "issue",
    "source_body_sha256",
    "source_comments_sha256",
    "audit_schema",
    "audit_digest",
    "audit_classification",
    "next_action",
    "authority",
    "execution_mode",
    "preferred_worker",
    "expected_pr_runner_label",
    "implementation_admitted",
    "state_ready_change_proposed",
    "readiness_gate",
    "mutation_batch",
)
_PREPARATION_STRING_FIELDS = (
    "audit_classification",
    "next_action",
    "authority",
    "execution_mode",
    "preferred_worker",
    "expected_pr_runner_label",
    "mutation_batch",
)


class _UniquePreparationLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys and ambiguous booleans."""


def _construct_unique_preparation_mapping(
    loader: yaml.SafeLoader, node: Any, deep: bool = False
) -> dict[Any, Any]:
    """Construct a mapping without silently accepting duplicate keys."""
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in mapping:
                raise ValueError(f"duplicate key {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as exc:
            raise ValueError("mapping key is not hashable") from exc
    return mapping


def _construct_strict_preparation_bool(loader: yaml.SafeLoader, node: Any) -> bool:
    """Accept only YAML's true/false spellings for packet boolean fields."""
    value = loader.construct_scalar(node)
    if value.lower() not in {"true", "false"}:
        raise ValueError("boolean values must be true or false")
    return value.lower() == "true"


_UniquePreparationLoader.add_constructor(
    "tag:yaml.org,2002:map", _construct_unique_preparation_mapping
)
_UniquePreparationLoader.add_constructor(
    "tag:yaml.org,2002:bool", _construct_strict_preparation_bool
)

RESOURCE_PREFIX = "resource:"
TYPE_PREFIX = "type:"
EVIDENCE_PREFIX = "evidence:"
DECISION_LABEL = "decision-required"
TRIAGE_LABEL = "needs-triage"
REVIEW_STATE_LABEL = "state:review"
PARENT_LABELS = frozenset({"epic", "parent", "type:epic"})
BLOCKED_LABELS = frozenset({"state:blocked", "state:blocked-external-input"})
BLOCKED_TRIAGE_BLOCK_RE = re.compile(
    r"<!--\s*blocked-triage-v1\b[^>]*-->.*?```(?:yaml|yml)\s*\n.*?```",
    re.IGNORECASE | re.DOTALL,
)
BLOCKED_BY_REFERENCE_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?blocked\s*-?\s*by\s*:\s*#[1-9][0-9]*\b"
)
_NON_BLOCKING_GATE_VALUE = r"(?:none|no\s+blockers?|n/?a|not\s+applicable|clear|resolved)"
NON_BLOCKING_GATE_RE = re.compile(
    r"(?im)(?:"
    r"(?<!\w)blocked\s*-?\s*by\s*:\s*" + _NON_BLOCKING_GATE_VALUE + r"(?=[ \t]*(?:[.!?]|$))|"
    r"^[ \t]*#{0,6}[ \t]*blocked\s*-?\s*by\s*:?[ \t]*\r?\n"
    r"(?:[ \t]*\r?\n)*[ \t]*" + _NON_BLOCKING_GATE_VALUE + r"[ \t]*(?:[.!?])?[ \t]*$"
    r")"
)

BLOCKER_TERMS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "provenance": (
        ("provenance", "checksum", "digest", "lineage", "seed", "compatibility"),
        (
            "blocked",
            "missing",
            "unavailable",
            "mismatch",
            "incompatible",
            "unsupported",
            "cannot verify",
            "not available",
        ),
    ),
    "rights": (
        ("license", "licensing", "rights", "permission", "consent", "redistribution"),
        (
            "blocked",
            "missing",
            "unavailable",
            "pending",
            "not granted",
            "cannot publish",
            "cannot release",
            "not available",
        ),
    ),
    "compute": (
        ("slurm", "sbatch", "gpu", "compute", "quota", "allocation"),
        (
            "blocked",
            "missing",
            "unavailable",
            "pending",
            "not authorized",
            "authorization unavailable",
            "authorization missing",
            "quota exhausted",
            "quota unavailable",
            "quota exceeded",
            "no allocation",
            "cannot submit",
        ),
    ),
    "external-input": (
        ("external data", "external asset", "checkpoint", "dataset", "model weights"),
        (
            "missing",
            "unavailable",
            "pending",
            "not available",
            "waiting for",
            "requires access",
        ),
    ),
}

DECISION_PATTERNS = (
    re.compile(
        r"\b(?:decision|maintainer\s+(?:decision|choice|approval)|"
        r"(?:owner|maintainer)\s+decisions?)"
        r"\s+(?:is\s+)?(?:required|needed|pending|requested)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:maintainer|owner)\b.{0,40}\b(?:must|needs?\s+to|should)\b"
        r".{0,40}\b(?:choose|select|approve|confirm|decide)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:choose|select)\s+(?:one|an?\s+option|the\s+(?:option|approach|policy|scope))\b"
        r"|\bapprove\s+or\s+waive\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\breopen(?:s|ed|ing)?\b.{0,80}\b(?:decision|ruling|gate)\b",
        re.IGNORECASE,
    ),
)
CANONICAL_RULING_RE = re.compile(
    r"^\s*ll7/robot_sf_ll7#(?P<issue>[1-9][0-9]*)\s*:\s*"
    r"(?P<token>[a-z0-9][a-z0-9._-]*)\s*$"
)
NON_AUTHORITATIVE_RULING_CONTEXT_RE = re.compile(
    r"\b(?:example|cop(?:y|ied|y-pasted)|quote|quoted|sample|historical|"
    r"do\s+not\s+apply|not\s+a\s+ruling)\b",
    re.IGNORECASE,
)
CONDITIONAL_DECISION_REVIVAL_RE = re.compile(
    r"\b(?:reopen(?:s|ed|ing)?|reapply)\b.{0,120}\bonly\s+if\b"
    r"|^\s*any\b.{0,160}\b(?:drift|change|mismatch|failure)\b.{0,120}"
    r"\breopen(?:s|ed|ing)?\b",
    re.IGNORECASE,
)
READY_HEADING_PATTERN = re.compile(
    r"^#{2,4}\s+(?:acceptance(?: criteria)?|definition of done|success criteria|"
    r"validation(?: / testing| command| commands)?|implementation plan)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
CHECKBOX_PATTERN = re.compile(r"^\s*[-*]\s+\[[ xX]\]\s+", re.MULTILINE)
UNCHECKED_CHECKBOX_PATTERN = re.compile(r"^\s*[-*]\s+\[ \]\s+", re.MULTILINE)
TERMINAL_REVIEW_STATUS_LINE_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:report|campaign|execution|run)\s+status\s*:\s*"
    r"(?P<status>.+)$",
    re.IGNORECASE,
)
TERMINAL_REVIEW_STATUS_PATTERN = re.compile(
    r"\b(?:"
    r"diagnostic[_ -]+ready[_ -]+for[_ -]+(?:domain[_ -]+)?(?:review|interpretation)"
    r"|terminal[_ -]+(?:review|interpretation)[_ -]+pending"
    r"|(?:domain[_ -]+review|interpretation)[_ -]+pending"
    r"|(?:complete|completed|terminal)[_ -]+for[_ -]+"
    r"(?:domain[_ -]+review|interpretation)"
    r")\b",
    re.IGNORECASE,
)
ISSUE_REF_PATTERN = re.compile(r"(?<![\w-])#(\d+)\b|(?:issue|issues)[ -](\d+)\b", re.IGNORECASE)
OPTION_LINE_PATTERN = re.compile(
    r"^\s*(?:[-*]\s+)?(?:\*\*)?"
    r"(?:\(([A-Z])\)|option\s+([A-Z])(?:[.:)\-])?)\s+"
    r"(?:\*\*)?(?P<label>.+?)(?:\*\*)?\s*$",
    re.IGNORECASE,
)
DECISION_SOURCE_PATTERN = re.compile(
    r"\b(?:decision|required|needed|pending|choose|select|approve|confirm|option|policy|scope)\b",
    re.IGNORECASE,
)
PARENT_TITLE_PATTERN = re.compile(
    r"\b(parent|roadmap|epic|tracking|multi[- ]slice|umbrella)\b", re.IGNORECASE
)
EXPLICIT_MERGED_CLOSE_PATTERN = re.compile(
    r"(?im)^\s*(?:close|completion|done)\s+"
    r"(?:condition|when|criteria)\s*:\s*.*\b(?:merged?|merge)\b.*\b(?:pr|pull request)\b"
)
PARENT_CLOSE_PATTERN = re.compile(
    r"(?im)^\s*parent\s+close\s+condition\s*:\s*all\s+(?:linked\s+)?children\s+closed\s*$"
)

Runner = Callable[[list[str], str | None], subprocess.CompletedProcess[str]]


@dataclass
class _RestRequestBudget:
    """Track the shared post-preflight REST request budget."""

    initial: int
    remaining: int
    requests_attempted: int = 0
    budget_exhausted: bool = False
    rate_limited: bool = False

    @classmethod
    def from_available(cls, available: object) -> _RestRequestBudget:
        """Normalize a quota-derived request budget without authorizing negatives."""
        try:
            normalized = max(0, int(available))
        except (TypeError, ValueError):
            normalized = 0
        return cls(initial=normalized, remaining=normalized)

    def reserve(self) -> bool:
        """Reserve one REST request, or record that the safe budget is exhausted."""
        if self.remaining <= 0:
            self.budget_exhausted = True
            return False
        self.remaining -= 1
        self.requests_attempted += 1
        return True

    def mark_budget_exhausted(self) -> None:
        """Record that more inventory was needed after the safe budget was spent."""
        self.remaining = 0
        self.budget_exhausted = True

    def mark_rate_limited(self) -> None:
        """Stop all later REST reads after an actual rate-limit response."""
        self.remaining = 0
        self.rate_limited = True


class _AuditDeadlineExceeded(TimeoutError):
    """Signal-driven interruption for one in-process audit phase."""


@contextmanager
def _deadline_interrupt(deadline: float | None) -> Iterator[None]:
    """Interrupt in-process work at the deadline when the host permits it.

    ``SIGALRM`` is intentionally limited to the main thread and to hosts with
    no pre-existing timer. The prior handler is restored before returning.
    Other embedding contexts retain the cooperative checks around each phase
    and invalidate any late result.
    """
    sigalrm = getattr(signal, "SIGALRM", None)
    setitimer = getattr(signal, "setitimer", None)
    getitimer = getattr(signal, "getitimer", None)
    itimer_real = getattr(signal, "ITIMER_REAL", None)
    if (
        deadline is None
        or sigalrm is None
        or setitimer is None
        or getitimer is None
        or itimer_real is None
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        yield
        return
    previous_handler = signal.getsignal(sigalrm)
    previous_timer = getitimer(itimer_real)
    if previous_timer[0] > 0:
        yield
        return

    def handle_deadline(_signum: int, _frame: object) -> None:
        raise _AuditDeadlineExceeded("issue-audit wall-time budget exhausted")

    handler_installed = False
    timer_installed = False
    try:
        signal.signal(sigalrm, handle_deadline)
        handler_installed = True
        setitimer(itimer_real, remaining)
        timer_installed = True
    except (OSError, OverflowError, ValueError):
        if timer_installed:
            setitimer(itimer_real, 0)
        if handler_installed:
            signal.signal(sigalrm, previous_handler)
        yield
        return
    try:
        yield
    finally:
        setitimer(itimer_real, 0)
        signal.signal(sigalrm, previous_handler)


def _deadline_from_seconds(max_wall_seconds: float | None) -> float | None:
    """Return an absolute monotonic deadline for an optional audit budget."""
    if max_wall_seconds is None:
        return None
    if not math.isfinite(max_wall_seconds) or max_wall_seconds < 0:
        raise ValueError("max_wall_seconds must be finite and non-negative")
    return time.monotonic() + max_wall_seconds


def _resolve_deadline(
    max_wall_seconds: float | None,
    deadline: float | None,
) -> float | None:
    """Resolve either a new relative budget or a shared absolute deadline."""
    if max_wall_seconds is not None and deadline is not None:
        raise ValueError("pass max_wall_seconds or deadline, not both")
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("deadline must be finite")
    return deadline if deadline is not None else _deadline_from_seconds(max_wall_seconds)


def _deadline_expired(deadline: float | None) -> bool:
    """Return whether a shared audit deadline has elapsed."""
    return deadline is not None and time.monotonic() >= deadline


def _deadline_timeout_result(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Return the common fail-closed result for work that misses the audit budget."""
    return subprocess.CompletedProcess(
        args,
        124,
        "",
        "issue-audit wall-time budget exhausted",
    )


def _deadline_timeout_inventory(repo: str, remote: str, *, reason: str) -> dict[str, Any]:
    """Return a minimal inventory artifact when discovery is interrupted."""
    return {
        "repo": repo,
        "remote": remote,
        "issues": [],
        "open_prs": [],
        "merged_prs": [],
        "labels": [],
        "claims": {},
        "worktrees": [],
        "jobs": [],
        "inventory": {
            "issues": {
                "available": False,
                "errors": [reason],
                "truncated": False,
            }
        },
    }


def _suppress_mutation_fields(value: object) -> None:
    """Clear every serialized mutation list in an incomplete audit result."""
    if isinstance(value, dict):
        if "mutations" in value:
            value["mutations"] = []
        for nested in value.values():
            _suppress_mutation_fields(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _suppress_mutation_fields(nested)


def _mark_plan_timed_out(plan: dict[str, Any], *, reason: str) -> None:
    """Convert a complete-looking plan into a mutation-free timeout artifact."""
    status_value = plan.get("classification_status")
    status = dict(status_value) if isinstance(status_value, Mapping) else {}
    issue_rows = plan.get("issues")
    classified_issues = (
        len(issue_rows) if isinstance(issue_rows, list) else status.get("classified_issues", 0)
    )
    total_issues = status.get("total_issues", classified_issues)
    remaining = status.get("remaining_issue_numbers", [])
    if not isinstance(remaining, list):
        remaining = []
    status.update(
        {
            "status": "timed_out",
            "reason": reason,
            "classified_issues": classified_issues,
            "total_issues": total_issues,
            "remaining_issue_numbers": remaining,
            "resume_from_issue": remaining[0] if remaining else None,
            "resume_supported": False,
            "resume_requires_fresh_full_inventory": True,
            "mutations_suppressed": True,
        }
    )
    plan["classification_status"] = status
    _suppress_mutation_fields(plan)
    truncated = set(plan.get("truncation_or_errors", []))
    truncated.add("classification")
    plan["truncation_or_errors"] = sorted(truncated)
    counts_value = plan.get("counts")
    counts = dict(counts_value) if isinstance(counts_value, Mapping) else {}
    counts["mutations"] = 0
    counts["truncated_or_error_sources"] = len(truncated)
    plan["counts"] = counts
    plan["plan_digest"] = compute_plan_digest(plan)


def _run_gh(
    args: list[str],
    input_text: str | None = None,
    *,
    timeout_seconds: float = DEFAULT_GH_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run gh with captured text output, returning failures to the caller."""
    if timeout_seconds <= 0:
        return subprocess.CompletedProcess(
            ["gh", *args],
            124,
            "",
            "gh command timed out before it started",
        )
    try:
        return subprocess.run(
            ["gh", *args],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            ["gh", *args],
            127,
            "",
            "gh CLI not found on PATH; install GitHub CLI and authenticate it",
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            ["gh", *args],
            124,
            "",
            f"gh command timed out after {exc.timeout}s",
        )


def _deadline_runner(
    runner: Runner,
    max_wall_seconds: float | None = None,
    *,
    deadline: float | None = None,
) -> Runner:
    """Wrap REST calls in an aggregate wall-time budget without weakening fail-closed reads."""
    effective_deadline = _resolve_deadline(max_wall_seconds, deadline)
    if effective_deadline is None:
        return runner

    def run(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        remaining = effective_deadline - time.monotonic()
        command = ["gh", *args]
        if remaining <= 0:
            return _deadline_timeout_result(command)
        if runner is _run_gh:
            result = _run_gh(
                args,
                input_text,
                timeout_seconds=min(DEFAULT_GH_TIMEOUT_SECONDS, remaining),
            )
        else:
            result = runner(args, input_text)
        if _deadline_expired(effective_deadline):
            return _deadline_timeout_result(command)
        return result

    return run


def _run_command(
    args: list[str],
    *,
    timeout_seconds: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    """Run a local discovery command without turning missing optional tools into writes."""
    if timeout_seconds <= 0:
        return subprocess.CompletedProcess(
            args,
            124,
            "",
            "command timed out before it started",
        )
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(args, 127, "", f"{args[0]} not found on PATH")
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args,
            124,
            "",
            f"{args[0]} command timed out after {exc.timeout}s",
        )


def _deadline_command_runner(
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
    deadline: float | None,
) -> Callable[[list[str]], subprocess.CompletedProcess[str]]:
    """Bound local discovery commands and reject results that finish past the deadline."""
    if deadline is None:
        return runner

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _deadline_timeout_result(args)
        if runner is _run_command:
            result = _run_command(args, timeout_seconds=min(30.0, remaining))
        else:
            result = runner(args)
        if _deadline_expired(deadline):
            return _deadline_timeout_result(args)
        return result

    return run


def _parse_json(result: subprocess.CompletedProcess[str], *, what: str) -> tuple[Any | None, str]:
    """Decode a command result into JSON or a useful fail-closed error."""
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return None, f"{what} failed: {detail or f'exit code {result.returncode}'}"
    try:
        return json.loads(result.stdout), ""
    except json.JSONDecodeError as exc:
        return None, f"{what} returned invalid JSON: {exc}"


def _is_rate_limit_error(error: str) -> bool:
    """Detect whether a command failure is a GitHub REST rate-limit error (issue #8304)."""
    lower = error.lower()
    return bool(
        re.search(r"\b(?:api\s+)?rate\s+limit\s+exceeded\b", lower)
        or re.search(r"\bsecondary\s+rate\s+limit\b", lower)
        or "abuse detection mechanism" in lower
        or re.search(r"\bhttp\s+429\b", lower)
    )


def _runner_or_default(runner: Runner | None) -> Runner:
    return runner or _run_gh


def _issue_ref_numbers(*values: object) -> set[int]:
    """Extract explicit issue references from titles, bodies, branches, and job names."""
    numbers: set[int] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        for match in ISSUE_REF_PATTERN.finditer(value):
            raw = match.group(1) or match.group(2)
            if raw:
                numbers.add(int(raw))
    return numbers


def _label_names(raw: object) -> list[str]:
    """Normalize GitHub issue or repository label rows."""
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, str) and item:
            names.append(item)
        elif isinstance(item, Mapping) and isinstance(item.get("name"), str):
            names.append(str(item["name"]))
    return sorted(set(names))


def _compact_excerpt(text: object, *, limit: int = MAX_SOURCE_EXCERPT) -> str:
    """Normalize and bound evidence text before placing it in a plan or envelope."""
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: max(1, limit - 3)].rstrip()}..."


def _issue_source_rows(issue: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return bounded source metadata for an issue body and its comments."""
    sources = [
        {
            "id": "body",
            "kind": "body",
            "url": str(issue.get("url") or ""),
            "author": str(issue.get("author") or ""),
            "created_at": "",
            "text": str(issue.get("body") or ""),
        }
    ]
    comments = issue.get("comments")
    if isinstance(comments, Sequence) and not isinstance(comments, (str, bytes)):
        for index, comment in enumerate(comments):
            if isinstance(comment, Mapping):
                sources.append(
                    {
                        "id": f"comment:{index}",
                        "kind": "comment",
                        "url": str(comment.get("url") or ""),
                        "author": str(comment.get("user") or comment.get("author") or ""),
                        "created_at": str(comment.get("created_at") or ""),
                        "text": str(comment.get("body") or ""),
                    }
                )
            elif isinstance(comment, str):
                sources.append(
                    {
                        "id": f"comment:{index}",
                        "kind": "comment",
                        "url": "",
                        "author": "",
                        "created_at": "",
                        "text": comment,
                    }
                )
    return sources


def _decision_source_rows(issue: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return source excerpts that support a pending maintainer decision."""
    sources: list[dict[str, str]] = []
    for source in _issue_source_rows(issue):
        matching_lines = [
            _compact_excerpt(line)
            for line in str(source["text"]).splitlines()
            if line.strip() and DECISION_SOURCE_PATTERN.search(line)
        ]
        for excerpt in matching_lines[:3]:
            sources.append(
                {
                    "source_id": source["id"],
                    "kind": source["kind"],
                    "url": source["url"],
                    "author": source["author"],
                    "created_at": source["created_at"],
                    "excerpt": excerpt,
                }
            )
    if not sources and DECISION_LABEL in set(_label_names(issue.get("labels"))):
        sources.append(
            {
                "source_id": "label:decision-required",
                "kind": "label",
                "url": str(issue.get("url") or ""),
                "author": "",
                "created_at": "",
                "excerpt": "decision-required label present",
            }
        )
    return sources


def _documented_options(issue: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract explicit option lines without inventing a maintainer policy."""
    options: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()
    for source in _issue_source_rows(issue):
        for line in str(source["text"]).splitlines():
            match = OPTION_LINE_PATTERN.match(line)
            if not match:
                continue
            token = str(match.group(1) or match.group(2) or "").upper()
            label = _compact_excerpt(match.group("label"))
            if not token or not label or token in seen_tokens:
                continue
            seen_tokens.add(token)
            options.append(
                {
                    "token": token,
                    "label": label,
                    "source_id": source["id"],
                    "source": {
                        "kind": source["kind"],
                        "url": source["url"],
                        "author": source["author"],
                        "created_at": source["created_at"],
                        "excerpt": _compact_excerpt(line),
                    },
                }
            )
    return options


def _normalize_issue(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Project a GitHub issue row onto the plan's stable issue shape."""
    url = str(raw.get("html_url") or raw.get("url") or "")
    raw_user = raw.get("user")
    author = str(
        (raw_user.get("login") if isinstance(raw_user, Mapping) else None)
        or raw.get("author")
        or ""
    )
    raw_assignees = raw.get("assignees")
    assignees = raw_assignees if isinstance(raw_assignees, list) else []
    raw_comments = raw.get("comments")
    comments = raw_comments if isinstance(raw_comments, list) else []
    return {
        "number": int(raw.get("number", 0)),
        "title": str(raw.get("title") or ""),
        "body": str(raw.get("body") or ""),
        "state": str(raw.get("state") or "").lower(),
        "updated_at": str(raw.get("updated_at") or ""),
        "url": url,
        "author": author,
        "labels": _label_names(raw.get("labels")),
        "assignees": sorted(
            str(item.get("login"))
            for item in assignees
            if isinstance(item, Mapping) and item.get("login")
        ),
        "comments": [
            {
                "body": str(item.get("body") or ""),
                "user": str(
                    (
                        item.get("user").get("login")
                        if isinstance(item.get("user"), Mapping)
                        else None
                    )
                    or item.get("author")
                    or ""
                ),
                "url": str(item.get("html_url") or item.get("url") or ""),
                "created_at": str(item.get("created_at") or ""),
            }
            for item in comments
            if isinstance(item, Mapping)
        ],
    }


def _repository_parts(repository: object) -> tuple[str, str] | None:
    """Return the owner and repository components of a GitHub ``owner/repo`` value."""
    if not isinstance(repository, str) or repository != repository.strip():
        return None
    parts = repository.split("/")
    if len(parts) != 2 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part or "") for part in parts):
        return None
    return parts[0], parts[1]


def _canonical_issue_url(url: object, *, number: int, repository: str) -> bool:
    """Return whether a URL is the expected public GitHub issue URL."""
    if not isinstance(url, str) or not url or url != url.strip():
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    parts = _repository_parts(repository)
    if (
        parts is None
        or parsed.scheme != "https"
        or parsed.netloc.casefold() != EXPECTED_GITHUB_HOST
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        return False
    owner, name = parts
    expected_path = f"/{owner}/{name}/issues/{number}"
    return parsed.path.casefold() in {
        expected_path.casefold(),
        f"{expected_path}/".casefold(),
    }


def _valid_issue_timestamp(value: object) -> bool:
    """Return whether a REST timestamp is non-empty and timezone-aware ISO-8601."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        return False
    timestamp = value
    if timestamp.endswith("Z"):
        timestamp = f"{timestamp[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _issue_identity_errors(raw: Mapping[str, Any], *, repository: str) -> list[str]:
    """Return identity errors shared by raw REST and normalized plan issue rows."""
    errors: list[str] = []
    number = raw.get("number")
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        errors.append("number must be a positive integer")
        return errors
    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append("title must be a non-empty string")
    state = raw.get("state")
    if not isinstance(state, str) or state.lower() != "open":
        errors.append("state must be the string 'open'")
    url = raw.get("html_url") if "html_url" in raw else raw.get("url")
    if not _canonical_issue_url(url, number=number, repository=repository):
        errors.append(f"URL must be https://{EXPECTED_GITHUB_HOST}/{repository}/issues/{number}")
    return errors


def _nested_login_errors(value: object, *, field: str) -> list[str]:
    """Validate a GitHub user-like nested mapping used by an issue row."""
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return [f"{field} must be an object or null"]
    login = value.get("login")
    if not isinstance(login, str) or not login.strip():
        return [f"{field}.login must be a non-empty string"]
    return []


def _issue_row_validation_errors(
    raw: Mapping[str, Any], *, repository: str, normalized: bool
) -> list[str]:
    """Validate issue fields before any normalization can coerce malformed values."""
    errors = _issue_identity_errors(raw, repository=repository)
    number = raw.get("number")
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        return errors

    if not isinstance(raw.get("updated_at"), str) or not _valid_issue_timestamp(
        raw.get("updated_at")
    ):
        errors.append("updated_at must be a non-empty timezone-aware ISO-8601 timestamp")

    for field in ("body", "author"):
        if field in raw and raw[field] is not None and not isinstance(raw[field], str):
            errors.append(f"{field} must be a string or null")

    raw_user = raw.get("user")
    if normalized:
        if "user" in raw and raw_user is not None and not isinstance(raw_user, str):
            errors.append("user must be a string or null in a normalized row")
    else:
        errors.extend(_nested_login_errors(raw_user, field="user"))

    raw_labels = raw.get("labels")
    if normalized:
        if not isinstance(raw_labels, list):
            errors.append("labels must be a list of strings")
        else:
            for index, label in enumerate(raw_labels):
                if not isinstance(label, str) or not label.strip():
                    errors.append(f"labels[{index}] must be a non-empty string")
    elif "labels" in raw:
        if not isinstance(raw_labels, list):
            errors.append("labels must be a list")
        else:
            for index, label in enumerate(raw_labels):
                if isinstance(label, str):
                    if not label.strip():
                        errors.append(f"labels[{index}] must be a non-empty string")
                elif isinstance(label, Mapping):
                    name = label.get("name")
                    if not isinstance(name, str) or not name.strip():
                        errors.append(f"labels[{index}].name must be a non-empty string")
                else:
                    errors.append(f"labels[{index}] must be a string or object")

    raw_assignees = raw.get("assignees")
    if "assignees" in raw:
        if not isinstance(raw_assignees, list):
            errors.append("assignees must be a list")
        else:
            for index, assignee in enumerate(raw_assignees):
                if normalized:
                    if not isinstance(assignee, str) or not assignee.strip():
                        errors.append(f"assignees[{index}] must be a non-empty string")
                elif not isinstance(assignee, Mapping):
                    errors.append(f"assignees[{index}] must be an object")
                else:
                    errors.extend(
                        f"assignees[{index}].{error.removeprefix('user.')}"
                        if error.startswith("user.")
                        else f"assignees[{index}]: {error}"
                        for error in _nested_login_errors(assignee, field="user")
                    )

    raw_comments = raw.get("comments")
    if "comments" in raw:
        if (
            isinstance(raw_comments, int)
            and not isinstance(raw_comments, bool)
            and raw_comments >= 0
        ):
            pass
        elif not isinstance(raw_comments, list):
            errors.append("comments must be a list")
        else:
            for index, comment in enumerate(raw_comments):
                if not isinstance(comment, Mapping):
                    errors.append(f"comments[{index}] must be an object")
                    continue
                for field in ("body", "author", "html_url", "url", "created_at"):
                    if (
                        field in comment
                        and comment[field] is not None
                        and not isinstance(comment[field], str)
                    ):
                        errors.append(f"comments[{index}].{field} must be a string or null")
                if comment.get("created_at") is not None and comment.get("created_at") != "":
                    if not _valid_issue_timestamp(comment.get("created_at")):
                        errors.append(f"comments[{index}].created_at must be a valid timestamp")
                comment_user = comment.get("user")
                if normalized:
                    if (
                        "user" in comment
                        and comment_user is not None
                        and not isinstance(comment_user, str)
                    ):
                        errors.append(f"comments[{index}].user must be a string or null")
                else:
                    errors.extend(
                        f"comments[{index}].{error}"
                        for error in _nested_login_errors(comment_user, field="user")
                    )
    return errors


def _has_canonical_issue_identity(raw: Mapping[str, Any], repository: str = DEFAULT_REPO) -> bool:
    """Return whether a row has positive identity and a repository-bound public URL."""
    return not _issue_identity_errors(raw, repository=repository)


def _is_canonical_issue_row(raw: Mapping[str, Any], repository: str = DEFAULT_REPO) -> bool:
    """Return whether a REST object is safe to normalize into the issue inventory."""
    return not _issue_row_validation_errors(raw, repository=repository, normalized=False)


def _plan_issue_row_numbers(plan: Mapping[str, Any]) -> tuple[set[int], list[str]]:
    """Validate normalized plan issue rows and return their canonical numbers."""
    if "issues" not in plan:
        return set(), []
    raw_rows = plan.get("issues")
    if not isinstance(raw_rows, list):
        return set(), ["plan issues must be a list"]
    repository = plan.get("repo", DEFAULT_REPO)
    if not isinstance(repository, str) or _repository_parts(repository) is None:
        return set(), ["plan repo must be a GitHub owner/repository string"]

    numbers: set[int] = set()
    errors: list[str] = []
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            errors.append(f"plan issue row {index} is not an object")
            continue
        row_errors = _issue_row_validation_errors(
            raw_row,
            repository=repository,
            normalized=True,
        )
        number = raw_row.get("number")
        if row_errors:
            errors.extend(f"plan issue row {index}: {error}" for error in row_errors)
        if isinstance(number, int) and not isinstance(number, bool) and number > 0:
            if number in numbers:
                errors.append(f"plan contains duplicate canonical issue row #{number}")
            numbers.add(number)
    return numbers, errors


def _pending_reference_number(item: Mapping[str, Any]) -> int | None:
    """Return a pending decision's exact issue number when its reference is valid."""
    raw_issue = item.get("issue")
    if not isinstance(raw_issue, str):
        return None
    match = re.fullmatch(r"#?([1-9][0-9]*)", raw_issue.strip())
    if not match:
        return None
    issue_number = int(match.group(1))
    raw_number = item.get("number")
    if raw_number is None:
        return issue_number
    if isinstance(raw_number, bool) or not isinstance(raw_number, int) or raw_number <= 0:
        return None
    return raw_number if raw_number == issue_number else None


def _positive_issue_number(value: object) -> int | None:
    """Return a positive integer issue number without coercing malformed values."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _pending_issue_binding_errors(
    item: Mapping[str, Any], row: Mapping[str, Any], *, index: int
) -> list[str]:
    """Ensure pending decision data is bound to the exact canonical plan row."""
    number = row.get("number")
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        return [f"pending decision {index} has no valid canonical issue row"]

    expected_decision_evidence = row.get("decision_evidence")
    if not isinstance(expected_decision_evidence, list):
        return [f"canonical issue row #{number} has invalid decision evidence"]
    expected_sources = row.get("decision_sources")
    if not isinstance(expected_sources, list):
        return [f"canonical issue row #{number} has invalid decision sources"]
    expected_options = row.get("documented_options")
    if not isinstance(expected_options, list):
        return [f"canonical issue row #{number} has invalid documented options"]
    if row.get("decision_required") is not True:
        return [f"canonical issue row #{number} is not marked decision-required"]
    expected = {
        "issue": f"#{number}",
        "number": number,
        "title": row.get("title"),
        "url": row.get("url"),
        "state": row.get("state"),
        "labels": _label_names(row.get("labels")),
        "classification": row.get("classification"),
        "decision_required": True,
        "question_source": "issue body/comments",
        "blocking_evidence": "; ".join(str(value) for value in expected_decision_evidence)
        or "decision gate detected",
        "decision_evidence": expected_decision_evidence,
        "evidence_sources": expected_sources,
        "documented_options": expected_options,
    }
    errors: list[str] = []
    for field, expected_value in expected.items():
        if field not in item:
            errors.append(f"pending decision {index} is missing {field}")
            continue
        actual_value = item[field]
        if field == "labels":
            if not isinstance(actual_value, list) or not all(
                isinstance(label, str) and label.strip() for label in actual_value
            ):
                errors.append(f"pending decision {index} has invalid labels")
                continue
            actual_value = _label_names(actual_value)
        elif field == "state":
            if not isinstance(actual_value, str):
                errors.append(f"pending decision {index} has invalid state")
                continue
            actual_value = actual_value.lower()
        if actual_value != expected_value:
            errors.append(
                f"pending decision {index} {field} does not match canonical issue row #{number}"
            )
    return errors


def _plan_issue_reference_errors(plan: Mapping[str, Any], *, issue_numbers: set[int]) -> list[str]:
    """Reject mutations and decisions absent from or forged against plan issue rows."""
    errors: list[str] = []
    raw_mutations = plan.get("mutations")
    if isinstance(raw_mutations, list):
        for index, mutation in enumerate(raw_mutations):
            if not isinstance(mutation, Mapping):
                continue
            number = mutation.get("issue")
            if (
                isinstance(number, int)
                and not isinstance(number, bool)
                and number > 0
                and number not in issue_numbers
            ):
                errors.append(f"mutation {index} references absent canonical issue #{number}")

    if "pending_decisions" not in plan:
        return errors
    pending = plan.get("pending_decisions")
    if not isinstance(pending, list):
        errors.append("plan pending_decisions must be a list")
        return errors
    raw_rows = plan.get("issues")
    rows_by_number = {
        row["number"]: row
        for row in raw_rows
        if isinstance(row, Mapping)
        and isinstance(row.get("number"), int)
        and not isinstance(row.get("number"), bool)
        and row["number"] in issue_numbers
    }
    for index, item in enumerate(pending):
        if not isinstance(item, Mapping):
            errors.append(f"pending decision {index} is not an object")
            continue
        number = _pending_reference_number(item)
        if number is None:
            errors.append(f"pending decision {index} has no exact positive issue reference")
        elif number not in issue_numbers:
            errors.append(f"pending decision {index} references absent canonical issue #{number}")
        elif number in rows_by_number:
            errors.extend(_pending_issue_binding_errors(item, rows_by_number[number], index=index))
    return errors


def _normalize_pr(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Project a GitHub pull request row onto the stable correlation shape."""
    head = raw.get("head") if isinstance(raw.get("head"), Mapping) else {}
    linked_issue_numbers = raw.get("linked_issue_numbers")
    linked_issue_numbers = linked_issue_numbers if isinstance(linked_issue_numbers, list) else []
    return {
        "number": int(raw.get("number", 0)),
        "title": str(raw.get("title") or ""),
        "body": str(raw.get("body") or ""),
        "state": str(raw.get("state") or "").lower(),
        "url": str(raw.get("html_url") or raw.get("url") or ""),
        "merged_at": str(raw.get("merged_at") or ""),
        "head_ref": str(head.get("ref") or raw.get("head_ref") or ""),
        "linked_issue_numbers": sorted(
            {int(number) for number in linked_issue_numbers if isinstance(number, int)}
        ),
    }


def paginate_rest(
    path: str,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    per_page: int = PER_PAGE,
    max_requests: int | None = None,
    runner: Runner | None = None,
    request_budget: _RestRequestBudget | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a bounded REST collection and report truncation instead of guessing."""
    if max_pages <= 0 or per_page <= 0:
        raise ValueError("max_pages and per_page must be positive")
    run = _runner_or_default(runner)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    pages_read = 0
    requests_attempted = 0
    raw_row_count = 0
    non_object_row_count = 0
    truncated = False
    rate_limited = False
    budget_exhausted = False
    request_limit_exhausted = False
    for page in range(1, max_pages + 1):
        separator = "&" if "?" in path else "?"
        endpoint = f"{path}{separator}per_page={per_page}&page={page}"
        if max_requests is not None and requests_attempted >= max_requests:
            request_limit_exhausted = True
            truncated = True
            if request_budget is not None and request_budget.remaining <= 0:
                request_budget.mark_budget_exhausted()
                budget_exhausted = True
            errors.append(f"{endpoint} skipped: direct request limit exhausted")
            break
        if request_budget is not None and not request_budget.reserve():
            budget_exhausted = True
            truncated = True
            errors.append(f"{endpoint} skipped: REST request budget exhausted")
            break
        requests_attempted += 1
        payload, error = _parse_json(run(["api", endpoint], None), what=endpoint)
        if error:
            errors.append(error)
            if _is_rate_limit_error(error):
                rate_limited = True
                truncated = True
                if request_budget is not None:
                    request_budget.mark_rate_limited()
            break
        if not isinstance(payload, list):
            errors.append(f"{endpoint} returned a non-list payload")
            break
        pages_read += 1
        raw_row_count += len(payload)
        non_object_row_count += sum(not isinstance(item, dict) for item in payload)
        rows.extend(item for item in payload if isinstance(item, dict))
        if len(payload) < per_page:
            break
    else:
        truncated = True
        if request_budget is not None and request_budget.remaining <= 0:
            request_budget.mark_budget_exhausted()
            budget_exhausted = True
    meta = {
        "pages_read": pages_read,
        "requests_attempted": requests_attempted,
        "per_page": per_page,
        "page_budget": max_pages,
        "row_count": len(rows),
        "raw_row_count": raw_row_count,
        "non_object_row_count": non_object_row_count,
        "truncated": truncated,
        "errors": errors,
    }
    if rate_limited:
        meta["rate_limited"] = True
        meta["quota_exhausted"] = True
    if budget_exhausted:
        meta["budget_exhausted"] = True
    if request_limit_exhausted:
        meta["request_limit_exhausted"] = True
    return rows, meta


def discover_open_issues(
    repo: str = DEFAULT_REPO,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    runner: Runner | None = None,
    request_budget: _RestRequestBudget | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover canonical open issues, filtering pull requests from the issues endpoint."""
    source = f"repos/{repo}/issues?state=open"
    rows, meta = paginate_rest(
        source,
        max_pages=max_pages,
        runner=runner,
        request_budget=request_budget,
    )
    malformed_object_row_count = 0
    issues: list[dict[str, Any]] = []
    for row in rows:
        if "pull_request" in row:
            continue
        if not _is_canonical_issue_row(row, repository=repo):
            malformed_object_row_count += 1
            continue
        issues.append(_normalize_issue(row))
    issues = sorted(issues, key=lambda item: item["number"])
    raw_row_count = int(meta.get("raw_row_count", 0))
    non_object_row_count = int(meta.get("non_object_row_count", 0))
    errors = list(meta.get("errors", []))
    if meta.get("truncated") or errors:
        source_status = ISSUE_SOURCE_STATUS_UNAVAILABLE
        source_reason = f"{source} did not provide a complete response" + (
            f": {errors[0]}" if errors else ""
        )
    elif non_object_row_count:
        source_status = ISSUE_SOURCE_STATUS_ANOMALOUS
        source_reason = (
            f"{source} returned {non_object_row_count} malformed non-object row(s); "
            "the canonical issue inventory is not trustworthy"
        )
    elif malformed_object_row_count:
        source_status = ISSUE_SOURCE_STATUS_ANOMALOUS
        source_reason = (
            f"{source} returned {malformed_object_row_count} malformed issue object row(s); "
            "the canonical issue inventory is not trustworthy"
        )
    elif not issues and raw_row_count:
        source_status = ISSUE_SOURCE_STATUS_ANOMALOUS
        source_reason = (
            f"{source} returned {raw_row_count} row(s) but no canonical issue rows; "
            "the empty queue is not proven"
        )
    elif issues:
        source_status = ISSUE_SOURCE_STATUS_COMPLETE
        source_reason = "canonical open-issue response is complete"
    elif not issues and int(meta.get("pages_read", 0)) > 0:
        source_status = ISSUE_SOURCE_STATUS_EMPTY
        source_reason = "successful empty response from the canonical open-issue source"
    else:
        source_status = ISSUE_SOURCE_STATUS_UNAVAILABLE
        source_reason = f"{source} returned no readable page"
    source_metadata = {
        **meta,
        "row_count": len(issues),
        "canonical_row_count": len(issues),
        "source": source,
        "source_kind": ISSUE_SOURCE_KIND,
        "source_status": source_status,
        "available": source_status in {ISSUE_SOURCE_STATUS_COMPLETE, ISSUE_SOURCE_STATUS_EMPTY},
        "source_status_reason": source_reason,
        "malformed_object_row_count": malformed_object_row_count,
    }
    if source_status == ISSUE_SOURCE_STATUS_EMPTY:
        source_metadata["source_proof"] = ISSUE_SOURCE_EMPTY_PROOF
    elif source_status == ISSUE_SOURCE_STATUS_COMPLETE:
        source_metadata["source_proof"] = "canonical_issue_rows"
    return issues, source_metadata


def discover_issue_comments(
    repo: str,
    issue_number: int,
    *,
    max_pages: int = DEFAULT_MAX_COMMENT_PAGES,
    runner: Runner | None = None,
    request_budget: _RestRequestBudget | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Read one complete, bounded issue comment thread through REST."""
    rows, meta = paginate_rest(
        f"repos/{repo}/issues/{issue_number}/comments",
        max_pages=max_pages,
        runner=runner,
        request_budget=request_budget,
    )
    comments: list[dict[str, str]] = []
    errors = list(meta.get("errors", []))
    non_object_row_count = int(meta.get("non_object_row_count", 0))
    if non_object_row_count:
        errors.append(
            f"comment response contained {non_object_row_count} malformed non-object row(s)"
        )
    malformed_object_row_count = int(meta.get("malformed_object_row_count", 0))
    for index, row in enumerate(rows):
        row_errors: list[str] = []
        for field in ("body", "author", "html_url", "url", "created_at"):
            if field in row and row[field] is not None and not isinstance(row[field], str):
                row_errors.append(f"{field} must be a string or null")
        if row.get("created_at") not in (None, "") and not _valid_issue_timestamp(
            row.get("created_at")
        ):
            row_errors.append("created_at must be a valid timestamp")
        row_errors.extend(_nested_login_errors(row.get("user"), field="user"))
        if row_errors:
            malformed_object_row_count += 1
            errors.extend(f"comment row {index}: {error}" for error in row_errors)
            continue
        user = row.get("user")
        login = user.get("login") if isinstance(user, Mapping) else None
        comments.append(
            {
                "body": str(row.get("body") or ""),
                "user": str(login or ""),
                "url": str(row.get("html_url") or row.get("url") or ""),
                "created_at": str(row.get("created_at") or ""),
            }
        )
    return comments, {
        **meta,
        "available": not errors and not meta.get("truncated"),
        "errors": errors,
        "malformed_object_row_count": malformed_object_row_count,
        "row_count": len(comments),
    }


def attach_issue_comments(
    repo: str,
    issues: list[dict[str, Any]],
    *,
    max_pages: int = DEFAULT_MAX_COMMENT_PAGES,
    runner: Runner | None = None,
    request_budget: _RestRequestBudget | None = None,
) -> dict[str, Any]:
    """Attach bounded REST comment threads to issues and aggregate read status."""
    metadata: dict[str, Any] = {
        "available": True,
        "issue_count": len(issues),
        "pages_read": 0,
        "row_count": 0,
        "truncated": False,
        "errors": [],
        "processed_issue_count": 0,
        "requests_attempted": 0,
        "non_object_row_count": 0,
        "malformed_object_row_count": 0,
    }
    for issue in issues:
        comments, comment_meta = discover_issue_comments(
            repo,
            int(issue["number"]),
            max_pages=max_pages,
            runner=runner,
            request_budget=request_budget,
        )
        issue["comments"] = comments
        metadata["pages_read"] += int(comment_meta.get("pages_read", 0))
        metadata["requests_attempted"] += int(comment_meta.get("requests_attempted", 0))
        metadata["row_count"] += len(comments)
        metadata["truncated"] = bool(metadata["truncated"] or comment_meta.get("truncated"))
        metadata["errors"].extend(comment_meta.get("errors", []))
        metadata["non_object_row_count"] += int(comment_meta.get("non_object_row_count", 0))
        metadata["malformed_object_row_count"] += int(
            comment_meta.get("malformed_object_row_count", 0)
        )
        metadata["processed_issue_count"] += 1
        if comment_meta.get("rate_limited"):
            metadata["rate_limited"] = True
            metadata["quota_exhausted"] = True
            metadata["quota_uncertain"] = True
            metadata["truncated"] = True
            break
        if comment_meta.get("budget_exhausted"):
            metadata["budget_exhausted"] = True
            metadata["quota_uncertain"] = True
            metadata["truncated"] = True
            break
    metadata["available"] = (
        not metadata["errors"]
        and not metadata["truncated"]
        and not metadata.get("rate_limited")
        and not metadata.get("budget_exhausted")
    )
    return metadata


def discover_pull_requests(
    repo: str = DEFAULT_REPO,
    *,
    state: str = "open",
    max_pages: int = DEFAULT_MAX_PAGES,
    runner: Runner | None = None,
    request_budget: _RestRequestBudget | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover open or closed pull requests through bounded REST pagination."""
    if state not in {"open", "closed"}:
        raise ValueError("state must be open or closed")
    rows, meta = paginate_rest(
        f"repos/{repo}/pulls?state={state}&sort=updated&direction=desc",
        max_pages=max_pages,
        runner=runner,
        request_budget=request_budget,
    )
    prs = [_normalize_pr(row) for row in rows if "/pull/" in str(row.get("html_url") or "")]
    return sorted(prs, key=lambda item: item["number"]), {**meta, "row_count": len(prs)}


def discover_issue_timeline_merged_prs(
    repo: str,
    issue_numbers: Iterable[int],
    *,
    max_pages: int = DEFAULT_MAX_TIMELINE_PAGES,
    max_requests: int | None = None,
    runner: Runner | None = None,
    request_budget: _RestRequestBudget | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover merged PRs linked to issues from bounded timeline events.

    A global ``pulls?state=closed`` scan is intentionally bounded.  When that
    scan is partial, issue timelines provide a narrower, complete-for-the-open-
    issue fallback without using GitHub search or guessing from issue numbers.
    GitHub's ``cross-referenced`` timeline payload includes the linked PR body
    and ``pull_request.merged_at`` fields needed for closure evidence.
    """
    numbers = sorted({int(number) for number in issue_numbers if int(number) > 0})
    run = _runner_or_default(runner)
    by_number: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    pages_read = 0
    requests_attempted = 0
    event_count = 0
    truncated = False
    rate_limited = False
    request_limit_exhausted = False

    for issue_number in numbers:
        if max_requests is not None and requests_attempted >= max_requests:
            request_limit_exhausted = True
            truncated = True
            if request_budget is not None and request_budget.remaining <= 0:
                request_budget.mark_budget_exhausted()
            break
        events, meta = paginate_rest(
            f"repos/{repo}/issues/{issue_number}/timeline",
            max_pages=max_pages,
            max_requests=(None if max_requests is None else max_requests - requests_attempted),
            runner=run,
            request_budget=request_budget,
        )
        pages_read += int(meta.get("pages_read", 0))
        requests_attempted += int(meta.get("requests_attempted", 0))
        event_count += len(events)
        truncated = bool(truncated or meta.get("truncated"))
        request_limit_exhausted = bool(
            request_limit_exhausted or meta.get("request_limit_exhausted")
        )
        errors.extend(f"issue {issue_number}: {error}" for error in meta.get("errors", []))
        if meta.get("rate_limited") or any(
            _is_rate_limit_error(error) for error in meta.get("errors", [])
        ):
            rate_limited = True
            truncated = True
            if request_budget is not None:
                request_budget.mark_rate_limited()
            break
        if meta.get("budget_exhausted"):
            truncated = True
            if request_budget is not None:
                request_budget.mark_budget_exhausted()
            break
        for event in events:
            if str(event.get("event") or "") != "cross-referenced":
                continue
            source = event.get("source")
            source_issue = source.get("issue") if isinstance(source, Mapping) else None
            if not isinstance(source_issue, Mapping):
                continue
            pull_request = source_issue.get("pull_request")
            if not isinstance(pull_request, Mapping) or not pull_request.get("merged_at"):
                continue
            raw_pr = dict(source_issue)
            raw_pr["state"] = "closed"
            raw_pr["merged_at"] = pull_request.get("merged_at")
            raw_pr["html_url"] = source_issue.get("html_url") or pull_request.get("html_url")
            raw_pr["linked_issue_numbers"] = [issue_number]
            normalized = _normalize_pr(raw_pr)
            normalized["coverage_source"] = "targeted_issue_timeline"
            normalized["timeline_issue"] = issue_number
            normalized["timeline_event_created_at"] = str(event.get("created_at") or "")
            if not normalized["number"]:
                continue
            existing = by_number.get(normalized["number"])
            if existing is None:
                by_number[normalized["number"]] = normalized
                continue
            existing["linked_issue_numbers"] = sorted(
                set(existing.get("linked_issue_numbers", []))
                | set(normalized.get("linked_issue_numbers", []))
            )

    metadata = {
        "available": not errors and not truncated,
        "issue_count": len(numbers),
        "pages_read": pages_read,
        "requests_attempted": requests_attempted,
        "page_budget": max_pages,
        "event_count": event_count,
        "row_count": len(by_number),
        "truncated": truncated,
        "errors": errors,
    }
    if rate_limited:
        metadata["rate_limited"] = True
        metadata["quota_exhausted"] = True
    if request_budget is not None and request_budget.budget_exhausted:
        metadata["budget_exhausted"] = True
    if request_limit_exhausted:
        metadata["request_limit_exhausted"] = True
    return sorted(by_number.values(), key=lambda item: item["number"]), metadata


def _merge_merged_pr_rows(
    *collections: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge global and targeted PR rows without losing linked issue evidence."""
    merged: dict[int, dict[str, Any]] = {}
    for collection in collections:
        for row in collection:
            number = int(row.get("number", 0))
            if not number:
                continue
            normalized = dict(row)
            linked = set(normalized.get("linked_issue_numbers", []))
            existing = merged.get(number)
            if existing is None:
                normalized["linked_issue_numbers"] = sorted(linked)
                merged[number] = normalized
                continue
            existing["linked_issue_numbers"] = sorted(
                set(existing.get("linked_issue_numbers", [])) | linked
            )
    return sorted(merged.values(), key=lambda item: item["number"])


def discover_repository_labels(
    repo: str = DEFAULT_REPO,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    runner: Runner | None = None,
    request_budget: _RestRequestBudget | None = None,
) -> tuple[set[str], dict[str, Any]]:
    """Read existing repository labels; never infer a label that was not returned."""
    rows, meta = paginate_rest(
        f"repos/{repo}/labels",
        max_pages=max_pages,
        runner=runner,
        request_budget=request_budget,
    )
    labels = {
        str(row["name"])
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    return labels, {**meta, "row_count": len(labels)}


def discover_claims(
    remote: str = DEFAULT_REMOTE,
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Discover remote issue-claim branches without claiming or changing anything."""
    run = command_runner or _run_command
    command = ["git", "ls-remote", "--heads", remote, "refs/heads/agent-claims/issue-*"]
    result = run(command)
    if result.returncode != 0:
        return {}, {"available": False, "errors": [(result.stderr or result.stdout).strip()]}
    claims: dict[int, dict[str, Any]] = {}
    prefix = "refs/heads/agent-claims/issue-"
    for line in (result.stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[1].startswith(prefix):
            continue
        suffix = parts[1][len(prefix) :]
        if suffix.isdigit():
            claims[int(suffix)] = {"claimed": True, "sha": parts[0], "ref": parts[1]}
    return claims, {"available": True, "count": len(claims), "errors": []}


def discover_worktrees(
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Discover local worktrees and expose branch/path text for issue correlation."""
    run = command_runner or _run_command
    command = ["git", "worktree", "list", "--porcelain"]
    result = run(command)
    if result.returncode != 0:
        return [], {"available": False, "errors": [(result.stderr or result.stdout).strip()]}
    worktrees: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in (result.stdout or "").splitlines() + [""]:
        if line.startswith("worktree "):
            if current:
                worktrees.append(current)
            current = {"path": line.removeprefix("worktree ")}
        elif line.startswith("branch "):
            current["branch"] = line.removeprefix("branch ")
        elif not line and current:
            worktrees.append(current)
            current = {}
    return worktrees, {"available": True, "count": len(worktrees), "errors": []}


def discover_jobs(
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Discover visible SLURM jobs when squeue is available.

    A missing squeue is normal on a non-SLURM machine.  For an issue that
    explicitly requires SLURM, the unavailable inventory is still surfaced so
    the classifier can fail closed instead of declaring readiness.
    """
    run = command_runner or _run_command
    command = ["squeue", "--noheader", "--format=%i|%j|%T|%S"]
    result = run(command)
    if result.returncode != 0:
        return [], {"available": False, "errors": [(result.stderr or result.stdout).strip()]}
    jobs: list[dict[str, str]] = []
    for line in (result.stdout or "").splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            jobs.append(
                {
                    "id": parts[0].strip(),
                    "name": parts[1].strip(),
                    "state": parts[2].strip(),
                    "start": parts[3].strip(),
                }
            )
    return jobs, {"available": True, "count": len(jobs), "errors": []}


def resolve_source_sha(
    repo_root: Path | None = None,
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> str:
    """Resolve the git HEAD commit SHA for the active source tree."""
    run = command_runner or _run_command
    result = run(["git", "rev-parse", "HEAD"])
    if result.returncode == 0:
        candidate = (result.stdout or "").strip()
        if re.fullmatch(r"^[0-9a-f]{40}$", candidate):
            return candidate
    return ""


def resolve_classifier_digest(path: Path | None = None) -> str:
    """Compute the SHA-256 digest of the issue-audit core classifier implementation."""
    target = path or Path(__file__).resolve()
    try:
        data = target.read_bytes()
        return hashlib.sha256(data).hexdigest()
    except OSError:
        return ""


def resolve_producer_identity(
    *,
    task_id: str | None = None,
    worktree: Path | str | None = None,
) -> dict[str, Any]:
    """Gather machine, user, and task identity for audit provenance."""
    hostname = socket.gethostname() or "unknown"
    username = os.environ.get("USER") or "unknown"
    resolved_task_id = (
        task_id
        or os.environ.get("AGENT_RUN_ID")
        or os.environ.get("CODEX_RUN_ID")
        or os.environ.get("CONVERSATION_ID")
        or os.environ.get("TASK_ID")
        or ""
    )
    worktree_str = str(worktree or Path.cwd())
    identity = f"{username}@{hostname}"
    if resolved_task_id:
        identity = f"{identity}:{resolved_task_id}"
    return {
        "identity": identity,
        "hostname": hostname,
        "username": username,
        "task_id": resolved_task_id or None,
        "worktree": worktree_str,
    }


def check_origin_main_freshness(
    remote: str = DEFAULT_REMOTE,
    *,
    fetch: bool = True,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> tuple[bool, str, str, str | None]:
    """Verify that HEAD has incorporated fetched <remote>/main.

    Returns (fresh: bool, head_sha: str, main_sha: str, error_detail: str | None).
    """
    run = command_runner or _run_command
    if fetch:
        fetch_result = run(["git", "fetch", remote, "main"])
        if fetch_result.returncode != 0:
            err = (fetch_result.stderr or fetch_result.stdout).strip() or "git fetch failed"
            return False, "", "", f"failed to fetch {remote}/main: {err}"

    head_res = run(["git", "rev-parse", "HEAD"])
    head_sha = (head_res.stdout or "").strip() if head_res.returncode == 0 else ""
    if not re.fullmatch(r"^[0-9a-f]{40}$", head_sha):
        return False, head_sha, "", "failed to resolve HEAD commit SHA"

    main_res = run(["git", "rev-parse", f"{remote}/main"])
    main_sha = (main_res.stdout or "").strip() if main_res.returncode == 0 else ""
    if not re.fullmatch(r"^[0-9a-f]{40}$", main_sha):
        return False, head_sha, main_sha, f"failed to resolve {remote}/main commit SHA"

    ancestor_res = run(["git", "merge-base", "--is-ancestor", f"{remote}/main", "HEAD"])
    if ancestor_res.returncode == 0:
        return True, head_sha, main_sha, None

    return (
        False,
        head_sha,
        main_sha,
        f"executing revision {head_sha} does not contain {remote}/main ({main_sha})",
    )


def _core_reset_handoff(
    *,
    retry_command: str,
    reset_at: object,
    now: float | None = None,
    subject: str = "GitHub core REST quota exhausted",
) -> dict[str, Any]:
    """Build reset metadata without issuing a second unbudgeted quota request."""
    safe_reset_at = reset_at if type(reset_at) is int and reset_at >= 0 else -1
    handoff = core_quota_reset_handoff(
        retry_command=retry_command,
        now=now,
        reset_at=safe_reset_at,
    )
    if subject != "GitHub core REST quota exhausted":
        handoff["handoff"] = str(handoff["handoff"]).replace(
            "GitHub core REST quota exhausted", subject
        )
    return handoff


def _build_plan_retry_command(
    repo: str,
    *,
    remote: str = DEFAULT_REMOTE,
    mode: str = "autonomous",
    max_pages: int = DEFAULT_MAX_PAGES,
    max_closed_pr_pages: int = DEFAULT_MAX_CLOSED_PR_PAGES,
    include_comments: bool = False,
    max_comment_pages: int = DEFAULT_MAX_COMMENT_PAGES,
    max_mutations: int = DEFAULT_MAX_MUTATIONS,
    max_wall_seconds: float | None = None,
    output: Path | None = None,
    read_only_diagnostic: bool = False,
) -> str:
    """Serialize the bounded plan invocation used by quota-reset handoffs."""
    command = [
        "uv",
        "run",
        "python",
        "scripts/dev/issue_audit_core.py",
        "plan",
        "--repo",
        repo,
        "--remote",
        remote,
        "--mode",
        mode,
        "--max-pages",
        str(max_pages),
        "--max-closed-pr-pages",
        str(max_closed_pr_pages),
    ]
    if read_only_diagnostic:
        command.append("--read-only-diagnostic")
    if include_comments:
        command.append("--include-comments")
    command.extend(
        (
            "--max-comment-pages",
            str(max_comment_pages),
            "--max-mutations",
            str(max_mutations),
        )
    )
    if max_wall_seconds is not None:
        command.extend(("--max-wall-seconds", str(max_wall_seconds)))
    if output is not None:
        command.extend(("--output", str(output)))
    return shlex.join(command)


def _skipped_rest_metadata(
    reason: str,
    *,
    page_budget: int,
    per_page: int = PER_PAGE,
    **extra: Any,
) -> dict[str, Any]:
    """Describe a REST collection that was not read, without implying completeness."""
    metadata: dict[str, Any] = {
        "available": False,
        "pages_read": 0,
        "requests_attempted": 0,
        "per_page": per_page,
        "page_budget": page_budget,
        "row_count": 0,
        "truncated": True,
        "errors": [reason],
        "quota_uncertain": True,
    }
    metadata.update(extra)
    return metadata


def _finalize_inventory_quota(
    quota_meta: dict[str, Any],
    request_budget: _RestRequestBudget,
) -> None:
    """Publish shared REST budget accounting and any reset-aware mid-run handoff."""
    quota_meta["request_budget"] = request_budget.initial
    quota_meta["requests_attempted"] = request_budget.requests_attempted
    quota_meta["requests_remaining"] = request_budget.remaining
    if request_budget.rate_limited:
        handoff = _core_reset_handoff(
            retry_command=str(quota_meta.get("retry_command") or "retry issue-audit plan"),
            reset_at=quota_meta.get("core_reset_at"),
            subject="GitHub core REST quota exhausted",
        )
        reason = "GitHub core REST quota exhausted during inventory discovery"
        errors = [str(error) for error in quota_meta.get("errors", [])]
        if reason not in errors:
            errors.append(reason)
        quota_meta.update(
            {
                "available": False,
                "status": "exhausted",
                "quota_exhausted": True,
                "quota_uncertain": True,
                "rate_limited": True,
                "next_action": "wait_for_reset_and_retry",
                "reason": reason,
                "errors": errors,
                "reset_in_seconds": handoff.get("reset_in_seconds"),
                "retry_after_utc": handoff.get("retry_after_utc"),
                "handoff": handoff.get("handoff", ""),
            }
        )
        return
    if request_budget.budget_exhausted:
        handoff = _core_reset_handoff(
            retry_command=str(quota_meta.get("retry_command") or "retry issue-audit plan"),
            reset_at=quota_meta.get("core_reset_at"),
            subject="GitHub core REST request budget exhausted",
        )
        reason = "GitHub core REST request budget exhausted before inventory completed"
        errors = [str(error) for error in quota_meta.get("errors", [])]
        if reason not in errors:
            errors.append(reason)
        quota_meta.update(
            {
                "available": False,
                "status": "insufficient",
                "quota_exhausted": False,
                "quota_uncertain": True,
                "budget_exhausted": True,
                "next_action": "wait_for_reset_and_retry",
                "reason": reason,
                "errors": errors,
                "reset_in_seconds": handoff.get("reset_in_seconds"),
                "retry_after_utc": handoff.get("retry_after_utc"),
                "handoff": handoff.get("handoff", ""),
            }
        )
        return
    if quota_meta.get("status") != "ok" or quota_meta.get("available") is False:
        quota_meta["quota_uncertain"] = True


def discover_core_quota(
    repo: str = DEFAULT_REPO,
    *,
    runner: Runner | None = None,
    min_core_remaining: int = DEFAULT_CORE_SAFETY_THRESHOLD,
    now: float | None = None,
    retry_command: str | None = None,
) -> tuple[RateLimitSnapshot, dict[str, Any]]:
    """Check GitHub core REST capacity before high-volume enrichment."""
    snapshot = fetch_rate_limit_snapshot(runner=runner)
    cmd = retry_command or _build_plan_retry_command(repo)
    if snapshot.status != "ok":
        actual_quota_exhausted = _is_rate_limit_error(snapshot.error)
        subject = (
            "GitHub core REST quota exhausted"
            if actual_quota_exhausted
            else "GitHub core REST quota preflight unavailable"
        )
        handoff = _core_reset_handoff(
            retry_command=cmd,
            now=now,
            reset_at=snapshot.core_reset_at,
            subject=subject,
        )
        meta = {
            "available": False,
            "status": snapshot.status,
            "core_remaining": snapshot.core_remaining,
            "core_reset_at": snapshot.core_reset_at,
            "reset_in_seconds": handoff.get("reset_in_seconds"),
            "retry_after_utc": handoff.get("retry_after_utc"),
            "retry_command": cmd,
            "next_action": (
                "wait_for_reset_and_retry" if actual_quota_exhausted else "retry_preflight"
            ),
            "handoff": handoff.get("handoff", ""),
            "reason": f"rate_limit_{snapshot.status}: {snapshot.error or 'core quota check failed'}",
            "errors": [snapshot.error or f"rate_limit_{snapshot.status}"],
            "quota_exhausted": actual_quota_exhausted,
            "quota_uncertain": True,
        }
        return snapshot, meta

    assert snapshot.core_remaining is not None
    available_budget = max(0, snapshot.core_remaining - min_core_remaining)
    if available_budget <= 0:
        handoff = _core_reset_handoff(
            retry_command=cmd,
            now=now,
            reset_at=snapshot.core_reset_at,
        )
        meta = {
            "available": False,
            "status": "exhausted",
            "core_remaining": snapshot.core_remaining,
            "core_reset_at": snapshot.core_reset_at,
            "reset_in_seconds": handoff.get("reset_in_seconds"),
            "retry_after_utc": handoff.get("retry_after_utc"),
            "retry_command": cmd,
            "next_action": "wait_for_reset_and_retry",
            "handoff": handoff.get("handoff", ""),
            "reason": (
                f"GitHub core REST quota exhausted "
                f"({snapshot.core_remaining} remaining <= safety margin {min_core_remaining})"
            ),
            "errors": [f"GitHub core REST quota exhausted ({snapshot.core_remaining} remaining)"],
            "quota_exhausted": True,
            "quota_uncertain": True,
        }
        return snapshot, meta

    meta = {
        "available": True,
        "status": "ok",
        "core_remaining": snapshot.core_remaining,
        "core_reset_at": snapshot.core_reset_at,
        "available_budget": available_budget,
        "min_core_remaining": min_core_remaining,
        "retry_command": cmd,
        "next_action": "none",
        "reason": "sufficient core quota available",
        "errors": [],
        "quota_exhausted": False,
        "quota_uncertain": False,
        "budget_exhausted": False,
    }
    return snapshot, meta


def discover_inventory(
    repo: str = DEFAULT_REPO,
    *,
    remote: str = DEFAULT_REMOTE,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_closed_pr_pages: int = DEFAULT_MAX_CLOSED_PR_PAGES,
    include_comments: bool = False,
    max_comment_pages: int = DEFAULT_MAX_COMMENT_PAGES,
    max_wall_seconds: float | None = None,
    deadline: float | None = None,
    runner: Runner | None = None,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    retry_command: str | None = None,
) -> dict[str, Any]:
    """Build the complete read-only inventory consumed by the shared classifier."""
    effective_deadline = _resolve_deadline(max_wall_seconds, deadline)
    rest_runner = _deadline_runner(runner or _run_gh, deadline=effective_deadline)
    local_runner = _deadline_command_runner(command_runner or _run_command, effective_deadline)
    effective_retry_command = retry_command or _build_plan_retry_command(
        repo,
        remote=remote,
        max_pages=max_pages,
        max_closed_pr_pages=max_closed_pr_pages,
        include_comments=include_comments,
        max_comment_pages=max_comment_pages,
        max_wall_seconds=max_wall_seconds,
    )
    _quota_snapshot, quota_meta = discover_core_quota(
        repo,
        runner=rest_runner,
        retry_command=effective_retry_command,
    )
    quota_available = bool(quota_meta.get("available"))
    request_budget = _RestRequestBudget.from_available(
        quota_meta.get("available_budget", 0) if quota_available else 0
    )
    preflight_reason = str(
        quota_meta.get("reason") or "core quota preflight did not authorize REST inventory"
    )
    initial_quota_exhausted = bool(quota_meta.get("quota_exhausted"))

    if quota_available:
        issues, issue_meta = discover_open_issues(
            repo,
            max_pages=max_pages,
            runner=rest_runner,
            request_budget=request_budget,
        )
    else:
        issues = []
        issue_meta = _skipped_rest_metadata(
            f"open issue inventory skipped: {preflight_reason}",
            page_budget=max_pages,
            quota_exhausted=initial_quota_exhausted,
        )

    if include_comments:
        if quota_available:
            comment_meta = attach_issue_comments(
                repo,
                issues,
                max_pages=max_comment_pages,
                runner=rest_runner,
                request_budget=request_budget,
            )
        else:
            comment_meta = _skipped_rest_metadata(
                f"comment inventory skipped: {preflight_reason}",
                page_budget=max_comment_pages,
                issue_count=len(issues),
                quota_exhausted=initial_quota_exhausted,
            )
    else:
        comment_meta = {"available": False, "reason": "not_requested", "errors": []}

    if quota_available:
        open_prs, open_pr_meta = discover_pull_requests(
            repo,
            state="open",
            max_pages=max_pages,
            runner=rest_runner,
            request_budget=request_budget,
        )
    else:
        open_prs = []
        open_pr_meta = _skipped_rest_metadata(
            f"open-PR inventory skipped: {preflight_reason}",
            page_budget=max_pages,
            quota_exhausted=initial_quota_exhausted,
        )

    closed_prs: list[dict[str, Any]] = []
    closed_pr_meta: dict[str, Any]
    if not quota_available:
        closed_pr_meta = _skipped_rest_metadata(
            f"closed-PR inventory skipped: {preflight_reason}",
            page_budget=max_closed_pr_pages,
            quota_exhausted=initial_quota_exhausted,
        )
    elif request_budget.rate_limited:
        closed_pr_meta = _skipped_rest_metadata(
            "closed-PR inventory skipped after an actual core REST rate-limit response",
            page_budget=max_closed_pr_pages,
            quota_exhausted=True,
        )
    elif request_budget.remaining <= 0:
        request_budget.mark_budget_exhausted()
        closed_pr_meta = _skipped_rest_metadata(
            "closed-PR inventory skipped after the shared REST request budget was spent",
            page_budget=max_closed_pr_pages,
            budget_exhausted=True,
        )
    else:
        budgeted_closed_pr_pages = min(max_closed_pr_pages, request_budget.remaining)
        closed_prs, closed_pr_meta = discover_pull_requests(
            repo,
            state="closed",
            max_pages=budgeted_closed_pr_pages,
            runner=rest_runner,
            request_budget=request_budget,
        )
        if (
            budgeted_closed_pr_pages < max_closed_pr_pages
            and closed_pr_meta.get("truncated")
            and closed_pr_meta.get("pages_read", 0) >= budgeted_closed_pr_pages
        ):
            closed_pr_meta["truncated"] = True
            closed_pr_meta["bounded_by_request_budget"] = True
            request_budget.mark_budget_exhausted()
        if closed_pr_meta.get("quota_exhausted") or any(
            _is_rate_limit_error(err) for err in closed_pr_meta.get("errors", [])
        ):
            request_budget.mark_rate_limited()
        if closed_pr_meta.get("budget_exhausted"):
            request_budget.mark_budget_exhausted()

    merged_prs = [pr for pr in closed_prs if pr.get("merged_at")]
    timeline_prs: list[dict[str, Any]] = []
    timeline_meta: dict[str, Any] = {
        "available": True,
        "reason": "global closed-PR inventory complete",
        "issue_count": len(issues),
        "pages_read": 0,
        "requests_attempted": 0,
        "page_budget": max_pages,
        "event_count": 0,
        "row_count": 0,
        "truncated": False,
        "errors": [],
    }
    global_closed_prs_complete = (
        not closed_pr_meta.get("truncated")
        and not closed_pr_meta.get("errors")
        and not closed_pr_meta.get("quota_exhausted")
        and not closed_pr_meta.get("budget_exhausted")
    )
    if not global_closed_prs_complete:
        timeline_page_budget = min(max_pages, DEFAULT_MAX_TIMELINE_PAGES)
        if not quota_available:
            timeline_meta = _skipped_rest_metadata(
                f"timeline enrichment skipped: {preflight_reason}",
                page_budget=timeline_page_budget,
                issue_count=len(issues),
                event_count=0,
                quota_exhausted=initial_quota_exhausted,
            )
        elif request_budget.rate_limited:
            timeline_meta = _skipped_rest_metadata(
                "timeline enrichment skipped after an actual core REST rate-limit response",
                page_budget=timeline_page_budget,
                issue_count=len(issues),
                event_count=0,
                quota_exhausted=True,
            )
        elif request_budget.remaining <= 0:
            request_budget.mark_budget_exhausted()
            timeline_meta = _skipped_rest_metadata(
                "timeline enrichment skipped after the shared REST request budget was spent",
                page_budget=timeline_page_budget,
                issue_count=len(issues),
                event_count=0,
                budget_exhausted=True,
            )
        else:
            timeline_prs, timeline_meta = discover_issue_timeline_merged_prs(
                repo,
                [int(issue["number"]) for issue in issues],
                max_pages=timeline_page_budget,
                max_requests=request_budget.remaining,
                runner=rest_runner,
                request_budget=request_budget,
            )
            timeline_meta["reason"] = "global closed-PR inventory partial"
            merged_prs = _merge_merged_pr_rows(merged_prs, timeline_prs)
            if timeline_meta.get("quota_exhausted") or any(
                _is_rate_limit_error(err) for err in timeline_meta.get("errors", [])
            ):
                request_budget.mark_rate_limited()
            if timeline_meta.get("budget_exhausted"):
                request_budget.mark_budget_exhausted()

    if quota_available and not request_budget.rate_limited and request_budget.remaining > 0:
        labels, label_meta = discover_repository_labels(
            repo,
            max_pages=max_pages,
            runner=rest_runner,
            request_budget=request_budget,
        )
    elif not quota_available:
        labels = set()
        label_meta = _skipped_rest_metadata(
            f"repository-label inventory skipped: {preflight_reason}",
            page_budget=max_pages,
            quota_exhausted=initial_quota_exhausted,
        )
    elif request_budget.rate_limited:
        labels = set()
        label_meta = _skipped_rest_metadata(
            "repository-label inventory skipped after an actual core REST rate-limit response",
            page_budget=max_pages,
            quota_exhausted=True,
        )
    else:
        request_budget.mark_budget_exhausted()
        labels = set()
        label_meta = _skipped_rest_metadata(
            "repository-label inventory skipped after the shared REST request budget was spent",
            page_budget=max_pages,
            budget_exhausted=True,
        )

    _finalize_inventory_quota(quota_meta, request_budget)
    quota_complete = bool(
        quota_meta.get("available")
        and quota_meta.get("status") == "ok"
        and not quota_meta.get("quota_uncertain")
        and not quota_meta.get("quota_exhausted")
    )

    coverage_complete = bool(
        (global_closed_prs_complete or timeline_meta.get("available")) and quota_complete
    )
    closure_coverage = {
        "complete_for_open_issues": coverage_complete,
        "mode": (
            "incomplete_quota_exhausted"
            if not quota_complete
            and (
                quota_meta.get("quota_exhausted")
                or quota_meta.get("status") in {"exhausted", "insufficient", "quota_blocked"}
            )
            else "incomplete"
            if not quota_complete
            else "global_closed_prs"
            if global_closed_prs_complete
            else "issue_timeline_fallback"
            if timeline_meta.get("available")
            else "incomplete"
        ),
        "global_closed_prs_complete": global_closed_prs_complete,
        "global_closed_prs_truncated": bool(closed_pr_meta.get("truncated")),
        "global_closed_prs_errors": list(closed_pr_meta.get("errors", [])),
        "timeline": timeline_meta,
        "quota": quota_meta,
    }
    claims, claim_meta = discover_claims(remote, command_runner=local_runner)
    worktrees, worktree_meta = discover_worktrees(command_runner=local_runner)
    jobs, job_meta = discover_jobs(command_runner=local_runner)
    return {
        "repo": repo,
        "remote": remote,
        "issues": issues,
        "open_prs": open_prs,
        "merged_prs": merged_prs,
        "labels": sorted(labels),
        "claims": claims,
        "worktrees": worktrees,
        "jobs": jobs,
        "quota": quota_meta,
        "inventory": {
            "issues": issue_meta,
            "comments": comment_meta,
            "open_prs": open_pr_meta,
            "closed_prs": closed_pr_meta,
            "issue_timeline_merged_prs": timeline_meta,
            "closure_coverage": closure_coverage,
            "quota": quota_meta,
            "labels": label_meta,
            "claims": claim_meta,
            "worktrees": worktree_meta,
            "jobs": job_meta,
        },
    }


def _comment_timestamp(value: object) -> datetime | None:
    """Parse a timezone-aware comment timestamp, or return ``None`` fail-closed."""
    if not isinstance(value, str) or not value.strip():
        return None
    timestamp = value.strip()
    if timestamp.endswith("Z"):
        timestamp = f"{timestamp[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _ordered_comment_texts(issue: Mapping[str, Any]) -> tuple[list[str], bool]:
    """Return comment bodies in chronological order when timestamps are complete."""
    comments = issue.get("comments")
    if not isinstance(comments, Sequence) or isinstance(comments, (str, bytes)):
        return [], True
    rows: list[tuple[datetime | None, int, str]] = []
    ordering_complete = True
    for index, comment in enumerate(comments):
        if isinstance(comment, Mapping):
            timestamp = _comment_timestamp(comment.get("created_at"))
            ordering_complete = ordering_complete and timestamp is not None
            rows.append((timestamp, index, str(comment.get("body") or "")))
        elif isinstance(comment, str):
            ordering_complete = False
            rows.append((None, index, comment))
        else:
            ordering_complete = False
            rows.append((None, index, ""))
    if ordering_complete:
        rows.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in rows], ordering_complete


def _text_for_issue(issue: Mapping[str, Any]) -> str:
    """Join body and chronologically ordered comments for evidence matching."""
    parts = [str(issue.get("title") or ""), str(issue.get("body") or "")]
    comment_texts, _ = _ordered_comment_texts(issue)
    parts.extend(comment_texts)
    return "\n".join(parts)


def _canonical_ruling_line_indices(lines: Sequence[str], issue_number: int) -> list[int]:
    """Return authoritative same-issue ruling lines from normalized text."""
    return [
        index
        for index, line in enumerate(lines)
        if (
            (match := CANONICAL_RULING_RE.fullmatch(line)) is not None
            and int(match.group("issue")) == issue_number
            and not any(
                NON_AUTHORITATIVE_RULING_CONTEXT_RE.search(context)
                for context in lines[max(0, index - 2) : index]
            )
        )
    ]


def _gate_evidence(text: str) -> list[dict[str, str]]:
    """Return current, issue-local provenance, rights, compute, and input gates."""
    evidence: list[dict[str, str]] = []
    normalized_text = NON_BLOCKING_GATE_RE.sub("", text)
    lines = [" ".join(line.lower().split()) for line in normalized_text.splitlines()]
    lines = [line for line in lines if line]
    gate_context = re.compile(
        r"\b(?:hard[- ]?)?(?:blocked[- ]+|gated\s+)(?:on|by|until|pending)\b"
        r"|\b(?:is|are|remains?|currently|still)\s+(?:hard[- ]?)?blocked\b"
        r"|\b(?:cannot|can't|unable to|not able to|not authorized to|not dispatchable)\b"
        r"[^|.;]{0,100}\b(?:submit|launch|run|proceed|stage|access|publish|release|verify|execute)\b"
        r"|\b(?:required|needed)\b[^|.;]{0,100}\b(?:missing|unavailable|pending|unset)\b"
        r"|\b(?:currently|availability:)\s*(?:missing|unavailable|unset)\b",
        re.IGNORECASE,
    )
    direct_blockers = {
        "blocked",
        "missing",
        "unavailable",
        "mismatch",
        "incompatible",
        "unsupported",
        "not available",
        "not granted",
        "cannot verify",
        "cannot publish",
        "cannot release",
        "cannot submit",
        "no allocation",
    }
    report_pattern = re.compile(
        r"\b\d+\s+of\s+\d+\b|\bgate-blocked\b|\bstate:(?:ready|running|blocked)\b"
        r"|\b(?:issues|rows|labels)\b.{0,60}\b(?:blocked|missing|unavailable)\b",
        re.IGNORECASE,
    )
    rule_pattern = re.compile(
        r"\bfail(?:s|ed)?\s+closed\b|\b(?:records?|rows?|evidence|claims?|outcomes?|"
        r"metadata|protocol|predeclared|execution|validates?|rejects?)\b"
        r"[^|.;]{0,50}\b(?:missing|unavailable|fallback|degraded|invalid)\b",
        re.IGNORECASE,
    )
    generic_gate_pattern = re.compile(
        r"\b(?:remains?|currently|is|are)?\s*(?:hard[- ]?)?blocked[- ]+"
        r"(?:on|by|until|pending)\b|\bhard[- ]gated\s+(?:on|by|until)\b",
        re.IGNORECASE,
    )
    for kind, (topics, blockers) in BLOCKER_TERMS.items():
        for line in lines:
            topic = next((term for term in topics if term in line), None)
            blocker = next((term for term in blockers if term in line), None)
            if not topic:
                continue

            def near(left: str, right: str) -> bool:
                return bool(
                    re.search(
                        rf"\b{re.escape(left)}\b[^|.;]{{0,80}}\b{re.escape(right)}\b",
                        line,
                    )
                    or re.search(
                        rf"\b{re.escape(right)}\b[^|.;]{{0,80}}\b{re.escape(left)}\b",
                        line,
                    )
                )

            report_line = bool(report_pattern.search(line))
            conditional_line = bool(re.search(r"\b(?:if|unless|when|otherwise)\b", line))
            explicit_gate = bool(gate_context.search(line))
            gate_related = any(
                near(topic, token)
                for token in ("blocked", "gated", "pending", "missing", "unavailable")
            )
            direct_status = bool(blocker and blocker in direct_blockers and near(topic, blocker))
            rule_line = bool(rule_pattern.search(line))
            # A bare "quota exhausted" or similar topic echo is not enough;
            # it must be attached to an explicit current gate.  Report tables
            # and conditional acceptance rules describe other records or
            # future failures, not the issue's present state.
            current_gate = (explicit_gate and gate_related) or (
                direct_status and not conditional_line and not rule_line
            )
            if kind == "external-input" and not gate_related:
                current_gate = False
            if kind == "external-input" and blocker == "missing" and not explicit_gate:
                current_gate = False
            if current_gate and not report_line:
                detail = blocker or "explicit gate"
                evidence.append(
                    {
                        "kind": kind,
                        "text": f"{topic} evidence includes {detail}",
                    }
                )
                break
    for line in lines:
        if generic_gate_pattern.search(line) and not report_pattern.search(line):
            evidence.append({"kind": "blocked", "text": "explicit current blocker"})
            break
    return evidence


def _blocked_reason_evidence(text: str) -> list[str]:
    """Return explicit machine-readable evidence that justifies a blocked label."""
    evidence: list[str] = []
    if BLOCKED_TRIAGE_BLOCK_RE.search(text):
        evidence.append("blocked-triage-v1 reason block present")
    blocked_by = BLOCKED_BY_REFERENCE_RE.findall(text)
    if blocked_by:
        evidence.append(f"Blocked-by reference present: {blocked_by[0].strip()}")
    return evidence


def _decision_evidence(
    text: str,
    labels: set[str],
    *,
    issue_number: int | None = None,
    comment_order_complete: bool = True,
) -> list[str]:
    """Return decision evidence while respecting a later canonical ruling."""
    evidence = ["decision-required label present"] if DECISION_LABEL in labels else []
    lines = [" ".join(line.split()) for line in text.splitlines()]
    resolution_pattern = re.compile(
        r"\b(?:decision-required(?:\s+label)?|decision|gate|deferral|approval)\b"
        r"[^.;]{0,100}\b(?:removed|resolved|settled|already exists|no longer needed|"
        r"not pending|remove|waiv(?:ed|ing)|approved|authorized|selected|made|deferred|"
        r"reaffirmed|reaffirm|in force)\b"
        r"|\bchoose\s+option\s+\([a-z]\)(?!\w)"
        r"|\b(?:remove|removed)\s+decision-required\b"
        r"|\breaffirm(?:ed|ation)?\b"
        r"|\bapprove\s+(?!or\s+waive\b)[^.;]{0,100}\b(?:nominal|exact|clean|"
        r"protocol|campaign|option)\b",
        re.IGNORECASE,
    )
    last_resolution = max(
        (index for index, line in enumerate(lines) if resolution_pattern.search(line)),
        default=-1,
    )
    if issue_number is not None and comment_order_complete:
        latest_ruling = max(
            _canonical_ruling_line_indices(lines, issue_number),
            default=-1,
        )
        last_resolution = max(last_resolution, latest_ruling)
    for index, line in enumerate(lines):
        if not line:
            continue
        if CONDITIONAL_DECISION_REVIVAL_RE.search(line):
            continue
        for pattern in DECISION_PATTERNS:
            match = pattern.search(line)
            if not match or index <= last_resolution:
                continue
            prefix = line[: match.start()]
            if re.search(r"\b(?:no|not|never|without)\b[^.;]{0,80}$", prefix, re.IGNORECASE):
                continue
            if match:
                evidence.append(f"issue text: {match.group(0).strip()[:180]}")
                break
    return evidence


def _ready_evidence(body: str) -> list[str]:
    """Return only concrete acceptance/validation signals supporting readiness."""
    evidence: list[str] = []
    if READY_HEADING_PATTERN.search(body):
        evidence.append("acceptance or validation heading present")
    if CHECKBOX_PATTERN.search(body) and not UNCHECKED_CHECKBOX_PATTERN.search(body):
        evidence.append("all issue checkboxes are marked complete")
    if re.search(r"(?im)^\s*(?:command|entry point|validation)\s*:", body):
        evidence.append("explicit command or validation entry point present")
    return evidence


def _invalid_preparation_packet(reason: str) -> dict[str, Any]:
    """Return a structured fail-closed result for a preparation marker."""
    return {
        "status": "invalid",
        "reason": (
            "goal-autopilot-preparation:v1 is invalid; readiness promotion withheld: " + reason
        ),
        "implementation_admitted": None,
        "state_ready_change_proposed": None,
        "next_action": None,
    }


def _parse_preparation_packet(body: str, *, issue_number: int) -> dict[str, Any]:
    """Parse the current issue preparation marker, if present.

    The preparation helper writes a marker around a YAML packet whose source
    body digest covers everything outside the marker.  A marker is therefore
    useful only when it is unique, structurally valid, and still bound to the
    current issue body.  The empty result deliberately preserves the audit's
    pre-marker behavior for issues that have never been prepared.
    """
    start_count = body.count(PREPARATION_MARKER_START)
    end_count = body.count(PREPARATION_MARKER_END)
    if start_count == 0 and end_count == 0:
        return {
            "status": "absent",
            "reason": "",
            "implementation_admitted": None,
            "state_ready_change_proposed": None,
            "next_action": None,
        }
    if start_count != 1 or end_count != 1:
        return _invalid_preparation_packet("marker is duplicated or unbalanced")

    marker_match = _PREPARATION_MARKER_RE.search(body)
    if marker_match is None:
        return _invalid_preparation_packet("marker boundaries are malformed")
    yaml_matches = list(_PREPARATION_YAML_RE.finditer(marker_match.group(0)))
    if len(yaml_matches) != 1:
        return _invalid_preparation_packet("marker must contain exactly one YAML block")

    try:
        loader = _UniquePreparationLoader(yaml_matches[0].group("content"))
        try:
            packet = loader.get_single_data()
        finally:
            loader.dispose()
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        return _invalid_preparation_packet(f"malformed YAML ({type(exc).__name__})")
    if not isinstance(packet, Mapping):
        return _invalid_preparation_packet("YAML packet must decode to a mapping")

    missing = [key for key in _PREPARATION_REQUIRED_FIELDS if key not in packet]
    if missing:
        return _invalid_preparation_packet("missing field(s): " + ", ".join(missing))
    if packet["schema"] != PREPARATION_PACKET_SCHEMA:
        return _invalid_preparation_packet("schema does not match the versioned packet contract")
    if packet["audit_schema"] != _PREPARATION_AUDIT_SCHEMA:
        return _invalid_preparation_packet("audit_schema does not match the preparation contract")
    if packet["repository"] != DEFAULT_REPO:
        return _invalid_preparation_packet("repository identity does not match the audit repo")
    if type(packet["issue"]) is not int or packet["issue"] != issue_number:
        return _invalid_preparation_packet("issue identity does not match the containing issue")
    for field in _PREPARATION_STRING_FIELDS:
        if not isinstance(packet[field], str):
            return _invalid_preparation_packet(f"{field} must be a string")
    for field in ("implementation_admitted", "state_ready_change_proposed"):
        if type(packet[field]) is not bool:
            return _invalid_preparation_packet(f"{field} must be a boolean")

    source_body_sha = packet["source_body_sha256"]
    if (
        not isinstance(source_body_sha, str)
        or _PREPARATION_SHA256_RE.fullmatch(source_body_sha) is None
    ):
        return _invalid_preparation_packet("source_body_sha256 is not a lowercase SHA-256 digest")
    source_comments_sha = packet["source_comments_sha256"]
    if source_comments_sha not in (None, ""):
        return _invalid_preparation_packet("source_comments_sha256 must be empty")
    audit_digest = packet["audit_digest"]
    if not isinstance(audit_digest, str) or _PREPARATION_SHA256_RE.fullmatch(audit_digest) is None:
        return _invalid_preparation_packet("audit_digest is not a lowercase SHA-256 digest")

    try:
        actual_source_sha = prepare_open_issue_contracts.preparation_source_body_sha256(body)
    except (TypeError, ValueError) as exc:
        return _invalid_preparation_packet(str(exc))
    if actual_source_sha != source_body_sha:
        return _invalid_preparation_packet(
            "source_body_sha256 does not match body content outside marker"
        )

    readiness_gate = packet["readiness_gate"]
    if packet["state_ready_change_proposed"]:
        if not isinstance(readiness_gate, Mapping):
            return _invalid_preparation_packet(
                "state_ready_change_proposed requires a readiness_gate mapping"
            )
        if readiness_gate.get("issue") not in {issue_number, str(issue_number)}:
            return _invalid_preparation_packet("readiness_gate issue does not match the packet")
        if readiness_gate.get("action") != "gate_readiness":
            return _invalid_preparation_packet("readiness_gate action is not gate_readiness")
        if readiness_gate.get("expected_body_sha256") != source_body_sha:
            return _invalid_preparation_packet(
                "readiness_gate expected_body_sha256 conflicts with source_body_sha256"
            )
        expected_labels = readiness_gate.get("expected_labels")
        if not isinstance(expected_labels, list) or any(
            not isinstance(label, str) for label in expected_labels
        ):
            return _invalid_preparation_packet(
                "readiness_gate expected_labels must be a string list"
            )
        if len(expected_labels) != len(set(expected_labels)):
            return _invalid_preparation_packet(
                "readiness_gate expected_labels must not contain duplicates"
            )
        if "state:ready" in expected_labels:
            return _invalid_preparation_packet(
                "readiness_gate expected_labels must not already contain state:ready"
            )
        if packet["next_action"] != "gate_readiness":
            return _invalid_preparation_packet(
                "state_ready_change_proposed conflicts with next_action"
            )
    elif readiness_gate not in (None, "", "None"):
        return _invalid_preparation_packet(
            "readiness_gate must be empty when state_ready_change_proposed is false"
        )

    withheld_fields = [
        field
        for field in ("implementation_admitted", "state_ready_change_proposed")
        if packet[field] is False
    ]
    reason = ""
    if withheld_fields:
        reason = (
            "goal-autopilot-preparation:v1 withholds readiness because "
            + ", ".join(f"{field}=false" for field in withheld_fields)
            + f"; next_action={packet['next_action'] or '(empty)'}"
        )
    return {
        "status": "valid",
        "reason": reason,
        "implementation_admitted": packet["implementation_admitted"],
        "state_ready_change_proposed": packet["state_ready_change_proposed"],
        "next_action": packet["next_action"],
    }


def _terminal_review_evidence(
    issue: Mapping[str, Any],
    *,
    issue_number: int,
    comment_order_complete: bool,
) -> list[str]:
    """Return explicit report-status evidence that execution awaits review.

    The status-line requirement is deliberate: prose about a future campaign,
    an acceptance criterion, or a hypothetical terminal state must not suppress
    dispatch.  A machine-readable status such as
    ``diagnostic_ready_for_domain_review`` is current evidence that the
    completed execution has crossed into interpretation or domain review.
    """
    comment_texts, observed_order_complete = _ordered_comment_texts(issue)
    sources = [str(issue.get("body") or ""), *comment_texts]
    latest_ruling_source = -1
    if comment_order_complete and observed_order_complete:
        for source_index, source in enumerate(sources):
            lines = [" ".join(line.split()) for line in source.splitlines()]
            if _canonical_ruling_line_indices(lines, issue_number):
                latest_ruling_source = source_index

    evidence: list[str] = []
    for source_index, source in enumerate(sources):
        if source_index <= latest_ruling_source:
            continue
        for raw_line in source.splitlines():
            line = " ".join(raw_line.split())
            if not line:
                continue
            match = TERMINAL_REVIEW_STATUS_LINE_PATTERN.match(line)
            if not match or not TERMINAL_REVIEW_STATUS_PATTERN.search(match.group("status")):
                continue
            evidence.append(f"terminal review status: {_compact_excerpt(line)}")
    return evidence[:3]


def _active_records(
    issue_number: int,
    *,
    open_prs: Iterable[Mapping[str, Any]],
    claims: Mapping[int, Mapping[str, Any]],
    worktrees: Iterable[Mapping[str, Any]],
    jobs: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Correlate active execution records with one issue number."""
    prs = [
        dict(pr)
        for pr in open_prs
        if issue_number in _issue_ref_numbers(pr.get("title"), pr.get("body"), pr.get("head_ref"))
    ]
    local_worktrees = [
        dict(row)
        for row in worktrees
        if issue_number in _issue_ref_numbers(row.get("path"), row.get("branch"))
    ]
    active_jobs = [
        dict(row)
        for row in jobs
        if issue_number
        in _issue_ref_numbers(row.get("name"), row.get("command"), row.get("job_name"))
    ]
    claim = dict(claims[issue_number]) if issue_number in claims else {}
    return {
        "open_prs": prs,
        "claims": [claim] if claim else [],
        "worktrees": local_worktrees,
        "jobs": active_jobs,
    }


def _merged_records(
    issue_number: int, merged_prs: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return merged PRs with explicit issue references in title/body/branch."""
    return [dict(pr) for pr in merged_prs if issue_number in _merged_pr_issue_numbers(pr)]


def _merged_pr_issue_numbers(pr: Mapping[str, Any]) -> set[int]:
    """Return normalized issue references for one merged PR row."""
    numbers: set[int] = set()
    raw_linked = pr.get("linked_issue_numbers")
    if isinstance(raw_linked, (list, tuple, set, frozenset)):
        for value in raw_linked:
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value > 0:
                numbers.add(value)
            elif isinstance(value, str) and value.isdigit() and int(value) > 0:
                numbers.add(int(value))
    numbers.update(_issue_ref_numbers(pr.get("title"), pr.get("body"), pr.get("head_ref")))
    return numbers


def _index_merged_prs(
    merged_prs: Sequence[Mapping[str, Any]],
    *,
    deadline: float | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], bool]:
    """Index merged PR references once, returning whether the budget was exhausted."""
    indexed: dict[int, list[dict[str, Any]]] = {}
    for pr in merged_prs:
        if _deadline_expired(deadline):
            return {}, True
        row = dict(pr)
        for issue_number in _merged_pr_issue_numbers(row):
            indexed.setdefault(issue_number, []).append(row)
        if _deadline_expired(deadline):
            return {}, True
    if _deadline_expired(deadline):
        return {}, True
    return indexed, False


def closure_evidence(
    issue: Mapping[str, Any],
    *,
    merged_prs: Sequence[Mapping[str, Any]],
    merged_pr_index: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
    open_issue_numbers: set[int] | None = None,
) -> dict[str, Any]:
    """Evaluate the narrow, documented conditions under which autonomous close is safe."""
    body = str(issue.get("body") or "")
    number = int(issue.get("number", 0))
    linked = (
        [dict(pr) for pr in merged_pr_index.get(number, ())]
        if merged_pr_index is not None
        else _merged_records(number, merged_prs)
    )
    coverage = {
        "coverage_sources": sorted(
            {str(pr.get("coverage_source") or "global_closed_prs") for pr in linked}
        ),
        "targeted_merged_prs": sorted(
            int(pr["number"])
            for pr in linked
            if pr.get("coverage_source") == "targeted_issue_timeline"
        ),
    }
    if not linked:
        return {
            "eligible": False,
            "reason": "no merged issue-linked PR",
            "merged_prs": [],
            **coverage,
        }
    if PARENT_TITLE_PATTERN.search(str(issue.get("title") or "")):
        if not PARENT_CLOSE_PATTERN.search(body):
            return {
                "eligible": False,
                "reason": "parent issue lacks documented all-children close condition",
                "merged_prs": [pr.get("number") for pr in linked],
                **coverage,
            }
        child_numbers = _issue_ref_numbers(body)
        child_numbers.discard(number)
        if open_issue_numbers is None or child_numbers & open_issue_numbers:
            return {
                "eligible": False,
                "reason": "documented parent close condition is not proven by the open inventory",
                "merged_prs": [pr.get("number") for pr in linked],
                "child_issues": sorted(child_numbers),
                **coverage,
            }
        return {
            "eligible": True,
            "reason": "documented parent close condition and no referenced child remains open",
            "merged_prs": [pr.get("number") for pr in linked],
            "child_issues": sorted(child_numbers),
            **coverage,
        }
    explicit = bool(EXPLICIT_MERGED_CLOSE_PATTERN.search(body))
    checked = bool(CHECKBOX_PATTERN.search(body)) and not bool(
        UNCHECKED_CHECKBOX_PATTERN.search(body)
    )
    if explicit or checked:
        reason = (
            "explicit merged-PR close condition"
            if explicit
            else "all acceptance checkboxes complete with merged issue-linked PR"
        )
        return {
            "eligible": True,
            "reason": reason,
            "merged_prs": [pr.get("number") for pr in linked],
            **coverage,
        }
    return {
        "eligible": False,
        "reason": "merged work exists but completion condition is not documented",
        "merged_prs": [pr.get("number") for pr in linked],
        **coverage,
    }


def _available(label: str, available_labels: set[str] | None) -> bool:
    """Allow a mutation only when the repository label inventory proves it exists."""
    return available_labels is not None and label in available_labels


def _completion_receipt_for_issue(
    completion_receipts: Mapping[object, object] | None,
    issue_number: int,
) -> Mapping[str, Any] | None:
    """Read a receipt entry from number, string-number, or ``#number`` keys."""
    if not isinstance(completion_receipts, Mapping):
        return None
    for key in (issue_number, str(issue_number), f"#{issue_number}"):
        value = completion_receipts.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _mutation(
    operation: str,
    issue_number: int,
    *,
    value: str | None,
    reason: str,
    evidence: Iterable[str] = (),
    blocked_reason: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a stable mutation row."""
    mutation = {
        "operation": operation,
        "issue": issue_number,
        "value": value,
        "reason": reason,
        "evidence": list(evidence),
    }
    reason_evidence = [item for item in blocked_reason if item]
    if reason_evidence:
        mutation["blocked_reason"] = reason_evidence
    return mutation


def _state_winner(
    state_labels: set[str],
    *,
    blocker: bool,
    external_blocker: bool,
    active: bool,
    ready: bool,
) -> str | None:
    """Choose a state only when repository evidence makes the choice unambiguous."""
    if blocker:
        return "state:blocked-external-input" if external_blocker else "state:blocked"
    if active:
        return "state:running"
    if ready:
        return "state:ready"
    if len(state_labels) == 1:
        return next(iter(state_labels))
    return None


@dataclass(frozen=True)
class Classification:
    """Normalized classification and safe operations for one open issue."""

    issue: int
    classification: str
    state_labels: tuple[str, ...]
    execution_state_labels: tuple[str, ...]
    resource_labels: tuple[str, ...]
    type_labels: tuple[str, ...]
    evidence_labels: tuple[str, ...]
    active: dict[str, list[dict[str, Any]]]
    blocker_evidence: tuple[dict[str, str], ...]
    blocked_reason_evidence: tuple[str, ...]
    blocked_label_decision: str
    decision_required: bool
    decision_evidence: tuple[str, ...]
    terminal_review_evidence: tuple[str, ...]
    readiness_evidence: tuple[str, ...]
    preparation_packet: dict[str, Any]
    closure: dict[str, Any]
    mutations: tuple[dict[str, Any], ...]
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible classification payload."""
        return {
            "issue": f"#{self.issue}",
            "number": self.issue,
            "classification": self.classification,
            "state_labels": list(self.state_labels),
            "execution_state_labels": list(self.execution_state_labels),
            "resource_labels": list(self.resource_labels),
            "type_labels": list(self.type_labels),
            "evidence_labels": list(self.evidence_labels),
            "active_evidence": self.active,
            "blocker_evidence": list(self.blocker_evidence),
            "blocked_reason_evidence": list(self.blocked_reason_evidence),
            "blocked_label_decision": self.blocked_label_decision,
            "decision_required": self.decision_required,
            "decision_evidence": list(self.decision_evidence),
            "terminal_review_evidence": list(self.terminal_review_evidence),
            "readiness_evidence": list(self.readiness_evidence),
            "preparation_packet": dict(self.preparation_packet),
            "closure_evidence": self.closure,
            "mutations": list(self.mutations),
            "findings": list(self.findings),
        }


def _issue_source_metadata_shape_failures(issue_meta: Mapping[str, Any]) -> list[str]:
    """Reject typed source metadata that a status branch would otherwise ignore."""
    failures: list[str] = []
    boolean_fields = (
        "available",
        "truncated",
        "rate_limited",
        "quota_exhausted",
        "budget_exhausted",
        "request_limit_exhausted",
        "quota_uncertain",
    )
    for field in boolean_fields:
        if field in issue_meta and type(issue_meta[field]) is not bool:
            failures.append(f"{field} must be a boolean when present")

    if "errors" in issue_meta:
        errors = issue_meta["errors"]
        if not isinstance(errors, list):
            failures.append("errors must be a list when present")
        elif any(not isinstance(error, str) or not error.strip() for error in errors):
            failures.append("errors must contain only non-empty strings")

    non_negative_integer_fields = (
        "pages_read",
        "requests_attempted",
        "per_page",
        "page_budget",
        "row_count",
        "raw_row_count",
        "canonical_row_count",
        "non_object_row_count",
        "malformed_object_row_count",
    )
    for field in non_negative_integer_fields:
        if field not in issue_meta:
            continue
        value = issue_meta[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            failures.append(f"{field} must be a non-negative integer when present")

    pages_read = issue_meta.get("pages_read")
    requests_attempted = issue_meta.get("requests_attempted")
    page_budget = issue_meta.get("page_budget")
    valid_pages_read = isinstance(pages_read, int) and not isinstance(pages_read, bool)
    valid_requests_attempted = isinstance(requests_attempted, int) and not isinstance(
        requests_attempted, bool
    )
    valid_page_budget = isinstance(page_budget, int) and not isinstance(page_budget, bool)
    if valid_pages_read and valid_page_budget and pages_read > page_budget:
        failures.append("pages_read must not exceed page_budget")
    if valid_requests_attempted and valid_pages_read and requests_attempted < pages_read:
        failures.append("requests_attempted must be at least pages_read")
    if valid_requests_attempted and valid_page_budget and requests_attempted > page_budget:
        failures.append("requests_attempted must not exceed page_budget")

    for field in ("source", "source_kind", "source_status", "source_proof", "source_status_reason"):
        if field in issue_meta and not isinstance(issue_meta[field], str):
            failures.append(f"{field} must be a string when present")
    return failures


def _empty_issue_source_contract_failures(
    issue_meta: Mapping[str, Any], *, source: str, canonical_row_count: int
) -> list[str]:
    """Return missing or contradictory fields in a successful-empty source contract."""
    failures: list[str] = []
    exact_fields = {
        "source": source,
        "source_kind": ISSUE_SOURCE_KIND,
        "source_status": ISSUE_SOURCE_STATUS_EMPTY,
        "source_proof": ISSUE_SOURCE_EMPTY_PROOF,
        "available": True,
        "truncated": False,
        "errors": [],
        "row_count": 0,
        "canonical_row_count": 0,
        "raw_row_count": 0,
        "non_object_row_count": 0,
        "malformed_object_row_count": 0,
    }
    if canonical_row_count != 0:
        failures.append(f"plan contains {canonical_row_count} canonical issue row(s)")
    for field, expected in exact_fields.items():
        if type(issue_meta.get(field)) is not type(expected) or issue_meta.get(field) != expected:
            failures.append(f"{field} must be {expected!r}")

    for field in ("pages_read", "requests_attempted", "per_page", "page_budget"):
        value = issue_meta.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            failures.append(f"{field} must be a positive integer")
    reason = issue_meta.get("source_status_reason")
    if not isinstance(reason, str) or not reason.strip():
        failures.append("source_status_reason must be a non-empty string")
    for field in (
        "rate_limited",
        "quota_exhausted",
        "budget_exhausted",
        "request_limit_exhausted",
        "quota_uncertain",
    ):
        if field in issue_meta and type(issue_meta[field]) is not bool:
            failures.append(f"{field} must be a boolean when present")
        elif issue_meta.get(field) is True:
            failures.append(f"{field} must not be true")
    return failures


def _complete_issue_source_contract_failures(
    issue_meta: Mapping[str, Any], *, source: str, canonical_row_count: int
) -> list[str]:
    """Return missing or contradictory fields in a successful non-empty source contract."""
    failures: list[str] = []
    exact_fields = {
        "source": source,
        "source_kind": ISSUE_SOURCE_KIND,
        "source_status": ISSUE_SOURCE_STATUS_COMPLETE,
        "source_proof": "canonical_issue_rows",
        "available": True,
        "truncated": False,
        "errors": [],
        "row_count": canonical_row_count,
        "canonical_row_count": canonical_row_count,
        "non_object_row_count": 0,
        "malformed_object_row_count": 0,
    }
    for field, expected in exact_fields.items():
        if type(issue_meta.get(field)) is not type(expected) or issue_meta.get(field) != expected:
            failures.append(f"{field} must be {expected!r}")

    raw_row_count = issue_meta.get("raw_row_count")
    if isinstance(raw_row_count, bool) or not isinstance(raw_row_count, int) or raw_row_count < 0:
        failures.append("raw_row_count must be a non-negative integer")
    elif raw_row_count < canonical_row_count:
        failures.append("raw_row_count must be at least canonical_row_count")

    for field in ("pages_read", "requests_attempted", "per_page", "page_budget"):
        value = issue_meta.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            failures.append(f"{field} must be a positive integer")
    reason = issue_meta.get("source_status_reason")
    if not isinstance(reason, str) or not reason.strip():
        failures.append("source_status_reason must be a non-empty string")
    for field in (
        "rate_limited",
        "quota_exhausted",
        "budget_exhausted",
        "request_limit_exhausted",
        "quota_uncertain",
    ):
        if field in issue_meta and type(issue_meta[field]) is not bool:
            failures.append(f"{field} must be a boolean when present")
        elif issue_meta.get(field) is True:
            failures.append(f"{field} must not be true")
    return failures


_QUOTA_REQUIRED_FIELDS = (
    "available",
    "status",
    "core_remaining",
    "core_reset_at",
    "available_budget",
    "min_core_remaining",
    "retry_command",
    "next_action",
    "reason",
    "errors",
    "quota_exhausted",
    "quota_uncertain",
    "budget_exhausted",
    "request_budget",
    "requests_attempted",
    "requests_remaining",
)


def _quota_metadata_contract_failures(
    raw_quota: object, *, field: str, required: bool = True
) -> list[str]:
    """Reject absent, partial, unhealthy, or inconsistent quota metadata."""
    if not isinstance(raw_quota, Mapping):
        return [
            f"{field} must be a complete object"
            if required
            else f"{field} must be an object when present"
        ]

    failures: list[str] = []
    if required:
        failures.extend(
            f"{field}.{name} is required"
            for name in _QUOTA_REQUIRED_FIELDS
            if name not in raw_quota
        )

    for name in ("available", "quota_uncertain", "quota_exhausted", "budget_exhausted"):
        if name in raw_quota and type(raw_quota[name]) is not bool:
            failures.append(f"{field}.{name} must be a boolean")
    for name in ("request_limit_exhausted", "rate_limited"):
        if name in raw_quota and type(raw_quota[name]) is not bool:
            failures.append(f"{field}.{name} must be a boolean when present")

    status = raw_quota.get("status")
    if "status" in raw_quota and not isinstance(status, str):
        failures.append(f"{field}.status must be a string")
    elif "status" in raw_quota and status != "ok":
        status_reason = (
            "unavailable or uncertain"
            if status in UNCERTAIN_QUOTA_STATUSES
            else "not an available quota status"
        )
        failures.append(f"{field}.status={status!r} is {status_reason}")
    if raw_quota.get("available") is not True and "available" in raw_quota:
        if type(raw_quota.get("available")) is bool:
            failures.append(f"{field}.available is false")
    if raw_quota.get("quota_uncertain") is True:
        failures.append(f"{field}.quota_uncertain is true")
    if raw_quota.get("quota_exhausted") is True:
        failures.append(f"{field}.quota_exhausted is true")
    if raw_quota.get("budget_exhausted") is True:
        failures.append(f"{field}.budget_exhausted is true")
    if raw_quota.get("request_limit_exhausted") is True:
        failures.append(f"{field}.request_limit_exhausted is true")
    if raw_quota.get("rate_limited") is True:
        failures.append(f"{field}.rate_limited is true")

    for name in ("core_remaining", "available_budget", "min_core_remaining"):
        value = raw_quota.get(name)
        if name in raw_quota and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            failures.append(f"{field}.{name} must be a non-negative integer")
    core_reset_at = raw_quota.get("core_reset_at")
    if (
        "core_reset_at" in raw_quota
        and core_reset_at is not None
        and (
            isinstance(core_reset_at, bool)
            or not isinstance(core_reset_at, int)
            or core_reset_at < 0
        )
    ):
        failures.append(f"{field}.core_reset_at must be a non-negative integer or null")

    for name in ("retry_command", "next_action", "reason"):
        value = raw_quota.get(name)
        if name in raw_quota and (not isinstance(value, str) or not value.strip()):
            failures.append(f"{field}.{name} must be a non-empty string")
    if raw_quota.get("next_action") == "none":
        pass
    elif "next_action" in raw_quota and isinstance(raw_quota.get("next_action"), str):
        failures.append(f"{field}.next_action must be 'none' for healthy quota")

    errors = raw_quota.get("errors")
    if "errors" in raw_quota:
        if not isinstance(errors, list) or any(
            not isinstance(error, str) or not error.strip() for error in errors
        ):
            failures.append(f"{field}.errors must be a list of non-empty strings")
        elif errors:
            failures.append(f"{field}.errors must be empty for healthy quota")

    request_values = {
        name: raw_quota.get(name)
        for name in ("request_budget", "requests_attempted", "requests_remaining")
    }
    for name, value in request_values.items():
        if name in raw_quota and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            failures.append(f"{field}.{name} must be a non-negative integer")
    if "request_budget" in raw_quota and isinstance(request_values["request_budget"], int):
        if request_values["request_budget"] <= 0:
            failures.append(f"{field}.request_budget must be positive")
    if all(
        isinstance(value, int) and not isinstance(value, bool) for value in request_values.values()
    ):
        request_budget = request_values["request_budget"]
        requests_attempted = request_values["requests_attempted"]
        requests_remaining = request_values["requests_remaining"]
        if requests_attempted + requests_remaining != request_budget:
            failures.append(
                f"{field}.requests_attempted + {field}.requests_remaining must equal "
                f"{field}.request_budget"
            )

    numeric_values_present = all(
        name in raw_quota
        and isinstance(raw_quota.get(name), int)
        and not isinstance(raw_quota.get(name), bool)
        for name in ("core_remaining", "available_budget", "min_core_remaining")
    )
    if numeric_values_present:
        core_remaining = int(raw_quota["core_remaining"])
        available_budget = int(raw_quota["available_budget"])
        min_core_remaining = int(raw_quota["min_core_remaining"])
        if core_remaining - min_core_remaining != available_budget:
            failures.append(
                f"{field}.available_budget must equal core_remaining - min_core_remaining"
            )
        if available_budget <= 0:
            failures.append(f"{field}.available_budget must be positive for healthy quota")
    if (
        isinstance(raw_quota.get("available_budget"), int)
        and not isinstance(raw_quota.get("available_budget"), bool)
        and isinstance(raw_quota.get("request_budget"), int)
        and not isinstance(raw_quota.get("request_budget"), bool)
        and raw_quota["request_budget"] != raw_quota["available_budget"]
    ):
        failures.append(f"{field}.request_budget must equal available_budget")

    contract_errors = raw_quota.get("contract_errors")
    if "contract_errors" in raw_quota:
        if not isinstance(contract_errors, list) or any(
            not isinstance(error, str) or not error.strip() for error in contract_errors
        ):
            failures.append(f"{field}.contract_errors must be a list of non-empty strings")
        elif contract_errors:
            failures.append(f"{field}.contract_errors is not empty")
    return failures


def _quota_contract_failures_from_inventory(
    inventory: Mapping[str, Any], *, required: bool = False
) -> list[str]:
    """Collect malformed or uncertain quota metadata from both inventory shapes."""
    failures: list[str] = []
    quota_present = False
    if "quota" in inventory:
        raw_quota = inventory.get("quota")
        quota_present = raw_quota not in (None, {})
        if quota_present or required:
            failures.extend(
                _quota_metadata_contract_failures(raw_quota, field="quota", required=required)
            )
    nested_inventory = inventory.get("inventory")
    if isinstance(nested_inventory, Mapping) and "quota" in nested_inventory:
        raw_quota = nested_inventory.get("quota")
        quota_present = quota_present or raw_quota not in (None, {})
        if raw_quota not in (None, {}) or required:
            failures.extend(
                _quota_metadata_contract_failures(
                    raw_quota,
                    field="inventory.quota",
                    required=required,
                )
            )
    if required and not quota_present:
        failures.append("quota metadata is required for a write-capable plan")
    return failures


def _quota_contract_failures_from_plan(
    plan: Mapping[str, Any], *, required: bool = False
) -> list[str]:
    """Collect malformed or uncertain quota metadata from a serialized plan."""
    failures: list[str] = []
    quota_present = False
    if "quota" in plan:
        raw_quota = plan.get("quota")
        quota_present = raw_quota not in (None, {})
        if quota_present or required:
            failures.extend(
                _quota_metadata_contract_failures(raw_quota, field="plan.quota", required=required)
            )
    inventory = plan.get("inventory")
    if isinstance(inventory, Mapping) and "quota" in inventory:
        raw_quota = inventory.get("quota")
        quota_present = quota_present or raw_quota not in (None, {})
        if raw_quota not in (None, {}) or required:
            failures.extend(
                _quota_metadata_contract_failures(
                    raw_quota,
                    field="plan.inventory.quota",
                    required=required,
                )
            )
    if required and not quota_present:
        failures.append("quota metadata is required for a write-capable plan")
    return failures


def _issue_inventory_source_assessment(
    inventory: Mapping[str, Any], *, canonical_row_count: int
) -> dict[str, Any]:
    """Assess whether the canonical open-issue source can prove its row count."""
    raw_repository = inventory.get("repo", DEFAULT_REPO)
    repository = raw_repository if isinstance(raw_repository, str) else DEFAULT_REPO
    source = f"repos/{repository}/issues?state=open"
    inventory_meta = inventory.get("inventory")
    issue_meta = inventory_meta.get("issues") if isinstance(inventory_meta, Mapping) else None
    base = {
        "source": source,
        "source_kind": ISSUE_SOURCE_KIND,
        "canonical_row_count": canonical_row_count,
    }

    def result(
        status: str,
        *,
        admissible: bool,
        reason: str | None = None,
        error_code: str | None = None,
        source_proof: str | None = None,
        metadata_reported: bool = True,
    ) -> dict[str, Any]:
        assessed = {
            **base,
            "status": status,
            "admissible": admissible,
            "metadata_reported": metadata_reported,
        }
        if source_proof:
            assessed["source_proof"] = source_proof
        if reason:
            assessed["reason"] = reason
        if error_code:
            assessed["error_code"] = error_code
        if not admissible:
            assessed["next_action"] = (
                "rerun the bounded issue-audit plan and inspect the raw open-issue response"
            )
        return assessed

    if (
        not isinstance(raw_repository, str)
        or not raw_repository
        or _repository_parts(repository) is None
    ):
        return result(
            ISSUE_SOURCE_STATUS_ANOMALOUS,
            admissible=False,
            error_code="issue_inventory_repository_invalid",
            reason="the requested repository must be a GitHub owner/repository string",
        )

    source_metadata_absent = (
        not isinstance(inventory_meta, Mapping) or "issues" not in inventory_meta
    )
    if source_metadata_absent or (isinstance(issue_meta, Mapping) and not issue_meta):
        if canonical_row_count:
            return result(
                ISSUE_SOURCE_STATUS_ANOMALOUS,
                admissible=False,
                error_code="issue_inventory_source_missing",
                reason=(
                    f"{source} returned canonical rows without source-contract metadata; "
                    "the source is not proven"
                ),
                metadata_reported=False,
            )
        return result(
            ISSUE_SOURCE_STATUS_ANOMALOUS,
            admissible=False,
            error_code="issue_inventory_source_missing",
            reason=(
                f"{source} returned zero canonical rows without source-contract metadata; "
                "the empty queue is not proven"
            ),
            metadata_reported=False,
        )

    if not isinstance(issue_meta, Mapping):
        return result(
            ISSUE_SOURCE_STATUS_ANOMALOUS,
            admissible=False,
            error_code="issue_inventory_source_metadata_invalid",
            reason=f"{source} reported non-object source metadata",
        )

    source_status = issue_meta.get("source_status")
    metadata_shape_failures = _issue_source_metadata_shape_failures(issue_meta)
    if metadata_shape_failures and source_status not in {
        ISSUE_SOURCE_STATUS_COMPLETE,
        ISSUE_SOURCE_STATUS_EMPTY,
    }:
        return result(
            ISSUE_SOURCE_STATUS_ANOMALOUS,
            admissible=False,
            error_code="issue_inventory_source_metadata_invalid",
            reason=(
                f"{source} reported malformed source metadata: "
                + "; ".join(metadata_shape_failures)
            ),
        )

    source_errors = issue_meta.get("errors", [])
    has_errors = bool(source_errors)
    reported_canonical_row_count = issue_meta.get("canonical_row_count")
    if (
        isinstance(reported_canonical_row_count, int)
        and not isinstance(reported_canonical_row_count, bool)
        and reported_canonical_row_count != canonical_row_count
    ):
        return result(
            ISSUE_SOURCE_STATUS_ANOMALOUS,
            admissible=False,
            error_code="issue_inventory_source_inconsistent",
            reason=(
                f"{source} reported {reported_canonical_row_count} canonical row(s), "
                f"but the plan contains {canonical_row_count}; the source contract is inconsistent"
            ),
        )
    if source_status == ISSUE_SOURCE_STATUS_COMPLETE and canonical_row_count > 0:
        contract_failures = metadata_shape_failures + _complete_issue_source_contract_failures(
            issue_meta,
            source=source,
            canonical_row_count=canonical_row_count,
        )
        if contract_failures:
            return result(
                ISSUE_SOURCE_STATUS_ANOMALOUS,
                admissible=False,
                error_code="issue_inventory_source_complete_unproven",
                reason=(
                    f"{source} reported complete status without a complete successful-source "
                    f"contract: {'; '.join(contract_failures)}"
                ),
            )
        return result(
            ISSUE_SOURCE_STATUS_COMPLETE,
            admissible=True,
            source_proof="canonical_issue_rows",
        )
    if source_status == ISSUE_SOURCE_STATUS_ANOMALOUS:
        return result(
            ISSUE_SOURCE_STATUS_ANOMALOUS,
            admissible=False,
            error_code="issue_inventory_source_anomalous",
            reason=str(
                issue_meta.get("source_status_reason")
                or f"{source} reported an anomalous source response"
            ),
        )
    if (
        issue_meta.get("available") is False
        or issue_meta.get("truncated") is True
        or has_errors
        or any(
            issue_meta.get(field) is True
            for field in (
                "rate_limited",
                "quota_exhausted",
                "budget_exhausted",
                "request_limit_exhausted",
            )
        )
    ):
        reason = str(issue_meta.get("source_status_reason") or "")
        if not reason and has_errors:
            reason = str(source_errors[0])
        return result(
            ISSUE_SOURCE_STATUS_UNAVAILABLE,
            admissible=False,
            error_code="issue_inventory_source_unavailable",
            reason=reason or f"{source} is unavailable or incomplete",
        )

    if source_status is None:
        return result(
            ISSUE_SOURCE_STATUS_ANOMALOUS,
            admissible=False,
            error_code=(
                "issue_inventory_source_anomalous"
                if canonical_row_count > 0
                else "issue_inventory_source_missing"
            ),
            reason=(
                f"{source} returned rows but no source status"
                if canonical_row_count > 0
                else f"{source} did not report source status"
            ),
        )

    if source_status == ISSUE_SOURCE_STATUS_EMPTY and canonical_row_count == 0:
        contract_failures = metadata_shape_failures + _empty_issue_source_contract_failures(
            issue_meta,
            source=source,
            canonical_row_count=canonical_row_count,
        )
        if contract_failures:
            count_conflict = any(
                field in issue_meta and issue_meta.get(field) != 0
                for field in (
                    "row_count",
                    "canonical_row_count",
                    "raw_row_count",
                    "non_object_row_count",
                )
            )
            return result(
                ISSUE_SOURCE_STATUS_ANOMALOUS,
                admissible=False,
                error_code=(
                    "issue_inventory_source_inconsistent"
                    if count_conflict
                    else "issue_inventory_source_empty_unproven"
                ),
                reason=(
                    f"{source} reported empty status without a complete successful-empty "
                    f"contract: {'; '.join(contract_failures)}"
                ),
            )
        return result(
            ISSUE_SOURCE_STATUS_EMPTY,
            admissible=True,
            source_proof=str(issue_meta.get("source_proof") or "explicit_source_contract"),
        )
    if source_status in {
        ISSUE_SOURCE_STATUS_UNAVAILABLE,
        ISSUE_SOURCE_STATUS_ANOMALOUS,
    }:
        return result(
            str(source_status),
            admissible=False,
            error_code=(
                "issue_inventory_source_unavailable"
                if source_status == ISSUE_SOURCE_STATUS_UNAVAILABLE
                else "issue_inventory_source_anomalous"
            ),
            reason=str(
                issue_meta.get("source_status_reason")
                or f"{source} reported source status {source_status!r}"
            ),
        )

    return result(
        ISSUE_SOURCE_STATUS_ANOMALOUS,
        admissible=False,
        error_code="issue_inventory_source_inconsistent",
        reason=(
            f"{source} reported source status {source_status!r} for "
            f"{canonical_row_count} canonical row(s); the empty queue is not proven"
        ),
    )


def classify_issue(
    issue: Mapping[str, Any],
    *,
    open_prs: Iterable[Mapping[str, Any]] = (),
    merged_prs: Iterable[Mapping[str, Any]] = (),
    claims: Mapping[int, Mapping[str, Any]] | None = None,
    worktrees: Iterable[Mapping[str, Any]] = (),
    jobs: Iterable[Mapping[str, Any]] = (),
    job_inventory_available: bool = True,
    open_issue_numbers: set[int] | None = None,
    available_labels: set[str] | None = None,
    completion_receipt: Mapping[str, Any] | None = None,
    repository: str = DEFAULT_REPO,
    merged_pr_index: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
) -> Classification:
    """Classify one issue and plan only evidence-supported autonomous repairs."""
    number = int(issue.get("number", 0))
    labels = set(_label_names(issue.get("labels")))
    body = str(issue.get("body") or "")
    title = str(issue.get("title") or "")
    text = _text_for_issue(issue)
    state_labels = tuple(shared_state_labels(labels))
    execution_state_labels = tuple(shared_execution_state_labels(labels))
    resource_labels = tuple(sorted(label for label in labels if label.startswith(RESOURCE_PREFIX)))
    type_labels = tuple(sorted(label for label in labels if label.startswith(TYPE_PREFIX)))
    evidence_labels = tuple(sorted(label for label in labels if label.startswith(EVIDENCE_PREFIX)))
    claims = claims or {}
    findings: list[str] = []

    active = _active_records(
        number,
        open_prs=open_prs,
        claims=claims,
        worktrees=worktrees,
        jobs=jobs,
    )
    active_now = any(active.values())
    blocker_evidence = _gate_evidence(text)
    blocked_reason_evidence = _blocked_reason_evidence(text)
    if "state:blocked-external-input" in labels:
        blocker_evidence.append(
            {"kind": "external-input", "text": "state:blocked-external-input label"}
        )
    if "state:blocked" in labels:
        blocker_evidence.append({"kind": "blocked", "text": "state:blocked label"})
    if "evidence:blocked" in labels:
        blocker_evidence.append({"kind": "blocked", "text": "evidence:blocked label"})
    if TRIAGE_LABEL in labels:
        blocker_evidence.append({"kind": "triage", "text": f"{TRIAGE_LABEL} label"})
    job_inventory_uncertain = "resource:slurm" in labels and not job_inventory_available
    if job_inventory_uncertain:
        findings.append(
            "SLURM job inventory unavailable; preserve this issue and do not promote it to ready"
        )

    _, comment_order_complete = _ordered_comment_texts(issue)
    terminal_review_evidence = _terminal_review_evidence(
        issue,
        issue_number=number,
        comment_order_complete=comment_order_complete,
    )
    decision_evidence = _decision_evidence(
        text,
        labels,
        issue_number=number,
        comment_order_complete=comment_order_complete,
    )
    decision_evidence.extend(terminal_review_evidence)
    if len(type_labels) > 1:
        decision_evidence.append("multiple mutually-exclusive type labels present")
    readiness_evidence = _ready_evidence(body)
    preparation_packet = _parse_preparation_packet(body, issue_number=number)
    parent_issue = bool(labels & PARENT_LABELS) or bool(PARENT_TITLE_PATTERN.search(title))
    if parent_issue:
        findings.append("parent issue cannot be promoted to state:ready")
    gate_blocked = bool(blocker_evidence)
    decision_required = bool(decision_evidence)
    ready = (
        bool(readiness_evidence)
        and not gate_blocked
        and not decision_required
        and not job_inventory_uncertain
        and not parent_issue
    )
    preparation_withholds_readiness = preparation_packet["status"] == "invalid" or (
        preparation_packet["status"] == "valid"
        and (
            preparation_packet["implementation_admitted"] is not True
            or preparation_packet["state_ready_change_proposed"] is not True
        )
    )
    if preparation_withholds_readiness:
        ready = False
        findings.append(str(preparation_packet["reason"]))
    closure_rows: Sequence[Mapping[str, Any]] = (
        () if merged_pr_index is not None else list(merged_prs)
    )
    closure = closure_evidence(
        issue,
        merged_prs=closure_rows,
        merged_pr_index=merged_pr_index,
        open_issue_numbers=open_issue_numbers,
    )
    if completion_receipt is None:
        receipt_admission = {
            "eligible": False,
            "reason": "exact-head completion receipt is required",
            "errors": ["exact-head completion receipt is required"],
        }
    else:
        receipt_admission = admit_completion_receipt(
            completion_receipt,
            expected_repository=repository,
            expected_issue=number,
            issue_contract=body,
        )
    closure["completion_receipt"] = receipt_admission
    working_state = "state:working" in state_labels
    if working_state and not receipt_admission["eligible"]:
        findings.append(
            "state:working downstream promotion withheld: " + str(receipt_admission["reason"])
        )
    execution_state_set = set(execution_state_labels)
    stale_running = "state:running" in execution_state_set and not active_now
    if stale_running:
        findings.append("state:running has no currently observed active record; preserved")
    state_set = execution_state_set
    ready = ready and not stale_running
    if working_state and not receipt_admission["eligible"]:
        ready = False
    external_blocker = any(item["kind"] == "external-input" for item in blocker_evidence)
    execution_records_active = any(active.get(kind) for kind in ("claims", "worktrees", "jobs"))
    terminal_review_override = bool(terminal_review_evidence) and not (
        execution_records_active or gate_blocked or job_inventory_uncertain
    )
    if terminal_review_override and active["open_prs"]:
        findings.append(
            "terminal review status supersedes open PR-only activity for dispatch classification"
        )
    blocked_label_decision = "not-applicable"
    winner = _state_winner(
        state_set,
        blocker=gate_blocked and bool(blocked_reason_evidence),
        external_blocker=external_blocker,
        active=active_now and not terminal_review_override,
        ready=ready,
    )
    if terminal_review_override:
        winner = None
    if working_state and not receipt_admission["eligible"] and "state:ready" in state_set:
        winner = None
    mutations: list[dict[str, Any]] = []

    if working_state and not receipt_admission["eligible"] and "state:ready" in state_set:
        if _available("state:ready", available_labels):
            mutations.append(
                _mutation(
                    "remove_label",
                    number,
                    value="state:ready",
                    reason="withhold downstream readiness until exact-head receipt verification",
                    evidence=[str(receipt_admission["reason"])],
                )
            )
        else:
            findings.append("cannot remove unavailable label state:ready")

    if terminal_review_override:
        for label in sorted(state_set.intersection({"state:ready", "state:running"})):
            if _available(label, available_labels):
                mutations.append(
                    _mutation(
                        "remove_label",
                        number,
                        value=label,
                        reason="terminal report status selects review instead of dispatch or execution",
                        evidence=terminal_review_evidence,
                    )
                )
            else:
                findings.append(f"cannot remove unavailable label {label}")
        if REVIEW_STATE_LABEL not in labels:
            if _available(REVIEW_STATE_LABEL, available_labels):
                mutations.append(
                    _mutation(
                        "add_label",
                        number,
                        value=REVIEW_STATE_LABEL,
                        reason="mark completed execution as awaiting domain review",
                        evidence=terminal_review_evidence,
                    )
                )
            else:
                findings.append(f"cannot add unavailable label {REVIEW_STATE_LABEL}")

    if len(state_set) > 1:
        if winner is None:
            decision_required = True
            decision_evidence.append("multiple state labels without decisive repository evidence")
        else:
            for label in sorted(state_set):
                if label != winner:
                    if _available(label, available_labels):
                        mutations.append(
                            _mutation(
                                "remove_label",
                                number,
                                value=label,
                                reason=f"remove contradictory state label; evidence selects {winner}",
                                evidence=[
                                    *(item["text"] for item in blocker_evidence),
                                    *readiness_evidence,
                                ],
                            )
                        )
                    else:
                        findings.append(f"cannot remove unavailable label {label}")

    if winner and winner not in state_set and _available(winner, available_labels):
        for label in sorted(state_set):
            if _available(label, available_labels):
                mutations.append(
                    _mutation(
                        "remove_label",
                        number,
                        value=label,
                        reason=f"replace stale state label; evidence selects {winner}",
                        evidence=[
                            *(item["text"] for item in blocker_evidence),
                            *readiness_evidence,
                            "active PR/claim/worktree/job evidence" if active_now else "",
                        ],
                    )
                )
            else:
                findings.append(f"cannot remove unavailable label {label}")
        mutations.append(
            _mutation(
                "add_label",
                number,
                value=winner,
                reason="repair missing state label from repository evidence",
                evidence=[
                    *(item["text"] for item in blocker_evidence),
                    *readiness_evidence,
                    "active PR/claim/worktree/job evidence" if active_now else "",
                ],
                blocked_reason=(blocked_reason_evidence if winner in BLOCKED_LABELS else ()),
            )
        )
    elif winner and winner not in state_set:
        findings.append(f"cannot add unavailable label {winner}")

    # A type mirror is safe only for a valid, complete archetype metadata block
    # and only when no type label already exists.  We do not repair malformed
    # metadata or choose among competing types.
    metadata = audit_archetype_metadata(body)
    parsed_metadata = metadata.parsed_metadata
    archetype = parsed_metadata.get("archetype") if isinstance(parsed_metadata, Mapping) else None
    target_type = SAFE_ARCHETYPE_LABEL_MAP.get(archetype) if isinstance(archetype, str) else None
    if not type_labels and not metadata.findings and target_type:
        if _available(target_type, available_labels):
            mutations.append(
                _mutation(
                    "add_label",
                    number,
                    value=target_type,
                    reason="mirror complete issue archetype metadata",
                    evidence=[f"archetype metadata: {archetype}"],
                )
            )
        else:
            findings.append(f"cannot add unavailable label {target_type}")

    # Explicit decision gates always win over readiness and are never guessed
    # away.  A proven blocker is visible through state:blocked when that label
    # exists, but a maintainer decision is not converted into an answer.
    if (
        decision_required
        and DECISION_LABEL not in labels
        and _available(DECISION_LABEL, available_labels)
    ):
        mutations.append(
            _mutation(
                "add_label",
                number,
                value=DECISION_LABEL,
                reason="make an explicit maintainer decision gate visible",
                evidence=decision_evidence,
            )
        )
    if gate_blocked and not state_set.intersection(
        {"state:blocked", "state:blocked-external-input"}
    ):
        target = (
            "state:blocked-external-input"
            if any(item["kind"] == "external-input" for item in blocker_evidence)
            else "state:blocked"
        )
        if _available(target, available_labels):
            if blocked_reason_evidence:
                blocked_label_decision = "apply"
                mutations.append(
                    _mutation(
                        "add_label",
                        number,
                        value=target,
                        reason="record a proven provenance, rights, compute, or external-input gate",
                        evidence=[item["text"] for item in blocker_evidence],
                        blocked_reason=blocked_reason_evidence,
                    )
                )
            else:
                blocked_label_decision = "declined-needs-triage"
                findings.append(
                    "declined state:blocked label because no blocked-triage-v1 or "
                    "Blocked-by reference is present"
                )
                if TRIAGE_LABEL not in labels and _available(TRIAGE_LABEL, available_labels):
                    mutations.append(
                        _mutation(
                            "add_label",
                            number,
                            value=TRIAGE_LABEL,
                            reason=(
                                "do not apply a blocked state without an explicit triage reason; "
                                "route the issue for triage"
                            ),
                            evidence=[item["text"] for item in blocker_evidence],
                        )
                    )
        else:
            findings.append(f"cannot add unavailable blocker label {target}")
    elif gate_blocked:
        blocked_label_decision = (
            "already-present" if blocked_reason_evidence else "existing-unexplained"
        )
        if not blocked_reason_evidence:
            if TRIAGE_LABEL not in labels and _available(TRIAGE_LABEL, available_labels):
                mutations.append(
                    _mutation(
                        "add_label",
                        number,
                        value=TRIAGE_LABEL,
                        reason=(
                            "surface an existing blocked label without an explicit triage reason "
                            "for maintainer review"
                        ),
                        evidence=[item["text"] for item in blocker_evidence],
                    )
                )
            findings.append("existing blocked state lacks a blocked-triage-v1 or Blocked-by reason")

    # Never promote an issue to ready while any active execution record or
    # unresolved decision exists.  This guard is intentionally redundant with
    # the winner calculation because it protects future rule additions.
    if ready and not active_now and not decision_required and not gate_blocked:
        if "state:ready" not in state_set and _available("state:ready", available_labels):
            mutations.append(
                _mutation(
                    "add_label",
                    number,
                    value="state:ready",
                    reason="issue has explicit acceptance/validation evidence and no active gate",
                    evidence=readiness_evidence,
                )
            )

    classification = (
        "decision-required"
        if decision_required
        else "blocked"
        if gate_blocked
        else ("running" if active_now else "ready" if ready else "unclassified")
    )
    if closure.get("eligible") and str(issue.get("state") or "").lower() == "open":
        if not decision_required and not gate_blocked and not active["open_prs"]:
            if receipt_admission["eligible"]:
                mutations.append(
                    _mutation(
                        "close_issue",
                        number,
                        value=None,
                        reason="documented closure condition and verified completion receipt are proven",
                        evidence=[
                            closure.get("reason", ""),
                            *(f"merged PR #{pr}" for pr in closure.get("merged_prs", [])),
                            f"completion receipt {receipt_admission['receipt_digest']}",
                            f"delivered head {receipt_admission['head_sha']}",
                        ],
                    )
                )
                classification = "complete"
            else:
                findings.append("autonomous close withheld: " + str(receipt_admission["reason"]))

    # Preserve a pre-existing running state when discovery has no active
    # evidence: stale state is uncertain, not permission to promote to ready.
    if stale_running and classification == "unclassified":
        classification = "running"

    # Multiple rules can independently notice the same missing state or
    # blocker. Keep one stable operation so a batch cannot apply duplicate
    # writes or report an empty evidence item.
    unique_mutations: list[dict[str, Any]] = []
    seen_mutations: set[tuple[str, int, str | None]] = set()
    for mutation in mutations:
        key = (
            str(mutation.get("operation")),
            int(mutation.get("issue", number)),
            mutation.get("value") if isinstance(mutation.get("value"), str) else None,
        )
        if key in seen_mutations:
            continue
        seen_mutations.add(key)
        mutation["evidence"] = [item for item in mutation.get("evidence", []) if item]
        unique_mutations.append(mutation)

    return Classification(
        issue=number,
        classification=classification,
        state_labels=state_labels,
        execution_state_labels=execution_state_labels,
        resource_labels=resource_labels,
        type_labels=type_labels,
        evidence_labels=evidence_labels,
        active=active,
        blocker_evidence=tuple(blocker_evidence),
        blocked_reason_evidence=tuple(blocked_reason_evidence),
        blocked_label_decision=blocked_label_decision,
        decision_required=decision_required,
        decision_evidence=tuple(decision_evidence),
        terminal_review_evidence=tuple(terminal_review_evidence),
        readiness_evidence=tuple(readiness_evidence),
        preparation_packet=preparation_packet,
        closure=closure,
        mutations=tuple(unique_mutations),
        findings=tuple(findings),
    )


def build_audit_plan(
    inventory: Mapping[str, Any],
    *,
    mode: str = "autonomous",
    max_mutations: int = DEFAULT_MAX_MUTATIONS,
    max_wall_seconds: float | None = None,
    deadline: float | None = None,
    source_sha: str | None = None,
    classifier_digest: str | None = None,
    producer: Mapping[str, Any] | None = None,
    read_only_diagnostic: bool = False,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Build the shared issue_audit_plan.v1 from a read-only inventory."""
    if mode not in {"autonomous", "interactive"}:
        raise ValueError("mode must be autonomous or interactive")
    effective_deadline = _resolve_deadline(max_wall_seconds, deadline)
    resolved_source_sha = (
        source_sha if source_sha is not None else resolve_source_sha(command_runner=command_runner)
    )
    resolved_classifier_digest = (
        classifier_digest if classifier_digest is not None else resolve_classifier_digest()
    )
    resolved_producer = dict(producer) if producer is not None else resolve_producer_identity()
    provenance_available = bool(
        resolved_source_sha
        and resolved_classifier_digest
        and isinstance(resolved_producer, Mapping)
        and resolved_producer.get("identity")
    )
    raw_issue_rows = inventory.get("issues", [])
    issue_row_errors: list[str] = []
    if not isinstance(raw_issue_rows, list):
        issue_row_errors.append("inventory issues must be a list")
        candidate_issues: list[Mapping[str, Any]] = []
    else:
        candidate_issues = []
        for index, item in enumerate(raw_issue_rows):
            if not isinstance(item, Mapping):
                issue_row_errors.append(f"inventory issue row {index} is not an object")
                continue
            candidate_issues.append(item)
    open_prs = [item for item in inventory.get("open_prs", []) if isinstance(item, Mapping)]
    merged_prs = [item for item in inventory.get("merged_prs", []) if isinstance(item, Mapping)]
    claims = inventory.get("claims") if isinstance(inventory.get("claims"), Mapping) else {}
    completion_receipts = (
        inventory.get("completion_receipts")
        if isinstance(inventory.get("completion_receipts"), Mapping)
        else {}
    )
    repository = str(inventory.get("repo") or DEFAULT_REPO)
    for index, issue in enumerate(candidate_issues):
        issue_row_errors.extend(
            f"inventory issue row {index}: {error}"
            for error in _issue_row_validation_errors(
                issue,
                repository=repository,
                normalized=True,
            )
        )
    issues = [] if issue_row_errors else candidate_issues
    worktrees = [item for item in inventory.get("worktrees", []) if isinstance(item, Mapping)]
    jobs = [item for item in inventory.get("jobs", []) if isinstance(item, Mapping)]
    available_labels = set(_label_names(inventory.get("labels")))
    issue_inventory_status = _issue_inventory_source_assessment(
        inventory,
        canonical_row_count=len(candidate_issues),
    )
    if issue_row_errors:
        issue_inventory_status = {
            **issue_inventory_status,
            "status": ISSUE_SOURCE_STATUS_ANOMALOUS,
            "admissible": False,
            "error_code": "issue_inventory_rows_invalid",
            "reason": "; ".join(issue_row_errors),
        }
    inventory_meta = inventory.get("inventory")
    inventory_meta = inventory_meta if isinstance(inventory_meta, Mapping) else {}
    job_meta = inventory_meta.get("jobs", {})
    job_available = bool(job_meta.get("available", True)) if isinstance(job_meta, Mapping) else True
    open_numbers = {int(item.get("number", 0)) for item in issues}
    classifications: list[dict[str, Any]] = []
    mutations: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    blocked_label_report: list[dict[str, Any]] = []
    ordered_issues = sorted(issues, key=lambda item: int(item.get("number", 0)))
    try:
        with _deadline_interrupt(effective_deadline):
            merged_pr_index, index_timed_out = _index_merged_prs(
                merged_prs,
                deadline=effective_deadline,
            )
    except _AuditDeadlineExceeded:
        merged_pr_index, index_timed_out = {}, True
    classification_timeout_reason = (
        "issue-audit wall-time budget exhausted while indexing merged-PR references"
        if index_timed_out
        else None
    )
    for issue in ordered_issues:
        if classification_timeout_reason is not None:
            break
        if _deadline_expired(effective_deadline):
            classification_timeout_reason = (
                "issue-audit wall-time budget exhausted "
                + ("before" if not classifications else "during")
                + " issue classification"
            )
            break
        try:
            with _deadline_interrupt(effective_deadline):
                classified = classify_issue(
                    issue,
                    open_prs=open_prs,
                    merged_prs=merged_prs,
                    merged_pr_index=merged_pr_index,
                    claims=claims,
                    worktrees=worktrees,
                    jobs=jobs,
                    job_inventory_available=job_available,
                    open_issue_numbers=open_numbers,
                    available_labels=available_labels,
                    completion_receipt=_completion_receipt_for_issue(
                        completion_receipts, int(issue["number"])
                    ),
                    repository=repository,
                )
        except _AuditDeadlineExceeded:
            classification_timeout_reason = (
                "issue-audit wall-time budget exhausted during issue classification"
            )
            break
        issue_labels = _label_names(issue.get("labels"))
        decision_sources = _decision_source_rows(issue)
        documented_options = _documented_options(issue)
        expected_issue = {
            "state": str(issue.get("state") or "").lower(),
            "updated_at": str(issue.get("updated_at") or ""),
            # Sorted label snapshot for timestamp-only drift tolerance at apply
            # time (issue #8295). Sorted for plan-digest determinism.
            "labels": sorted(issue_labels),
        }
        issue_mutations = [
            {**mutation, "expected_issue": expected_issue.copy()}
            for mutation in classified.mutations
        ]
        row = {
            "number": int(issue.get("number", 0)),
            "title": str(issue.get("title") or ""),
            "url": str(issue.get("url") or ""),
            "state": str(issue.get("state") or "").lower(),
            "updated_at": str(issue.get("updated_at") or ""),
            "labels": issue_labels,
            "decision_sources": decision_sources,
            "documented_options": documented_options,
            **classified.to_dict(),
            "mutations": issue_mutations,
        }
        classifications.append(row)
        mutations.extend(issue_mutations)
        if classified.blocked_label_decision != "not-applicable":
            blocked_label_report.append(
                {
                    "issue": classified.issue,
                    "decision": classified.blocked_label_decision,
                    "blocker_evidence": list(classified.blocker_evidence),
                    "reason_evidence": list(classified.blocked_reason_evidence),
                    "fallback_label": (
                        TRIAGE_LABEL
                        if classified.blocked_label_decision == "declined-needs-triage"
                        else None
                    ),
                }
            )
        if classified.decision_required:
            pending.append(
                {
                    "issue": f"#{classified.issue}",
                    "number": classified.issue,
                    "title": str(issue.get("title") or ""),
                    "url": str(issue.get("url") or ""),
                    "state": str(issue.get("state") or "").lower(),
                    "labels": issue_labels,
                    "classification": classified.classification,
                    "decision_required": True,
                    "question_source": "issue body/comments",
                    "blocking_evidence": "; ".join(classified.decision_evidence)
                    or "decision gate detected",
                    "decision_evidence": list(classified.decision_evidence),
                    "evidence_sources": decision_sources,
                    "documented_options": documented_options,
                    "safe_mutations_applied": [],
                }
            )
        if _deadline_expired(effective_deadline):
            classification_timeout_reason = (
                "issue-audit wall-time budget exhausted during issue classification"
            )
            break
    if classification_timeout_reason is None and _deadline_expired(effective_deadline):
        classification_timeout_reason = (
            "issue-audit wall-time budget exhausted while finalizing the audit plan"
        )
    quota_meta: dict[str, Any] = {}
    if isinstance(inventory.get("quota"), Mapping):
        quota_meta.update(inventory["quota"])
    if isinstance(inventory_meta.get("quota"), Mapping):
        quota_meta.update(inventory_meta["quota"])
    quota_contract_failures = _quota_contract_failures_from_inventory(
        inventory,
        required=(not read_only_diagnostic and bool(mutations or pending)),
    )
    if quota_contract_failures:
        quota_meta["contract_errors"] = sorted(set(quota_contract_failures))
    quota_exhausted = bool(
        quota_meta.get("quota_exhausted") or quota_meta.get("status") == "exhausted"
    )
    quota_uncertain = bool(
        quota_meta.get("quota_uncertain")
        or quota_meta.get("budget_exhausted")
        or quota_exhausted
        or quota_meta.get("status")
        in {
            "exhausted",
            "insufficient",
            "quota_blocked",
            "failed",
            "malformed",
            "unavailable",
        }
        or (isinstance(quota_meta.get("available"), bool) and not quota_meta["available"])
        or bool(quota_contract_failures)
    )
    if quota_uncertain:
        quota_meta["quota_uncertain"] = True
    truncated: list[str] = []
    inventory_uncertainties: list[str] = []
    issue_inventory_incomplete = not bool(issue_inventory_status["admissible"])
    if issue_inventory_incomplete:
        inventory_uncertainties.append("issues")
        truncated.append("issues")
    for name, meta in inventory_meta.items():
        if not isinstance(meta, Mapping):
            continue
        if name == "comments" and meta.get("reason") == "not_requested":
            continue
        if name == "jobs" and meta.get("available") is False:
            inventory_uncertainties.append(name)
            continue
        source_uncertain = bool(
            meta.get("truncated") or meta.get("errors") or meta.get("available") is False
        )
        if source_uncertain:
            truncated.append(name)
            if name != "jobs":
                inventory_uncertainties.append(name)
    if quota_uncertain:
        inventory_uncertainties.append("core_quota")
        truncated.append("core_quota")
    remaining_issue_numbers = [
        int(issue.get("number", 0)) for issue in ordered_issues[len(classifications) :]
    ]
    classification_timed_out = classification_timeout_reason is not None
    if not read_only_diagnostic and not provenance_available:
        truncated.append("provenance_unavailable")
        inventory_uncertainties.append("provenance_unavailable")
    incomplete_inventory = (
        classification_timed_out
        or quota_uncertain
        or bool(truncated)
        or read_only_diagnostic
        or not provenance_available
    )
    if read_only_diagnostic and classification_timeout_reason is None:
        effective_reason: str | None = (
            "issue-audit running in read-only diagnostic mode; mutations suppressed"
        )
    elif not provenance_available and classification_timeout_reason is None:
        effective_reason = "issue-audit source provenance is unavailable; mutations suppressed"
    elif issue_inventory_incomplete and classification_timeout_reason is None:
        effective_reason = str(issue_inventory_status["reason"])
    elif incomplete_inventory and classification_timeout_reason is None:
        effective_reason = "issue-audit inventory is incomplete; mutations suppressed"
    else:
        effective_reason = classification_timeout_reason
    classification_status = {
        "status": (
            "timed_out"
            if classification_timed_out
            else "incomplete"
            if issue_inventory_incomplete
            else "complete"
        ),
        "reason": effective_reason,
        "classified_issues": len(classifications),
        "total_issues": len(ordered_issues),
        "remaining_issue_numbers": remaining_issue_numbers,
        "resume_from_issue": remaining_issue_numbers[0] if remaining_issue_numbers else None,
        "resume_supported": False,
        "resume_requires_fresh_full_inventory": True,
        "mutations_suppressed": incomplete_inventory,
    }
    if incomplete_inventory:
        # Partial or uncertain inventory is useful for diagnosis, but no mutation
        # from an incomplete pass is safe to carry into an apply operation.
        mutations = []
        _suppress_mutation_fields(classifications)
        if classification_timed_out:
            truncated.append("classification")
    if len(mutations) > max_mutations:
        mutations = mutations[:max_mutations]
        truncated.append("mutation_budget")
    plan = {
        "schema": PLAN_SCHEMA,
        "repo": str(inventory.get("repo") or DEFAULT_REPO),
        "mode": mode,
        "source_sha": resolved_source_sha,
        "classifier_digest": resolved_classifier_digest,
        "producer": resolved_producer,
        "diagnostic_mode": read_only_diagnostic,
        "provenance": {
            "schema": PROVENANCE_SCHEMA,
            "source_sha": resolved_source_sha,
            "classifier_digest": resolved_classifier_digest,
            "producer": resolved_producer,
        },
        "project5": {"writes": False, "owner": "gh-issue-sequencer"},
        "label_policy": {
            "create_missing": False,
            "mutually_exclusive": [sorted(EXECUTION_STATE_LABELS), TYPE_PREFIX],
            "composable": [RESOURCE_PREFIX, EVIDENCE_PREFIX, sorted(STATE_QUALIFIER_LABELS)],
            "preserve_state_qualifiers": True,
        },
        "inventory": dict(inventory_meta),
        "issue_inventory_status": issue_inventory_status,
        "classification_status": classification_status,
        "inventory_coverage": (
            dict(inventory_meta.get("closure_coverage"))
            if isinstance(inventory_meta.get("closure_coverage"), Mapping)
            else {}
        ),
        "inventory_uncertainties": sorted(set(inventory_uncertainties)),
        "quota": dict(quota_meta),
        "issues": classifications,
        "mutations": mutations,
        "blocked_label_report": blocked_label_report,
        "pending_decisions": pending,
        "truncation_or_errors": sorted(set(truncated)),
        "counts": {
            "open_issues": len(classifications),
            "mutations": len(mutations),
            "blocked_label_decisions": len(blocked_label_report),
            "pending_decisions": len(pending),
            "truncated_or_error_sources": len(set(truncated)),
        },
    }
    if inventory.get(LEGACY_ISSUE_INVENTORY_MARKER) is True:
        plan[LEGACY_ISSUE_INVENTORY_MARKER] = True
    if incomplete_inventory:
        _suppress_mutation_fields(plan)
    elif _deadline_expired(effective_deadline):
        _mark_plan_timed_out(
            plan,
            reason="issue-audit wall-time budget exhausted while finalizing the audit plan",
        )
    else:
        try:
            with _deadline_interrupt(effective_deadline):
                plan["plan_digest"] = compute_plan_digest(plan)
        except _AuditDeadlineExceeded:
            _mark_plan_timed_out(
                plan,
                reason="issue-audit wall-time budget exhausted while finalizing the audit plan",
            )
        if _deadline_expired(effective_deadline):
            _mark_plan_timed_out(
                plan,
                reason="issue-audit wall-time budget exhausted while finalizing the audit plan",
            )
    if "plan_digest" not in plan:
        plan["plan_digest"] = compute_plan_digest(plan)
    return plan


def label_api_path(repo: str, issue_number: int, label: str) -> str:
    """Return the REST label endpoint with every label character URI-escaped."""
    return f"repos/{repo}/issues/{issue_number}/labels/{quote(label, safe='')}"


def _blocked_label_plan_errors(mutations: Sequence[object]) -> list[dict[str, Any]]:
    """Reject blocked-label writes that are not bound to explicit reason evidence."""
    errors: list[dict[str, Any]] = []
    for index, mutation in enumerate(mutations):
        if not isinstance(mutation, Mapping):
            continue
        if mutation.get("operation") != "add_label" or mutation.get("value") not in BLOCKED_LABELS:
            continue
        reason_evidence = mutation.get("blocked_reason")
        valid = isinstance(reason_evidence, list) and any(
            isinstance(item, str) and item.strip() for item in reason_evidence
        )
        if not valid:
            errors.append(
                {
                    "index": index,
                    "mutation": dict(mutation),
                    "error": (
                        "blocked label mutation requires blocked_reason evidence from "
                        "blocked-triage-v1 or a Blocked-by reference"
                    ),
                }
            )
    return errors


def _is_absent_label_delete(result: subprocess.CompletedProcess[str]) -> bool:
    """Recognize GitHub's idempotent missing-label delete response only."""
    if result.returncode == 0:
        return False
    detail = (result.stderr or result.stdout).strip()
    return bool(
        "404" in detail
        and re.search(r"\blabel\b.*\b(?:does not exist|not found)\b", detail, re.IGNORECASE)
    )


def _mutation_issue_preconditions(
    mutations: Sequence[object],
) -> tuple[dict[int, dict[str, str]], list[dict[str, Any]]]:
    """Validate mutation shape and one state/version snapshot per issue batch."""
    preconditions: dict[int, dict[str, str]] = {}
    errors: list[dict[str, Any]] = []
    for index, mutation in enumerate(mutations):
        if not isinstance(mutation, Mapping):
            errors.append({"index": index, "error": "mutation is not an object"})
            continue
        raw_issue = mutation.get("issue")
        if isinstance(raw_issue, bool) or not isinstance(raw_issue, int):
            errors.append(
                {
                    "index": index,
                    "error": "mutation issue must be an exact positive integer",
                }
            )
            continue
        number = raw_issue
        if number <= 0:
            errors.append({"index": index, "issue": number, "error": "issue must be positive"})
            continue
        operation = str(mutation.get("operation") or "")
        value = mutation.get("value")
        if operation not in {"add_label", "remove_label", "close_issue"}:
            errors.append(
                {
                    "index": index,
                    "issue": number,
                    "operation": operation,
                    "error": f"unsupported mutation: {operation}",
                }
            )
            continue
        if operation in {"add_label", "remove_label"} and not (isinstance(value, str) and value):
            errors.append(
                {
                    "index": index,
                    "issue": number,
                    "operation": operation,
                    "error": f"{operation} requires a non-empty string value",
                }
            )
            continue
        if operation == "close_issue" and value is not None:
            errors.append(
                {
                    "index": index,
                    "issue": number,
                    "operation": operation,
                    "error": "close_issue requires a null value",
                }
            )
            continue
        raw_expected = mutation.get("expected_issue")
        if not isinstance(raw_expected, Mapping):
            errors.append(
                {
                    "index": index,
                    "issue": number,
                    "error": "mutation requires an expected_issue state/version snapshot",
                }
            )
            continue
        expected = {
            "state": str(raw_expected.get("state") or "").lower(),
            "updated_at": str(raw_expected.get("updated_at") or ""),
        }
        if expected["state"] not in {"open", "closed"} or not expected["updated_at"]:
            errors.append(
                {
                    "index": index,
                    "issue": number,
                    "expected_issue": expected,
                    "error": "expected_issue requires state=open|closed and non-empty updated_at",
                }
            )
            continue
        raw_labels = raw_expected.get("labels")
        if raw_labels is not None:
            if not isinstance(raw_labels, list) or not all(
                isinstance(entry, str) for entry in raw_labels
            ):
                errors.append(
                    {
                        "index": index,
                        "issue": number,
                        "expected_issue": expected,
                        "error": "expected_issue labels must be a string list when present",
                    }
                )
                continue
            expected["labels"] = sorted(raw_labels)
        previous = preconditions.setdefault(number, expected)
        if previous != expected:
            errors.append(
                {
                    "index": index,
                    "issue": number,
                    "expected_issue": expected,
                    "first_expected_issue": previous,
                    "error": "one issue mutation batch has inconsistent expected_issue snapshots",
                }
            )
    return preconditions, errors


def apply_mutations(
    plan: Mapping[str, Any],
    *,
    max_mutations: int = DEFAULT_MAX_MUTATIONS,
    runner: Runner | None = None,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    apply_source_sha: str | None = None,
    apply_classifier_digest: str | None = None,
    apply_producer: Mapping[str, Any] | None = None,
    check_freshness: bool = False,
    enforce_provenance: bool = True,
    remote: str = DEFAULT_REMOTE,
) -> dict[str, Any]:
    """Apply a bounded plan and read back every touched issue.

    The caller is responsible for choosing the skill authority.  This function
    only executes explicit operations in the plan and refuses a truncated plan,
    missing repository, or unsupported operation.
    """
    raw_mutations = plan.get("mutations")
    planned_count = len(raw_mutations) if isinstance(raw_mutations, list) else 0

    curr_source_sha = (
        apply_source_sha
        if apply_source_sha is not None
        else resolve_source_sha(command_runner=command_runner)
    )
    curr_classifier_digest = (
        apply_classifier_digest
        if apply_classifier_digest is not None
        else resolve_classifier_digest()
    )
    curr_producer = (
        dict(apply_producer) if apply_producer is not None else resolve_producer_identity()
    )
    plan_source_sha = str(plan.get("source_sha") or "")
    plan_classifier_digest = str(plan.get("classifier_digest") or "")
    plan_producer = dict(plan["producer"]) if isinstance(plan.get("producer"), Mapping) else {}
    recorded_digest = str(plan.get("plan_digest") or "")

    def empty_counts(failed: int) -> dict[str, int]:
        """Return the stable no-write count shape for an early refusal."""
        return {
            "planned": planned_count,
            "applied": 0,
            "already_applied": 0,
            "failed": failed,
            "stale_state_issues": 0,
            "skipped_stale_mutations": 0,
        }

    def refuse(
        reason: str,
        *,
        failed: int = 1,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        """Return a stable structured refusal without attempting a write."""
        refusal = {
            "schema": "issue_audit_apply.v1",
            "ok": False,
            "reason": reason,
            "source_sha": curr_source_sha,
            "classifier_digest": curr_classifier_digest,
            "producer": curr_producer,
            "plan_digest": recorded_digest,
            "plan_source_sha": plan_source_sha,
            "plan_classifier_digest": plan_classifier_digest,
            "applied": [],
            "already_applied": [],
            "stale_states": [],
            "timestamp_drift_bypassed": [],
            "failures": [reason],
            "readback": [],
            "counts": empty_counts(failed),
        }
        if error_code:
            refusal["error_code"] = error_code
        return refusal

    if plan.get("schema") != PLAN_SCHEMA:
        return refuse(f"expected {PLAN_SCHEMA}")
    if plan.get("diagnostic_mode"):
        return refuse("cannot apply mutations from a read-only diagnostic plan")
    if not isinstance(raw_mutations, list):
        return refuse("plan mutations must be a list")
    if len(raw_mutations) > max_mutations:
        return refuse("plan exceeds mutation budget")
    raw_truncation_errors = plan.get("truncation_or_errors", [])
    if not isinstance(raw_truncation_errors, list):
        return refuse("plan truncation_or_errors must be a list")

    plan_issue_rows = plan.get("issues")
    canonical_row_count = len(plan_issue_rows) if isinstance(plan_issue_rows, list) else 0
    plan_issue_numbers, plan_issue_row_errors = _plan_issue_row_numbers(plan)
    issue_inventory_status = _issue_inventory_source_assessment(
        {
            "repo": plan.get("repo", DEFAULT_REPO),
            "inventory": plan.get("inventory"),
            LEGACY_ISSUE_INVENTORY_MARKER: plan.get(LEGACY_ISSUE_INVENTORY_MARKER),
        },
        canonical_row_count=canonical_row_count,
    )
    recorded_issue_inventory_status = plan.get("issue_inventory_status")
    recorded_status_inadmissible = isinstance(recorded_issue_inventory_status, Mapping) and (
        recorded_issue_inventory_status.get("admissible") is False
        or recorded_issue_inventory_status.get("status")
        in {ISSUE_SOURCE_STATUS_UNAVAILABLE, ISSUE_SOURCE_STATUS_ANOMALOUS}
    )
    if recorded_status_inadmissible:
        refusal = refuse(
            str(
                recorded_issue_inventory_status.get("reason")
                or "canonical open-issue source is not admissible"
            ),
            error_code=str(
                recorded_issue_inventory_status.get("error_code")
                or issue_inventory_status.get("error_code")
                or "issue_inventory_source_unavailable"
            ),
        )
        refusal["issue_inventory_status"] = dict(recorded_issue_inventory_status)
        return refusal
    if not issue_inventory_status["admissible"]:
        refusal = refuse(
            str(issue_inventory_status["reason"]),
            error_code=str(issue_inventory_status["error_code"]),
        )
        refusal["issue_inventory_status"] = issue_inventory_status
        return refusal
    pending_decisions = plan.get("pending_decisions")
    if canonical_row_count == 0 and (
        planned_count > 0
        or (
            pending_decisions is not None
            and (not isinstance(pending_decisions, list) or bool(pending_decisions))
        )
    ):
        return refuse(
            "an empty canonical issue inventory cannot carry mutations or pending decisions",
            error_code="issue_inventory_empty_with_work",
        )
    if plan_issue_row_errors:
        return refuse(
            "; ".join(plan_issue_row_errors),
            error_code="issue_inventory_rows_invalid",
        )
    reference_errors = _plan_issue_reference_errors(plan, issue_numbers=plan_issue_numbers)
    if reference_errors:
        return refuse(
            "; ".join(reference_errors),
            error_code="issue_inventory_reference_mismatch",
        )

    recorded_digest = str(plan.get("plan_digest") or "")
    if not recorded_digest:
        return {
            "schema": "issue_audit_apply.v1",
            "ok": False,
            "reason": "plan is missing plan_digest; regenerate it before applying",
            "applied": [],
            "already_applied": [],
            "stale_states": [],
            "timestamp_drift_bypassed": [],
            "failures": ["missing plan_digest"],
            "readback": [],
            "counts": empty_counts(1),
        }
    try:
        current_digest = compute_plan_digest(plan)
    except ValueError as exc:
        return {
            "schema": "issue_audit_apply.v1",
            "ok": False,
            "reason": str(exc),
            "applied": [],
            "already_applied": [],
            "stale_states": [],
            "timestamp_drift_bypassed": [],
            "failures": [str(exc)],
            "readback": [],
            "counts": empty_counts(1),
        }
    if recorded_digest != current_digest:
        return {
            "schema": "issue_audit_apply.v1",
            "ok": False,
            "reason": "stale plan_digest does not match the plan contents; regenerate it before applying",
            "applied": [],
            "already_applied": [],
            "stale_states": [],
            "timestamp_drift_bypassed": [],
            "failures": ["stale plan_digest"],
            "readback": [],
            "counts": empty_counts(1),
        }
    if raw_truncation_errors:
        return {
            "schema": "issue_audit_apply.v1",
            "ok": False,
            "reason": "inventory or mutation plan is incomplete",
            "applied": [],
            "already_applied": [],
            "stale_states": [],
            "timestamp_drift_bypassed": [],
            "failures": list(raw_truncation_errors),
            "readback": [],
            "counts": empty_counts(len(raw_truncation_errors)),
        }
    mutations = raw_mutations
    blocked_label_errors = _blocked_label_plan_errors(mutations)
    if blocked_label_errors:
        return {
            "schema": "issue_audit_apply.v1",
            "ok": False,
            "reason": "plan contains an unreasoned blocked-label mutation",
            "applied": [],
            "already_applied": [],
            "stale_states": [],
            "timestamp_drift_bypassed": [],
            "failures": blocked_label_errors,
            "readback": [],
            "counts": empty_counts(len(blocked_label_errors)),
        }
    preconditions, precondition_plan_errors = _mutation_issue_preconditions(mutations)
    if precondition_plan_errors:
        return {
            "schema": "issue_audit_apply.v1",
            "ok": False,
            "reason": "plan contains invalid mutations or issue mutation preconditions",
            "applied": [],
            "already_applied": [],
            "stale_states": [],
            "timestamp_drift_bypassed": [],
            "failures": precondition_plan_errors,
            "readback": [],
            "counts": empty_counts(len(precondition_plan_errors)),
        }

    if enforce_provenance and planned_count > 0:
        if not plan_source_sha:
            return refuse("plan is missing source_sha; regenerate it before applying")
        if not plan_classifier_digest:
            return refuse("plan is missing classifier_digest; regenerate it before applying")
        if not plan_producer or not plan_producer.get("identity"):
            return refuse("plan is missing producer identity; regenerate it before applying")
        if not curr_source_sha:
            return refuse("current source_sha is unavailable; cannot verify apply-time integrity")
        if not curr_classifier_digest:
            return refuse(
                "current classifier_digest is unavailable; cannot verify apply-time integrity"
            )
        if not curr_producer or not curr_producer.get("identity"):
            return refuse(
                "current producer identity is unavailable; cannot verify apply-time integrity"
            )
        if plan_source_sha != curr_source_sha:
            return refuse(
                f"plan-time source SHA ({plan_source_sha}) does not match apply-time source SHA ({curr_source_sha})"
            )
        if plan_classifier_digest != curr_classifier_digest:
            return refuse(
                f"plan-time classifier digest ({plan_classifier_digest}) does not match apply-time classifier digest ({curr_classifier_digest})"
            )
        if check_freshness:
            fresh, _head_sha, _main_sha, err = check_origin_main_freshness(
                remote, command_runner=command_runner
            )
            if not fresh:
                return refuse(
                    f"autonomous apply refused: executing revision does not contain current {remote}/main ({err or 'stale revision'}). Refresh with: git fetch {remote} main && git merge {remote}/main"
                )

    quota_contract_failures = _quota_contract_failures_from_plan(
        plan,
        required=planned_count > 0,
    )
    if quota_contract_failures:
        return refuse(
            "quota metadata is not admissible: " + "; ".join(sorted(set(quota_contract_failures))),
            error_code="quota_unavailable",
        )

    repo = str(plan.get("repo") or DEFAULT_REPO)
    run = _runner_or_default(runner)
    applied: list[dict[str, Any]] = []
    already_applied: list[dict[str, Any]] = []
    stale_states: list[dict[str, Any]] = []
    timestamp_drift_bypassed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    touched: set[int] = set()
    expectations: dict[int, dict[str, Any]] = {}
    preflighted: set[int] = set()
    skipped_batches: set[int] = set()

    def register_expected(number: int, operation: str, value: object) -> None:
        """Track successful and idempotent operations for the readback gate."""
        touched.add(number)
        expected = expectations.setdefault(
            number,
            {
                "add_labels": set(),
                "remove_labels": set(),
                "closed": False,
                "expected_state": preconditions[number]["state"],
            },
        )
        if operation == "add_label" and isinstance(value, str):
            expected["add_labels"].add(value)  # type: ignore[union-attr]
        elif operation == "remove_label" and isinstance(value, str):
            expected["remove_labels"].add(value)  # type: ignore[union-attr]
        elif operation == "close_issue":
            expected["closed"] = True

    for mutation in mutations:
        if not isinstance(mutation, Mapping):
            failures.append({"mutation": mutation, "error": "mutation is not an object"})
            continue
        number = mutation["issue"]
        operation = str(mutation["operation"])
        value = mutation.get("value")
        if number not in preflighted:
            preflighted.add(number)
            expected_issue = preconditions[number]
            preflight = run(["api", f"repos/{repo}/issues/{number}"], None)
            live_issue, preflight_error = _parse_json(preflight, what=f"pre-write issue {number}")
            if preflight_error or not isinstance(live_issue, Mapping):
                skipped_batches.add(number)
                failures.append(
                    {
                        "issue": number,
                        "disposition": "precondition_unavailable",
                        "expected_issue": expected_issue,
                        "error": preflight_error or "invalid payload",
                    }
                )
            else:
                observed_issue = {
                    "state": str(live_issue.get("state") or "").lower(),
                    "updated_at": str(live_issue.get("updated_at") or ""),
                }
                if "labels" in expected_issue:
                    # Timestamp-only drift tolerance (issue #8295): automation
                    # comments advance updated_at without touching the mutation
                    # target. Compare only the semantic fields (state plus the
                    # label set); a mismatch stays fail-closed stale.
                    observed_labels = sorted(_label_names(live_issue.get("labels")))
                    observed_issue["labels"] = observed_labels
                    semantic_match = (
                        observed_issue["state"] == expected_issue["state"]
                        and observed_labels == expected_issue["labels"]
                    )
                else:
                    # Legacy plan without a label snapshot: keep the strict
                    # state/version comparison.
                    semantic_match = observed_issue == expected_issue
                if not semantic_match:
                    skipped_batches.add(number)
                    stale = {
                        "issue": number,
                        "disposition": "stale_state",
                        "expected_issue": expected_issue,
                        "observed_issue": observed_issue,
                        "skipped_mutations": sum(
                            1
                            for candidate in mutations
                            if isinstance(candidate, Mapping) and candidate.get("issue") == number
                        ),
                    }
                    if "labels" in expected_issue:
                        stale["drift_kind"] = "semantic"
                        stale["retry"] = {
                            "action": "regenerate_plan",
                            "reason": "labels or state changed since the plan was built",
                            "detail": "re-run the plan step that produced this artifact, "
                            "then re-apply; do not retry this artifact",
                        }
                    stale_states.append(stale)
                    failures.append(stale)
                elif observed_issue["updated_at"] != expected_issue["updated_at"]:
                    timestamp_drift_bypassed.append(
                        {
                            "issue": number,
                            "expected_updated_at": expected_issue["updated_at"],
                            "observed_updated_at": observed_issue["updated_at"],
                        }
                    )
        if number in skipped_batches:
            continue
        if operation == "add_label" and isinstance(value, str) and value:
            endpoint = f"repos/{repo}/issues/{number}/labels"
            result = run(
                ["api", "-X", "POST", endpoint, "--input", "-"],
                json.dumps({"labels": [value]}),
            )
        elif operation == "remove_label" and isinstance(value, str) and value:
            result = run(["api", "-X", "DELETE", label_api_path(repo, number, value)], None)
        elif operation == "close_issue":
            result = run(
                ["api", "-X", "PATCH", f"repos/{repo}/issues/{number}", "--input", "-"],
                json.dumps({"state": "closed"}),
            )
        else:
            failures.append(
                {"mutation": dict(mutation), "error": f"unsupported mutation: {operation}"}
            )
            continue
        if result.returncode != 0:
            if (
                operation == "remove_label"
                and isinstance(value, str)
                and _is_absent_label_delete(result)
            ):
                already_applied.append(
                    {
                        **dict(mutation),
                        "skipped_reason": "already_absent",
                    }
                )
                register_expected(number, operation, value)
                continue
            failures.append(
                {
                    "mutation": dict(mutation),
                    "error": (result.stderr or result.stdout).strip()
                    or f"exit code {result.returncode}",
                }
            )
            continue
        applied.append(dict(mutation))
        register_expected(number, operation, value)

    readback: list[dict[str, Any]] = []
    for number in sorted(touched):
        result = run(["api", f"repos/{repo}/issues/{number}"], None)
        payload, error = _parse_json(result, what=f"readback issue {number}")
        if error or not isinstance(payload, Mapping):
            readback.append({"issue": number, "ok": False, "error": error or "invalid payload"})
            continue
        labels = _label_names(payload.get("labels"))
        state = str(payload.get("state") or "").lower()
        expected = expectations[number]
        added = sorted(set(expected["add_labels"]) & set(labels))  # type: ignore[arg-type]
        missing_additions = sorted(set(expected["add_labels"]) - set(labels))  # type: ignore[arg-type]
        removed = sorted(set(expected["remove_labels"]) - set(labels))  # type: ignore[arg-type]
        missing_removals = sorted(set(expected["remove_labels"]) & set(labels))  # type: ignore[arg-type]
        expected_closed = bool(expected["closed"])
        expected_state = "closed" if expected_closed else str(expected["expected_state"])
        state_ok = state == expected_state
        row_ok = not missing_additions and not missing_removals and state_ok
        readback.append(
            {
                "issue": number,
                "ok": row_ok,
                "state": state,
                "labels": labels,
                "verified": {
                    "added": added,
                    "missing_additions": missing_additions,
                    "removed": removed,
                    "missing_removals": missing_removals,
                    "closed": state == "closed" if expected_closed else None,
                    "expected_state": expected_state,
                    "state_matches": state_ok,
                },
            }
        )
    return {
        "schema": "issue_audit_apply.v1",
        "ok": not failures and all(row.get("ok") for row in readback),
        "source_sha": curr_source_sha,
        "classifier_digest": curr_classifier_digest,
        "producer": curr_producer,
        "plan_digest": recorded_digest,
        "plan_source_sha": plan_source_sha,
        "plan_classifier_digest": plan_classifier_digest,
        "applied": applied,
        "already_applied": already_applied,
        "stale_states": stale_states,
        "timestamp_drift_bypassed": timestamp_drift_bypassed,
        "failures": failures,
        "readback": readback,
        "counts": {
            "planned": len(mutations),
            "applied": len(applied),
            "already_applied": len(already_applied),
            "failed": len(failures),
            "stale_state_issues": len(stale_states),
            "skipped_stale_mutations": sum(
                int(row.get("skipped_mutations", 0)) for row in stale_states
            ),
        },
    }


def build_pending_decision_queue(
    plan: Mapping[str, Any],
    *,
    applied_mutations: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return the machine-readable queue consumed by the interactive skill."""
    pending = plan.get("pending_decisions")
    if not isinstance(pending, list):
        return []
    applied_by_issue: dict[str, list[dict[str, Any]]] = {}
    for mutation in applied_mutations:
        if not isinstance(mutation, Mapping):
            continue
        try:
            issue = f"#{int(mutation['issue'])}"
        except (KeyError, TypeError, ValueError):
            continue
        applied_by_issue.setdefault(issue, []).append(dict(mutation))
    queue: list[dict[str, Any]] = []
    for item in pending:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        issue = str(row.get("issue") or "")
        row["safe_mutations_applied"] = applied_by_issue.get(issue, [])
        queue.append(row)
    return sorted(queue, key=lambda row: _pending_issue_number(row.get("issue")))


def compute_plan_digest(plan: Mapping[str, Any]) -> str:
    """Return the canonical digest used to bind an envelope to one plan."""
    payload = {key: value for key, value in plan.items() if key != "plan_digest"}
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"plan is not JSON-canonicalizable: {exc}") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _pending_issue_number(value: object) -> int:
    """Normalize a pending queue issue reference for deterministic ordering."""
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else 2**31 - 1


def select_next_pending_decision(
    plan: Mapping[str, Any],
    *,
    issue_scope: Iterable[int] | None = None,
    applied_mutations: Iterable[Mapping[str, Any]] = (),
) -> tuple[int, int, dict[str, Any]] | None:
    """Select one queue entry by issue number, never by Project #5 ordering."""
    queue = build_pending_decision_queue(plan, applied_mutations=applied_mutations)
    allowed = {int(number) for number in issue_scope} if issue_scope is not None else None
    scoped_queue = [
        row
        for row in queue
        if allowed is None or _pending_issue_number(row.get("issue")) in allowed
    ]
    if not scoped_queue:
        return None
    return 0, len(scoped_queue), scoped_queue[0]


def _envelope_inventory_errors(plan: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    """Return inventory failures that make this decision envelope unsafe."""
    errors: list[str] = []
    truncation = plan.get("truncation_or_errors")
    if isinstance(truncation, list) and truncation:
        errors.append("plan inventory is truncated or contains errors")
    uncertainties = set(plan.get("inventory_uncertainties") or [])
    labels = set(_label_names(row.get("labels")))
    if "jobs" in uncertainties and "resource:slurm" in labels:
        errors.append("SLURM inventory is unavailable for a resource:slurm issue")
    return errors


def build_decision_envelope(
    plan: Mapping[str, Any],
    *,
    issue_scope: Iterable[int] | None = None,
    expected_plan_digest: str | None = None,
    applied_mutations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    """Build the next factual, machine-readable maintainer decision envelope."""
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"expected {PLAN_SCHEMA} plan")
    recorded_digest = str(plan.get("plan_digest") or "")
    if not recorded_digest:
        raise ValueError("plan is missing plan_digest; regenerate it before presenting a decision")
    current_digest = compute_plan_digest(plan)
    if recorded_digest != current_digest:
        raise ValueError("plan_digest does not match the plan contents")
    if expected_plan_digest and expected_plan_digest != current_digest:
        raise ValueError("plan digest is stale; refresh the inventory before presenting a decision")

    plan_issue_rows = plan.get("issues")
    canonical_row_count = len(plan_issue_rows) if isinstance(plan_issue_rows, list) else 0
    plan_issue_numbers, plan_issue_row_errors = _plan_issue_row_numbers(plan)
    issue_inventory_status = _issue_inventory_source_assessment(
        {
            "repo": plan.get("repo", DEFAULT_REPO),
            "inventory": plan.get("inventory"),
            LEGACY_ISSUE_INVENTORY_MARKER: plan.get(LEGACY_ISSUE_INVENTORY_MARKER),
        },
        canonical_row_count=canonical_row_count,
    )
    recorded_issue_inventory_status = plan.get("issue_inventory_status")
    recorded_status_inadmissible = isinstance(recorded_issue_inventory_status, Mapping) and (
        recorded_issue_inventory_status.get("admissible") is False
        or recorded_issue_inventory_status.get("status")
        in {ISSUE_SOURCE_STATUS_UNAVAILABLE, ISSUE_SOURCE_STATUS_ANOMALOUS}
    )
    if not issue_inventory_status["admissible"] or recorded_status_inadmissible:
        reason = str(
            (
                recorded_issue_inventory_status.get("reason")
                if recorded_status_inadmissible
                else issue_inventory_status.get("reason")
            )
            or "canonical open-issue source is not admissible"
        )
        raise ValueError(f"issue inventory is not admissible: {reason}")
    pending_decisions = plan.get("pending_decisions")
    if canonical_row_count == 0 and (
        pending_decisions is not None
        and (not isinstance(pending_decisions, list) or bool(pending_decisions))
    ):
        raise ValueError(
            "issue inventory is empty but the plan contains pending decisions; regenerate it"
        )
    if plan_issue_row_errors:
        raise ValueError(
            "issue inventory rows are not admissible: " + "; ".join(plan_issue_row_errors)
        )
    reference_errors = _plan_issue_reference_errors(plan, issue_numbers=plan_issue_numbers)
    if reference_errors:
        raise ValueError(
            "issue inventory references are not admissible: " + "; ".join(reference_errors)
        )

    quota_contract_failures = _quota_contract_failures_from_plan(
        plan,
        required=isinstance(pending_decisions, list) and bool(pending_decisions),
    )
    if quota_contract_failures:
        raise ValueError(
            "quota metadata is not admissible: " + "; ".join(sorted(set(quota_contract_failures)))
        )

    selected = select_next_pending_decision(
        plan,
        issue_scope=issue_scope,
        applied_mutations=applied_mutations,
    )
    if selected is None:
        return None
    index, total, row = selected
    issue_number = _pending_issue_number(row.get("issue"))
    if issue_number >= 2**31 - 1:
        raise ValueError("pending decision has no valid issue number")
    inventory_errors = _envelope_inventory_errors(plan, row)
    sources = row.get("evidence_sources")
    sources = (
        [dict(source) for source in sources if isinstance(source, Mapping)]
        if isinstance(sources, list)
        else []
    )
    options = row.get("documented_options")
    options = (
        [dict(option) for option in options if isinstance(option, Mapping)]
        if isinstance(options, list)
        else []
    )
    source_excerpt = next(
        (str(source.get("excerpt") or "") for source in sources if source.get("excerpt")),
        str(row.get("blocking_evidence") or "decision gate detected"),
    )
    if inventory_errors:
        status = "blocked_incomplete_inventory"
    elif len(options) < 2:
        status = "needs_clarification"
    else:
        status = "ready"
    return {
        "schema": ENVELOPE_SCHEMA,
        "plan_schema": PLAN_SCHEMA,
        "plan_digest": current_digest,
        "repo": str(plan.get("repo") or DEFAULT_REPO),
        "status": status,
        "queue": {
            "position": index + 1,
            "total": total,
            "remaining_after": max(0, total - index - 1),
            "ordering": "issue_number_ascending",
        },
        "issue": {
            "number": issue_number,
            "display": f"#{issue_number}",
            "title": str(row.get("title") or ""),
            "url": str(row.get("url") or ""),
            "state": str(row.get("state") or "").lower(),
            "labels": _label_names(row.get("labels")),
            "classification": str(row.get("classification") or "decision-required"),
        },
        "decision": {
            "required": True,
            "question": f"Which documented option should be applied to #{issue_number}?",
            "question_source": str(row.get("question_source") or "issue body/comments"),
            "blocking_evidence": str(row.get("blocking_evidence") or "decision gate detected"),
            "source_excerpt": source_excerpt,
            "evidence_sources": sources,
            "documented_options": options,
            "safe_mutations_applied": list(row.get("safe_mutations_applied") or []),
        },
        "answer_contract": {
            "format": f"#{issue_number}: <option-token>",
            "allowed_tokens": [
                str(option.get("token")) for option in options if option.get("token")
            ],
            "exact_match_required": True,
        },
        "verification": {
            "refresh_issue_before_apply": True,
            "compare_state_and_labels": True,
            "rerun_shared_classifier": True,
            "project5_writes": False,
        },
        "forbidden_inferences": [
            "research priority",
            "provenance or benchmark interpretation",
            "rights or compute authorization",
            "Project #5 ordering or field values",
        ],
        "inventory_errors": inventory_errors,
    }


def _envelope_pending_binding_errors(
    envelope: Mapping[str, Any],
    plan: Mapping[str, Any],
    expected_issue: Mapping[str, Any],
) -> list[str]:
    """Bind an envelope issue to the current admissible pending plan row."""
    errors: list[str] = []
    quota_contract_failures = _quota_contract_failures_from_plan(plan, required=True)
    if quota_contract_failures:
        errors.append(
            "current plan quota metadata is not admissible: "
            + "; ".join(sorted(set(quota_contract_failures)))
        )
    raw_plan_issue_rows = plan.get("issues")
    if not isinstance(raw_plan_issue_rows, list):
        errors.append("current plan issues must be a list")
        plan_issue_rows: list[object] = []
        canonical_row_count = 0
    else:
        plan_issue_rows = raw_plan_issue_rows
        canonical_row_count = len(plan_issue_rows)
    plan_issue_numbers, plan_issue_row_errors = _plan_issue_row_numbers(plan)
    errors.extend(f"current plan: {error}" for error in plan_issue_row_errors)

    issue_inventory_status = _issue_inventory_source_assessment(
        {
            "repo": plan.get("repo", DEFAULT_REPO),
            "inventory": plan.get("inventory"),
            "quota": plan.get("quota"),
            LEGACY_ISSUE_INVENTORY_MARKER: plan.get(LEGACY_ISSUE_INVENTORY_MARKER),
        },
        canonical_row_count=canonical_row_count,
    )
    if not issue_inventory_status["admissible"]:
        errors.append(
            "current plan issue inventory is not admissible: "
            + str(issue_inventory_status.get("reason") or "source contract failed")
        )
    recorded_status = plan.get("issue_inventory_status")
    if isinstance(recorded_status, Mapping) and (
        recorded_status.get("admissible") is False
        or recorded_status.get("status")
        in {ISSUE_SOURCE_STATUS_UNAVAILABLE, ISSUE_SOURCE_STATUS_ANOMALOUS}
    ):
        errors.append("current plan records an inadmissible issue inventory")

    pending = plan.get("pending_decisions")
    if not isinstance(pending, list):
        errors.append("current plan pending_decisions must be a list")
        return errors
    reference_errors = _plan_issue_reference_errors(plan, issue_numbers=plan_issue_numbers)
    errors.extend(f"current plan: {error}" for error in reference_errors)

    expected_number = _positive_issue_number(expected_issue.get("number"))
    matching_pending = [
        (index, item)
        for index, item in enumerate(pending)
        if isinstance(item, Mapping)
        and expected_number is not None
        and _pending_reference_number(item) == expected_number
    ]
    if not matching_pending:
        errors.append("envelope issue is not a current pending canonical issue")
        return errors
    if len(matching_pending) > 1:
        errors.append("envelope issue has multiple current pending canonical rows")
        return errors

    pending_index, pending_row = matching_pending[0]
    current_rows = {
        row.get("number"): row
        for row in plan_issue_rows
        if isinstance(row, Mapping) and _positive_issue_number(row.get("number")) is not None
    }
    current_row = current_rows.get(expected_number)
    if not isinstance(current_row, Mapping):
        errors.append("envelope issue is not a current pending canonical issue")
        return errors
    pending_binding_errors = _pending_issue_binding_errors(
        pending_row,
        current_row,
        index=pending_index,
    )
    errors.extend(f"current plan: {error}" for error in pending_binding_errors)
    expected_projection = {
        "number": expected_number,
        "display": f"#{expected_number}",
        "title": pending_row.get("title"),
        "url": pending_row.get("url"),
        "state": str(pending_row.get("state") or "").lower(),
        "labels": _label_names(pending_row.get("labels")),
        "classification": pending_row.get("classification"),
    }
    for field, current_value in expected_projection.items():
        envelope_value = expected_issue.get(field)
        if field == "state":
            envelope_value = str(envelope_value or "").lower()
        elif field == "labels":
            envelope_value = _label_names(envelope_value)
        if envelope_value != current_value:
            errors.append(f"envelope issue {field} does not match the current pending row")
    if envelope.get("repo") != plan.get("repo"):
        errors.append("envelope repo does not match the current plan")
    return errors


def validate_decision_envelope(
    envelope: Mapping[str, Any],
    *,
    plan: Mapping[str, Any] | None = None,
    live_issue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check plan binding and live issue state before applying an answer."""
    errors: list[str] = []
    expected_issue = envelope.get("issue") if isinstance(envelope.get("issue"), Mapping) else {}
    if envelope.get("schema") != ENVELOPE_SCHEMA:
        errors.append(f"expected {ENVELOPE_SCHEMA} envelope")
    envelope_digest = str(envelope.get("plan_digest") or "")
    if not envelope_digest:
        errors.append("envelope is missing plan_digest")
    if plan is not None:
        try:
            observed_digest = compute_plan_digest(plan)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if envelope_digest != observed_digest:
                errors.append("envelope plan digest does not match the current plan")
        errors.extend(_envelope_pending_binding_errors(envelope, plan, expected_issue))
    else:
        errors.append("plan is required to bind the envelope to a current pending canonical row")
    repository = plan.get("repo") if plan is not None else envelope.get("repo")
    if not isinstance(repository, str) or _repository_parts(repository) is None:
        errors.append("envelope repo is not a valid GitHub owner/repository")
    expected_number = _positive_issue_number(expected_issue.get("number"))
    if expected_number is None:
        errors.append("envelope issue number is not a positive integer")
        expected_number_for_url = 0
    else:
        expected_number_for_url = expected_number
    expected_url = expected_issue.get("url")
    if (
        isinstance(repository, str)
        and _repository_parts(repository) is not None
        and expected_number is not None
        and not _canonical_issue_url(
            expected_url,
            number=expected_number_for_url,
            repository=repository,
        )
    ):
        errors.append("envelope issue URL is not bound to its repo and issue number")
    if live_issue is not None:
        actual_number = _positive_issue_number(live_issue.get("number"))
        if actual_number is None:
            errors.append("live issue number is not a positive integer")
        if actual_number != expected_number:
            errors.append("live issue number does not match the envelope")
        expected_title = expected_issue.get("title")
        actual_title = live_issue.get("title")
        if expected_title != actual_title:
            errors.append("live issue title changed since the envelope was created")
        actual_url = (
            live_issue.get("html_url") if "html_url" in live_issue else live_issue.get("url")
        )
        if expected_url != actual_url:
            errors.append("live issue URL changed since the envelope was created")
        if (
            isinstance(repository, str)
            and _repository_parts(repository) is not None
            and actual_number is not None
            and not _canonical_issue_url(
                actual_url,
                number=actual_number,
                repository=repository,
            )
        ):
            errors.append("live issue URL is not bound to its repo and issue number")
        expected_state = str(expected_issue.get("state") or "").lower()
        actual_state = str(live_issue.get("state") or "").lower()
        if expected_state and actual_state != expected_state:
            errors.append("live issue state changed since the envelope was created")
        expected_labels = set(_label_names(expected_issue.get("labels")))
        actual_labels = set(_label_names(live_issue.get("labels")))
        if expected_labels != actual_labels:
            errors.append("live issue labels changed since the envelope was created")
    return {"ok": not errors, "errors": errors}


def parse_decision_answer(envelope: Mapping[str, Any], answer: str) -> dict[str, str]:
    """Parse the exact issue/option answer contract emitted by an envelope."""
    if envelope.get("status") != "ready":
        raise ValueError("envelope is not ready for an answer")
    match = re.fullmatch(r"\s*#(\d+)\s*:\s*([A-Za-z][A-Za-z0-9_-]*)\s*", answer or "")
    if not match:
        raise ValueError("answer must match '#<issue-number>: <option-token>'")
    issue_number = int(match.group(1))
    issue = envelope.get("issue") if isinstance(envelope.get("issue"), Mapping) else {}
    expected_number = _pending_issue_number(issue.get("number"))
    if issue_number != expected_number:
        raise ValueError("answer issue number does not match the envelope")
    token = match.group(2).upper()
    contract = envelope.get("answer_contract")
    allowed = {
        str(value).upper()
        for value in (contract.get("allowed_tokens") if isinstance(contract, Mapping) else [])
    }
    if token not in allowed:
        raise ValueError(f"answer option {token!r} is not documented by the envelope")
    return {"issue": f"#{issue_number}", "option": token}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _render_plan(plan: dict[str, Any], deadline: float | None) -> str:
    """Render a plan only after admitting it as complete under the shared deadline."""
    timeout_reason = "issue-audit wall-time budget exhausted while serializing the audit plan"
    if _deadline_expired(deadline) and plan["classification_status"]["status"] == "complete":
        _mark_plan_timed_out(plan, reason=timeout_reason)
    try:
        with _deadline_interrupt(deadline):
            rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    except _AuditDeadlineExceeded:
        _mark_plan_timed_out(plan, reason=timeout_reason)
        rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if _deadline_expired(deadline) and plan["classification_status"]["status"] == "complete":
        _mark_plan_timed_out(plan, reason=timeout_reason)
        rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    return rendered


def main(argv: Sequence[str] | None = None) -> int:
    """Expose bounded plan and apply operations for skills and CI checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="discover and emit an issue-audit plan")
    plan_parser.add_argument("--repo", default=DEFAULT_REPO)
    plan_parser.add_argument("--remote", default=DEFAULT_REMOTE)
    plan_parser.add_argument("--mode", choices=("autonomous", "interactive"), default="autonomous")
    plan_parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    plan_parser.add_argument(
        "--max-closed-pr-pages",
        type=int,
        default=DEFAULT_MAX_CLOSED_PR_PAGES,
        help="independent page budget for repository-wide closed-PR history",
    )
    plan_parser.add_argument(
        "--include-comments",
        action="store_true",
        help="include bounded REST comment threads in the issue evidence inventory",
    )
    plan_parser.add_argument("--max-comment-pages", type=int, default=DEFAULT_MAX_COMMENT_PAGES)
    plan_parser.add_argument("--max-mutations", type=int, default=DEFAULT_MAX_MUTATIONS)
    plan_parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=DEFAULT_MAX_AUDIT_WALL_SECONDS,
        help=(
            "aggregate audit wall-time budget across discovery and classification; "
            "zero emits a fail-closed empty plan"
        ),
    )
    plan_parser.add_argument(
        "--read-only-diagnostic",
        action="store_true",
        help=(
            "run in read-only diagnostic mode without requiring origin/main freshness; "
            "suppresses all mutations"
        ),
    )
    plan_parser.add_argument("--output", type=Path)
    apply_parser = subparsers.add_parser("apply", help="apply a previously emitted plan")
    apply_parser.add_argument("plan", type=Path)
    apply_parser.add_argument("--remote", default=DEFAULT_REMOTE)
    apply_parser.add_argument("--max-mutations", type=int, default=DEFAULT_MAX_MUTATIONS)
    apply_parser.add_argument(
        "--no-freshness-check",
        action="store_true",
        help="bypass origin/main freshness check during manual maintainer apply",
    )
    envelope_parser = subparsers.add_parser(
        "envelope", help="emit the next maintainer decision envelope from a plan"
    )
    envelope_parser.add_argument("plan", type=Path)
    envelope_parser.add_argument("--issue", dest="issue_scope", action="append", type=int)
    envelope_parser.add_argument("--expected-plan-digest")
    envelope_parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds < 0:
            parser.error("--max-wall-seconds must be finite and non-negative")
        if (
            args.mode == "autonomous"
            and not args.read_only_diagnostic
            and args.max_wall_seconds > 0
        ):
            should_fetch = args.max_wall_seconds >= 1.0
            fresh, head_sha, main_sha, _err = check_origin_main_freshness(
                args.remote, fetch=should_fetch
            )
            if not fresh:
                sys.stderr.write(
                    f"issue-audit: executing revision {head_sha} does not contain current "
                    f"{args.remote}/main ({main_sha}). Refresh with: git fetch {args.remote} main "
                    f"&& git merge {args.remote}/main (or pass --read-only-diagnostic for "
                    "read-only inspection)\n"
                )
                return 2
        elif args.read_only_diagnostic:
            head_sha = resolve_source_sha()
            sys.stderr.write(
                f"issue-audit: running in read-only diagnostic mode on revision {head_sha or 'unknown'}; "
                "all mutations are suppressed\n"
            )
        deadline = _deadline_from_seconds(args.max_wall_seconds)
        retry_command = _build_plan_retry_command(
            args.repo,
            remote=args.remote,
            mode=args.mode,
            max_pages=args.max_pages,
            max_closed_pr_pages=args.max_closed_pr_pages,
            include_comments=args.include_comments,
            max_comment_pages=args.max_comment_pages,
            max_mutations=args.max_mutations,
            max_wall_seconds=args.max_wall_seconds,
            output=args.output,
            read_only_diagnostic=args.read_only_diagnostic,
        )
        try:
            with _deadline_interrupt(deadline):
                inventory = discover_inventory(
                    args.repo,
                    remote=args.remote,
                    max_pages=args.max_pages,
                    max_closed_pr_pages=args.max_closed_pr_pages,
                    include_comments=args.include_comments,
                    max_comment_pages=args.max_comment_pages,
                    deadline=deadline,
                    retry_command=retry_command,
                )
        except _AuditDeadlineExceeded:
            inventory = _deadline_timeout_inventory(
                args.repo,
                args.remote,
                reason="issue-audit wall-time budget exhausted during inventory discovery",
            )
        plan = build_audit_plan(
            inventory,
            mode=args.mode,
            max_mutations=args.max_mutations,
            deadline=deadline,
            read_only_diagnostic=args.read_only_diagnostic,
        )
        rendered = _render_plan(plan, deadline)
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        quota = plan.get("quota")
        if isinstance(quota, Mapping) and (
            quota.get("quota_uncertain")
            or quota.get("quota_exhausted")
            or quota.get("status")
            in {"exhausted", "insufficient", "quota_blocked", "failed", "malformed", "unavailable"}
        ):
            retry_after = quota.get("retry_after_utc") or "unknown"
            reset_in = quota.get("reset_in_seconds")
            in_seconds_str = f" in ~{reset_in}s" if reset_in is not None else ""
            headline = (
                "GitHub core REST quota exhausted"
                if quota.get("quota_exhausted") or quota.get("status") == "exhausted"
                else "GitHub core REST request budget/preflight is uncertain"
            )
            reason = quota.get("reason") or headline
            cmd = quota.get("retry_command") or "retry later"
            sys.stderr.write(
                f"issue-audit: {headline} ({reason}). "
                f"Reset at {retry_after}{in_seconds_str}. Next action: {cmd}\n"
            )
        issue_inventory_status = plan.get("issue_inventory_status")
        if isinstance(issue_inventory_status, Mapping) and not issue_inventory_status.get(
            "admissible", True
        ):
            status = str(issue_inventory_status.get("status") or "unavailable")
            reason = str(issue_inventory_status.get("reason") or "source contract failed")
            next_action = str(
                issue_inventory_status.get("next_action") or "rerun the bounded issue-audit plan"
            )
            sys.stderr.write(
                "issue-audit: canonical open-issue inventory "
                f"{status}; no empty queue admitted ({reason}). "
                f"Next action: {next_action}\n"
            )
        return 2 if plan["truncation_or_errors"] else 0
    if args.command == "envelope":
        try:
            envelope = build_decision_envelope(
                _load_json(args.plan),
                issue_scope=args.issue_scope,
                expected_plan_digest=args.expected_plan_digest,
            )
        except ValueError as exc:
            print(json.dumps({"schema": ENVELOPE_SCHEMA, "error": str(exc)}, indent=2))
            return 2
        payload = envelope or {
            "schema": ENVELOPE_SCHEMA,
            "status": "empty",
            "reason": "no pending decisions in the requested scope",
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0 if envelope is None or envelope["status"] == "ready" else 2
    result = apply_mutations(
        _load_json(args.plan),
        max_mutations=args.max_mutations,
        check_freshness=not args.no_freshness_check,
        remote=args.remote,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
