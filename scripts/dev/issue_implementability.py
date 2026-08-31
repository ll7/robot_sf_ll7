#!/usr/bin/env python3
"""Check whether one GitHub issue is safe for autonomous implementation claim."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from scripts.dev import gh_issue_rest, issue_claim, issue_dependency_packet

SCHEMA = "issue_implementability.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"
READY_LABEL = "state:ready"
STATE_LABEL_PREFIX = "state:"
EXECUTION_HEADINGS = frozenset({"execution", "execution contract"})
EXECUTION_FIELDS = frozenset({"owning_repo", "mutation_repos", "route_required", "external_inputs"})
LOCAL_ROUTE = "local"
MULTI_REPOSITORY_ROUTE = "multi_repository"
ROUTE_PREFLIGHT_TTL_SECONDS = 30 * 60
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PARENT_LABELS = frozenset({"epic", "parent", "type:epic"})
HUMAN_DECISION_LABELS = frozenset(
    {
        "author-decision",
        "decision-required",
        "state:author-decision",
        "state:blocked-human-decision",
    }
)
COMPUTE_LABELS = frozenset(
    {
        "campaign",
        "needs-campaign",
        "resource:slurm",
        "routing:needs-compute",
        "slurm",
    }
)
EXTERNAL_LABELS = frozenset(
    {
        "resource:external-data",
        "state:blocked-external-input",
    }
)
WORKING_LABELS = frozenset({"state:running", "state:working"})
REVIEW_LABELS = frozenset({"needs-review", "state:review"})
BLOCKING_LABELS = frozenset(
    {
        "blocked",
        "deferred",
        "duplicate",
        "evidence:blocked",
        "invalid",
        "needs-triage",
        "state:blocked",
        "state:hold",
        "state:blocked-no-code-slice",
        "state:parked",
        "state:deferred",
        "wontfix",
    }
)

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "objective": (
        "ask",
        "current failure mode",
        "goal",
        "goal problem",
        "objective",
        "plain language summary",
        "problem",
        "purpose",
        "question",
        "research question",
        "summary",
    ),
    "scope": (
        "allowed paths",
        "forbidden work",
        "in scope",
        "non goals",
        "out of scope",
        "scope",
        "scope boundary",
    ),
    "inputs": (
        "affected files",
        "candidate paths",
        "canonical entry points",
        "current evidence",
        "exact source",
        "exact surface",
        "inputs",
        "prerequisites",
        "proposed implementation surface",
        "required changes",
        "required contract fields",
    ),
    "acceptance": (
        "acceptance criteria",
        "completion",
        "definition of done",
        "required outputs",
        "success metrics",
    ),
    "verification": (
        "proof",
        "testing",
        "validation",
        "validation gates",
        "validation proof",
        "validation testing",
        "verification",
        "verification gates",
    ),
}

HEADING_RE = re.compile(r"^(?P<marks>#{1,6})[ \t]+(?P<title>.+?)[ \t]*#*[ \t]*$")
LEADING_NUMBER_RE = re.compile(r"^\d+(?:[.)]|\s+-)\s*")
SPACE_RE = re.compile(r"\s+")
PARENT_TITLE_RE = re.compile(r"^\s*\[(?:epic|parent)\]", re.IGNORECASE)


def _normalize_heading(value: str) -> str:
    """Return a stable comparison form for one Markdown heading."""
    text = value.strip().lower()
    text = re.sub(r"[`*_]", "", text)
    text = LEADING_NUMBER_RE.sub("", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[/|:—–-]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def _heading_sections(body: str) -> list[tuple[str, str]]:
    """Return Markdown sections, including empty ones, outside fenced blocks."""
    lines = body.splitlines(keepends=True)
    spans: list[tuple[int, int, str]] = []
    offset = 0
    fence: str | None = None
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            offset += len(line)
            continue
        if fence is None:
            match = HEADING_RE.match(line.rstrip("\r\n"))
            if match is not None:
                spans.append((offset, offset + len(line), _normalize_heading(match.group("title"))))
        offset += len(line)

    sections: list[tuple[str, str]] = []
    for index, (_start, content_start, heading) in enumerate(spans):
        next_start = spans[index + 1][0] if index + 1 < len(spans) else len(body)
        sections.append((heading, body[content_start:next_start].strip()))
    return sections


def _heading_records(body: str) -> list[tuple[str, str]]:
    """Return non-empty Markdown sections while ignoring headings inside fences."""
    return [(heading, content) for heading, content in _heading_sections(body) if content]


def _heading_matches(heading: str, alias: str) -> bool:
    """Return whether a normalized heading exactly identifies a contract section."""
    return heading == alias


def inspect_contract(body: str) -> dict[str, Any]:
    """Inspect required implementation-contract sections without inferring intent."""
    records = _heading_records(body)
    headings = sorted({heading for heading, _ in records})
    fields: dict[str, dict[str, Any]] = {}
    missing_fields: list[str] = []
    for field, aliases in FIELD_ALIASES.items():
        matched = sorted(
            {
                heading
                for heading, _ in records
                if any(_heading_matches(heading, alias) for alias in aliases)
            }
        )
        fields[field] = {"present": bool(matched), "matched_headings": matched}
        if not matched:
            missing_fields.append(field)
    return {
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "headings": headings,
        "fields": fields,
        "missing_fields": missing_fields,
        "complete": not missing_fields,
    }


def preflight_body_text(body: str) -> dict[str, Any]:
    """Run the deterministic zero-write preflight for one issue body.

    Returns a stable JSON-ready verdict with ``ready``, the exact ``missing_fields``
    (objective, scope, inputs, acceptance, verification), and the body digest so a
    worker can repair the local draft before any GitHub create request. This guard
    creates no labels, comments, projects, claims, or issues; the live
    ``goal_issue_admission`` boundary remains responsible for state, claims,
    blockers, and freshness.
    """
    contract = inspect_contract(body)
    missing_fields = list(contract["missing_fields"])
    return {
        "schema": "issue_body_preflight.v1",
        "ready": not missing_fields,
        "missing_fields": missing_fields,
        "body_sha256": contract["body_sha256"],
    }


def preflight_body_file(path: str | Path) -> dict[str, Any]:
    """Read one issue-body file from disk and run the offline preflight."""
    body = Path(path).read_text(encoding="utf-8")
    return preflight_body_text(body)


def _parse_route_timestamp(value: object) -> dt.datetime | None:
    """Parse an RFC 3339 timestamp used by a route-preflight artifact."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(dt.UTC)


def _route_preflight_status(  # noqa: C901
    route_preflight: Mapping[str, Any] | None,
    *,
    expected_config_digest: str | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Return a credential-free freshness verdict for a multi-repository route plan."""
    if route_preflight is None:
        return {
            "status": "missing",
            "reason": "fresh multi-repository route preflight is required",
        }
    selected_route = route_preflight.get("selected_route")
    if not isinstance(selected_route, Mapping) or not selected_route:
        return {
            "status": "invalid",
            "reason": "route preflight has no selected route",
        }
    if (
        "is_worker_executable" in selected_route
        and selected_route["is_worker_executable"] is not True
    ):
        return {
            "status": "invalid",
            "reason": "route preflight selected route is not worker-executable",
        }
    route_status = route_preflight.get("status") or route_preflight.get("route_status")
    if route_status is not None and route_status not in {"available", "passed", "ok"}:
        return {
            "status": "stale",
            "reason": f"route preflight status is not usable: {route_status}",
        }
    config_digest = route_preflight.get("config_digest") or route_preflight.get(
        "route_config_digest"
    )
    if not isinstance(config_digest, str) or SHA256_RE.fullmatch(config_digest) is None:
        return {
            "status": "stale",
            "reason": "route preflight has no valid routing-config digest",
        }
    if expected_config_digest and config_digest != expected_config_digest:
        return {
            "status": "stale",
            "reason": "route preflight routing-config digest differs from the current digest",
        }
    recorded_at = route_preflight.get("checked_at") or route_preflight.get(
        "created_at", route_preflight.get("timestamp")
    )
    parsed_at = _parse_route_timestamp(recorded_at)
    if parsed_at is None:
        return {
            "status": "stale",
            "reason": "route preflight has no timezone-qualified timestamp",
        }
    ttl_value = route_preflight.get("ttl_seconds", ROUTE_PREFLIGHT_TTL_SECONDS)
    if isinstance(ttl_value, bool) or not isinstance(ttl_value, (int, float)) or ttl_value <= 0:
        return {
            "status": "invalid",
            "reason": "route preflight ttl_seconds must be positive",
        }
    if ttl_value > ROUTE_PREFLIGHT_TTL_SECONDS:
        return {
            "status": "invalid",
            "reason": f"route preflight ttl_seconds exceeds {ROUTE_PREFLIGHT_TTL_SECONDS}s",
        }
    current = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    expires_at = parsed_at + dt.timedelta(seconds=float(ttl_value))
    if current >= expires_at:
        return {
            "status": "stale",
            "reason": "route preflight has expired",
            "config_digest": config_digest,
            "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        }
    return {
        "status": "fresh",
        "reason": "route preflight is within its freshness window",
        "config_digest": config_digest,
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
    }


def inspect_execution_contract(  # noqa: C901, PLR0912
    body: str, *, repository: str = DEFAULT_REPO
) -> dict[str, Any]:
    """Inspect the optional repository-ownership execution declaration.

    Existing issue bodies remain single-repository local by default.  An explicit
    ``Execution`` section opts into strict ownership and route validation.
    """
    sections = _heading_sections(body)
    matching = [
        (heading, content) for heading, content in sections if heading in EXECUTION_HEADINGS
    ]
    if not matching:
        return {
            "present": False,
            "source": "implicit",
            "valid": True,
            "owning_repo": repository,
            "mutation_repos": [repository],
            "route_required": LOCAL_ROUTE,
            "external_inputs": [],
            "missing_fields": [],
            "errors": [],
        }

    errors: list[str] = []
    if len(matching) > 1:
        errors.append("execution contract appears in more than one section")
    content = matching[0][1]
    fenced = re.findall(r"(?ms)^\s*```(?:yaml|yml)\s*\n(.*?)^\s*```\s*$", content)
    if len(fenced) > 1:
        errors.append("execution contract contains more than one YAML block")
    yaml_text = fenced[0] if fenced else content
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        parsed = None
        errors.append(f"execution contract is not valid YAML: {exc}")
    if isinstance(parsed, Mapping) and isinstance(parsed.get("execution"), Mapping):
        values = parsed["execution"]
        unknown = sorted(str(key) for key in parsed if key != "execution")
    else:
        values = parsed
        unknown = []
    if unknown:
        errors.append(
            "execution contract has unsupported top-level field(s): " + ", ".join(unknown)
        )
    if not isinstance(values, Mapping):
        errors.append("execution contract must be a YAML object")
        values = {}
    unknown_fields = sorted(str(key) for key in values if key not in EXECUTION_FIELDS)
    if unknown_fields:
        errors.append("execution contract has unsupported field(s): " + ", ".join(unknown_fields))
    missing_fields = sorted(field for field in EXECUTION_FIELDS if field not in values)
    if missing_fields:
        errors.append("execution contract is missing: " + ", ".join(missing_fields))

    owning_repo = values.get("owning_repo")
    if not isinstance(owning_repo, str) or not owning_repo.strip():
        errors.append("execution.owning_repo must be a non-empty repository name")
        owning_repo = None
    else:
        owning_repo = owning_repo.strip()

    mutation_repos = values.get("mutation_repos")
    if not isinstance(mutation_repos, list) or any(
        not isinstance(repo, str) or not repo.strip() for repo in mutation_repos
    ):
        errors.append("execution.mutation_repos must be a list of repository names")
        mutation_repos = []
    else:
        mutation_repos = sorted({repo.strip() for repo in mutation_repos})

    route_required = values.get("route_required")
    if isinstance(route_required, str):
        route_required = route_required.strip().lower().replace("-", "_")
    if route_required not in {LOCAL_ROUTE, MULTI_REPOSITORY_ROUTE}:
        errors.append("execution.route_required must be 'local' or 'multi_repository'")
        route_required = None

    external_inputs = values.get("external_inputs")
    if not isinstance(external_inputs, list):
        errors.append("execution.external_inputs must be a list")
        external_inputs = []

    return {
        "present": True,
        "source": "explicit",
        "valid": not errors,
        "owning_repo": owning_repo,
        "mutation_repos": mutation_repos,
        "route_required": route_required,
        "external_inputs": external_inputs,
        "missing_fields": missing_fields,
        "errors": errors,
    }


def _normalize_labels(raw: Any) -> list[str]:
    """Normalize REST or offline label values to sorted names."""
    if not isinstance(raw, list):
        raise ValueError("labels must be a list")
    labels: list[str] = []
    for value in raw:
        if isinstance(value, str):
            name = value
        elif isinstance(value, dict) and isinstance(value.get("name"), str):
            name = value["name"]
        else:
            raise ValueError("each label must be a string or an object with a string name")
        name = name.strip()
        if name:
            labels.append(name)
    return sorted(set(labels))


def _normalize_assignees(raw: Any) -> list[str]:
    """Normalize REST or offline assignee values to sorted logins."""
    if not isinstance(raw, list):
        raise ValueError("assignees must be a list")
    assignees: list[str] = []
    for value in raw:
        if isinstance(value, str):
            login = value
        elif isinstance(value, dict) and isinstance(value.get("login"), str):
            login = value["login"]
        else:
            raise ValueError("each assignee must be a string or an object with a string login")
        login = login.strip()
        if login:
            assignees.append(login)
    return sorted(set(assignees))


def normalize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one issue payload."""
    number = issue.get("number")
    if type(number) is not int or number <= 0:
        raise ValueError("issue number must be a positive integer")
    title = issue.get("title")
    body = issue.get("body")
    state = issue.get("state")
    url = issue.get("url", "")
    if not isinstance(title, str):
        raise ValueError("issue title must be a string")
    if not isinstance(body, str):
        raise ValueError("issue body must be a string")
    if not isinstance(state, str):
        raise ValueError("issue state must be a string")
    if not isinstance(url, str):
        raise ValueError("issue url must be a string")
    return {
        "number": number,
        "title": title.strip(),
        "body": body,
        "state": state.strip().upper(),
        "url": url,
        "labels": _normalize_labels(issue.get("labels", [])),
        "assignees": _normalize_assignees(issue.get("assignees", [])),
    }


def _has_blocked_prefix(labels: set[str]) -> bool:
    """Return whether any label uses the explicit blocker namespace."""
    return any(label.startswith("blocked:") for label in labels)


def _state_labels(labels: set[str]) -> list[str]:
    """Return all lifecycle labels in deterministic order."""
    return sorted(label for label in labels if label.startswith(STATE_LABEL_PREFIX))


def _pending_decision_heading(contract: dict[str, Any], labels: set[str]) -> bool:
    """Detect an unresolved decision heading unless a ruling label is present."""
    if "ruled" in labels or "domain-approved" in labels:
        return False
    decision_headings = {
        "decision required",
        "maintainer decision required",
        "required maintainer decision",
    }
    return bool(decision_headings & set(contract["headings"]))


def _classify_issue(
    normalized: dict[str, Any],
    claim: dict[str, Any],
    contract: dict[str, Any],
    labels: set[str],
    execution_contract: dict[str, Any],
    *,
    repository: str,
    route_preflight: Mapping[str, Any] | None,
    now: dt.datetime | None,
) -> tuple[str, list[str], str]:
    """Classify one normalized issue using the documented precedence order."""
    execution_status = {
        "status": "not_required",
        "reason": "single-repository local execution",
    }
    execution_classification: tuple[str, list[str], str] | None = None
    if execution_contract.get("valid") is not True:
        execution_classification = (
            "needs_spec",
            ["invalid execution contract: " + "; ".join(execution_contract.get("errors", []))],
            "needs_spec",
        )
    else:
        owning_repo = execution_contract.get("owning_repo")
        mutation_repos = set(execution_contract.get("mutation_repos", []))
        route_required = execution_contract.get("route_required")
        external_inputs = execution_contract.get("external_inputs", [])
        if external_inputs:
            execution_classification = (
                "blocked",
                [
                    "execution contract declares external inputs; resolve them before local "
                    "implementation"
                ],
                "external_input_missing",
            )
        elif (
            owning_repo == repository
            and mutation_repos == {repository}
            and route_required == LOCAL_ROUTE
        ):
            pass
        elif route_required == MULTI_REPOSITORY_ROUTE:
            execution_status = _route_preflight_status(
                route_preflight,
                now=now,
            )
            if execution_status["status"] != "fresh":
                admission_reason = (
                    "wrong_owner_repo" if route_preflight is None else "stale_route_state"
                )
                execution_classification = (
                    "wrong_owner_repo",
                    [
                        f"execution owner/mutation scope is not local to {repository}",
                        execution_status["reason"],
                    ],
                    admission_reason,
                )
        else:
            execution_classification = (
                "wrong_owner_repo",
                [
                    f"execution contract requires mutation outside {repository} or declares "
                    "an incompatible local route"
                ],
                "wrong_owner_repo",
            )

    rules = [
        (
            normalized["state"] != "OPEN",
            "closed",
            f"issue state is {normalized['state'] or 'unknown'}",
            "closed",
        ),
        (claim.get("ok") is not True, "error", "claim state is unavailable", "claim_unavailable"),
        (
            claim.get("claimed") is True,
            "already_claimed",
            "an atomic issue claim already exists",
            "already_claimed",
        ),
        (
            bool(normalized["assignees"]),
            "assigned",
            "the issue already has an assignee",
            "assigned",
        ),
        (
            len(_state_labels(labels)) != 1,
            "state_conflict",
            (
                "exactly one state:* label is required; found "
                + (", ".join(_state_labels(labels)) or "none")
            ),
            "state_label_conflict",
        ),
        (
            bool(labels & PARENT_LABELS) or PARENT_TITLE_RE.match(normalized["title"]) is not None,
            "parent",
            "parent or epic issues are not implementation leaves",
            "parent_not_leaf",
        ),
        (
            bool(labels & HUMAN_DECISION_LABELS) or _pending_decision_heading(contract, labels),
            "human_decision",
            "a maintainer or author decision is required",
            "human_decision",
        ),
        (
            bool(labels & COMPUTE_LABELS),
            "needs_compute",
            "the issue is routed to compute or campaign execution",
            "needs_compute",
        ),
        (
            bool(labels & EXTERNAL_LABELS),
            "blocked",
            "the issue requires external input",
            "external_input_missing",
        ),
        (
            "state:running" in labels and claim.get("claimed") is not True,
            "stale_running",
            "state:running requires a current atomic claim or covering PR",
            "stale_running_state",
        ),
        (
            bool(labels & WORKING_LABELS),
            "working",
            "the issue is already in an active work state",
            "active_work",
        ),
        (
            bool(labels & REVIEW_LABELS),
            "review",
            "the issue is already in review",
            "covering_pr_open",
        ),
        (
            bool(labels & BLOCKING_LABELS) or _has_blocked_prefix(labels),
            "blocked",
            "a blocking workflow label is present",
            "blocked",
        ),
        (
            execution_classification is not None,
            *(execution_classification or ("error", ["execution gate failed"], "error")),
        ),
        (
            READY_LABEL not in labels,
            "needs_ready_label",
            f"required label {READY_LABEL!r} is absent",
            "needs_ready_label",
        ),
        (
            not contract["complete"],
            "needs_spec",
            "missing implementation-contract fields: " + ", ".join(contract["missing_fields"]),
            "needs_spec",
        ),
    ]
    for condition, classification, reason, admission_reason in rules:
        if condition:
            return classification, [reason], admission_reason
    if execution_status["status"] == "fresh":
        return (
            "ready",
            ["fresh multi-repository route preflight permits routed claim admission"],
            "claimable",
        )
    return "ready", ["issue state and execution contract permit claim admission"], "claimable"


def evaluate_issue(
    issue: dict[str, Any],
    claim: dict[str, Any],
    *,
    dependency_evaluation: Mapping[str, Any] | None = None,
    repository: str = DEFAULT_REPO,
    route_preflight: Mapping[str, Any] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Return a deterministic, fail-closed issue implementability report."""
    normalized = normalize_issue(issue)
    contract = inspect_contract(normalized["body"])
    execution_contract = inspect_execution_contract(normalized["body"], repository=repository)
    if execution_contract.get("route_required") == MULTI_REPOSITORY_ROUTE:
        execution_contract["route_preflight"] = _route_preflight_status(
            route_preflight,
            now=now,
        )
    else:
        execution_contract["route_preflight"] = {
            "status": "not_required",
            "reason": "single-repository local execution",
        }
    labels = set(normalized["labels"])
    classification, reasons, admission_reason = _classify_issue(
        normalized,
        claim,
        contract,
        labels,
        execution_contract,
        repository=repository,
        route_preflight=route_preflight,
        now=now,
    )

    report = {
        "schema": SCHEMA,
        "issue": {
            "number": normalized["number"],
            "title": normalized["title"],
            "state": normalized["state"],
            "url": normalized["url"],
            "labels": normalized["labels"],
            "assignees": normalized["assignees"],
        },
        "claim": {
            "ok": claim.get("ok"),
            "claimed": claim.get("claimed"),
            "claim_ref": claim.get("claim_ref"),
            "sha": claim.get("sha"),
        },
        "contract": contract,
        "execution_contract": execution_contract,
        "classification": classification,
        "admission_reason": admission_reason,
        "reasons": reasons,
        "ready": classification == "ready",
        "write_allowed": classification == "ready",
    }
    if dependency_evaluation is not None:
        return issue_dependency_packet.apply_dependency_gate(report, dependency_evaluation)
    return report


def fetch_live_issue(number: int, *, repo: str) -> dict[str, Any]:
    """Fetch one issue through the canonical REST-backed normalized reader."""
    payload = gh_issue_rest.fetch_issue(number, repo=repo)
    if not isinstance(payload, dict):
        raise ValueError("issue reader returned a non-object payload")
    if payload.get("status") != "ok":
        raise RuntimeError(str(payload.get("error", "issue read failed")))
    return payload


def _resolve_issue_dependency_packet(
    issue: Mapping[str, Any], *, repo: str, repo_root: Path | str | None
) -> Mapping[str, Any] | None:
    """Resolve an explicitly embedded or referenced packet for one live issue."""
    body = issue.get("body", "")
    packet, extraction_errors = issue_dependency_packet.extract_packet_from_issue_body(
        body if isinstance(body, str) else "", repo_root=repo_root
    )
    if extraction_errors:
        return issue_dependency_packet.invalid_packet_evaluation(extraction_errors)
    if packet is None:
        return None
    return issue_dependency_packet.resolve_packet(
        packet,
        repo_root=repo_root,
        expected_repository=repo,
        expected_issue=issue.get("number") if isinstance(issue.get("number"), int) else None,
    )


def live_issue_report(
    number: int,
    *,
    repo: str,
    remote: str,
    repo_root: Path | str | None = None,
    route_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read one live issue and gate claimability on any explicit dependency packet."""
    issue = fetch_live_issue(number, repo=repo)
    claim = issue_claim.status_issue(number, remote=remote)
    dependency_evaluation = _resolve_issue_dependency_packet(
        issue, repo=repo, repo_root=repo_root or Path.cwd()
    )
    return evaluate_issue(
        issue,
        claim,
        dependency_evaluation=dependency_evaluation,
        repository=repo,
        route_preflight=route_preflight,
    )


def _parse_claimed(value: str) -> dict[str, Any]:
    """Return a normalized offline claim-state fixture."""
    if value == "unknown":
        return {"ok": False, "claimed": None, "claim_ref": None, "sha": None}
    claimed = value == "true"
    return {
        "ok": True,
        "claimed": claimed,
        "claim_ref": "offline/claim" if claimed else None,
        "sha": "offline" if claimed else None,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "issue",
        type=int,
        nargs="?",
        default=0,
        help="Positive GitHub issue number (not required with --preflight-body).",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository as OWNER/REPO.")
    parser.add_argument(
        "--remote", default=DEFAULT_REMOTE, help="Git remote used for claim status."
    )
    parser.add_argument(
        "--preflight-body",
        type=Path,
        help=(
            "Offline zero-write mode: validate one issue-body file against the "
            "canonical contract before any GitHub create request and exit."
        ),
    )
    parser.add_argument("--body-file", help="Offline mode: read the issue body from this file.")
    parser.add_argument("--title", default="offline issue", help="Offline issue title.")
    parser.add_argument("--state", default="OPEN", help="Offline issue state.")
    parser.add_argument("--label", action="append", default=[], help="Offline label; repeatable.")
    parser.add_argument(
        "--assignee", action="append", default=[], help="Offline assignee; repeatable."
    )
    parser.add_argument(
        "--claimed",
        choices=("false", "true", "unknown"),
        default="false",
        help="Offline atomic-claim state.",
    )
    parser.add_argument(
        "--route-preflight-json",
        type=Path,
        help="Optional fresh route-plan JSON for an explicitly multi-repository issue.",
    )
    return parser


def _load_route_preflight(path: Path | None) -> Mapping[str, Any] | None:
    """Load one route-plan object without exposing provider or credential data."""
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("route preflight JSON must be an object")
    return payload


def _error_report(number: int, message: str) -> dict[str, Any]:
    """Return a stable fail-closed error payload."""
    return {
        "schema": SCHEMA,
        "issue": {"number": number},
        "classification": "error",
        "admission_reason": "error",
        "reasons": [message],
        "ready": False,
        "write_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    if args.preflight_body is not None:
        try:
            report = preflight_body_file(args.preflight_body)
        except OSError as exc:
            report = {
                "schema": "issue_body_preflight.v1",
                "ready": False,
                "missing_fields": [],
                "body_sha256": "",
                "error": str(exc),
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        if report.get("ready") is True:
            return 0
        return 2
    try:
        if args.issue <= 0:
            raise ValueError("issue number must be positive")
        if args.body_file:
            body = Path(args.body_file).read_text(encoding="utf-8")
            issue = {
                "number": args.issue,
                "title": args.title,
                "body": body,
                "state": args.state,
                "url": "",
                "labels": args.label,
                "assignees": args.assignee,
            }
            report = evaluate_issue(
                issue,
                _parse_claimed(args.claimed),
                repository=args.repo,
                route_preflight=_load_route_preflight(args.route_preflight_json),
            )
        else:
            report = live_issue_report(
                args.issue,
                repo=args.repo,
                remote=args.remote,
                route_preflight=_load_route_preflight(args.route_preflight_json),
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        report = _error_report(args.issue, str(exc))

    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("ready") is True:
        return 0
    if report.get("classification") == "error":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
