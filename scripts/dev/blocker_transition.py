#!/usr/bin/env python3
"""Plan fail-closed transitions for stale issue and PR blocker states.

The planner is intentionally a pure projection over already-observed issue,
dependency, ruling, child, and PR records.  It does not resolve a dependency,
make a ruling, or decide whether scientific work is authorized.  Report and
plan modes are read-only.  Apply mode replaces only the target issue's labels
after an exact live state/label compare-and-swap and verifies the writeback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

from scripts.dev._gh_rest import parse_json, run_gh_api

SCHEMA = "blocker_transition_plan.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"

BLOCKER_CLASSES = (
    "human_decision",
    "ruled_pending_child",
    "dependency_predicate",
    "implementation_defect",
    "stale_base_or_metadata",
    "domain_approval",
    "external_input",
    "compute_required",
    "parked_or_deferred",
    "transient_ci",
    "invalid_or_conflicting_state",
    "none",
)
BLOCKER_PRIORITY = {
    "invalid_or_conflicting_state": 0,
    "ruled_pending_child": 1,
    "dependency_predicate": 2,
    "stale_base_or_metadata": 3,
    "human_decision": 4,
    "domain_approval": 5,
    "external_input": 6,
    "compute_required": 7,
    "parked_or_deferred": 8,
    "transient_ci": 9,
    "implementation_defect": 10,
    "none": 11,
}

STATE_LABELS = frozenset(
    {
        "state:blocked",
        "state:blocked-no-code-slice",
        "state:hold",
        "state:parked",
        "state:ready",
        "state:review",
        "state:running",
        "state:working",
    }
)
BLOCKING_LABELS = frozenset(
    {
        "blocked",
        "decision-required",
        "dependency:has-blockers",
        "domain-review-required",
        "evidence:blocked",
        "needs-triage",
        "state:blocked",
        "state:blocked-external-input",
        "state:blocked-no-code-slice",
        "state:hold",
    }
)
COMPUTE_LABELS = frozenset(
    {"campaign", "needs-campaign", "resource:compute", "resource:slurm", "slurm"}
)
EXTERNAL_LABELS = frozenset({"resource:external-data", "state:blocked-external-input"})
PARKED_LABELS = frozenset({"deferred", "state:parked", "state:deferred"})
CI_PENDING_VALUES = frozenset(
    {"expected", "in_progress", "pending", "queued", "requested", "waiting"}
)

ApiRunner = Callable[[str, object | None, str | None], CompletedProcess[str]]
SourceRevalidator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class TransitionError(RuntimeError):
    """Raised when a transition plan or exact apply cannot be trusted."""


def _text(value: Any) -> str:
    """Normalize nullable values without turning null into a claim."""
    return "" if value is None else str(value).strip()


def _labels(value: Any) -> list[str]:
    """Normalize labels from either issue snapshots or REST objects."""
    if not isinstance(value, list):
        raise TransitionError("labels must be a list")
    names: set[str] = set()
    for item in value:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, Mapping) and isinstance(item.get("name"), str):
            name = str(item["name"]).strip()
        else:
            raise TransitionError("labels must contain strings or named objects")
        if name:
            names.add(name)
    return sorted(names)


def _issue(issue: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the issue identity used by a plan."""
    number = issue.get("number")
    if type(number) is not int or number < 1:
        raise TransitionError("issue number must be a positive integer")
    state = _text(issue.get("state")).upper()
    if not state:
        raise TransitionError("issue state must be present")
    body_observed = isinstance(issue.get("body"), str)
    body = _text(issue.get("body"))
    return {
        "number": number,
        "title": _text(issue.get("title")),
        "state": state,
        "url": _text(issue.get("url") or issue.get("html_url")),
        "labels": _labels(issue.get("labels", [])),
        "body_observed": body_observed,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "updated_at": _text(issue.get("updated_at") or issue.get("updatedAt")),
    }


def _records(value: Sequence[Mapping[str, Any]] | None, *, name: str) -> list[dict[str, Any]]:
    """Normalize optional observed records without resolving their meaning."""
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TransitionError(f"{name} must be a list of objects")
    records: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TransitionError(f"{name} must contain objects")
        records.append(dict(item))
    return records


def _dependency_status(record: Mapping[str, Any]) -> str:
    """Classify one observed dependency without treating a link as proof."""
    predicate = record.get("predicate")
    if isinstance(predicate, Mapping) and isinstance(predicate.get("satisfied"), bool):
        return "satisfied" if predicate["satisfied"] else "unsatisfied"
    if isinstance(record.get("predicate_satisfied"), bool):
        return "satisfied" if record["predicate_satisfied"] else "unsatisfied"
    if record.get("satisfied") is True:
        return "satisfied"
    if record.get("satisfied") is False:
        return "unsatisfied"
    if record.get("available") is False:
        return "unknown"
    if record.get("merged") is True or record.get("closed") is True:
        return "satisfied"
    status = _text(record.get("status") or record.get("state")).casefold()
    if status in {"closed", "merged", "resolved", "satisfied", "complete", "completed"}:
        return "satisfied"
    if status in {"open", "blocked", "pending", "failed", "unsatisfied", "in_progress"}:
        return "unsatisfied"
    return "unknown"


def _ruling_details(ruling: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a ruling observation without accepting ambiguous tokens."""
    if not isinstance(ruling, Mapping):
        return {
            "provided": False,
            "valid": False,
            "token": "",
            "carrier": "",
            "digest": "",
            "reason_codes": [],
        }
    token_values = []
    for key in ("token", "ruling_token"):
        value = _text(ruling.get(key))
        if value:
            token_values.append(value)
    raw_tokens = ruling.get("tokens")
    if isinstance(raw_tokens, list):
        token_values.extend(value for value in (_text(item) for item in raw_tokens) if value)
    unique_tokens = sorted(set(token_values))
    valid = ruling.get("valid") is True or _text(ruling.get("status")).casefold() in {
        "approved",
        "recorded",
        "valid",
    }
    token = unique_tokens[0] if len(unique_tokens) == 1 else ""
    carrier = _text(ruling.get("carrier") or ruling.get("source_url") or ruling.get("source"))
    reason_codes: list[str] = []
    if len(unique_tokens) > 1:
        reason_codes.append("conflicting_ruling_tokens")
    if valid and not token:
        reason_codes.append("ruling_token_missing_or_ambiguous")
    if valid and not carrier:
        reason_codes.append("ruling_carrier_missing")
    if not valid and (token_values or carrier):
        reason_codes.append("ruling_observation_not_valid")
    return {
        "provided": True,
        "valid": valid and len(unique_tokens) == 1 and bool(carrier),
        "token": token,
        "carrier": carrier,
        "digest": _text(
            ruling.get("digest") or ruling.get("ruling_digest") or ruling.get("source_digest")
        ),
        "reason_codes": sorted(set(reason_codes)),
    }


def _child_contract_ready(child: Mapping[str, Any]) -> bool:
    """Require an explicit executable-child signal before parent cleanup."""
    if child.get("contract_ready") is True or child.get("implementable") is True:
        return True
    contract = child.get("contract")
    if isinstance(contract, Mapping):
        return contract.get("complete") is True or contract.get("ready") is True
    return False


def _child_status(child: Mapping[str, Any]) -> str:
    """Return a conservative child terminal state."""
    if child.get("merged") is True or child.get("closed") is True:
        return "resolved"
    state = _text(child.get("state")).casefold()
    if state in {"closed", "merged", "complete", "completed"}:
        return "resolved"
    return "open" if _child_contract_ready(child) else "missing_contract"


def _conflict_reason_codes(labels: set[str]) -> list[str]:
    """Return explicit contradictory active-state combinations."""
    conflicts: list[tuple[set[str], str]] = [
        ({"state:ready", "state:blocked"}, "ready_and_blocked"),
        ({"state:ready", "state:working"}, "ready_and_working"),
        ({"state:running", "state:working"}, "running_and_working"),
        ({"state:blocked", "state:working"}, "blocked_and_working"),
    ]
    return sorted(code for required, code in conflicts if required <= labels)


def _condition(
    blocker_class: str,
    reason_codes: list[str],
    *,
    dependencies: list[dict[str, Any]],
    children: list[dict[str, Any]],
    affected_prs: list[dict[str, Any]],
) -> tuple[str, str, str]:
    """Return the exact owner, next action, and permitted state for a class."""
    unresolved_children = [child for child in children if _child_status(child) != "resolved"]
    owners = {
        "human_decision": "maintainer or domain owner",
        "ruled_pending_child": "bounded child implementation owner",
        "dependency_predicate": "dependency owner or maintainer",
        "implementation_defect": "owning issue or PR",
        "stale_base_or_metadata": "PR refresh/reconciliation owner",
        "domain_approval": "domain-aware approver",
        "external_input": "external-input or rights custodian",
        "compute_required": "authorized compute operator",
        "parked_or_deferred": "maintainer; preserve revival condition",
        "transient_ci": "CI or PR owner",
        "invalid_or_conflicting_state": "maintainer or queue owner",
        "none": "none",
    }
    actions = {
        "human_decision": "obtain the named maintainer/domain decision; do not dispatch",
        "ruled_pending_child": (
            "implement or identify the bounded child contract before changing the parent state"
        ),
        "dependency_predicate": "re-evaluate the typed dependency predicate and preserve its exact revision",
        "implementation_defect": "route the defect to the owning issue or PR and re-run admission",
        "stale_base_or_metadata": "refresh the affected PR base or metadata, then re-run exact-head review",
        "domain_approval": "obtain independent domain-aware approval before any claim or promotion",
        "external_input": "stage and verify the required external input or rights evidence",
        "compute_required": "obtain the separately authorized compute execution and durable receipt",
        "parked_or_deferred": "preserve the revival condition and do not activate the parked lane",
        "transient_ci": "wait for terminal CI or inspect the exact failing run before routing",
        "invalid_or_conflicting_state": "stop and repair the contradictory state through an exact-item review",
        "none": "no blocker transition is proposed; retain current state",
    }
    if blocker_class == "ruled_pending_child" and unresolved_children:
        actions[blocker_class] = "execute the named bounded child, then re-evaluate the parent"
    if blocker_class == "dependency_predicate" and dependencies:
        actions[blocker_class] = (
            "re-read every dependency predicate; a link alone is not satisfaction"
        )
    if blocker_class == "stale_base_or_metadata" and affected_prs:
        actions[blocker_class] = (
            "refresh the affected PR from current origin/main and rebuild its exact-head proof"
        )
    permitted = {
        "human_decision": "blocked",
        "ruled_pending_child": "blocked",
        "dependency_predicate": "blocked",
        "implementation_defect": "blocked",
        "stale_base_or_metadata": "review",
        "domain_approval": "blocked",
        "external_input": "blocked",
        "compute_required": "blocked",
        "parked_or_deferred": "parked",
        "transient_ci": "review",
        "invalid_or_conflicting_state": "blocked",
        "none": "unchanged",
    }[blocker_class]
    return owners[blocker_class], actions[blocker_class], permitted


def _label_delta(
    issue: Mapping[str, Any],
    *,
    blocker_class: str,
    ruling_valid: bool,
    child_present: bool,
    all_dependencies_satisfied: bool,
    implementability_ready: bool,
) -> dict[str, list[str]]:
    """Return conservative label additions/removals for an exact-item plan."""
    labels = set(issue["labels"])
    additions: set[str] = set()
    removals: set[str] = set()
    if ruling_valid:
        additions.update({"ruled", "parent", "dependency:has-blockers", "state:blocked"})
        removals.update(
            {"decision-required", "blocked:needs-maintainer", "state:blocked-no-code-slice"}
        )
        if not child_present:
            additions.add("needs-triage")
    if (
        blocker_class == "dependency_predicate"
        and all_dependencies_satisfied
        and implementability_ready
    ):
        additions.add("state:ready")
        removals.update({"state:blocked", "dependency:has-blockers", "needs-triage"})
    if blocker_class == "invalid_or_conflicting_state":
        additions.add("needs-triage")
    if blocker_class in {"human_decision", "domain_approval", "external_input", "compute_required"}:
        removals.add("state:ready")
    return {
        "add": sorted(additions - labels),
        "remove": sorted(removals & labels),
    }


def _dependency_conflict_codes(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Find contradictory status fields without resolving the dependency."""
    conflicts: list[str] = []
    satisfied_values = {"closed", "complete", "completed", "merged", "resolved", "satisfied"}
    unsatisfied_values = {"blocked", "failed", "in_progress", "open", "pending", "unsatisfied"}
    for record in records:
        status = _text(record.get("status") or record.get("state")).casefold()
        explicit = record.get("satisfied")
        if explicit is True and status in unsatisfied_values:
            conflicts.append("dependency_satisfied_but_status_unsatisfied")
        if explicit is False and status in satisfied_values:
            conflicts.append("dependency_unsatisfied_but_status_satisfied")
    return sorted(set(conflicts))


def _child_link_conflict_codes(
    issue_number: int, children: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Detect explicit parent/child link drift without guessing an intended parent."""
    return sorted(
        {
            "parent_child_link_drift"
            for child in children
            if child.get("parent_number") is not None and child.get("parent_number") != issue_number
        }
    )


def _source_keys(
    *,
    ruling_details: Mapping[str, Any],
    dependencies: Sequence[Mapping[str, Any]],
    children: Sequence[Mapping[str, Any]],
    affected_prs: Sequence[Mapping[str, Any]],
    implementability: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Create the source identity used by plan freshness and revalidation."""
    return {
        "ruling": {
            "valid": ruling_details.get("valid") is True,
            "token": _text(ruling_details.get("token")),
            "carrier": _text(ruling_details.get("carrier")),
            "digest": _text(ruling_details.get("digest")),
        },
        "dependencies": sorted(
            [
                {
                    "number": item.get("number"),
                    "predicate": item.get("predicate")
                    if isinstance(item.get("predicate"), (str, int, float, bool))
                    else "",
                    "state": _text(item.get("state") or item.get("status")).upper(),
                    "available": item.get("available")
                    if isinstance(item.get("available"), bool)
                    else None,
                    "merged": item.get("merged") if isinstance(item.get("merged"), bool) else None,
                    "closed": item.get("closed") if isinstance(item.get("closed"), bool) else None,
                    "revision": _text(
                        item.get("revision") or item.get("head_sha") or item.get("sha")
                    ),
                    "artifact_digest": _text(item.get("artifact_digest") or item.get("digest")),
                    "packet_id": _text(item.get("packet_id") or item.get("id")),
                    "packet_digest": _text(item.get("packet_digest") or item.get("source_digest")),
                    "satisfied": item.get("satisfied")
                    if isinstance(item.get("satisfied"), bool)
                    else item.get("predicate_satisfied")
                    if isinstance(item.get("predicate_satisfied"), bool)
                    else None,
                }
                for item in dependencies
            ],
            key=lambda item: str(item.get("number")),
        ),
        "children": sorted(
            [
                {
                    "number": item.get("number"),
                    "state": _text(item.get("state")).upper(),
                    "merged": item.get("merged") if isinstance(item.get("merged"), bool) else None,
                    "closed": item.get("closed") if isinstance(item.get("closed"), bool) else None,
                    "revision": _text(
                        item.get("revision") or item.get("head_sha") or item.get("sha")
                    ),
                    "contract_ready": _child_contract_ready(item),
                }
                for item in children
            ],
            key=lambda item: str(item.get("number")),
        ),
        "affected_prs": sorted(
            [
                {
                    "number": item.get("number"),
                    "head_sha": _text(item.get("head_sha") or item.get("revision")),
                    "base_sha": _text(item.get("base_sha")),
                    "stale_base": item.get("stale_base")
                    if isinstance(item.get("stale_base"), bool)
                    else None,
                    "metadata_stale": item.get("metadata_stale")
                    if isinstance(item.get("metadata_stale"), bool)
                    else None,
                    "owner": _text(
                        item.get("owner")
                        or item.get("implementation_owner")
                        or item.get("assignee")
                    ),
                }
                for item in affected_prs
            ],
            key=lambda item: str(item.get("number")),
        ),
        "implementability": {
            "ready": implementability.get("ready") is True
            if isinstance(implementability, Mapping)
            else False,
            "body_sha256": _text((implementability or {}).get("body_sha256"))
            if isinstance(implementability, Mapping)
            else "",
            "contract_digest": _text((implementability or {}).get("contract_digest"))
            if isinstance(implementability, Mapping)
            else "",
            "ci_status": _text((implementability or {}).get("ci_status"))
            if isinstance(implementability, Mapping)
            else "",
        },
    }


def _digest_payload(payload: Mapping[str, Any]) -> str:
    """Hash the stable plan payload without observation timestamps or its digest."""
    stable = dict(payload)
    stable.pop("plan_digest", None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _meaningful(value: Any) -> bool:
    """Return whether a source projection carries an observation to revalidate."""
    if isinstance(value, Mapping):
        return any(_meaningful(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value)
    return value not in (None, "", False)


def plan_transition(  # noqa: C901, PLR0912, PLR0915
    issue: Mapping[str, Any],
    *,
    dependencies: Sequence[Mapping[str, Any]] | None = None,
    ruling: Mapping[str, Any] | None = None,
    children: Sequence[Mapping[str, Any]] | None = None,
    affected_prs: Sequence[Mapping[str, Any]] | None = None,
    implementability: Mapping[str, Any] | None = None,
    mode: str = "report-only",
    authorized: bool = False,
) -> dict[str, Any]:
    """Build one deterministic blocker transition plan."""
    if mode not in {"report-only", "plan", "apply"}:
        raise TransitionError(f"unsupported mode {mode!r}")
    normalized = _issue(issue)
    dependency_rows = _records(dependencies, name="dependencies")
    child_rows = _records(children, name="children")
    pr_rows = _records(affected_prs, name="affected_prs")
    ruling_details = _ruling_details(ruling)
    ruling_valid = bool(ruling_details["valid"])
    dependency_states = [_dependency_status(item) for item in dependency_rows]
    unresolved_dependencies = [
        item
        for item, status in zip(dependency_rows, dependency_states, strict=True)
        if status != "satisfied"
    ]
    all_dependencies_satisfied = bool(dependency_rows) and not unresolved_dependencies
    child_present = bool(child_rows)
    unresolved_children = [child for child in child_rows if _child_status(child) != "resolved"]
    child_contract_ready = any(_child_contract_ready(child) for child in child_rows)
    implementability_ready = (
        isinstance(implementability, Mapping) and implementability.get("ready") is True
    )
    labels = set(normalized["labels"])
    reason_candidates: list[tuple[str, list[str]]] = []
    conflicts = _conflict_reason_codes(labels)
    conflicts.extend(ruling_details["reason_codes"])
    conflicts.extend(_dependency_conflict_codes(dependency_rows))
    conflicts.extend(_child_link_conflict_codes(normalized["number"], child_rows))
    if ruling_valid and child_contract_ready and "state:blocked-no-code-slice" in labels:
        conflicts.append("ruled_child_exists_with_no_code_slice")
    if conflicts:
        reason_candidates.append(("invalid_or_conflicting_state", sorted(set(conflicts))))
    if ruling_valid and (not child_present or unresolved_children):
        reasons = ["valid_ruling_requires_bounded_child"]
        if not child_present:
            reasons.append("missing_bounded_child")
        else:
            if any(_child_status(child) == "missing_contract" for child in child_rows):
                reasons.append("bounded_child_contract_missing")
            if any(_child_status(child) == "open" for child in child_rows):
                reasons.append("bounded_child_not_resolved")
        reason_candidates.append(("ruled_pending_child", reasons))
    if unresolved_dependencies or "dependency:has-blockers" in labels:
        reasons = (
            ["dependency_observation_missing"]
            if not dependency_rows
            else [f"dependency_{status}" for status in dependency_states if status != "satisfied"]
        )
        if not reasons:
            reasons = ["dependency_recheck_required"]
        reason_candidates.append(("dependency_predicate", sorted(set(reasons))))
    if any(
        _text(row.get("stale_base") or row.get("metadata_stale")).casefold() == "true"
        for row in pr_rows
    ):
        reason_candidates.append(("stale_base_or_metadata", ["affected_pr_stale_base_or_metadata"]))
    if "decision-required" in labels or "state:blocked-human-decision" in labels:
        reason_candidates.append(("human_decision", ["human_decision_label_present"]))
    if "domain-review-required" in labels and "domain-approved" not in labels:
        reason_candidates.append(("domain_approval", ["domain_approval_required"]))
    if labels & EXTERNAL_LABELS:
        reason_candidates.append(("external_input", ["external_input_label_present"]))
    if labels & COMPUTE_LABELS:
        reason_candidates.append(("compute_required", ["compute_or_campaign_label_present"]))
    if labels & PARKED_LABELS:
        reason_candidates.append(("parked_or_deferred", ["parked_or_deferred_label_present"]))
    ci_status = _text((implementability or {}).get("ci_status")).casefold()
    if ci_status in CI_PENDING_VALUES:
        reason_candidates.append(("transient_ci", [f"ci_status_{ci_status}"]))
    if not reason_candidates and any(label in BLOCKING_LABELS for label in labels):
        reason_candidates.append(("implementation_defect", ["unowned_blocker_requires_triage"]))
    if reason_candidates:
        blocker_class, reason_codes = min(
            reason_candidates,
            key=lambda item: (BLOCKER_PRIORITY[item[0]], item[0]),
        )
    else:
        blocker_class, reason_codes = "none", []
    owner, next_action, next_state = _condition(
        blocker_class,
        reason_codes,
        dependencies=dependency_rows,
        children=child_rows,
        affected_prs=pr_rows,
    )
    delta = _label_delta(
        normalized,
        blocker_class=blocker_class,
        ruling_valid=ruling_valid,
        child_present=child_present,
        all_dependencies_satisfied=all_dependencies_satisfied,
        implementability_ready=implementability_ready,
    )
    expected = {
        "number": normalized["number"],
        "state": normalized["state"],
        "labels": normalized["labels"],
        "body_observed": normalized["body_observed"],
        "body_sha256": normalized["body_sha256"],
    }
    source_keys = _source_keys(
        ruling_details=ruling_details,
        dependencies=dependency_rows,
        children=child_rows,
        affected_prs=pr_rows,
        implementability=implementability,
    )
    freshness_keys = {
        "issue_body_sha256": normalized["body_sha256"],
        "issue_body_observed": normalized["body_observed"],
        "issue_updated_at": normalized["updated_at"],
        **source_keys,
    }
    proposed_state = (
        "ready"
        if blocker_class == "dependency_predicate"
        and all_dependencies_satisfied
        and implementability_ready
        else next_state
    )
    terminal_delegated = normalized["state"] == "CLOSED"
    if terminal_delegated:
        blocker_class = "none"
        reason_codes = ["terminal_reconciliation_delegated"]
        owner = "terminal-label reconciliation owner #7651"
        next_action = "delegate terminal-label reconciliation to #7651; do not mutate here"
        proposed_state = "unchanged"
        delta = {"add": [], "remove": []}
    required_child_contract = {
        "status": "not_required",
        "parent_issue": normalized["number"],
        "required_fields": [],
        "executable": False,
    }
    if ruling_valid:
        required_child_contract = {
            "status": "ready" if child_contract_ready else "required",
            "parent_issue": normalized["number"],
            "required_fields": [
                "bounded scope and owner",
                "inputs and canonical source paths",
                "acceptance criteria",
                "verification command and expected proof",
                "authority and resource boundary",
            ],
            "executable": child_contract_ready,
            "ruling_token": ruling_details["token"],
            "ruling_carrier": ruling_details["carrier"],
            "observed_children": [item.get("number") for item in child_rows],
        }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "repository": _text(issue.get("repository") or issue.get("repo")),
        "item": normalized,
        "source_observations": {
            "ruling_valid": ruling_valid,
            "ruling": ruling_details,
            "dependencies": dependency_states,
            "dependency_conflicts": _dependency_conflict_codes(dependency_rows),
            "child_link_conflicts": _child_link_conflict_codes(normalized["number"], child_rows),
            "children_present": child_present,
            "unresolved_children": [item.get("number") for item in unresolved_children],
            "child_contract_ready": child_contract_ready,
            "implementability_ready": implementability_ready,
        },
        "blocker_class": blocker_class,
        "secondary_blockers": [
            {"class": candidate, "reason_codes": codes}
            for candidate, codes in reason_candidates
            if candidate != blocker_class
        ],
        "reason_codes": sorted(set(reason_codes)),
        "authority_owner": owner,
        "required_child_or_pr_links": {
            "children": sorted(
                {item.get("number") for item in child_rows if item.get("number") is not None}
            ),
            "dependencies": sorted(
                {item.get("number") for item in dependency_rows if item.get("number") is not None}
            ),
            "affected_prs": sorted(
                {item.get("number") for item in pr_rows if item.get("number") is not None}
            ),
            "affected_pr_owners": [
                {
                    "number": item.get("number"),
                    "owner": _text(
                        item.get("owner")
                        or item.get("implementation_owner")
                        or item.get("assignee")
                    ),
                }
                for item in source_keys["affected_prs"]
                if item.get("number") is not None
            ],
        },
        "required_child_contract": required_child_contract,
        "next_action": next_action,
        "next_permitted_state": proposed_state,
        "freshness_keys": freshness_keys,
        "revalidation_triggers": [
            "issue body, state, or labels change",
            "ruling carrier or token changes",
            "dependency state, revision, artifact, or predicate changes",
            "child or affected PR head/base changes",
            "implementability or CI result changes",
        ],
        "expected_before_mutation": expected,
        "proposed_label_delta": delta,
        "no_write": not (mode == "apply" and authorized),
        "mode": mode,
        "apply_authorized": mode == "apply" and authorized,
        "terminal_reconciliation": {
            "delegated": terminal_delegated,
            "owner": "#7651",
            "no_write": terminal_delegated,
        },
    }
    payload["plan_digest"] = _digest_payload(payload)
    return payload


def _default_api_runner(
    path: str, payload: object | None = None, method: str | None = None
) -> CompletedProcess[str]:
    """Run one bounded REST request for exact apply mode."""
    return run_gh_api(path, payload, method=method, timeout=30, timeout_context="transition apply")


def _read_live_issue(issue_number: int, *, repo: str, runner: ApiRunner) -> dict[str, Any]:
    """Read the exact live issue used by the compare-and-swap guard."""
    path = f"repos/{repo}/issues/{issue_number}"
    result = runner(path, None, None)
    data, error = parse_json(result, what=f"live issue #{issue_number}")
    if error or not isinstance(data, Mapping):
        raise TransitionError(error or "live issue response was not an object")
    return {
        "number": data.get("number"),
        "state": _text(data.get("state")).upper(),
        "labels": _labels(data.get("labels", [])),
        "body": _text(data.get("body")),
        "updated_at": _text(data.get("updated_at") or data.get("updatedAt")),
    }


def _revalidate_sources(
    plan: Mapping[str, Any],
    *,
    source_revalidator: SourceRevalidator | None,
) -> None:
    """Require a current source read when the plan depends on external observations."""
    freshness = plan.get("freshness_keys")
    if not isinstance(freshness, Mapping):
        raise TransitionError("transition plan is missing freshness keys")
    expected_sources = {
        key: freshness.get(key)
        for key in ("ruling", "dependencies", "children", "affected_prs", "implementability")
    }
    has_external_observation = any(_meaningful(value) for value in expected_sources.values())
    if not has_external_observation:
        return
    if source_revalidator is None:
        raise TransitionError(
            "source revalidation is required before applying a ruling or dependency transition"
        )
    try:
        observed_sources = source_revalidator(expected_sources)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise TransitionError(f"source revalidation failed: {exc}") from exc
    if not isinstance(observed_sources, Mapping):
        raise TransitionError("source revalidation must return an object")
    for key, expected in expected_sources.items():
        if observed_sources.get(key) != expected:
            raise TransitionError(f"source observation drifted for {key}")


def apply_transition(  # noqa: C901
    plan: Mapping[str, Any],
    *,
    repo: str = DEFAULT_REPO,
    expected_plan_digest: str,
    authorized: bool,
    runner: ApiRunner | None = None,
    source_revalidator: SourceRevalidator | None = None,
) -> dict[str, Any]:
    """Apply one exact label delta after a live compare-and-swap read."""
    if not authorized:
        raise TransitionError("exact-item apply requires explicit authorization")
    if expected_plan_digest != plan.get("plan_digest"):
        raise TransitionError("expected plan digest does not match the supplied plan")
    if plan.get("schema") != SCHEMA:
        raise TransitionError("unsupported transition plan schema")
    if plan.get("mode") != "apply" or plan.get("apply_authorized") is not True:
        raise TransitionError("supplied plan was not created for authorized apply")
    terminal_reconciliation = plan.get("terminal_reconciliation")
    if (
        isinstance(terminal_reconciliation, Mapping)
        and terminal_reconciliation.get("delegated") is True
    ):
        raise TransitionError("terminal label reconciliation is delegated to #7651")
    item = plan.get("item")
    expected = plan.get("expected_before_mutation")
    delta = plan.get("proposed_label_delta")
    if (
        not isinstance(item, Mapping)
        or not isinstance(expected, Mapping)
        or not isinstance(delta, Mapping)
    ):
        raise TransitionError("transition plan is missing exact apply fields")
    number = item.get("number")
    if type(number) is not int or number < 1 or expected.get("number") != number:
        raise TransitionError("transition plan item identity is invalid")
    if expected.get("body_observed") is not True:
        raise TransitionError("exact apply requires an observed issue body")
    run = runner or _default_api_runner
    observed = _read_live_issue(number, repo=repo, runner=run)
    if observed["state"] != expected.get("state"):
        raise TransitionError("live issue state drifted before transition apply")
    if observed["labels"] != expected.get("labels"):
        raise TransitionError("live issue labels drifted before transition apply")
    observed_body_sha256 = hashlib.sha256(observed["body"].encode("utf-8")).hexdigest()
    if observed_body_sha256 != expected.get("body_sha256"):
        raise TransitionError("live issue body drifted before transition apply")
    _revalidate_sources(plan, source_revalidator=source_revalidator)
    additions = _labels(delta.get("add", []))
    removals = set(_labels(delta.get("remove", [])))
    target = sorted((set(observed["labels"]) | set(additions)) - removals)
    if target == observed["labels"]:
        return {
            "schema": "blocker_transition_apply.v1",
            "status": "unchanged",
            "issue": number,
            "plan_digest": plan["plan_digest"],
            "labels": target,
            "state": observed["state"],
            "readback": True,
            "write_attempted": False,
        }
    path = f"repos/{repo}/issues/{number}/labels"
    result = run(path, {"labels": target}, "PUT")
    data, error = parse_json(result, what=f"replace labels for issue #{number}")
    if error or not isinstance(data, list):
        raise TransitionError(error or "label replacement response was not a list")
    confirmed = _labels(data)
    if confirmed != target:
        raise TransitionError("label replacement response did not match target labels")
    readback = _read_live_issue(number, repo=repo, runner=run)
    if readback["labels"] != target or readback["state"] != expected.get("state"):
        raise TransitionError("live issue label/state readback did not match transition target")
    return {
        "schema": "blocker_transition_apply.v1",
        "status": "applied" if target != observed["labels"] else "unchanged",
        "issue": number,
        "plan_digest": plan["plan_digest"],
        "labels": target,
        "state": readback["state"],
        "readback": True,
        "write_attempted": True,
    }


def _load_json(path: str | None, *, default: Any) -> Any:
    """Load an optional JSON fixture."""
    if not path:
        return default
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransitionError(f"unable to load JSON fixture {path}: {exc}") from exc


def _live_issue_json(issue_number: int, *, repo: str) -> dict[str, Any]:
    """Read one live issue for the CLI without authorizing any write."""
    result = _default_api_runner(f"repos/{repo}/issues/{issue_number}")
    data, error = parse_json(result, what=f"issue #{issue_number}")
    if error or not isinstance(data, Mapping):
        raise TransitionError(error or "issue response was not an object")
    return dict(data)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--mode", choices=("report-only", "plan", "apply"), default="report-only")
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--expected-plan-digest", default="")
    parser.add_argument("--issue-json", help="offline issue JSON fixture")
    parser.add_argument("--dependencies-json")
    parser.add_argument("--ruling-json")
    parser.add_argument("--children-json")
    parser.add_argument("--affected-prs-json")
    parser.add_argument("--implementability-json")
    parser.add_argument(
        "--revalidated-sources-json",
        help="offline/current source projection for exact apply revalidation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run report, plan, or exact-item apply mode."""
    args = _build_parser().parse_args(argv)
    try:
        issue = (
            _load_json(args.issue_json, default=None)
            if args.issue_json
            else _live_issue_json(args.issue, repo=args.repo)
        )
        if not isinstance(issue, Mapping):
            raise TransitionError("issue JSON must be an object")
        plan = plan_transition(
            issue,
            dependencies=_load_json(args.dependencies_json, default=[]),
            ruling=_load_json(args.ruling_json, default=None),
            children=_load_json(args.children_json, default=[]),
            affected_prs=_load_json(args.affected_prs_json, default=[]),
            implementability=_load_json(args.implementability_json, default=None),
            mode=args.mode,
            authorized=args.authorize,
        )
        result: Mapping[str, Any] = plan
        if args.mode == "apply":
            if not args.expected_plan_digest:
                raise TransitionError("--expected-plan-digest is required in apply mode")
            revalidated_sources = _load_json(args.revalidated_sources_json, default=None)
            source_revalidator: SourceRevalidator | None = None
            if revalidated_sources is not None:
                if not isinstance(revalidated_sources, Mapping):
                    raise TransitionError("revalidated source JSON must be an object")

                def _fixture_source_revalidator(_expected: Mapping[str, Any]) -> Mapping[str, Any]:
                    return revalidated_sources

                source_revalidator = _fixture_source_revalidator
            result = apply_transition(
                plan,
                repo=args.repo,
                expected_plan_digest=args.expected_plan_digest,
                authorized=args.authorize,
                source_revalidator=source_revalidator,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, TransitionError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "error", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
