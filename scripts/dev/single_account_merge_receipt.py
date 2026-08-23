#!/usr/bin/env python3
"""Build and verify the bounded single-account merge receipt.

The receipt is a control-plane record for one exact pull-request head.  It is
deliberately narrower than merge authority: it may document a waiver of a
second human implementation reviewer, but it cannot waive hosted checks,
metadata, threads, requested reviewers, explicit holds, or any domain,
scientific/evidence, legal/release, security, dependency, or draft gate.

The module owns the expected-head merge operation used by repository-side
callers.  Report-only and validation paths only read state.  A pre-merge
receipt digest excludes the terminal merge result, so recording GitHub's
returned merge SHA never reconstructs or changes the evidence observed before
the compare-and-swap operation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "single_account_merge_receipt.v1"
VERIFY_SCHEMA = "single_account_merge_receipt_verification.v1"
AUTHORITY_FIXTURE_SCHEMA = "single_account_merge_authority_fixture.v1"

SUCCESS_CONCLUSIONS = frozenset({"success", "neutral", "skipped"})
PENDING_STATUSES = frozenset(
    {"expected", "in_progress", "pending", "queued", "requested", "waiting"}
)
EVIDENCE_STATES = frozenset(
    {
        "accepted",
        "missing",
        "pending",
        "unavailable",
        "stale",
        "conflicting",
        "malformed",
        "superseded",
        "dismissed",
        "withdrawn",
        "failure",
    }
)
HOLD_KEYS = (
    "merge",
    "dependency",
    "draft",
    "domain",
    "scientific_evidence",
    "legal_release",
    "security",
)
APPROVED_AUTOMATED_IDENTITIES = frozenset({"coderabbit", "coderabbitai"})
REVIEW_KIND_RANK = {
    "check_run": 0,
    "review_event": 1,
    "static_report": 2,
    "machine_comment": 3,
}
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MACHINE_REVIEW_RE = re.compile(
    r"single-account-review\s*:\s*accepted\s*@\s*(?P<head>[0-9a-fA-F]{40})\b"
    r".*?metadata\s*[:=]\s*(?P<metadata>[0-9a-fA-F]{64})\b"
    r".*?(?:evidence|digest)\s*[:=]\s*(?P<evidence>[0-9a-fA-F]{64})\b",
    re.IGNORECASE | re.DOTALL,
)
_DO_NOT_MERGE_RE = re.compile(r"\[\s*do\s+not\s+merge\s*\]", re.IGNORECASE)
_DIRECT_MERGE_RE = re.compile(
    r"pulls/(?:\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*|<[^>]+>|[A-Za-z0-9_.-]+)/merge"
)

GhApi = Callable[[str, str, dict[str, Any] | None], tuple[Any | None, str | None]]


def _canonical_json(value: Any) -> str:
    """Return the stable JSON representation used by all receipt digests."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value using the receipt canonicalization."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _now_utc() -> str:
    """Return a second-precision UTC observation timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _string(value: Any) -> str:
    """Normalize arbitrary fixture/API values to a string."""
    return str(value or "")


def _full_sha(value: Any) -> bool:
    """Return whether *value* is a full Git object SHA."""
    return isinstance(value, str) and bool(_SHA_RE.fullmatch(value))


def _digest(value: Any) -> bool:
    """Return whether *value* is a full SHA-256 digest."""
    return isinstance(value, str) and bool(_DIGEST_RE.fullmatch(value))


def _identity(record: Mapping[str, Any]) -> str:
    """Extract the stable identity from a check, review, report, or comment."""
    user = record.get("user") if isinstance(record.get("user"), Mapping) else {}
    author = record.get("author") if isinstance(record.get("author"), Mapping) else {}
    app = record.get("app") if isinstance(record.get("app"), Mapping) else {}
    for candidate in (
        record.get("identity"),
        record.get("source"),
        record.get("login"),
        user.get("login"),
        author.get("login"),
        app.get("slug"),
        app.get("name"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _association(record: Mapping[str, Any]) -> str:
    """Normalize a GitHub author-association field."""
    return _string(record.get("authorAssociation") or record.get("author_association")).upper()


def _record_digest(record: Mapping[str, Any]) -> str:
    """Create an evidence digest without retaining review prose in the receipt."""
    compact: dict[str, Any] = {}
    for key in sorted(record):
        if key in {"body", "body_excerpt", "diff_hunk", "patch"}:
            continue
        value = record[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, Mapping):
            compact[key] = {
                str(nested_key): nested_value
                for nested_key, nested_value in value.items()
                if nested_key not in {"body", "body_excerpt", "diff_hunk", "patch"}
            }
        else:
            compact[key] = value
    body = record.get("body") or record.get("body_excerpt")
    if isinstance(body, str):
        compact["body_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return _sha256_json(compact)


def _carrier(
    *,
    record: Mapping[str, Any],
    kind: str,
    head_sha: str,
    metadata_digest: str,
    verdict: str = "accepted",
) -> dict[str, Any]:
    """Project one accepted carrier onto the immutable receipt shape."""
    identity = _identity(record)
    evidence_digest = _string(record.get("evidence_digest") or record.get("digest"))
    if not _digest(evidence_digest):
        evidence_digest = _record_digest(record)
    return {
        "identity": identity,
        "kind": kind,
        "head_sha": head_sha,
        "metadata_digest": metadata_digest,
        "evidence_digest": evidence_digest,
        "verdict": verdict,
    }


def _approved_automated(record: Mapping[str, Any]) -> bool:
    """Return whether a check/report explicitly belongs to an approved source."""
    if record.get("approved_reviewer") is True or record.get("approved_source") is True:
        return True
    identity = _identity(record).lower().replace(" ", "")
    return identity in APPROVED_AUTOMATED_IDENTITIES


def _head_of(record: Mapping[str, Any]) -> str:
    """Extract the exact commit binding from one evidence record."""
    return _string(record.get("head_sha") or record.get("commit_id") or record.get("commit_sha"))


def _review_metadata(record: Mapping[str, Any]) -> str:
    """Extract the metadata epoch bound by one review carrier."""
    return _string(record.get("metadata_digest") or record.get("metadata"))


def _candidate_state(  # noqa: C901, PLR0912 - explicit fail-closed carrier states.
    record: Mapping[str, Any],
    *,
    kind: str,
    head_sha: str,
    metadata_digest: str,
    waiver_actor: str,
) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Classify one candidate carrier without treating prose as approval."""
    reasons: list[str] = []
    if record.get("superseded") is True:
        return None, "superseded", ["review_carrier_superseded"]

    candidate_head = _head_of(record)
    if not candidate_head:
        return None, "malformed", ["review_carrier_head_missing"]
    if candidate_head.lower() != head_sha.lower():
        return None, "stale", ["review_carrier_stale_head"]

    candidate_metadata = _review_metadata(record)
    if not candidate_metadata:
        return None, "malformed", ["review_carrier_metadata_missing"]
    if candidate_metadata.lower() != metadata_digest.lower():
        return None, "stale", ["review_carrier_stale_metadata"]

    identity = _identity(record)
    if not identity:
        return None, "malformed", ["review_carrier_identity_missing"]
    if kind in {"check_run", "static_report", "machine_comment"} and not _approved_automated(
        record
    ):
        return None, "unavailable", ["review_carrier_source_not_approved"]

    if kind == "check_run":
        status = _string(record.get("status")).lower()
        conclusion = _string(record.get("conclusion")).lower()
        if status in PENDING_STATUSES or status != "completed":
            return None, "pending", ["review_check_run_not_terminal"]
        if conclusion not in SUCCESS_CONCLUSIONS:
            return None, "failure", ["review_check_run_not_success"]
    elif kind == "review_event":
        state = _string(record.get("state")).upper()
        if state == "DISMISSED" or record.get("dismissed") is True:
            return None, "dismissed", ["review_event_dismissed"]
        if state in {"WITHDRAWN", "RETRACTED"} or record.get("withdrawn") is True:
            return None, "withdrawn", ["review_event_withdrawn"]
        if state in {"PENDING", "PENDING_REVIEW"}:
            return None, "pending", ["review_event_pending"]
        if state != "APPROVED":
            return None, "missing", ["review_event_not_approved"]
        if waiver_actor and identity.lower() == waiver_actor.lower():
            return None, "conflicting", ["owner_self_review_not_independent"]
        if record.get("approved_reviewer") is not True and _association(record) not in {
            "MEMBER",
            "COLLABORATOR",
        }:
            return None, "unavailable", ["reviewer_not_approved_source"]
    elif kind == "static_report":
        verdict = _string(record.get("verdict")).lower()
        if verdict in {"pending", "queued"}:
            return None, "pending", ["static_review_pending"]
        if verdict != "accepted":
            return None, "conflicting", ["static_review_not_accepted"]
    else:
        body = _string(record.get("body") or record.get("body_excerpt"))
        match = _MACHINE_REVIEW_RE.search(body)
        if match is None:
            return None, "malformed", ["machine_review_marker_malformed"]
        if match.group("head").lower() != head_sha.lower():
            return None, "stale", ["machine_review_marker_stale_head"]
        if match.group("metadata").lower() != metadata_digest.lower():
            return None, "stale", ["machine_review_marker_stale_metadata"]

    return (
        _carrier(record=record, kind=kind, head_sha=head_sha, metadata_digest=metadata_digest),
        "accepted",
        reasons,
    )


def _direct_review_source(
    source: Mapping[str, Any], *, head_sha: str, metadata_digest: str
) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Validate a preclassified carrier supplied by a trusted caller or fixture."""
    kind = _string(source.get("kind") or source.get("carrier_kind"))
    if kind not in REVIEW_KIND_RANK:
        return None, "malformed", ["review_carrier_kind_invalid"]
    carrier = source.get("carrier") if isinstance(source.get("carrier"), Mapping) else source
    state = _string(source.get("status") or source.get("verdict")).lower()
    if state != "accepted":
        return (
            None,
            state if state in EVIDENCE_STATES else "malformed",
            ["review_carrier_not_accepted"],
        )
    candidate_head = _string(carrier.get("head_sha"))
    candidate_metadata = _string(carrier.get("metadata_digest"))
    evidence_digest = _string(carrier.get("evidence_digest"))
    identity = _string(carrier.get("identity"))
    if candidate_head.lower() != head_sha.lower():
        return None, "stale", ["review_carrier_stale_head"]
    if candidate_metadata.lower() != metadata_digest.lower():
        return None, "stale", ["review_carrier_stale_metadata"]
    if not identity or not _digest(evidence_digest):
        return None, "malformed", ["review_carrier_fields_malformed"]
    return (
        {
            "identity": identity,
            "kind": kind,
            "head_sha": head_sha,
            "metadata_digest": metadata_digest,
            "evidence_digest": evidence_digest,
            "verdict": "accepted",
        },
        "accepted",
        [],
    )


def classify_implementation_review(  # noqa: C901 - precedence and carrier states are explicit.
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the highest-precedence independent implementation review carrier.

    The input is intentionally generic so live REST/GraphQL snapshots and
    offline fixtures use the same contract.  A source must bind both the
    exact head and the current PR metadata digest.  Check runs and review
    events outrank static reports and machine comments; an owner-authored
    comment or self-review never becomes independent merely because it is
    present in the snapshot.
    """
    head_sha = _string(evidence.get("head_sha"))
    metadata_digest = _string(evidence.get("metadata_digest"))
    if not _full_sha(head_sha) or not _digest(metadata_digest):
        return {
            "status": "malformed",
            "carrier": None,
            "reason_codes": ["review_context_head_or_metadata_malformed"],
        }

    direct = evidence.get("review_source")
    if isinstance(direct, Mapping):
        carrier, status, reasons = _direct_review_source(
            direct, head_sha=head_sha, metadata_digest=metadata_digest
        )
        if carrier is not None:
            return {
                "status": "accepted",
                "carrier": carrier,
                "reason_codes": [],
                "precedence": REVIEW_KIND_RANK[carrier["kind"]],
            }
        return {"status": status, "carrier": None, "reason_codes": reasons}

    candidates: list[tuple[int, dict[str, Any]]] = []
    observations: list[tuple[str, list[str]]] = []
    collections = (
        ("check_run", evidence.get("check_runs")),
        ("review_event", evidence.get("reviews")),
        ("static_report", evidence.get("static_reports")),
        ("machine_comment", evidence.get("comments")),
    )
    for kind, raw_items in collections:
        if raw_items is None:
            continue
        if not isinstance(raw_items, list):
            observations.append(("malformed", [f"{kind}_collection_malformed"]))
            continue
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                observations.append(("malformed", [f"{kind}_carrier_malformed"]))
                continue
            carrier, status, reasons = _candidate_state(
                raw_item,
                kind=kind,
                head_sha=head_sha,
                metadata_digest=metadata_digest,
                waiver_actor=_string(evidence.get("waiver_actor")),
            )
            observations.append((status, reasons))
            if carrier is not None:
                candidates.append((REVIEW_KIND_RANK[kind], carrier))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]["identity"], item[1]["evidence_digest"]))
        best_rank, best = candidates[0]
        same_identity = [
            carrier
            for rank, carrier in candidates
            if rank == best_rank and carrier["identity"] == best["identity"]
        ]
        if len({carrier["evidence_digest"] for carrier in same_identity}) > 1:
            return {
                "status": "conflicting",
                "carrier": None,
                "reason_codes": ["same_precedence_review_carrier_conflict"],
            }
        return {
            "status": "accepted",
            "carrier": best,
            "reason_codes": [],
            "precedence": best_rank,
        }

    if not observations:
        return {"status": "missing", "carrier": None, "reason_codes": ["review_carrier_missing"]}
    status_order = (
        "conflicting",
        "dismissed",
        "withdrawn",
        "superseded",
        "malformed",
        "stale",
        "pending",
        "failure",
        "unavailable",
        "missing",
    )
    for selected in status_order:
        reasons = sorted(
            {reason for status, codes in observations if status == selected for reason in codes}
        )
        if reasons:
            return {"status": selected, "carrier": None, "reason_codes": reasons}
    return {"status": "missing", "carrier": None, "reason_codes": ["review_carrier_missing"]}


def normalize_required_checks(  # noqa: C901, PLR0912 - each hosted-check state remains distinct.
    raw_checks: Any, *, head_sha: str
) -> dict[str, Any]:
    """Normalize required hosted checks and keep every failure state distinct."""
    if not _full_sha(head_sha):
        return {
            "status": "malformed",
            "head_sha": head_sha,
            "checks": [],
            "reason_codes": ["head_sha_malformed"],
        }
    if isinstance(raw_checks, Mapping) and isinstance(raw_checks.get("checks"), list):
        normalized = copy.deepcopy(dict(raw_checks))
        if _string(normalized.get("head_sha")).lower() != head_sha.lower():
            normalized["status"] = "stale"
            normalized["reason_codes"] = ["required_checks_stale_head"]
        return normalized
    if not isinstance(raw_checks, list) or not raw_checks:
        return {
            "status": "unavailable",
            "head_sha": head_sha,
            "checks": [],
            "reason_codes": ["required_checks_unavailable"],
        }
    checks: list[dict[str, Any]] = []
    statuses: list[str] = []
    reasons: list[str] = []
    for index, raw in enumerate(raw_checks):
        if not isinstance(raw, Mapping):
            statuses.append("malformed")
            reasons.append(f"required_check_{index}_malformed")
            continue
        name = _string(raw.get("name") or raw.get("context"))
        observed_head = _head_of(raw)
        status = _string(raw.get("status")).lower()
        conclusion = _string(raw.get("conclusion")).lower()
        if not name or not observed_head:
            state = "malformed"
            reasons.append(f"required_check_{name or index}_malformed")
        elif observed_head.lower() != head_sha.lower():
            state = "stale"
            reasons.append(f"required_check_{name}_stale_head")
        elif status in PENDING_STATUSES or status != "completed":
            state = "pending"
            reasons.append(f"required_check_{name}_pending")
        elif conclusion not in SUCCESS_CONCLUSIONS:
            state = "failure"
            reasons.append(f"required_check_{name}_not_success")
        else:
            state = "success"
        statuses.append(state)
        checks.append(
            {
                "name": name,
                "head_sha": observed_head,
                "status": status or "missing",
                "conclusion": conclusion or None,
                "state": state,
                "required": raw.get("required", True) is not False,
                "identity": _identity(raw),
                "details_url": _string(
                    raw.get("details_url") or raw.get("html_url") or raw.get("target_url")
                ),
            }
        )
    if "malformed" in statuses:
        overall = "malformed"
    elif "stale" in statuses:
        overall = "stale"
    elif "pending" in statuses:
        overall = "pending"
    elif "failure" in statuses:
        overall = "failure"
    elif statuses and all(state == "success" for state in statuses):
        overall = "success"
    else:
        overall = "unavailable"
    return {
        "status": overall,
        "head_sha": head_sha,
        "checks": checks,
        "reason_codes": sorted(set(reasons)),
    }


def _normalize_thread_resolution(value: Any) -> dict[str, Any]:
    """Normalize the complete actionable review-thread result."""
    if isinstance(value, bool):
        return {"status": "resolved" if value else "unresolved", "unresolved": 0 if value else 1}
    if isinstance(value, Mapping):
        status = _string(value.get("status")).lower()
        if status not in {"resolved", "unresolved", "pending", "unavailable", "malformed"}:
            return {"status": "malformed", "unresolved": None}
        return {
            "status": status,
            "unresolved": value.get("unresolved"),
            "evidence_digest": _string(value.get("evidence_digest")),
        }
    return {"status": "unavailable", "unresolved": None}


def _normalize_requested(value: Any, *, label: str) -> dict[str, Any]:
    """Normalize reviewer/team request state without guessing missing data."""
    if isinstance(value, Mapping):
        status = _string(value.get("status")).lower()
        if status not in {"clear", "requested", "pending", "unavailable", "malformed"}:
            status = "malformed"
        return {
            "status": status,
            "count": value.get("count"),
            "identities": sorted(str(item) for item in value.get("identities", []) if item),
        }
    if isinstance(value, list):
        identities = sorted(
            _identity(item) if isinstance(item, Mapping) else _string(item) for item in value
        )
        identities = [item for item in identities if item]
        return {
            "status": "requested" if identities else "clear",
            "count": len(identities),
            "identities": identities,
        }
    return {"status": "unavailable", "count": None, "identities": [], "label": label}


def _hold(
    status: str, *, reason_codes: list[str] | None = None, source: str = ""
) -> dict[str, Any]:
    """Build one explicit hold disposition."""
    return {
        "status": status,
        "reason_codes": sorted(set(reason_codes or [])),
        "source": source,
    }


def derive_holds(evidence: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Derive separate hold dimensions from a trusted live snapshot.

    Absence of a hold label is a clear disposition only when the corresponding
    live labels/body were actually supplied.  Missing source fields remain
    ``unavailable`` and therefore block the receipt.
    """
    raw_labels = evidence.get("labels")
    labels: set[str]
    if isinstance(raw_labels, list):
        labels = {
            _string(item.get("name")) if isinstance(item, Mapping) else _string(item)
            for item in raw_labels
        }
        labels.discard("")
    else:
        labels = set()
    body = _string(evidence.get("body"))
    explicit = evidence.get("explicit_holds")
    explicit_reasons = (
        sorted(str(item) for item in explicit if item) if isinstance(explicit, list) else []
    )
    body_has_hold = bool(_DO_NOT_MERGE_RE.search(body))
    merge_reasons = list(explicit_reasons)
    if "merge-ready" not in labels:
        merge_reasons.append("merge_ready_label_missing")
    if body_has_hold:
        merge_reasons.append("do_not_merge_marker")
    holds: dict[str, dict[str, Any]] = {
        "merge": _hold(
            "held" if merge_reasons else "clear",
            reason_codes=merge_reasons,
            source="labels_and_body",
        ),
        "dependency": _hold(
            "held" if "dependency:has-blockers" in labels else "clear",
            reason_codes=["dependency_has_blockers"] if "dependency:has-blockers" in labels else [],
            source="labels",
        ),
        "draft": _hold(
            "held"
            if evidence.get("draft") is True
            else "clear"
            if type(evidence.get("draft")) is bool
            else "unavailable",
            reason_codes=["pr_is_draft"] if evidence.get("draft") is True else [],
            source="pull_request",
        ),
        "domain": _hold(
            "held"
            if "domain-review-required" in labels
            or any("domain-approval:pending" in item for item in explicit_reasons)
            else "clear",
            reason_codes=["domain_review_required"] if "domain-review-required" in labels else [],
            source="labels_and_body",
        ),
        "scientific_evidence": _hold(
            "held"
            if labels
            & {
                "scientific-review-required",
                "evidence-review-required",
                "benchmark-review-required",
            }
            else "clear",
            reason_codes=sorted(
                labels
                & {
                    "scientific-review-required",
                    "evidence-review-required",
                    "benchmark-review-required",
                }
            ),
            source="labels",
        ),
        "legal_release": _hold(
            "held"
            if labels
            & {"legal-review-required", "release-review-required", "license-review-required"}
            else "clear",
            reason_codes=sorted(
                labels
                & {"legal-review-required", "release-review-required", "license-review-required"}
            ),
            source="labels",
        ),
        "security": _hold(
            "held" if labels & {"security-review-required", "security-hold"} else "clear",
            reason_codes=sorted(labels & {"security-review-required", "security-hold"}),
            source="labels",
        ),
    }
    supplied = evidence.get("holds")
    if isinstance(supplied, Mapping):
        for key in HOLD_KEYS:
            if isinstance(supplied.get(key), Mapping):
                value = dict(supplied[key])
                status = _string(value.get("status")).lower()
                holds[key] = _hold(
                    status
                    if status in EVIDENCE_STATES or status in {"clear", "held"}
                    else "malformed",
                    reason_codes=[str(item) for item in value.get("reason_codes", []) if item],
                    source=_string(value.get("source")) or "caller",
                )
    return holds


def _normalize_waiver(value: Any, *, observed_at: str) -> dict[str, Any]:
    """Normalize the optional bounded single-account waiver."""
    if value is None:
        return {"used": False, "actor": None, "reason": None, "observed_at": observed_at}
    if not isinstance(value, Mapping):
        return {
            "used": True,
            "actor": None,
            "reason": None,
            "observed_at": observed_at,
            "status": "malformed",
        }
    used = value.get("used") is True
    actor = _string(value.get("actor") or value.get("waiver_actor")) or None
    reason = _string(value.get("reason") or value.get("waiver_reason")) or None
    timestamp = _string(value.get("observed_at") or observed_at)
    return {"used": used, "actor": actor, "reason": reason, "observed_at": timestamp}


def _cas_request(repository: str, pr_number: int, head_sha: str, value: Any) -> dict[str, Any]:
    """Normalize the one supported expected-head compare-and-swap request."""
    path = f"repos/{repository}/pulls/{pr_number}/merge"
    if isinstance(value, Mapping):
        request = value.get("request") if isinstance(value.get("request"), Mapping) else value
        method = _string(request.get("method")).upper() or "PUT"
        request_path = _string(request.get("path")) or path
        payload = (
            dict(request.get("payload")) if isinstance(request.get("payload"), Mapping) else {}
        )
    else:
        method = "PUT"
        request_path = path
        payload = {}
    payload.setdefault("sha", head_sha)
    payload.setdefault("merge_method", "squash")
    return {
        "method": method,
        "path": request_path,
        "payload": payload,
        "expected_head_sha": head_sha,
    }


def _premerge_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable portion hashed by ``receipt_digest``."""
    projection = copy.deepcopy(dict(receipt))
    projection.pop("receipt_digest", None)
    projection.pop("merge_result", None)
    projection.pop("status", None)
    return projection


def receipt_digest(receipt: Mapping[str, Any]) -> str:
    """Compute the canonical pre-merge digest for a receipt."""
    return _sha256_json(_premerge_projection(receipt))


def _normalize_review_source(value: Any, *, head_sha: str, metadata_digest: str) -> dict[str, Any]:
    """Normalize a classified implementation-review source."""
    if (
        isinstance(value, Mapping)
        and value.get("status") in EVIDENCE_STATES
        and isinstance(value.get("carrier"), Mapping)
    ):
        return {
            "status": _string(value.get("status")).lower(),
            "carrier": copy.deepcopy(value.get("carrier"))
            if isinstance(value.get("carrier"), Mapping)
            else None,
            "reason_codes": sorted(str(item) for item in value.get("reason_codes", []) if item),
            "precedence": value.get("precedence"),
        }
    if isinstance(value, Mapping):
        return classify_implementation_review(
            {"head_sha": head_sha, "metadata_digest": metadata_digest, "review_source": value}
        )
    return {"status": "unavailable", "carrier": None, "reason_codes": ["review_source_unavailable"]}


def build_receipt(  # noqa: PLR0913 - schema fields are intentionally explicit and auditable.
    *,
    repository: str,
    pr_number: int,
    head_sha: str,
    base_sha: str,
    current_base_sha: str,
    metadata_digest: str,
    required_checks: Any,
    review_source: Any,
    thread_resolution: Any,
    requested_reviewers: Any,
    requested_teams: Any,
    holds: Any,
    waiver: Any = None,
    expected_head_cas: Any = None,
    gate_audit: Mapping[str, Any] | None = None,
    pr_state: str | None = None,
    pr_merged_at: str | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Produce one deterministic receipt, including blocked evidence states."""
    timestamp = observed_at or _now_utc()
    checks = normalize_required_checks(required_checks, head_sha=head_sha)
    review = _normalize_review_source(
        review_source, head_sha=head_sha, metadata_digest=metadata_digest
    )
    threads = _normalize_thread_resolution(thread_resolution)
    reviewers = _normalize_requested(requested_reviewers, label="reviewers")
    teams = _normalize_requested(requested_teams, label="teams")
    normalized_holds = (
        derive_holds({"holds": holds}) if isinstance(holds, Mapping) else derive_holds({})
    )
    normalized_waiver = _normalize_waiver(waiver, observed_at=timestamp)
    cas = _cas_request(repository, pr_number, head_sha, expected_head_cas)
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "repository": repository,
        "pr_number": pr_number,
        "head_sha": head_sha,
        "base_sha": base_sha,
        "current_base_sha": current_base_sha,
        "metadata_digest": metadata_digest,
        "required_checks": checks,
        "implementation_review": review,
        "thread_resolution": threads,
        "requested_reviewers": reviewers,
        "requested_teams": teams,
        "holds": normalized_holds,
        "waiver": normalized_waiver,
        "expected_head_cas": {"request": cas, "status": "not_applied"},
        "gate_audit": copy.deepcopy(dict(gate_audit)) if gate_audit is not None else None,
        "pr_state": _string(pr_state).upper() or None,
        "pr_merged_at": pr_merged_at,
        "observed_at": timestamp,
        "merge_result": {"status": "not_applied", "returned_merged_sha": None},
    }
    reasons = _premerge_reasons(receipt, structural_only=True)
    receipt["status"] = "ready" if not reasons else "blocked"
    receipt["reason_codes"] = reasons
    receipt["receipt_digest"] = receipt_digest(receipt)
    return receipt


def _premerge_reasons(  # noqa: C901, PLR0912, PLR0915 - every waiver/hold dimension fails closed independently.
    receipt: Mapping[str, Any], *, structural_only: bool = False
) -> list[str]:
    """Return deterministic pre-merge blockers for one receipt."""
    reasons: list[str] = []
    if not _full_sha(_string(receipt.get("head_sha"))):
        reasons.append("head_sha_missing_or_malformed")
    if not _full_sha(_string(receipt.get("base_sha"))):
        reasons.append("base_sha_missing_or_malformed")
    if not _full_sha(_string(receipt.get("current_base_sha"))):
        reasons.append("current_base_sha_missing_or_malformed")
    if not _digest(_string(receipt.get("metadata_digest"))):
        reasons.append("metadata_digest_missing_or_malformed")

    pr_state = _string(receipt.get("pr_state")).upper()
    pr_merged_at = receipt.get("pr_merged_at")
    if pr_state not in {"OPEN", "CLOSED", "MERGED"}:
        reasons.append("pr_state_unavailable")
    if pr_merged_at is not None and (not isinstance(pr_merged_at, str) or not pr_merged_at.strip()):
        reasons.append("pr_merged_at_malformed")
    if pr_state != "OPEN" or pr_merged_at:
        reasons.append("pr_not_open")
    if pr_state == "MERGED" or pr_merged_at:
        reasons.append("pr_already_merged")

    gate_audit = receipt.get("gate_audit")
    if not isinstance(gate_audit, Mapping):
        reasons.append("merge_queue_gate_unavailable")
    else:
        gate_passed = (
            gate_audit.get("passed") is True
            if "passed" in gate_audit
            else _string(gate_audit.get("status")).lower() == "success"
        )
        if not gate_passed:
            reasons.append("merge_queue_gate_not_passed")
            raw_gate_reasons = gate_audit.get("reasons")
            if isinstance(raw_gate_reasons, list):
                reasons.extend(
                    f"merge_queue_gate_{reason}"
                    for reason in raw_gate_reasons
                    if isinstance(reason, str) and reason
                )

    checks = (
        receipt.get("required_checks")
        if isinstance(receipt.get("required_checks"), Mapping)
        else {}
    )
    if checks.get("status") != "success":
        reasons.append(f"required_checks_{_string(checks.get('status')) or 'unavailable'}")
    if _string(checks.get("head_sha")).lower() != _string(receipt.get("head_sha")).lower():
        reasons.append("required_checks_head_mismatch")
    raw_check_items = checks.get("checks")
    if not isinstance(raw_check_items, list) or not raw_check_items:
        reasons.append("required_checks_entries_missing")
    else:
        for index, check in enumerate(raw_check_items):
            if not isinstance(check, Mapping):
                reasons.append(f"required_check_{index}_malformed")
                continue
            if check.get("state") != "success":
                reasons.append(
                    f"required_check_{_string(check.get('name')) or index}_"
                    f"{_string(check.get('state')) or 'malformed'}"
                )
            if _string(check.get("head_sha")).lower() != _string(receipt.get("head_sha")).lower():
                reasons.append(
                    f"required_check_{_string(check.get('name')) or index}_head_mismatch"
                )
    review = (
        receipt.get("implementation_review")
        if isinstance(receipt.get("implementation_review"), Mapping)
        else {}
    )
    if review.get("status") != "accepted":
        reasons.append(f"implementation_review_{_string(review.get('status')) or 'unavailable'}")
    else:
        carrier = review.get("carrier") if isinstance(review.get("carrier"), Mapping) else {}
        if _string(carrier.get("head_sha")).lower() != _string(receipt.get("head_sha")).lower():
            reasons.append("implementation_review_head_mismatch")
        if (
            _string(carrier.get("metadata_digest")).lower()
            != _string(receipt.get("metadata_digest")).lower()
        ):
            reasons.append("implementation_review_metadata_mismatch")
        if not _string(carrier.get("identity")):
            reasons.append("implementation_review_identity_missing")
        if _string(carrier.get("kind")) not in REVIEW_KIND_RANK:
            reasons.append("implementation_review_kind_invalid")
        if _string(carrier.get("verdict")).lower() != "accepted":
            reasons.append("implementation_review_verdict_invalid")
        if not _digest(_string(carrier.get("evidence_digest"))):
            reasons.append("implementation_review_evidence_digest_malformed")

    threads = (
        receipt.get("thread_resolution")
        if isinstance(receipt.get("thread_resolution"), Mapping)
        else {}
    )
    if threads.get("status") != "resolved":
        reasons.append(f"review_threads_{_string(threads.get('status')) or 'unavailable'}")
    for field in ("requested_reviewers", "requested_teams"):
        requests = receipt.get(field) if isinstance(receipt.get(field), Mapping) else {}
        if requests.get("status") != "clear":
            reasons.append(f"{field}_{_string(requests.get('status')) or 'unavailable'}")
        elif requests.get("count") not in {0, None} or requests.get("identities"):
            reasons.append(f"{field}_clear_state_inconsistent")
    holds = receipt.get("holds") if isinstance(receipt.get("holds"), Mapping) else {}
    for key in HOLD_KEYS:
        disposition = holds.get(key) if isinstance(holds.get(key), Mapping) else {}
        if disposition.get("status") != "clear":
            reasons.append(f"hold_{key}_{_string(disposition.get('status')) or 'unavailable'}")

    waiver = receipt.get("waiver") if isinstance(receipt.get("waiver"), Mapping) else {}
    if waiver.get("used") is True:
        if not _string(waiver.get("actor")):
            reasons.append("waiver_actor_missing")
        if not _string(waiver.get("reason")):
            reasons.append("waiver_reason_missing")
        if not _string(waiver.get("observed_at")):
            reasons.append("waiver_timestamp_missing")
    cas_block = (
        receipt.get("expected_head_cas")
        if isinstance(receipt.get("expected_head_cas"), Mapping)
        else {}
    )
    cas = cas_block.get("request") if isinstance(cas_block.get("request"), Mapping) else {}
    if cas.get("method") != "PUT":
        reasons.append("expected_head_cas_method_invalid")
    if _string(cas.get("expected_head_sha")).lower() != _string(receipt.get("head_sha")).lower():
        reasons.append("expected_head_cas_head_mismatch")
    payload = cas.get("payload") if isinstance(cas.get("payload"), Mapping) else {}
    if _string(payload.get("sha")).lower() != _string(receipt.get("head_sha")).lower():
        reasons.append("expected_head_cas_payload_mismatch")
    if not structural_only and receipt.get("status") not in {"ready", "applied", "merged"}:
        reasons.append("receipt_status_not_ready")
    return sorted(set(reasons))


def validate_receipt(receipt: Any) -> dict[str, Any]:
    """Validate receipt structure and the immutable pre-merge digest."""
    reasons: list[str] = []
    if not isinstance(receipt, Mapping):
        return {
            "schema": VERIFY_SCHEMA,
            "status": "invalid",
            "passed": False,
            "reasons": ["receipt_not_object"],
        }
    if receipt.get("schema") != RECEIPT_SCHEMA:
        reasons.append("receipt_schema_mismatch")
    required = {
        "repository",
        "pr_number",
        "head_sha",
        "base_sha",
        "current_base_sha",
        "metadata_digest",
        "required_checks",
        "implementation_review",
        "thread_resolution",
        "requested_reviewers",
        "requested_teams",
        "holds",
        "waiver",
        "expected_head_cas",
        "observed_at",
        "merge_result",
        "receipt_digest",
        "status",
        "reason_codes",
    }
    reasons.extend(f"receipt_field_missing:{field}" for field in sorted(required - set(receipt)))
    if not isinstance(receipt.get("repository"), str) or not _string(receipt.get("repository")):
        reasons.append("repository_missing_or_malformed")
    if isinstance(receipt.get("pr_number"), bool) or not isinstance(receipt.get("pr_number"), int):
        reasons.append("pr_number_missing_or_malformed")
    if not _digest(_string(receipt.get("receipt_digest"))):
        reasons.append("receipt_digest_missing_or_malformed")
    elif _string(receipt.get("receipt_digest")).lower() != receipt_digest(receipt).lower():
        reasons.append("receipt_digest_mismatch")
    holds = receipt.get("holds")
    if not isinstance(holds, Mapping):
        reasons.append("holds_missing_or_malformed")
    else:
        reasons.extend(f"hold_field_missing:{key}" for key in HOLD_KEYS if key not in holds)
    status = "valid" if not reasons else "invalid"
    return {
        "schema": VERIFY_SCHEMA,
        "status": status,
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "receipt_digest": _string(receipt.get("receipt_digest")),
    }


def verify_receipt(  # noqa: C901, PLR0912 - revalidation compares every immutable evidence dimension.
    receipt: Mapping[str, Any],
    *,
    live_evidence: Mapping[str, Any] | None = None,
    require_merged: bool = False,
) -> dict[str, Any]:
    """Verify a receipt and optionally compare it with a fresh live snapshot."""
    structural = validate_receipt(receipt)
    reasons = list(structural.get("reasons", []))
    if structural.get("passed") is True:
        reasons.extend(_premerge_reasons(receipt))

    if live_evidence is not None:
        if (
            _string(live_evidence.get("head_sha")).lower()
            != _string(receipt.get("head_sha")).lower()
        ):
            reasons.append("live_head_sha_changed")
        if (
            _string(live_evidence.get("base_sha")).lower()
            != _string(receipt.get("base_sha")).lower()
        ):
            reasons.append("live_pr_base_sha_changed")
        if (
            _string(live_evidence.get("current_base_sha")).lower()
            != _string(receipt.get("current_base_sha")).lower()
        ):
            reasons.append("live_current_base_sha_changed")
        if (
            _string(live_evidence.get("metadata_digest")).lower()
            != _string(receipt.get("metadata_digest")).lower()
        ):
            reasons.append("live_metadata_digest_changed")
        if (
            _string(live_evidence.get("pr_state")).upper()
            != _string(receipt.get("pr_state")).upper()
        ):
            reasons.append("live_pr_state_changed")
        if live_evidence.get("pr_merged_at") != receipt.get("pr_merged_at"):
            reasons.append("live_pr_merged_at_changed")
        fresh_checks = normalize_required_checks(
            live_evidence.get("required_checks"), head_sha=_string(live_evidence.get("head_sha"))
        )
        if _canonical_json(fresh_checks) != _canonical_json(receipt.get("required_checks")):
            reasons.append("live_required_checks_changed")
        fresh_review = _normalize_review_source(
            live_evidence.get("review_source"),
            head_sha=_string(live_evidence.get("head_sha")),
            metadata_digest=_string(live_evidence.get("metadata_digest")),
        )
        if _canonical_json(fresh_review) != _canonical_json(receipt.get("implementation_review")):
            reasons.append("live_implementation_review_changed")
        fresh_threads = _normalize_thread_resolution(live_evidence.get("thread_resolution"))
        if fresh_threads.get("status") != (receipt.get("thread_resolution") or {}).get("status"):
            reasons.append("live_review_threads_changed")
        fresh_reviewers = _normalize_requested(
            live_evidence.get("requested_reviewers"), label="reviewers"
        )
        fresh_teams = _normalize_requested(live_evidence.get("requested_teams"), label="teams")
        if fresh_reviewers.get("status") != (receipt.get("requested_reviewers") or {}).get(
            "status"
        ):
            reasons.append("live_requested_reviewers_changed")
        if fresh_teams.get("status") != (receipt.get("requested_teams") or {}).get("status"):
            reasons.append("live_requested_teams_changed")
        if _canonical_json(live_evidence.get("gate_audit")) != _canonical_json(
            receipt.get("gate_audit")
        ):
            reasons.append("live_gate_audit_changed")
        fresh_holds = derive_holds(live_evidence)
        if _canonical_json(fresh_holds) != _canonical_json(receipt.get("holds")):
            reasons.append("live_hold_disposition_changed")

    merge_result = (
        receipt.get("merge_result") if isinstance(receipt.get("merge_result"), Mapping) else {}
    )
    if require_merged:
        if merge_result.get("status") != "merged":
            reasons.append("merged_result_missing")
        if not _full_sha(_string(merge_result.get("returned_merged_sha"))):
            reasons.append("returned_merged_sha_missing_or_malformed")
    elif merge_result.get("status") not in {"not_applied", "merged"}:
        reasons.append("merge_result_invalid_for_premerge_verification")

    unique = sorted(set(reasons))
    passed = not unique
    return {
        "schema": VERIFY_SCHEMA,
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "reasons": unique,
        "receipt_digest": _string(receipt.get("receipt_digest")),
        "head_sha": _string(receipt.get("head_sha")),
        "current_base_sha": _string(receipt.get("current_base_sha")),
    }


def record_merge_result(
    receipt: Mapping[str, Any],
    *,
    status: str,
    returned_merged_sha: str | None = None,
    response: Mapping[str, Any] | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Append the terminal merge result while preserving the pre-merge digest."""
    result = copy.deepcopy(dict(receipt))
    result["merge_result"] = {
        "status": status,
        "returned_merged_sha": returned_merged_sha,
        "response": copy.deepcopy(dict(response)) if response is not None else None,
        "observed_at": observed_at or _now_utc(),
    }
    result["status"] = "merged" if status == "merged" else "applied"
    result["reason_codes"] = list(result.get("reason_codes", []))
    return result


def apply_guarded_merge(
    receipt: Mapping[str, Any],
    *,
    repository: str,
    api: GhApi,
    observed_at: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Apply exactly one expected-head squash merge and verify its terminal state."""
    verification = verify_receipt(receipt)
    if verification.get("passed") is not True:
        return None, "receipt blocked: " + ", ".join(verification.get("reasons", []))
    pr_number = receipt.get("pr_number")
    head_sha = _string(receipt.get("head_sha"))
    response, error = api(
        "PUT",
        f"repos/{repository}/pulls/{pr_number}/merge",
        {"sha": head_sha, "merge_method": "squash"},
    )
    if error:
        return record_merge_result(
            receipt, status="failed", response={"error": error}, observed_at=observed_at
        ), error
    if not isinstance(response, Mapping) or response.get("merged") is not True:
        message = (
            _string(response.get("message"))
            if isinstance(response, Mapping)
            else "merge response was not an object"
        )
        return record_merge_result(
            receipt, status="failed", response={"message": message}, observed_at=observed_at
        ), message
    merged_sha = _string(response.get("sha") or response.get("merge_commit_sha"))
    if not _full_sha(merged_sha):
        message = "merge response did not include a full returned merge SHA"
        return record_merge_result(
            receipt, status="failed", response=dict(response), observed_at=observed_at
        ), message
    verified, verify_error = api("GET", f"repos/{repository}/pulls/{pr_number}", None)
    if verify_error or not isinstance(verified, Mapping):
        message = (
            f"remote merge verification failed: {verify_error or 'response was not an object'}"
        )
        return record_merge_result(
            receipt, status="failed", response=dict(response), observed_at=observed_at
        ), message
    if verified.get("state") != "closed" or verified.get("merged") is not True:
        message = "remote merge response is not closed/merged"
        return record_merge_result(
            receipt, status="failed", response=dict(response), observed_at=observed_at
        ), message
    merged_receipt = record_merge_result(
        receipt,
        status="merged",
        returned_merged_sha=merged_sha,
        response=dict(response),
        observed_at=observed_at,
    )
    return {
        "pr": pr_number,
        "merge_commit_sha": merged_sha,
        "receipt": merged_receipt,
    }, None


def detect_post_merge_incident(
    receipt: Mapping[str, Any], *, observed_merged_sha: str | None = None
) -> dict[str, Any]:
    """Classify an invalid post-merge receipt without repairing its evidence."""
    verification = verify_receipt(receipt, require_merged=True)
    merge_result = (
        receipt.get("merge_result") if isinstance(receipt.get("merge_result"), Mapping) else {}
    )
    if (
        observed_merged_sha
        and _string(merge_result.get("returned_merged_sha")) != observed_merged_sha
    ):
        verification["passed"] = False
        verification["status"] = "blocked"
        verification.setdefault("reasons", []).append("post_merge_sha_mismatch")
    if verification.get("passed") is True:
        return {"status": "healthy", "waiver_reuse": "allowed", "verification": verification}
    return {
        "status": "incident",
        "waiver_reuse": "blocked",
        "action": "preserve_receipt_and_choose_guarded_revert_or_exact_head_fix_forward",
        "verification": verification,
    }


def build_live_evidence(
    pr_number: int, *, repository: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Re-read canonical merge-gate evidence for a report/validate/apply run."""
    from scripts.dev.merge_queue_gate import (  # local import avoids a module cycle for pure helpers
        evaluate_merge_gate,
        fetch_main_sha,
        fetch_pr_snapshot,
        fetch_threads_resolved,
    )

    snapshot, error = fetch_pr_snapshot(pr_number, repo=repository)
    if error or not snapshot:
        return None, error or "PR snapshot unavailable"
    current_base_sha = fetch_main_sha(repo=repository)
    if not current_base_sha:
        return None, "current main SHA unavailable"
    threads, thread_error = fetch_threads_resolved(pr_number, repo=repository)
    if thread_error or threads is None:
        return None, thread_error or "review-thread state unavailable"
    gate = evaluate_merge_gate(
        snapshot,
        main_sha=current_base_sha,
        threads_resolved=threads,
        reviewers_requested=snapshot.get("reviewers_requested")
        if isinstance(snapshot.get("reviewers_requested"), bool)
        else None,
    )
    review_evidence = (
        snapshot.get("review_evidence")
        if isinstance(snapshot.get("review_evidence"), Mapping)
        else {}
    )
    review_source = classify_implementation_review(
        {
            "head_sha": snapshot.get("head_sha"),
            "metadata_digest": snapshot.get("metadata_digest"),
            **review_evidence,
        }
    )
    return {
        "repository": repository,
        "pr_number": int(snapshot.get("number") or pr_number),
        "head_sha": snapshot.get("head_sha"),
        "base_sha": snapshot.get("base_sha"),
        "current_base_sha": current_base_sha,
        "pr_state": snapshot.get("pr_state"),
        "pr_merged_at": snapshot.get("pr_merged_at"),
        "metadata_digest": snapshot.get("metadata_digest"),
        "required_checks": snapshot.get("required_checks"),
        "review_source": review_source,
        "thread_resolution": {"status": "resolved" if threads else "unresolved"},
        "requested_reviewers": snapshot.get("requested_reviewers"),
        "requested_teams": snapshot.get("requested_teams"),
        "holds": derive_holds(snapshot),
        "gate_audit": gate.to_dict(),
    }, None


def build_receipt_from_stack_entry(
    repository: str,
    entry: Mapping[str, Any],
    *,
    current_base_sha: str,
    waiver_actor: str = "",
    waiver_reason: str = "",
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Build a receipt from the complete `stacked_prs` exact-head snapshot."""
    waiver = {
        "used": bool(waiver_actor or waiver_reason),
        "actor": waiver_actor or None,
        "reason": waiver_reason or None,
    }
    return build_receipt(
        repository=repository,
        pr_number=int(entry["pr"]),
        head_sha=_string(entry.get("head_sha")),
        base_sha=_string(entry.get("base_sha")),
        current_base_sha=current_base_sha,
        metadata_digest=_string(entry.get("metadata_digest")),
        required_checks=entry.get("required_checks"),
        review_source=entry.get("implementation_review"),
        thread_resolution=entry.get("review_threads"),
        requested_reviewers={
            "status": "requested" if entry.get("requested_reviewer_count") else "clear",
            "count": entry.get("requested_reviewer_count", 0),
        },
        requested_teams={
            "status": "requested" if entry.get("requested_team_count") else "clear",
            "count": entry.get("requested_team_count", 0),
        },
        holds=entry.get("holds") or derive_holds(entry),
        waiver=waiver,
        expected_head_cas={"expected_base_sha": current_base_sha},
        gate_audit=entry.get("merge_queue_gate"),
        pr_state=_string(entry.get("state")) or None,
        pr_merged_at=entry.get("merged_at"),
        observed_at=observed_at,
    )


def _run_gh_api(
    method: str, path: str, payload: dict[str, Any] | None = None
) -> tuple[Any | None, str | None]:
    """Run the one bounded GitHub API operation used by ``--apply``."""
    args = ["gh", "api"]
    if method == "GET":
        args.append(path)
    else:
        args.extend(["--method", method, path, "--input", "-"])
    try:
        result = subprocess.run(
            args,
            input=json.dumps(payload) if method != "GET" else None,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    if result.returncode != 0:
        return None, result.stderr.strip() or f"gh api exited with code {result.returncode}"
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"gh api returned invalid JSON: {exc}"


def validate_merge_authority_fixture(  # noqa: C901, PLR0912 - explicit policy checks.
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate that every declared merge caller routes through the receipt owner.

    This is deliberately a repository-policy check rather than a source parser
    for arbitrary Python.  It verifies the small set of declared callers and
    scans runtime scripts and workflow guidance for direct merge endpoints that
    could bypass the receipt's expected-head compare-and-swap operation.
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    fixture_path = root / "scripts/dev/single_account_merge_authority_fixture.v1.json"
    reasons: list[str] = []
    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema": AUTHORITY_FIXTURE_SCHEMA,
            "status": "blocked",
            "passed": False,
            "reasons": [f"fixture_unavailable:{exc}"],
        }
    if not isinstance(fixture, Mapping):
        return {
            "schema": AUTHORITY_FIXTURE_SCHEMA,
            "status": "blocked",
            "passed": False,
            "reasons": ["fixture_not_object"],
        }
    if fixture.get("schema") != AUTHORITY_FIXTURE_SCHEMA:
        reasons.append("fixture_schema_mismatch")
    if fixture.get("receipt_schema") != RECEIPT_SCHEMA:
        reasons.append("fixture_receipt_schema_mismatch")
    owner_rel = _string(fixture.get("receipt_owner"))
    owner_path = root / owner_rel if owner_rel else root / "__missing_receipt_owner__"
    if not owner_rel or not owner_path.is_file():
        reasons.append("receipt_owner_missing")
        owner_text = ""
    else:
        owner_text = owner_path.read_text(encoding="utf-8")
        if "def apply_guarded_merge" not in owner_text:
            reasons.append("receipt_owner_apply_entrypoint_missing")
        if RECEIPT_SCHEMA not in owner_text:
            reasons.append("receipt_owner_schema_missing")

    callers = fixture.get("merge_callers")
    if not isinstance(callers, list) or not callers:
        reasons.append("merge_callers_missing")
        callers = []
    for index, raw_caller in enumerate(callers):
        if not isinstance(raw_caller, Mapping):
            reasons.append(f"merge_caller_{index}_malformed")
            continue
        caller_rel = _string(raw_caller.get("path"))
        caller_path = root / caller_rel if caller_rel else root / "__missing_caller__"
        if not caller_rel or not caller_path.is_file():
            reasons.append(f"merge_caller_{index}_missing")
            continue
        caller_text = caller_path.read_text(encoding="utf-8")
        if raw_caller.get("requires_receipt") is not True:
            reasons.append(f"merge_caller_{caller_rel}_receipt_requirement_missing")
        if (
            RECEIPT_SCHEMA not in caller_text
            and owner_rel not in caller_text
            and "apply_guarded_merge" not in caller_text
        ):
            reasons.append(f"merge_caller_{caller_rel}_does_not_reference_receipt")

    forbidden = fixture.get("forbidden_direct_merge_patterns")
    if not isinstance(forbidden, list) or not forbidden:
        reasons.append("forbidden_merge_patterns_missing")

    scan_roots = (
        root / "scripts/dev",
        root / "docs",
        root / ".agents/skills",
        root / ".opencode/skills",
    )
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            continue
        suffixes = {".py"} if scan_root.name == "dev" else {".md"}
        for candidate in sorted(scan_root.rglob("*")):
            if not candidate.is_file() or candidate.suffix not in suffixes:
                continue
            if candidate.resolve() == owner_path.resolve():
                continue
            try:
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                reasons.append(f"merge_authority_scan_failed:{candidate.relative_to(root)}:{exc}")
                continue
            for line_number, line in enumerate(lines, start=1):
                if _DIRECT_MERGE_RE.search(line) or "gh pr merge" in line:
                    reasons.append(
                        "direct_merge_bypass:"
                        + str(candidate.relative_to(root))
                        + f":{line_number}"
                    )

    unique = sorted(set(reasons))
    return {
        "schema": AUTHORITY_FIXTURE_SCHEMA,
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "receipt_owner": owner_rel,
        "merge_callers": [
            _string(item.get("path")) for item in callers if isinstance(item, Mapping)
        ],
        "reasons": unique,
    }


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load one receipt file without repairing or normalizing it."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "receipt file must contain a JSON object"
    return value, None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a human-readable JSON artifact to an explicit caller path."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, required=True, help="pull-request number")
    parser.add_argument("--repo", default="ll7/robot_sf_ll7", help="owner/repo")
    parser.add_argument(
        "--mode", choices=("report-only", "validate", "apply"), default="report-only"
    )
    parser.add_argument("--receipt-file", type=Path, help="receipt JSON file for validate/apply")
    parser.add_argument("--output", type=Path, help="write a report-only receipt to this path")
    parser.add_argument("--waiver-actor", default="")
    parser.add_argument("--waiver-reason", default="")
    return parser


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - CLI modes are explicit and fail closed.
    """Run a read-only receipt report, validation, or guarded apply."""
    args = _parser().parse_args(argv)
    if args.mode in {"validate", "apply"} and args.receipt_file is None:
        print("--receipt-file is required for validate/apply", file=sys.stderr)
        return 2
    if args.mode == "report-only":
        evidence, error = build_live_evidence(args.pr, repository=args.repo)
        if error or evidence is None:
            print(
                json.dumps(
                    {"status": "error", "error": error or "evidence unavailable"}, sort_keys=True
                )
            )
            return 1
        receipt = build_receipt(
            **evidence,
            waiver={
                "used": bool(args.waiver_actor or args.waiver_reason),
                "actor": args.waiver_actor,
                "reason": args.waiver_reason,
            },
        )
        if args.output:
            _write_json(args.output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt.get("status") == "ready" else 1

    receipt, error = _load_json(args.receipt_file)
    if error or receipt is None:
        print(
            json.dumps({"status": "error", "error": error or "receipt unavailable"}, sort_keys=True)
        )
        return 1
    evidence, error = build_live_evidence(args.pr, repository=args.repo)
    if error or evidence is None:
        print(
            json.dumps(
                {"status": "error", "error": error or "live evidence unavailable"}, sort_keys=True
            )
        )
        return 1
    verification = verify_receipt(receipt, live_evidence=evidence)
    if args.mode == "validate":
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 0 if verification.get("passed") is True else 1
    if verification.get("passed") is not True:
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 1
    merged, merge_error = apply_guarded_merge(receipt, repository=args.repo, api=_run_gh_api)
    if merge_error or merged is None:
        output: dict[str, Any] = {
            "status": "error",
            "error": merge_error or "merge failed",
        }
        if isinstance(merged, Mapping) and "receipt_digest" in merged:
            output["receipt"] = merged
        print(json.dumps(output, sort_keys=True))
        return 1
    print(json.dumps(merged, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
