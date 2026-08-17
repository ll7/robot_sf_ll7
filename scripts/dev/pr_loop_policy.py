#!/usr/bin/env python3
"""Machine-checkable state policy for autonomous PR loops.

Classifies PR state from compact snapshots and recommends one bounded action:
stop, continue, reroute, escalate, wait_ci, inspect_failed_ci, verify_artifacts,
refresh_snapshot, mark_ready_candidate, await_gate_verdict, reconcile_pr_metadata,
await_changed_coverage,
or no_action.

Every PolicyDecision also emits a high-level ``flow_decision`` — one of exactly
``stop``, ``continue``, ``reroute``, or ``escalate`` — for machine consumption.

Review state (e.g. CHANGES_REQUESTED, APPROVED, COMMENTED) from the snapshot
is incorporated: CHANGES_REQUESTED forces a non-continue flow decision.

Gate-verdict contract (issue #6019): an exact head is only eligible to advance
toward merge when every required check is green AND a current exact-head
``gate-verdict: accepted @ <head_sha>`` trailer exists. The dispatcher rejects
(fail closed) any head missing such a trailer, classifying it as
``pending_gate_verdict`` instead of ``ready_to_merge``. Changed-line coverage
contract (issue #7293): merge-admission snapshots also require a current
``changed-coverage: passed @ <head_sha>`` trailer, or an exact-head
``changed-coverage: not-required @ <head_sha> reason=<code>`` exception.
After the gate verdict is current, a matching ``pr-metadata: reconciled @
<digest>`` trailer is also required so final title/body state is reviewed.

Accepts a compact PR queue snapshot JSON (as emitted by snapshot_pr_queue.py)
or a single-PR mock, and emits JSON or concise text with the next action and
stop reason. Dry-run mode never calls gh or mutates state.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.dev.pr_metadata import extract_metadata_digests
from scripts.dev.route_efficiency_report import (
    EXPECTED_ARTIFACT_KEYS,
    has_validation_success,
    is_complete_artifact_set,
)

DEFAULT_MAX_ACTIONS = 5

VALID_ACTIONS = frozenset(
    {
        "stop",
        "continue",
        "reroute",
        "escalate",
        "wait_ci",
        "inspect_failed_ci",
        "verify_artifacts",
        "refresh_snapshot",
        "mark_ready_candidate",
        "await_gate_verdict",
        "await_changed_coverage",
        "reconcile_pr_metadata",
        "no_action",
    }
)

VALID_FLOW_DECISIONS = frozenset({"stop", "continue", "reroute", "escalate"})

VALID_STATES = frozenset(
    {
        "pending_ci",
        "failed_ci",
        "failed_validation",
        "missing_artifacts",
        "stale_worktree",
        "stale_merge_base",
        "pending_gate_verdict",
        "pending_pr_metadata",
        "pending_changed_coverage",
        "ready_to_merge",
        "no_action",
    }
)

# Minimum overlap (hex chars) required to treat an abbreviated trailer SHA as a
# match for a longer head SHA. Seven mirrors git's default short SHA width.
GATE_VERDICT_MIN_SHA_OVERLAP = 7

# Matches ``gate-verdict: accepted @ <sha>`` trailers embedded in comment or
# review body excerpts, capturing the hex SHA. The verdict word is matched
# case-insensitively so a human- or bot-authored ``Accepted`` still satisfies
# the contract; surrounding markdown/code fences are tolerated.
_GATE_VERDICT_RE = re.compile(
    r"gate-verdict\s*:\s*accepted\s*@\s*([0-9a-fA-F]{7,40})\b",
    re.IGNORECASE,
)
GATE_VERDICT_RE = _GATE_VERDICT_RE
_CHANGED_COVERAGE_RE = re.compile(
    r"changed-coverage\s*:\s*(passed|not-required)\s*@\s*"
    r"([0-9a-fA-F]{7,40})(?![0-9a-fA-F])"
    r"(?:\s+reason=([a-z0-9][a-z0-9._-]*))?",
    re.IGNORECASE,
)
CHANGED_COVERAGE_RE = _CHANGED_COVERAGE_RE
_BASE_POLICY_RE = re.compile(
    r"base-policy\s*:\s*(ordinary-cas|current-base)\s*@\s*([0-9a-fA-F]{7,40})\b",
    re.IGNORECASE,
)
BASE_POLICY_RE = _BASE_POLICY_RE


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """A deterministic policy recommendation for one PR."""

    pr: int
    action: str
    state: str
    flow_decision: str
    reason: str
    actions_remaining: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision as a plain dict."""
        return asdict(self)


def _extract_manifest_compact_artifacts(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return compact artifact records from a routed-worker manifest."""
    compact = manifest.get("compact_artifacts")
    if isinstance(compact, dict):
        return compact
    attempts = manifest.get("attempted_routes")
    if not isinstance(attempts, list):
        return {}
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        if attempt.get("run_dir") == manifest.get("chosen_run_dir"):
            compact = attempt.get("compact_artifacts")
            return compact if isinstance(compact, dict) else {}
    return {}


def _compact_artifacts_present(compact_artifacts: dict[str, Any] | None) -> bool | None:
    """Return compact artifact completeness, or None when no manifest was supplied."""
    if compact_artifacts is None:
        return None
    return is_complete_artifact_set(compact_artifacts)


def _validation_failed(compact_artifacts: dict[str, Any] | None) -> bool:
    """Return True when the validation artifact is present and not successful."""
    if compact_artifacts is None:
        return False
    return has_validation_success(compact_artifacts) is not True


def _artifact_state(
    artifacts: Any,
    *,
    compact_artifacts: dict[str, Any] | None,
) -> str | None:
    """Return a PR-loop state derived from artifact evidence, if any."""
    compact_present = _compact_artifacts_present(compact_artifacts)
    if compact_present is False:
        return "missing_artifacts"
    if _validation_failed(compact_artifacts):
        return "failed_validation"
    if artifacts is not None and not artifacts:
        return "missing_artifacts"
    return None


def _compact_artifacts_from_pr(
    pr: dict[str, Any],
    compact_artifacts: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return explicit compact artifacts or compact artifacts embedded in a PR dict."""
    if compact_artifacts is not None:
        return compact_artifacts
    compact_raw = pr.get("compact_artifacts")
    return compact_raw if isinstance(compact_raw, dict) else None


def _gate_verdict_sha_from_item(item: Any) -> str | None:
    """Return the accepted SHA from one explicit gate-verdict record, if any."""
    if isinstance(item, str):
        return None
    if not isinstance(item, dict):
        return None
    verdict = str(item.get("verdict", "")).lower()
    accepted_flag = item.get("accepted")
    sha = str(item.get("sha") or item.get("head_sha") or "")
    if sha and (verdict == "accepted" or accepted_flag is True):
        return sha
    return None


def _explicit_gate_verdict_texts(pr: dict[str, Any]) -> list[str]:
    """Return synthesized trailer texts from explicit gate-verdict fields."""
    texts: list[str] = []
    explicit_list = pr.get("gate_verdicts")
    if isinstance(explicit_list, list):
        for item in explicit_list:
            if isinstance(item, str):
                texts.append(item)
                continue
            sha = _gate_verdict_sha_from_item(item)
            if sha:
                texts.append(f"gate-verdict: accepted @ {sha}")
    explicit = pr.get("gate_verdict")
    sha = _gate_verdict_sha_from_item(explicit)
    if sha:
        texts.append(f"gate-verdict: accepted @ {sha}")
    return texts


_TRUSTED_GATE_VERDICT_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}


def _snapshot_body_texts(pr: dict[str, Any]) -> list[str]:
    """Return comment/review body excerpts that may carry a gate-verdict trailer."""
    texts: list[str] = []
    for key in ("comment_snapshot", "review_snapshot"):
        snapshot = pr.get(key)
        if not isinstance(snapshot, dict):
            continue
        latest = snapshot.get("latest")
        if not isinstance(latest, list):
            continue
        for entry in latest:
            if not isinstance(entry, dict):
                continue
            association = str(
                entry.get("author_association") or entry.get("authorAssociation") or ""
            ).upper()
            if association not in _TRUSTED_GATE_VERDICT_ASSOCIATIONS:
                continue
            if isinstance(entry.get("body_excerpt"), str):
                texts.append(entry["body_excerpt"])
    return texts


def _gate_verdict_texts(pr: dict[str, Any]) -> list[str]:
    """Return candidate text blobs that may carry a gate-verdict trailer.

    Sources, in priority order:
      - explicit ``gate_verdicts`` list of strings or ``{"sha": ...}`` dicts,
      - explicit ``gate_verdict`` dict (``{"verdict": "accepted", "sha": ...}``
        or ``{"accepted": True, "head_sha": ...}``),
      - compact ``comment_snapshot.latest[].body_excerpt`` blobs,
      - compact ``review_snapshot.latest[].body_excerpt`` blobs.

    The snapshot producer truncates bodies to ``COMMENT_BODY_LIMIT`` (180), which
    is ample for a ``gate-verdict: accepted @ <40-char-sha>`` trailer.
    """
    return _explicit_gate_verdict_texts(pr) + _snapshot_body_texts(pr)


def _accepted_gate_verdict_shas(pr: dict[str, Any]) -> set[str]:
    """Return the set of lowercased SHAs with an accepted gate-verdict trailer."""
    shas: set[str] = set()
    for text in _gate_verdict_texts(pr):
        for match in _GATE_VERDICT_RE.finditer(text):
            shas.add(match.group(1).lower())
    return shas


def _sha_matches_head(trailer_sha: str, head_sha: str) -> bool:
    """Return True when trailer_sha identifies the same commit as head_sha.

    A trailer may carry either a full 40-char SHA or git's abbreviated short
    form. Both are compared case-insensitively; an abbreviated trailer is only
    accepted when it is a prefix of the head SHA with at least
    ``GATE_VERDICT_MIN_SHA_OVERLAP`` hex chars of overlap. Anything shorter is
    ambiguous and fails closed.
    """
    trailer = trailer_sha.lower()
    head = head_sha.lower()
    if not trailer or not head:
        return False
    if trailer == head:
        return True
    if len(trailer) < GATE_VERDICT_MIN_SHA_OVERLAP:
        return False
    return head.startswith(trailer)


def has_current_accepted_gate_verdict(pr: dict[str, Any], head_sha: str) -> bool:
    """Return True iff a current exact-head ``gate-verdict: accepted`` trailer exists.

    Fail-closed by design: an empty head SHA, a missing trailer, or a trailer
    whose SHA does not identify the exact head all return False. This is the
    gate described in issue #6019 — the dispatcher must reject any exact head
    unless every required check is green AND such a trailer is present.
    """
    if not isinstance(pr, dict) or not head_sha:
        return False
    accepted = _accepted_gate_verdict_shas(pr)
    return any(_sha_matches_head(sha, head_sha) for sha in accepted)


def _explicit_changed_coverage_texts(pr: dict[str, Any]) -> list[str]:
    """Return synthesized changed-coverage trailers from explicit snapshot fields."""
    texts: list[str] = []
    explicit_list = pr.get("changed_coverage_verdicts")
    if isinstance(explicit_list, list):
        texts.extend(item for item in explicit_list if isinstance(item, str))
        for item in explicit_list:
            if not isinstance(item, dict):
                continue
            verdict = str(item.get("verdict", "")).lower()
            sha = str(item.get("sha") or item.get("head_sha") or "")
            reason = str(item.get("reason") or "")
            if verdict in {"passed", "not-required"} and sha:
                trailer = f"changed-coverage: {verdict} @ {sha}"
                if reason:
                    trailer += f" reason={reason}"
                texts.append(trailer)
    explicit = pr.get("changed_coverage_verdict")
    if isinstance(explicit, str):
        texts.append(explicit)
    elif isinstance(explicit, dict):
        verdict = str(explicit.get("verdict", "")).lower()
        sha = str(explicit.get("sha") or explicit.get("head_sha") or "")
        reason = str(explicit.get("reason") or "")
        if verdict in {"passed", "not-required"} and sha:
            trailer = f"changed-coverage: {verdict} @ {sha}"
            if reason:
                trailer += f" reason={reason}"
            texts.append(trailer)
    return texts


def _changed_coverage_texts(pr: dict[str, Any]) -> list[str]:
    """Return trusted explicit and compact review/comment changed-coverage text."""
    return _explicit_changed_coverage_texts(pr) + _snapshot_body_texts(pr)


def _changed_coverage_records(pr: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return parsed changed-coverage records as ``(verdict, sha, reason)`` tuples."""
    records: list[tuple[str, str, str]] = []
    for text in _changed_coverage_texts(pr):
        for match in _CHANGED_COVERAGE_RE.finditer(text):
            verdict, sha, reason = match.groups()
            records.append((verdict.lower(), sha.lower(), (reason or "").lower()))
    return records


def current_changed_coverage_verdict(pr: dict[str, Any], head_sha: str) -> tuple[str, str]:
    """Return the exact-head changed-coverage status and exception reason.

    The status is one of ``passed``, ``not_required``, ``missing``, ``stale``,
    ``invalid``, or ``ambiguous``. A ``not-required`` verdict is valid only
    when it carries a machine-readable ``reason=`` token. Multiple conflicting
    current-head verdicts fail closed as ``ambiguous``.
    """
    if not isinstance(pr, dict) or not head_sha:
        return "missing", ""
    records = _changed_coverage_records(pr)
    current = [record for record in records if _sha_matches_head(record[1], head_sha)]
    if not current:
        return ("stale", "") if records else ("missing", "")

    statuses = {(verdict, reason) for verdict, _, reason in current}
    if len(statuses) > 1:
        return "ambiguous", ""
    verdict, _, reason = current[0]
    if verdict == "passed":
        return "passed", ""
    if reason:
        return "not_required", reason
    return "invalid", ""


def has_current_changed_coverage_verdict(pr: dict[str, Any], head_sha: str) -> bool:
    """Return whether exact-head coverage proof or a justified exception exists."""
    status, _ = current_changed_coverage_verdict(pr, head_sha)
    return status in {"passed", "not_required"}


def _explicit_metadata_verdict_texts(pr: dict[str, Any]) -> list[str]:
    """Return synthesized metadata-trailer texts from explicit snapshot fields."""
    texts: list[str] = []
    explicit_list = pr.get("metadata_verdicts")
    if isinstance(explicit_list, list):
        for item in explicit_list:
            if isinstance(item, str):
                texts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            digest = str(item.get("digest") or item.get("metadata_digest") or "")
            verdict = str(item.get("verdict", "")).lower()
            if digest and (verdict in {"accepted", "reconciled"} or item.get("accepted") is True):
                texts.append(f"pr-metadata: reconciled @ {digest}")
    explicit = pr.get("metadata_verdict")
    if isinstance(explicit, str):
        texts.append(explicit)
    elif isinstance(explicit, dict):
        digest = str(explicit.get("digest") or explicit.get("metadata_digest") or "")
        verdict = str(explicit.get("verdict", "")).lower()
        if digest and (verdict in {"accepted", "reconciled"} or explicit.get("accepted") is True):
            texts.append(f"pr-metadata: reconciled @ {digest}")
    return texts


def _metadata_verdict_texts(pr: dict[str, Any]) -> list[str]:
    """Return candidate metadata trailers from trusted explicit/snapshot sources."""
    return _explicit_metadata_verdict_texts(pr) + _snapshot_body_texts(pr)


def _metadata_verdict_digests(pr: dict[str, Any]) -> set[str]:
    """Return lowercased metadata digests from trusted trailer carriers."""
    digests: set[str] = set()
    for text in _metadata_verdict_texts(pr):
        digests.update(extract_metadata_digests(text))
    return digests


def has_any_pr_metadata_verdict(pr: dict[str, Any]) -> bool:
    """Return whether any trusted reconciled metadata trailer is present."""
    return bool(_metadata_verdict_digests(pr))


def has_current_pr_metadata_verdict(pr: dict[str, Any], digest: str) -> bool:
    """Return whether a trusted trailer binds the exact current title/body digest."""
    if not isinstance(pr, dict) or not isinstance(digest, str) or not digest:
        return False
    return digest.lower() in _metadata_verdict_digests(pr)


def _label_names(pr: dict[str, Any]) -> list[str]:
    """Return compact label-name strings from a PR dict."""
    labels = pr.get("labels") or []
    if not isinstance(labels, list):
        return []
    return [str(label) for label in labels]


def _base_freshness_provenance(pr: dict[str, Any]) -> tuple[str, str, str]:
    """Return normalized base-freshness verdict, PR base SHA, and current main SHA.

    ``pr_queue_snapshot.v2`` records authoritative base freshness under the
    nested ``base_freshness`` object. Older snapshots only exposed top-level
    ``base_sha``/``main_sha`` provenance; keep that compatibility path without
    treating legacy missing values as a new hard blocker.
    """
    raw = pr.get("base_freshness")
    if isinstance(raw, dict):
        verdict = str(raw.get("verdict", "") or "")
        base_sha = str(raw.get("base_sha", "") or "")
        current_main_sha = str(raw.get("current_main_sha", "") or "")
        return verdict, base_sha, current_main_sha

    base_sha = str(pr.get("base_sha", "") or "")
    main_sha = str(pr.get("main_sha", "") or "")
    if base_sha and main_sha:
        return ("fresh" if base_sha == main_sha else "stale"), base_sha, main_sha
    return "", base_sha, main_sha


def _base_freshness_state(pr: dict[str, Any]) -> str | None:
    """Return a fail-closed state for blocking base-freshness verdicts."""
    verdict, _, _ = _base_freshness_provenance(pr)
    if verdict in {"stale", "missing-base", "unavailable-current-main"}:
        return "stale_merge_base"
    return None


def _base_policy_texts(pr: dict[str, Any]) -> list[str]:
    """Return explicit and trusted review evidence for the risk-tiered base policy."""
    texts: list[str] = []
    explicit = pr.get("base_policy")
    if isinstance(explicit, str):
        texts.append(explicit)
    elif isinstance(explicit, list):
        texts.extend(item for item in explicit if isinstance(item, str))
    texts.extend(_snapshot_body_texts(pr))
    return texts


def has_current_ordinary_base_policy(pr: dict[str, Any], head_sha: str) -> bool:
    """Return whether trusted exact-head evidence selects the ordinary CAS path."""
    if not isinstance(pr, dict) or not head_sha:
        return False
    for text in _base_policy_texts(pr):
        for match in _BASE_POLICY_RE.finditer(text):
            policy, sha = match.groups()
            if policy.lower() == "ordinary-cas" and _sha_matches_head(sha, head_sha):
                return True
    return False


def _merge_ready_state(
    pr: dict[str, Any],
    *,
    label_names: list[str],
    overall: str,
    head_sha: str,
) -> str | None:
    """Return the merge-readiness state for a green, merge-ready PR, or None.

    Risk-tiered stale base (issue #6272): missing or unavailable snapshot
    provenance, and stale base-sensitive PRs, fail closed as
    ``stale_merge_base``. A stale ordinary PR may proceed only when trusted
    exact-head evidence records ``base-policy: ordinary-cas @ <head_sha>``;
    the guarded merger then performs the immediate current-main CAS.

    Gate-verdict contract (issue #6019): reject any exact head unless a current
    exact-head ``gate-verdict: accepted @ <head_sha>`` trailer exists. The final
    title/body pair must also have a current ``pr-metadata: reconciled @
    <digest>`` trailer. Changed-line coverage must also be current when the
    snapshot marks ``changed_coverage_required``; docs-only and other explicit
    exceptions use a reasoned ``not-required`` trailer. Fail closed when any
    required evidence is missing or stale.
    """
    if "merge-ready" not in label_names or overall != "success":
        return None
    base_state = _base_freshness_state(pr)
    base_verdict, _, _ = _base_freshness_provenance(pr)
    if (
        base_state == "stale_merge_base"
        and base_verdict == "stale"
        and has_current_ordinary_base_policy(pr, head_sha)
    ):
        base_state = None
    if base_state is not None:
        return base_state
    if pr.get("review_threads_admission") == "fail_closed_unknown":
        return "unknown_review_threads"
    if not has_current_accepted_gate_verdict(pr, head_sha):
        return "pending_gate_verdict"
    metadata_digest = str(pr.get("metadata_digest", "") or "")
    if not has_current_pr_metadata_verdict(pr, metadata_digest):
        return "pending_pr_metadata"
    if pr.get("changed_coverage_required") is True:
        coverage_status, _ = current_changed_coverage_verdict(pr, head_sha)
        if coverage_status not in {"passed", "not_required"}:
            return "pending_changed_coverage"
    return "ready_to_merge"


def classify_pr_state(
    pr: dict[str, Any],
    *,
    compact_artifacts: dict[str, Any] | None = None,
) -> str:
    """Classify a single PR into a machine-checkable loop state.

    Pure function: no side effects, no GitHub calls.
    """
    if not isinstance(pr, dict):
        return "no_action"
    compact_artifacts = _compact_artifacts_from_pr(pr, compact_artifacts)
    status = str(pr.get("status", ""))
    if status == "error":
        return "no_action"
    checks = pr.get("checks") or {}
    overall = str(checks.get("overall", ""))
    label_names = _label_names(pr)
    is_draft = bool(pr.get("draft", False))
    head_sha = str(pr.get("head_sha", ""))
    expected = str(pr.get("expected_head_sha", ""))
    artifacts = pr.get("artifacts")
    if is_draft:
        return "no_action"
    if overall == "failure":
        return "failed_ci"
    if overall == "pending":
        return "pending_ci"
    if expected and head_sha and head_sha != expected:
        return "stale_worktree"
    artifact_state = _artifact_state(artifacts, compact_artifacts=compact_artifacts)
    if artifact_state is not None:
        return artifact_state
    return (
        _merge_ready_state(
            pr,
            label_names=label_names,
            overall=overall,
            head_sha=head_sha,
        )
        or "no_action"
    )


def _review_state(pr: dict[str, Any]) -> str:
    """Extract review state string from a PR dict.

    Supports two formats:
      - scalar: ``review_state`` or ``review`` field with a single string value.
      - dict: ``reviews`` field mapping review conclusions to counts, e.g.
        ``{"CHANGES_REQUESTED": 1, "APPROVED": 1}``.

    Returns the uppercased review state or empty string if absent.
    Priority: CHANGES_REQUESTED > CHANGES_REQUESTED (via scalar) > any other dict
    key with count > 0 > scalar value > empty.
    """
    raw = pr.get("review_state") or pr.get("review") or ""
    reviews_dict = pr.get("reviews")
    if isinstance(reviews_dict, dict):
        if reviews_dict.get("CHANGES_REQUESTED", 0) > 0:
            return "CHANGES_REQUESTED"
        for key, count in reviews_dict.items():
            if isinstance(count, (int, float)) and count > 0:
                return str(key).upper()
    return str(raw).upper() if raw else ""


def _compute_flow_decision(
    state: str,
    *,
    review_state: str,
    budget_exhausted: bool,
) -> str:
    """Map classified state + review state to a high-level flow decision.

    Exactly one of: stop, continue, reroute, escalate.

    Deterministic rules:
      - budget_exhausted -> stop
      - CHANGES_REQUESTED -> escalate (review blocker overrides other routing)
      - pending_ci -> continue
      - pending_gate_verdict -> continue (wait for current exact-head gate verdict)
      - pending_pr_metadata -> continue (reconcile final title/body metadata)
      - pending_changed_coverage -> continue (obtain exact-head coverage proof)
      - ready_to_merge -> continue
      - failed_ci, failed_validation, missing_artifacts, stale_worktree -> reroute
      - no_action -> stop
    """
    if budget_exhausted:
        return "stop"
    if review_state == "CHANGES_REQUESTED":
        return "escalate"
    match state:
        case (
            "pending_ci"
            | "pending_gate_verdict"
            | "pending_pr_metadata"
            | "pending_changed_coverage"
            | "ready_to_merge"
        ):
            return "continue"
        case "unknown_review_threads":
            return "continue"
        case (
            "failed_ci"
            | "failed_validation"
            | "missing_artifacts"
            | "stale_worktree"
            | "stale_merge_base"
        ):
            return "reroute"
        case _:
            return "stop"


def recommend_action(  # noqa: C901
    state: str,
    *,
    pr_number: int,
    actions_remaining: int,
    has_merge_ready: bool = False,
    ci_success: bool = False,
    review_state: str = "",
    stale_base_sha: str = "",
    current_main_sha: str = "",
) -> PolicyDecision:
    """Map a classified state to a deterministic next action.

    Pure function: no side effects.
    """
    budget_exhausted = actions_remaining <= 0
    flow_decision = _compute_flow_decision(
        state,
        review_state=review_state,
        budget_exhausted=budget_exhausted,
    )
    if budget_exhausted:
        return PolicyDecision(
            pr=pr_number,
            action="stop",
            state=state,
            flow_decision="stop",
            reason="loop budget exhausted",
            actions_remaining=0,
        )
    remaining = actions_remaining - 1
    if review_state == "CHANGES_REQUESTED":
        return PolicyDecision(
            pr=pr_number,
            action="escalate",
            state=state,
            flow_decision=flow_decision,
            reason="review changes requested; escalate before continuing loop automation",
            actions_remaining=remaining,
        )
    match state:
        case "pending_ci":
            return PolicyDecision(
                pr=pr_number,
                action="wait_ci",
                state=state,
                flow_decision=flow_decision,
                reason="CI checks still pending",
                actions_remaining=remaining,
            )
        case "failed_ci":
            return PolicyDecision(
                pr=pr_number,
                action="inspect_failed_ci",
                state=state,
                flow_decision=flow_decision,
                reason="CI checks failed; inspect failures before retry",
                actions_remaining=remaining,
            )
        case "failed_validation":
            return PolicyDecision(
                pr=pr_number,
                action="verify_artifacts",
                state=state,
                flow_decision=flow_decision,
                reason="validation artifact present but reports failure",
                actions_remaining=remaining,
            )
        case "missing_artifacts":
            return PolicyDecision(
                pr=pr_number,
                action="verify_artifacts",
                state=state,
                flow_decision=flow_decision,
                reason="required artifacts not present",
                actions_remaining=remaining,
            )
        case "stale_worktree":
            return PolicyDecision(
                pr=pr_number,
                action="refresh_snapshot",
                state=state,
                flow_decision=flow_decision,
                reason="PR head SHA does not match expected snapshot",
                actions_remaining=remaining,
            )
        case "ready_to_merge":
            return PolicyDecision(
                pr=pr_number,
                action="mark_ready_candidate",
                state=state,
                flow_decision=flow_decision,
                reason="CI green, merge-ready label, and current exact-head gate verdict present",
                actions_remaining=remaining,
            )
        case "pending_gate_verdict":
            return PolicyDecision(
                pr=pr_number,
                action="await_gate_verdict",
                state=state,
                flow_decision=flow_decision,
                reason=(
                    "CI green and merge-ready but no current exact-head "
                    "gate-verdict: accepted trailer; reject head"
                ),
                actions_remaining=remaining,
            )
        case "pending_pr_metadata":
            return PolicyDecision(
                pr=pr_number,
                action="reconcile_pr_metadata",
                state=state,
                flow_decision=flow_decision,
                reason=(
                    "CI green, merge-ready, and current gate verdict exist but the final PR "
                    "title/body lacks a matching trusted pr-metadata trailer"
                ),
                actions_remaining=remaining,
            )
        case "pending_changed_coverage":
            return PolicyDecision(
                pr=pr_number,
                action="await_changed_coverage",
                state=state,
                flow_decision=flow_decision,
                reason=(
                    "CI green, merge-ready, and current gate/metadata verdicts exist but "
                    "exact-head changed-line coverage proof or a reasoned exception is missing"
                ),
                actions_remaining=remaining,
            )
        case "unknown_review_threads":
            return PolicyDecision(
                pr=pr_number,
                action="await_review_threads",
                state=state,
                flow_decision=flow_decision,
                reason=(
                    "REST fallback cannot refresh GraphQL-only review threads; "
                    "retry after the GraphQL quota resets"
                ),
                actions_remaining=remaining,
            )
        case "stale_merge_base":
            stale = stale_base_sha or "?"
            current = current_main_sha or "?"
            return PolicyDecision(
                pr=pr_number,
                action="refresh_snapshot",
                state=state,
                flow_decision=flow_decision,
                reason=(
                    f"merge base SHA {stale} does not match "
                    f"current main SHA {current}; "
                    "PR must be rebased onto main"
                ),
                actions_remaining=remaining,
            )
        case _:
            return PolicyDecision(
                pr=pr_number,
                action="no_action",
                state="no_action",
                flow_decision=flow_decision,
                reason="nothing actionable for this PR",
                actions_remaining=remaining,
            )


def _pr_number(pr: dict[str, Any]) -> int:
    """Extract PR number safely."""
    try:
        return int(pr.get("number", 0))
    except (TypeError, ValueError):
        return 0


def evaluate_queue(
    prs: list[dict[str, Any]],
    *,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    expected_head_shas: dict[int, str] | None = None,
    artifact_presence: dict[int, bool] | None = None,
    compact_artifacts: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate a PR queue and emit per-PR decisions under a loop budget.

    Pure function: reads snapshot dicts, never calls external APIs.
    """
    decisions: list[dict[str, Any]] = []
    actions_used = 0
    expected_shas = expected_head_shas or {}
    artifacts = artifact_presence or {}
    compact_by_pr = compact_artifacts or {}
    for pr in prs:
        num = _pr_number(pr)
        enriched: dict[str, Any] = dict(pr)
        if num in expected_shas:
            enriched["expected_head_sha"] = expected_shas[num]
        if num in artifacts:
            enriched["artifacts"] = artifacts[num]
        compact = compact_by_pr.get(num)
        if compact is not None:
            enriched["compact_artifacts"] = compact
        state = classify_pr_state(enriched, compact_artifacts=compact)
        review = _review_state(enriched)
        labels = enriched.get("labels") or []
        label_names = [str(label) for label in labels] if isinstance(labels, list) else []
        checks = enriched.get("checks") or {}
        has_merge = "merge-ready" in label_names
        ci_ok = str(checks.get("overall", "")) == "success"
        remaining = max_actions - actions_used
        if remaining <= 0:
            decisions.append(
                PolicyDecision(
                    pr=num,
                    action="stop",
                    state="no_action",
                    flow_decision="stop",
                    reason="loop budget exhausted",
                    actions_remaining=0,
                ).to_dict()
            )
            break
        stale_base = ""
        current_main = ""
        if state == "stale_merge_base":
            _, stale_base, current_main = _base_freshness_provenance(enriched)
        decision = recommend_action(
            state,
            pr_number=num,
            actions_remaining=remaining,
            has_merge_ready=has_merge,
            ci_success=ci_ok,
            review_state=review,
            stale_base_sha=stale_base,
            current_main_sha=current_main,
        )
        decisions.append(decision.to_dict())
        actions_used += 1
    return {
        "schema": "pr_loop_policy.v1",
        "max_actions": max_actions,
        "actions_used": actions_used,
        "decisions": decisions,
    }


def format_text(result: dict[str, Any]) -> str:
    """Format a compact human-readable policy summary."""
    lines = [
        f"max_actions: {result['max_actions']}  actions_used: {result['actions_used']}",
    ]
    for d in result["decisions"]:
        lines.append(
            f"PR #{d['pr']}: {d['action']} (state={d['state']}, flow={d['flow_decision']}) "
            f"— {d['reason']} [remaining={d['actions_remaining']}]"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--snapshot",
        help="Path to a compact PR queue snapshot JSON file.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read snapshot JSON from stdin instead of a file.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=DEFAULT_MAX_ACTIONS,
        help=f"Loop budget: maximum actions before stop (default {DEFAULT_MAX_ACTIONS}).",
    )
    parser.add_argument(
        "--expected-sha",
        nargs="*",
        metavar="PR=SHA",
        help="Expected head SHAs as PR=SHA pairs for staleness detection.",
    )
    parser.add_argument(
        "--artifact-present",
        nargs="*",
        metavar="PR=true|false",
        help="Artifact presence as PR=true|false pairs.",
    )
    parser.add_argument(
        "--manifest",
        nargs="*",
        metavar="PR=PATH",
        help="Routed-worker manifest paths as PR=PATH pairs; overrides --artifact-present.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def _parse_pairs(pairs: list[str] | None, *, as_bool: bool = False) -> dict[int, Any]:
    """Parse PR=VALUE pairs into a dict."""
    result: dict[int, Any] = {}
    if not pairs:
        return result
    for pair in pairs:
        if "=" not in pair:
            continue
        key_str, _, value = pair.partition("=")
        try:
            key = int(key_str)
        except ValueError:
            continue
        if as_bool:
            result[key] = value.lower() in ("true", "1", "yes")
        else:
            result[key] = value
    return result


def _resolve_manifest_path(path_str: str, *, target_repo: Path) -> Path:
    """Resolve a manifest path inside the target repository."""
    path = Path(path_str)
    unresolved = path if path.is_absolute() else target_repo / path
    if unresolved.is_symlink():
        raise ValueError("manifest path must not be a symlink")
    resolved = unresolved.resolve(strict=False)
    if not resolved.is_relative_to(target_repo.resolve()):
        raise ValueError("manifest path must resolve inside target repository")
    return resolved


def _read_compact_artifact_text(
    artifact_path: str,
    *,
    target_repo: Path,
    max_chars: int = 4000,
) -> str | None:
    """Read a compact artifact text file from inside the target repository."""
    try:
        resolved = _resolve_manifest_path(artifact_path, target_repo=target_repo)
    except ValueError:
        return None
    if not resolved.is_file():
        return None
    try:
        return resolved.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError:
        return None


def _hydrate_validation_result(
    compact: dict[str, Any],
    *,
    target_repo: Path,
) -> dict[str, Any]:
    """Populate validation.result from the compact validation artifact file when absent."""
    validation = compact.get("validation")
    if not isinstance(validation, dict) or isinstance(validation.get("result"), str):
        return compact
    path = validation.get("path")
    if not isinstance(path, str):
        return compact
    result = _read_compact_artifact_text(path, target_repo=target_repo)
    if result is None:
        return compact
    hydrated = dict(compact)
    hydrated["validation"] = {**validation, "result": result}
    return hydrated


def load_manifest_artifacts(
    manifest_pairs: list[str] | None,
    *,
    target_repo: str | Path = ".",
) -> tuple[dict[int, bool], dict[int, dict[str, Any]], list[str]]:
    """Load routed-worker manifest pairs into PR artifact policy inputs."""
    artifact_presence: dict[int, bool] = {}
    compact_artifacts: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    repo_root = Path(target_repo).resolve()
    if not manifest_pairs:
        return artifact_presence, compact_artifacts, warnings

    for pair in manifest_pairs:
        if "=" not in pair:
            warnings.append(f"ignored malformed manifest pair: {pair}")
            continue
        key_str, _, path_str = pair.partition("=")
        try:
            pr_number = int(key_str)
        except ValueError:
            warnings.append(f"ignored manifest pair with non-integer PR: {pair}")
            continue
        try:
            manifest_path = _resolve_manifest_path(path_str, target_repo=repo_root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"ignored manifest for PR {pr_number}: {exc}")
            continue
        if not isinstance(manifest, dict):
            warnings.append(f"ignored manifest for PR {pr_number}: JSON root is not an object")
            continue
        compact = _hydrate_validation_result(
            _extract_manifest_compact_artifacts(manifest),
            target_repo=repo_root,
        )
        compact_artifacts[pr_number] = compact
        artifact_presence[pr_number] = is_complete_artifact_set(compact)
        missing = sorted(
            key
            for key in EXPECTED_ARTIFACT_KEYS
            if not isinstance(compact.get(key), dict) or compact[key].get("present") is not True
        )
        if missing:
            warnings.append(f"manifest for PR {pr_number} missing artifacts: {', '.join(missing)}")
    return artifact_presence, compact_artifacts, warnings


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    raw: dict[str, Any] | list[dict[str, Any]] | None = None
    if args.stdin:
        try:
            raw = json.load(sys.stdin)
        except json.JSONDecodeError as exc:
            print(f"invalid JSON on stdin: {exc}", file=sys.stderr)
            return 1
    elif args.snapshot:
        try:
            with open(args.snapshot) as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            print(f"file not found: {args.snapshot}", file=sys.stderr)
            return 1
        except json.JSONDecodeError as exc:
            print(f"invalid JSON in {args.snapshot}: {exc}", file=sys.stderr)
            return 1
    else:
        print("provide --snapshot or --stdin", file=sys.stderr)
        return 1
    if isinstance(raw, dict) and "prs" in raw:
        prs: list[dict[str, Any]] = raw["prs"]
    elif isinstance(raw, list):
        prs = raw
    else:
        print("snapshot must contain a 'prs' array or be a JSON array of PR dicts", file=sys.stderr)
        return 1
    expected_shas = _parse_pairs(args.expected_sha)
    artifact_presence = _parse_pairs(args.artifact_present, as_bool=True)
    manifest_presence, compact_artifacts, manifest_warnings = load_manifest_artifacts(
        args.manifest,
        target_repo=Path.cwd(),
    )
    artifact_presence.update(manifest_presence)
    for warning in manifest_warnings:
        print(warning, file=sys.stderr)
    result = evaluate_queue(
        prs,
        max_actions=args.max_actions,
        expected_head_shas=expected_shas,
        artifact_presence=artifact_presence,
        compact_artifacts=compact_artifacts,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(format_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
