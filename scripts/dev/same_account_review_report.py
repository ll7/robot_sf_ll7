#!/usr/bin/env python3
"""Project app-custodied same-account implementation reviews into static reports.

Issue #8677 keeps the existing ``static_report`` receipt carrier and uses the
already installed OpenAI ChatGPT Codex Connector GitHub App only as authenticated
producer/custody provenance.  The visible repository-owner account remains the
publisher.  Direct owner prose, copied markers, edited comments, and caller-
injected approval flags are never promoted by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

APPROVED_APP_ID = 1_144_995
APPROVED_APP_SLUG = "chatgpt-codex-connector"
APPROVED_APP_OWNER = "openai"
APPROVED_PRODUCER_IDENTITY = f"{APPROVED_APP_OWNER}/{APPROVED_APP_SLUG}"
REPORT_HEADING = "## Independent implementation review"
REPORT_SECTION_HEADINGS = ("### Scope", "### Findings", "### Validation")
_REPORT_PREFIX_RE = re.compile(r"(?m)^single-account-review:\s*")
_REPORT_BLOCK_RE = re.compile(
    r"(?m)^single-account-review: "
    r"(?P<verdict>accepted|changes_requested) @ (?P<head>[0-9a-fA-F]{40})$\n"
    r"^metadata: (?P<metadata>[0-9a-fA-F]{64})$\n"
    r"^evidence: (?P<evidence>[0-9a-fA-F]{64})$\n"
    r"^unresolved-correctness-findings: (?P<findings>0|[1-9][0-9]*)$"
)
_HEAD_LINE_RE = re.compile(
    r"(?m)^single-account-review:\s*(?:accepted|changes_requested)\s*@\s*"
    r"(?P<value>[0-9a-fA-F]{40})\s*$"
)
_METADATA_LINE_RE = re.compile(r"(?m)^metadata:\s*(?P<value>[0-9a-fA-F]{64})\s*$")
_EVIDENCE_LINE_RE = re.compile(r"(?m)^evidence:\s*(?P<value>[0-9a-fA-F]{64})\s*$")
_MAX_COMMENT_PAGES = 10
_PAGE_SIZE = 100


class GhRunner(Protocol):
    """Minimal ``gh`` runner contract used by the production gate and tests."""

    def __call__(self, args: list[str], timeout: int = 30) -> Any:
        """Run one GitHub CLI invocation."""
        ...


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON bytes for one evidence binding."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _repository_parts(repository: str) -> tuple[str, str]:
    """Return a normalized ``owner/repository`` pair or raise ``ValueError``."""
    owner, separator, name = repository.strip().partition("/")
    if separator != "/" or not owner or not name or "/" in name:
        raise ValueError(f"invalid repository identifier: {repository!r}")
    return owner.lower(), name.lower()


def _report_match(body: str) -> re.Match[str]:
    """Return the one canonical review marker block or raise ``ValueError``."""
    matches = list(_REPORT_BLOCK_RE.finditer(body))
    if len(matches) != 1:
        raise ValueError("review report must contain exactly one canonical marker block")
    return matches[0]


def _validate_report_shape(body: str, match: re.Match[str]) -> None:
    """Require an implementation assessment and internally consistent verdict."""
    marker_start = match.start()
    report_text = body[:marker_start].strip()
    missing_headings = [
        heading for heading in (REPORT_HEADING, *REPORT_SECTION_HEADINGS) if heading not in report_text
    ]
    if missing_headings:
        joined = ", ".join(missing_headings)
        raise ValueError(f"review report is missing required headings: {joined}")
    if len(report_text) < 160:
        raise ValueError("review report is too short to establish an implementation assessment")

    verdict = match.group("verdict")
    findings = int(match.group("findings"))
    if verdict == "accepted" and findings != 0:
        raise ValueError("accepted review report must declare zero unresolved correctness findings")
    if verdict == "changes_requested" and findings == 0:
        raise ValueError("changes-requested report must declare at least one unresolved finding")


def _normalized_report_body(body: str, match: re.Match[str]) -> str:
    """Replace the self-referential evidence value with a stable zero placeholder."""
    start, end = match.span("evidence")
    return f"{body[:start]}{'0' * 64}{body[end:]}"


def report_evidence_digest(*, body: str, repository: str, pr_number: int) -> str:
    """Compute the original report digest bound to the approved custody route.

    The report text is bound to its declared head, metadata epoch, verdict and
    unresolved-finding count, plus the exact repository/PR, publisher, and fixed
    existing GitHub App identity.  The evidence field itself is normalized to
    zeroes before hashing so the producer can bind the report before publishing
    it once through GitHub.
    """
    owner, name = _repository_parts(repository)
    if type(pr_number) is not int or pr_number < 1:
        raise ValueError("PR number must be a positive integer")
    match = _report_match(body)
    _validate_report_shape(body, match)
    payload = {
        "app": {
            "id": APPROVED_APP_ID,
            "owner": APPROVED_APP_OWNER,
            "slug": APPROVED_APP_SLUG,
        },
        "head_sha": match.group("head").lower(),
        "metadata_digest": match.group("metadata").lower(),
        "pr_number": pr_number,
        "producer_identity": APPROVED_PRODUCER_IDENTITY,
        "publisher_identity": owner,
        "report_body": _normalized_report_body(body, match),
        "repository": f"{owner}/{name}",
        "unresolved_correctness_findings": int(match.group("findings")),
        "verdict": match.group("verdict"),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def bind_report_evidence(*, body: str, repository: str, pr_number: int) -> str:
    """Return ``body`` with its canonical evidence digest inserted once."""
    match = _report_match(body)
    digest = report_evidence_digest(body=body, repository=repository, pr_number=pr_number)
    start, end = match.span("evidence")
    return f"{body[:start]}{digest}{body[end:]}"


def _string(value: Any) -> str:
    """Return a stripped string or an empty string."""
    return value.strip() if isinstance(value, str) else ""


def _nested_mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping view or an empty mapping."""
    return value if isinstance(value, Mapping) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse a timezone-aware GitHub timestamp."""
    text = _string(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _is_target_app(comment: Mapping[str, Any]) -> bool:
    """Return whether a comment claims the existing approved app tuple."""
    app = _nested_mapping(comment.get("performed_via_github_app"))
    owner = _nested_mapping(app.get("owner"))
    return (
        app.get("id") == APPROVED_APP_ID
        and _string(app.get("slug")).lower() == APPROVED_APP_SLUG
        and _string(owner.get("login")).lower() == APPROVED_APP_OWNER
    )


def _looks_like_report(comment: Mapping[str, Any]) -> bool:
    """Return whether an app comment enters the static-report supersession lane."""
    body = _string(comment.get("body"))
    return bool(body and (REPORT_HEADING in body or _REPORT_PREFIX_RE.search(body)))


def _comment_order(comment: Mapping[str, Any]) -> tuple[datetime, int] | None:
    """Return an immutable newest-report ordering key when both fields validate."""
    created_at = _parse_timestamp(comment.get("created_at"))
    comment_id = comment.get("id")
    if created_at is None or type(comment_id) is not int or comment_id < 1:
        return None
    return created_at, comment_id


def _custody_reason(
    comment: Mapping[str, Any], *, repository: str, pr_number: int
) -> str | None:
    """Validate immutable GitHub source/custody fields for one app report."""
    owner, name = _repository_parts(repository)
    comment_id = comment.get("id")
    if type(comment_id) is not int or comment_id < 1:
        return "static_report_comment_id_malformed"
    if not _string(comment.get("node_id")):
        return "static_report_comment_node_id_missing"
    if not _is_target_app(comment):
        return "static_report_app_identity_unapproved"

    publisher = _nested_mapping(comment.get("user"))
    if _string(publisher.get("login")).lower() != owner:
        return "static_report_publisher_not_repository_owner"

    api_root = f"https://api.github.com/repos/{owner}/{name}"
    expected_api_url = f"{api_root}/issues/comments/{comment_id}"
    expected_issue_url = f"{api_root}/issues/{pr_number}"
    expected_html_url = (
        f"https://github.com/{owner}/{name}/pull/{pr_number}#issuecomment-{comment_id}"
    )
    if _string(comment.get("url")).lower() != expected_api_url:
        return "static_report_comment_url_mismatch"
    if _string(comment.get("issue_url")).lower() != expected_issue_url:
        return "static_report_pr_url_mismatch"
    if _string(comment.get("html_url")).lower() != expected_html_url:
        return "static_report_html_url_mismatch"

    created_at = _parse_timestamp(comment.get("created_at"))
    updated_at = _parse_timestamp(comment.get("updated_at"))
    if created_at is None or updated_at is None:
        return "static_report_timestamp_malformed"
    if created_at != updated_at:
        return "static_report_comment_edited"
    if comment.get("minimized") is True:
        return "static_report_comment_minimized"
    return None


def _partial_marker_value(pattern: re.Pattern[str], body: str) -> str:
    """Extract one unambiguous marker value for fail-closed diagnostics."""
    matches = list(pattern.finditer(body))
    return matches[0].group("value") if len(matches) == 1 else ""


def _project_comment(
    comment: Mapping[str, Any], *, repository: str, pr_number: int
) -> dict[str, Any]:
    """Project one app report without manufacturing missing bindings."""
    owner, _name = _repository_parts(repository)
    raw_body = comment.get("body")
    body = raw_body if isinstance(raw_body, str) else ""
    custody_reason = _custody_reason(comment, repository=repository, pr_number=pr_number)

    record: dict[str, Any] = {
        "identity": APPROVED_PRODUCER_IDENTITY,
        "publisher_identity": owner,
        "approved_source": custody_reason is None,
        "head_sha": _partial_marker_value(_HEAD_LINE_RE, body),
        "metadata_digest": _partial_marker_value(_METADATA_LINE_RE, body),
        "evidence_digest": _partial_marker_value(_EVIDENCE_LINE_RE, body),
        "verdict": "malformed",
        "source_comment_id": comment.get("id"),
        "source_url": _string(comment.get("html_url")),
        "producer": {
            "app_id": APPROVED_APP_ID,
            "app_owner": APPROVED_APP_OWNER,
            "app_slug": APPROVED_APP_SLUG,
        },
        "custody": {
            "created_at": comment.get("created_at"),
            "updated_at": comment.get("updated_at"),
            "status": "accepted" if custody_reason is None else "unavailable",
            "reason_code": custody_reason,
        },
        "body": body,
    }

    try:
        match = _report_match(body)
        _validate_report_shape(body, match)
    except ValueError as exc:
        record["projection_reason"] = str(exc)
        return record

    record.update(
        {
            "head_sha": match.group("head"),
            "metadata_digest": match.group("metadata"),
            "evidence_digest": match.group("evidence"),
            "unresolved_correctness_findings": int(match.group("findings")),
        }
    )
    if custody_reason is not None:
        record["verdict"] = match.group("verdict")
        return record

    expected_digest = report_evidence_digest(
        body=body,
        repository=repository,
        pr_number=pr_number,
    )
    if match.group("evidence").lower() != expected_digest:
        record["verdict"] = "invalid_evidence"
        record["projection_reason"] = "static_report_evidence_digest_mismatch"
        return record

    record["verdict"] = match.group("verdict")
    return record


def project_static_reports_from_comments(
    comments: Sequence[Mapping[str, Any]], *, repository: str, pr_number: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project target-app reports, preserving newest-report supersession.

    Only comments carrying the exact existing GitHub App tuple enter this lane.
    A malformed ordering field makes the lane unavailable instead of allowing an
    older acceptance to survive an unorderable newer report.
    """
    owner, _name = _repository_parts(repository)
    candidates = [
        comment for comment in comments if _is_target_app(comment) and _looks_like_report(comment)
    ]
    provenance: dict[str, Any] = {
        "source": "rest_issue_comments",
        "status": "missing",
        "producer_identity": APPROVED_PRODUCER_IDENTITY,
        "publisher_identity": owner,
        "approved_app": {
            "id": APPROVED_APP_ID,
            "owner": APPROVED_APP_OWNER,
            "slug": APPROVED_APP_SLUG,
        },
        "candidate_count": len(candidates),
        "selected_comment_id": None,
        "reason_codes": [],
    }
    if not candidates:
        return [], provenance

    ordered: list[tuple[tuple[datetime, int], Mapping[str, Any]]] = []
    for comment in candidates:
        key = _comment_order(comment)
        if key is None:
            provenance["status"] = "unavailable"
            provenance["reason_codes"] = ["static_report_ordering_malformed"]
            return (
                [
                    {
                        "identity": APPROVED_PRODUCER_IDENTITY,
                        "approved_source": False,
                        "head_sha": "",
                        "metadata_digest": "",
                        "evidence_digest": "",
                        "verdict": "malformed",
                        "projection_reason": "static_report_ordering_malformed",
                    }
                ],
                provenance,
            )
        ordered.append((key, comment))

    ordered.sort(key=lambda item: item[0])
    projected = [
        _project_comment(comment, repository=repository, pr_number=pr_number)
        for _key, comment in ordered
    ]
    for report in projected[:-1]:
        report["superseded"] = True
    latest = projected[-1]
    provenance["selected_comment_id"] = latest.get("source_comment_id")
    custody = _nested_mapping(latest.get("custody"))
    if latest.get("approved_source") is not True:
        provenance["status"] = "unavailable"
        reason = _string(custody.get("reason_code")) or _string(latest.get("projection_reason"))
        provenance["reason_codes"] = [reason or "static_report_custody_unavailable"]
    elif latest.get("verdict") in {"accepted", "changes_requested"}:
        provenance["status"] = "accepted"
    else:
        provenance["status"] = "malformed"
        reason = _string(latest.get("projection_reason"))
        provenance["reason_codes"] = [reason or "static_report_projection_malformed"]
    return projected, provenance


def _fetch_rest_comments(
    gh: GhRunner, *, repository: str, pr_number: int
) -> tuple[list[Mapping[str, Any]] | None, str | None]:
    """Fetch the complete PR issue-comment collection through bounded REST pages."""
    _repository_parts(repository)
    if type(pr_number) is not int or pr_number < 1:
        return None, "static_report_pr_number_malformed"
    rows: list[Mapping[str, Any]] = []
    for page in range(1, _MAX_COMMENT_PAGES + 1):
        endpoint = (
            f"repos/{repository}/issues/{pr_number}/comments?"
            f"per_page={_PAGE_SIZE}&page={page}"
        )
        result = gh(["api", endpoint], timeout=45)
        if getattr(result, "returncode", 1) != 0:
            diagnostic = _string(getattr(result, "stderr", ""))
            return None, diagnostic or "static_report_comment_fetch_failed"
        try:
            payload = json.loads(str(getattr(result, "stdout", "")))
        except json.JSONDecodeError:
            return None, "static_report_comment_response_not_json"
        if not isinstance(payload, list) or any(not isinstance(item, Mapping) for item in payload):
            return None, "static_report_comment_response_malformed"
        rows.extend(payload)
        if len(payload) < _PAGE_SIZE:
            return rows, None
    return None, "static_report_comment_pagination_incomplete"


def fetch_same_account_static_reports(
    gh: GhRunner, *, repository: str, pr_number: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch and project production same-account static-report evidence.

    Source unavailability is represented in provenance and returns no approved
    report.  Other carrier classes can still be evaluated by the receipt; this
    route alone never makes the entire PR snapshot authoritative.
    """
    owner, _name = _repository_parts(repository)
    comments, error = _fetch_rest_comments(gh, repository=repository, pr_number=pr_number)
    if error or comments is None:
        return [], {
            "source": "rest_issue_comments",
            "status": "unavailable",
            "producer_identity": APPROVED_PRODUCER_IDENTITY,
            "publisher_identity": owner,
            "approved_app": {
                "id": APPROVED_APP_ID,
                "owner": APPROVED_APP_OWNER,
                "slug": APPROVED_APP_SLUG,
            },
            "candidate_count": 0,
            "selected_comment_id": None,
            "reason_codes": [error or "static_report_comment_fetch_failed"],
        }
    return project_static_reports_from_comments(
        comments,
        repository=repository,
        pr_number=pr_number,
    )


def _read_text(path: str) -> str:
    """Read report text from a path or stdin when ``path`` is ``-``."""
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _write_text(path: str, text: str) -> None:
    """Write report text to a path or stdout when ``path`` is ``-``."""
    if path == "-":
        sys.stdout.write(text)
        return
    Path(path).write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Bind or verify one static-report evidence block before publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bind", "check"))
    parser.add_argument("--repo", required=True, help="repository in owner/name form")
    parser.add_argument("--pr", required=True, type=int, help="pull-request number")
    parser.add_argument("--input", default="-", help="input Markdown path, or - for stdin")
    parser.add_argument("--output", default="-", help="output path for bind, or - for stdout")
    args = parser.parse_args(argv)

    try:
        body = _read_text(args.input)
        if args.mode == "bind":
            _write_text(
                args.output,
                bind_report_evidence(body=body, repository=args.repo, pr_number=args.pr),
            )
            return 0
        match = _report_match(body)
        expected = report_evidence_digest(body=body, repository=args.repo, pr_number=args.pr)
        if match.group("evidence").lower() != expected:
            print("static report evidence digest mismatch", file=sys.stderr)
            return 1
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("static report evidence binding: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
