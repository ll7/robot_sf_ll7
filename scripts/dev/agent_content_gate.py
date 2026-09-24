#!/usr/bin/env python3
"""Canonical fail-closed author trust gate for agent content ingestion.

Plain-language summary
----------------------
Agent workflows read issue bodies, pull-request bodies, comments, and review
comments and then place that text into prompts, capsules, and loop decisions.
Only content authored by our own operator account (``ll7``) is auto-ingested.
Every other author, including bots and GitHub Actions accounts, is untrusted by
default. Our own user can explicitly opt foreign content in with a flag, and
that opt-in is recorded with attribution.

The gate fails closed: missing, deleted, or ambiguous author metadata is never
trusted.

Flags
-----
- Label ``agent:digest`` on the issue or pull request. The label opts the whole
  thread in; the receipt records ``label:agent:digest`` as the flag source.
- An own-user comment or review body containing ``agent-digest: allow``. The
  marker is honored only when the author is our own user; a marker inside
  foreign-authored text is ignored.

Receipt
-------
:func:`receipt_for_rows` returns a JSON-serializable
``agent_content_gate_receipt.v1`` object that names every included and excluded
row with a stable reason (``own_user``, ``untrusted_author``, ``missing_author``,
or ``flagged_by_own_user``) plus flag attribution.

CLI
---
::

    python scripts/dev/agent_content_gate.py --check --fixture rows.json --format json

The fixture is either a JSON row list or ``{"labels": [...], "rows": [...]}``.
Exit code 0 means the receipt was built; a nonzero exit means the fixture was
malformed and no content is trusted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

OWN_USER = "ll7"
LABEL_FLAG = "agent:digest"
COMMENT_FLAG_MARKER = "agent-digest: allow"
SCHEMA = "agent_content_gate_receipt.v1"

CLASS_OWN_USER = "own_user"
CLASS_UNTRUSTED = "untrusted"
CLASS_FLAGGED = "flagged_by_own_user"
TRUSTED_CLASSIFICATIONS = frozenset({CLASS_OWN_USER, CLASS_FLAGGED})

REASON_OWN_USER = "own_user"
REASON_UNTRUSTED_AUTHOR = "untrusted_author"
REASON_MISSING_AUTHOR = "missing_author"
REASON_FLAGGED = "flagged_by_own_user"

COMMENT_KINDS = frozenset({"issue_comment", "pr_comment", "review_comment"})
FLAG_SOURCE_LABEL = f"label:{LABEL_FLAG}"
FLAG_SOURCE_COMMENT_MARKER = f"comment_marker:{COMMENT_FLAG_MARKER}"


def extract_login(value: object) -> str:
    """Return a non-empty author login from a raw or nested author value.

    Accepts a plain login string or a GitHub user object. A ``None`` user, a
    deleted user, or an object without a login returns ``""`` so the caller can
    classify the row as missing-author and fail closed.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("login", "author", "user", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def row_login(row: Mapping[str, Any]) -> str:
    """Return the row author login, or ``""`` when no author metadata exists."""
    for key in ("author", "user"):
        if key in row:
            login = extract_login(row.get(key))
            if login:
                return login
    login = extract_login(row.get("login"))
    return login


def row_kind(row: Mapping[str, Any]) -> str:
    """Return the normalized row kind, or ``unknown`` when it is absent."""
    value = row.get("kind")
    return value.strip() if isinstance(value, str) and value.strip() else "unknown"


def row_identifier(row: Mapping[str, Any]) -> str:
    """Return a stable identifier for a row within its thread."""
    raw_id = row.get("id")
    if raw_id is not None and raw_id != "":
        return str(raw_id)
    url = row.get("url")
    if isinstance(url, str) and url:
        return url
    return row_kind(row)


def is_own_user_row(row: Mapping[str, Any]) -> bool:
    """Return whether the row author is our own operator account."""
    return row_login(row).casefold() == OWN_USER.casefold()


def contains_flag_marker(row: Mapping[str, Any]) -> bool:
    """Return whether a row body contains the own-user digest marker."""
    body = row.get("body")
    return isinstance(body, str) and COMMENT_FLAG_MARKER.casefold() in body.casefold()


def label_flag(labels: Iterable[object]) -> dict[str, Any] | None:
    """Return the own-user label flag when ``agent:digest`` is present."""
    if any(str(label).casefold() == LABEL_FLAG.casefold() for label in labels):
        return {
            "source": FLAG_SOURCE_LABEL,
            "author": None,
            "created_at": None,
            "row_id": None,
        }
    return None


def thread_flags(
    rows: Sequence[Mapping[str, Any]], *, labels: Iterable[object] = ()
) -> list[dict[str, Any]]:
    """Return every explicit own-user flag that applies to a thread.

    The issue/PR label opts the whole thread in. The ``agent-digest: allow``
    marker is accepted only from own-user-authored comments or reviews; a
    marker inside foreign-authored text is ignored.
    """
    flags: list[dict[str, Any]] = []
    label = label_flag(labels)
    if label is not None:
        flags.append(label)
    for row in rows:
        if row_kind(row) in COMMENT_KINDS and is_own_user_row(row) and contains_flag_marker(row):
            created_at = row.get("created_at") or row.get("updated_at")
            flags.append(
                {
                    "source": FLAG_SOURCE_COMMENT_MARKER,
                    "author": row_login(row),
                    "created_at": str(created_at) if created_at else None,
                    "row_id": row_identifier(row),
                }
            )
    return flags


def _receipt_entry(
    row: Mapping[str, Any],
    *,
    classification: str,
    reason: str,
    flag: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build one stable receipt entry for a classified row."""
    created_at = row.get("created_at")
    return {
        "id": row_identifier(row),
        "kind": row_kind(row),
        "author": row_login(row),
        "url": str(row.get("url") or ""),
        "created_at": str(created_at) if created_at else "",
        "classification": classification,
        "reason": reason,
        "flag": dict(flag) if flag is not None else None,
    }


def classify_row(
    row: Mapping[str, Any], *, flags: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """Classify one content row and fail closed on missing author metadata.

    Returns a receipt entry with ``classification`` and ``reason``. A row with
    no login is untrusted with reason ``missing_author`` even when flags are
    present, because the flag can only include a known author.
    """
    login = row_login(row)
    if not login:
        return _receipt_entry(
            row,
            classification=CLASS_UNTRUSTED,
            reason=REASON_MISSING_AUTHOR,
            flag=None,
        )
    if login.casefold() == OWN_USER.casefold():
        return _receipt_entry(
            row,
            classification=CLASS_OWN_USER,
            reason=REASON_OWN_USER,
            flag=None,
        )
    if flags:
        return _receipt_entry(
            row,
            classification=CLASS_FLAGGED,
            reason=REASON_FLAGGED,
            flag=flags[0],
        )
    return _receipt_entry(
        row,
        classification=CLASS_UNTRUSTED,
        reason=REASON_UNTRUSTED_AUTHOR,
        flag=None,
    )


def receipt_for_rows(
    rows: Sequence[Mapping[str, Any]], *, labels: Iterable[object] = ()
) -> dict[str, Any]:
    """Build a machine-readable digest receipt for a bounded set of rows.

    Raises ``ValueError`` when a row is not an object so callers can fail
    closed instead of partially trusting malformed input.
    """
    normalized: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be an object")
        normalized.append(row)
    flags = thread_flags(normalized, labels=labels)
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in normalized:
        entry = classify_row(row, flags=flags)
        if entry["classification"] in TRUSTED_CLASSIFICATIONS:
            included.append(entry)
        else:
            excluded.append(entry)
    return {
        "schema": SCHEMA,
        "own_user": OWN_USER,
        "flags": flags,
        "included": included,
        "excluded": excluded,
        "counts": {
            "total": len(normalized),
            "included": len(included),
            "excluded": len(excluded),
        },
    }


def receipt_entry(receipt: Mapping[str, Any], row_id: object) -> dict[str, Any] | None:
    """Return the receipt entry for one row identifier, or ``None`` when absent."""
    wanted = str(row_id)
    for section in ("included", "excluded"):
        for entry in receipt.get(section) or []:
            if isinstance(entry, Mapping) and str(entry.get("id")) == wanted:
                return dict(entry)
    return None


def unclassified_receipt(*, reason: str) -> dict[str, Any]:
    """Return a fail-closed receipt for content with no classifiable rows.

    Used when a transport returns rendered text without structured author
    metadata; the receipt makes explicit that auto-ingestion is not allowed.
    """
    return {
        "schema": SCHEMA,
        "own_user": OWN_USER,
        "flags": [],
        "included": [],
        "excluded": [],
        "counts": {"total": 0, "included": 0, "excluded": 0},
        "status": "fail_closed",
        "reason": reason,
        "auto_ingest_allowed": False,
    }


def load_fixture(text: str) -> tuple[list[Mapping[str, Any]], list[object]]:
    """Parse a CLI fixture into rows and thread labels.

    Raises ``ValueError`` for malformed JSON shapes or non-object rows so the
    CLI can exit nonzero without trusting any content.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"fixture is not valid JSON: {exc}") from exc
    if isinstance(payload, list):
        rows: object = payload
        labels: object = []
    elif isinstance(payload, dict):
        rows = payload.get("rows")
        labels = payload.get("labels") or []
    else:
        raise ValueError("fixture must be a row list or an object with a rows list")
    if not isinstance(rows, list):
        raise ValueError("fixture rows must be a list")
    if not isinstance(labels, list):
        raise ValueError("fixture labels must be a list")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"fixture row {index} must be an object")
    return rows, labels


def render_markdown(receipt: Mapping[str, Any]) -> str:
    """Render a compact Markdown view of a digest receipt."""
    lines = [
        "# Author Trust Gate Receipt",
        "",
        f"- schema: `{receipt.get('schema', SCHEMA)}`",
        f"- own user: `{receipt.get('own_user', OWN_USER)}`",
        f"- included: {receipt.get('counts', {}).get('included', 0)}",
        f"- excluded: {receipt.get('counts', {}).get('excluded', 0)}",
        "",
        "| Row | Kind | Author | Classification | Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in [*(receipt.get("included") or []), *(receipt.get("excluded") or [])]:
        row = " | ".join(
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in (
                entry.get("id", ""),
                entry.get("kind", ""),
                entry.get("author", ""),
                entry.get("classification", ""),
                entry.get("reason", ""),
            )
        )
        lines.append(f"| {row} |")
    return "\n".join(lines) + "\n"


def _cmd_check(args: argparse.Namespace) -> int:
    """Implement the fixture classification command."""
    if not args.check:
        print("--check is required; this helper never mutates content", file=sys.stderr)
        return 2
    try:
        if args.fixture == "-":
            text = sys.stdin.read()
        else:
            text = Path(args.fixture).read_text(encoding="utf-8")
        rows, labels = load_fixture(text)
        receipt = receipt_for_rows(rows, labels=labels)
    except (OSError, ValueError) as exc:
        error = f"author content gate fixture error: {exc}"
        if args.format == "json":
            print(json.dumps({"schema": SCHEMA, "status": "error", "error": error}))
        else:
            print(error, file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        sys.stdout.write(render_markdown(receipt))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="agent_content_gate.py",
        description="Fail-closed author trust gate for agent content ingestion.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Classify the fixture and emit a digest receipt; required.",
    )
    parser.add_argument(
        "--fixture",
        default="-",
        help="Fixture JSON path, or '-' to read stdin (default: -).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Receipt output format (default: json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    return _cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
