"""Regression coverage for the issue snapshot command in the token-efficient profile."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.dev.snapshot_issue_batch import _parse_args, main

DOC_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "templates" / "token_efficient_thread_profile.md"
)
SNAPSHOT_COMMAND_RE = re.compile(
    r"`uv run python scripts/dev/snapshot_issue_batch\.py (?P<args>[^`]+)`"
)


def _snapshot_invocations(doc_text: str) -> list[str]:
    """Return argument strings for documented issue snapshot invocations."""
    return [match.group("args").strip() for match in SNAPSHOT_COMMAND_RE.finditer(doc_text)]


def test_documented_issue_snapshot_uses_claimable_selector() -> None:
    """The broad-discovery issue snapshot command must use ``--claimable``."""
    invocations = _snapshot_invocations(DOC_PATH.read_text(encoding="utf-8"))

    assert any("--claimable" in args for args in invocations)


def test_documented_issue_snapshot_is_not_selector_less() -> None:
    """The documented issue snapshot must not use the rejected bare ``--json`` form."""
    invocations = _snapshot_invocations(DOC_PATH.read_text(encoding="utf-8"))

    for invocation in invocations:
        parsed = _parse_args(shlex.split(invocation))
        assert parsed.claimable or parsed.issues, (
            f"documented snapshot_issue_batch.py invocation lacks a selector: {invocation!r}"
        )


def test_documented_issue_snapshot_succeeds_against_cli() -> None:
    """The documented broad-discovery command drives the production parser offline."""
    invocations = _snapshot_invocations(DOC_PATH.read_text(encoding="utf-8"))
    claimable_invocations = [
        invocation for invocation in invocations if _parse_args(shlex.split(invocation)).claimable
    ]
    assert claimable_invocations, "expected a --claimable issue snapshot invocation"

    issue_payload = [
        {
            "number": 6031,
            "title": "example claimable issue",
            "state": "OPEN",
            "url": "https://github.test/issues/6031",
            "labels": [],
            "assignees": [],
        }
    ]
    with (
        patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh,
        patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as mock_claims,
    ):
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_payload), stderr="")
        mock_claims.return_value = {
            6031: {"ok": True, "claimed": False, "claim_ref": None, "sha": None}
        }
        rc = main(shlex.split(claimable_invocations[0]))

    assert rc == 0
