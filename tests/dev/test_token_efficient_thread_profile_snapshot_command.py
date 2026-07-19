"""Regression tests for the PR queue snapshot command documented in the token-efficient profile.

Issue #6028 (friction): the ``goal-pr-review`` queue snapshot command documented in
``docs/templates/token_efficient_thread_profile.md`` must succeed against the current
``scripts/dev/snapshot_pr_queue.py`` CLI. The bare ``snapshot_pr_queue.py --json`` form
previously documented there is rejected by the helper ("at least one PR number is required")
because the CLI requires either ``--active`` or explicit PR numbers. These tests pin the
documented form to a selector the CLI accepts so the friction cannot silently return.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.dev.snapshot_pr_queue import main

DOC_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "templates" / "token_efficient_thread_profile.md"
)
SNAPSHOT_COMMAND_RE = re.compile(
    r"`uv run python scripts/dev/snapshot_pr_queue\.py (?P<args>[^`]+)`"
)


def _snapshot_invocations(doc_text: str) -> list[str]:
    """Return the argument strings for each documented ``snapshot_pr_queue.py`` invocation."""
    return [match.group("args").strip() for match in SNAPSHOT_COMMAND_RE.finditer(doc_text)]


def test_documented_snapshot_command_is_present() -> None:
    """The token-efficient profile must document at least one PR queue snapshot command."""
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    invocations = _snapshot_invocations(doc_text)

    assert invocations, "expected a snapshot_pr_queue.py invocation in the token-efficient profile"


def test_documented_snapshot_command_uses_active_selector() -> None:
    """The documented broad-discovery command must use the CLI's ``--active`` selector."""
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    invocations = _snapshot_invocations(doc_text)

    assert any("--active" in args for args in invocations), (
        "documented snapshot_pr_queue.py invocation must use --active for broad queue discovery"
    )


def test_documented_snapshot_command_is_not_the_bare_rejected_form() -> None:
    """The documented command must not be the selector-less ``--json`` form the CLI rejects.

    Regression for issue #6028: ``snapshot_pr_queue.py --json`` without ``--active`` or a PR
    number exits with status 1 ("at least one PR number is required").
    """
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    invocations = _snapshot_invocations(doc_text)

    for args in invocations:
        tokens = args.split()
        has_selector = bool(set(tokens) & {"--active", "--prs"}) or any(
            token.isdigit() for token in tokens
        )
        assert has_selector, (
            f"documented snapshot_pr_queue.py invocation lacks a CLI selector: {args!r}"
        )


def test_documented_snapshot_command_succeeds_against_cli(capsys) -> None:  # type: ignore[no-untyped-def]
    """The documented queue snapshot command must exit 0 against the current CLI.

    This is the literal issue #6028 acceptance criterion ("the documented command succeeds
    against the current CLI"). The ``--active`` form is exercised with ``gh`` mocked so the
    test stays offline while still driving the real ``main()`` argument and selector gates.
    """
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    invocations = _snapshot_invocations(doc_text)
    active_invocations = [args for args in invocations if "--active" in args.split()]
    assert active_invocations, (
        "documented snapshot_pr_queue.py invocation must use --active so the command succeeds"
    )
    active_invocation = active_invocations[0]

    pr_payload = [
        {
            "number": 6027,
            "title": "example PR",
            "state": "OPEN",
            "isDraft": False,
            "url": "https://github.test/pull/6027",
            "labels": [],
            "headRefName": "feature",
            "headRefOid": "cafe00",
            "mergeable": "MERGEABLE",
            "statusCheckRollup": [],
            "reviews": [],
            "comments": [],
        }
    ]
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        rc = main(active_invocation.split())

    assert rc == 0, (
        f"documented command 'snapshot_pr_queue.py {active_invocation}' failed against the CLI"
    )
    # ``main`` should not have printed a selector-required error for the documented form.
    assert "at least one PR" not in capsys.readouterr().err
