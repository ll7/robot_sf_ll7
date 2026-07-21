"""Fail-closed truncation coverage for ``project_priority_score`` item listing.

Issue #5048 extends the shared ``gh ... list --limit N`` guard (from #4991 /
PR #5040) to ``GhProjectClient.item_list``. Because this list drives Priority
Score write-backs, a result at the cap (indistinguishable from a full page)
must fail closed rather than silently skip items beyond the limit.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.dev._gh_pagination import GhListTruncated
from scripts.tools.project_priority_score import GhProjectClient


def _items_payload(count: int) -> str:
    """Return a gh project item-list JSON payload with ``count`` issue items."""
    return json.dumps(
        {
            "items": [
                {"id": f"item{index}", "content": {"type": "Issue", "number": index}}
                for index in range(count)
            ]
        }
    )


def test_item_list_fails_closed_at_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exactly ``limit`` project items raise so a partial sync never runs."""

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=_items_payload(3), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    client = GhProjectClient()
    with pytest.raises(GhListTruncated) as exc_info:
        client.item_list(owner="ll7", project_number=5, limit=3)
    message = str(exc_info.value)
    assert "--limit 3" in message
    assert "gh project item-list" in message


def test_item_list_passes_below_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fewer items than the cap return cleanly with no raise."""

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=_items_payload(2), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    client = GhProjectClient()
    items = client.item_list(owner="ll7", project_number=5, limit=20)
    assert len(items) == 2


def test_targeted_lookup_finds_issue_beyond_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """`item_list_until_issue` finds an issue past the truncation cap cheaply.

    Issue #5870: a project with more items than the default cap must not force a
    full untruncated page for a targeted ``--issue-number`` lookup. The helper
    uses the bounded project query surface and verifies the exact issue number.
    """

    calls: list[list[str]] = []

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        items = [{"id": "item250", "content": {"type": "Issue", "number": 250}}]
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=json.dumps({"items": items}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    client = GhProjectClient()
    found = client.item_list_until_issue(owner="ll7", project_number=5, issue_number=250, limit=25)
    assert len(found) == 1
    assert found[0]["content"]["number"] == 250
    assert len(calls) == 1
    assert "--query" in calls[0]
    assert calls[0][calls[0].index("--query") + 1] == "250"
    assert calls[0][calls[0].index("--limit") + 1] == "25"


def test_targeted_lookup_missing_issue_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing issue yields an empty list, not an ambiguous full scan."""

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=json.dumps({"items": []}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    client = GhProjectClient()
    assert (
        client.item_list_until_issue(owner="ll7", project_number=5, issue_number=999, limit=100)
        == []
    )


def test_targeted_lookup_fails_closed_when_query_is_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capped query cannot claim that a missing exact issue is absent."""

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=_items_payload(2), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    client = GhProjectClient()
    with pytest.raises(GhListTruncated):
        client.item_list_until_issue(owner="ll7", project_number=5, issue_number=999, limit=2)


def test_full_project_sync_still_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unscoped sync retains the explicit truncation guard past the cap.

    This regression guards issue #5870's compatibility constraint: making
    targeted lookups pagination-safe must not weaken the fail-closed full-sync
    path. A capped page must still raise GhListTruncated for unscoped sync.
    """

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=_items_payload(400), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    client = GhProjectClient()
    with pytest.raises(GhListTruncated):
        client.item_list(owner="ll7", project_number=5, limit=400)
