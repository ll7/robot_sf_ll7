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
