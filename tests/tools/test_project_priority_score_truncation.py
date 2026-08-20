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
from scripts.dev.github_quota import RateLimitSnapshot
from scripts.tools import project_priority_score
from scripts.tools.project_priority_score import GhProjectClient, ProjectQuotaBlockedError


@pytest.fixture(autouse=True)
def _healthy_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pagination tests deterministic without live quota calls."""
    monkeypatch.setattr(
        project_priority_score,
        "read_rate_limit",
        lambda: RateLimitSnapshot(
            status="ok",
            graphql_remaining=4_000,
            graphql_reset_at=1_800_000_000,
            core_remaining=4_000,
            core_reset_at=1_800_000_000,
        ),
    )


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


def _graphql_item(index: int) -> dict[str, object]:
    """Return one ProjectV2 GraphQL item fixture."""

    return {
        "id": f"item{index}",
        "content": {"__typename": "Issue", "number": index, "title": f"Issue {index}"},
        "fieldValues": {
            "nodes": [],
            "pageInfo": {"hasNextPage": False, "endCursor": None},
        },
    }


def _graphql_items_payload(
    items: list[dict[str, object]], *, has_next_page: bool, end_cursor: str | None
) -> str:
    """Return one explicit-cursor ProjectV2 item page."""

    return json.dumps(
        {
            "data": {
                "node": {
                    "__typename": "ProjectV2",
                    "items": {
                        "nodes": items,
                        "pageInfo": {
                            "hasNextPage": has_next_page,
                            "endCursor": end_cursor,
                        },
                    },
                }
            }
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


@pytest.mark.parametrize("item_count", [999, 1000])
def test_paginated_item_list_accepts_full_page_with_explicit_completion(
    monkeypatch: pytest.MonkeyPatch,
    item_count: int,
) -> None:
    """A full logical page is complete when cursor metadata says pagination ended."""

    calls: list[list[str]] = []
    items = [_graphql_item(index) for index in range(item_count)]

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=_graphql_items_payload(items, has_next_page=False, end_cursor="final"),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    client = GhProjectClient()

    found = client.item_list_paginated(
        owner="ll7",
        project_number=5,
        project_id="project-id",
        limit=1000,
    )

    assert len(found) == item_count
    assert len(calls) == 1
    assert calls[0][:3] == ["gh", "api", "graphql"]


def test_paginated_item_list_accumulates_cursor_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A continuation cursor includes items beyond the first page boundary."""

    calls: list[list[str]] = []
    pages = {
        None: _graphql_items_payload(
            [_graphql_item(index) for index in range(1000)],
            has_next_page=True,
            end_cursor="cursor-1",
        ),
        "cursor-1": _graphql_items_payload(
            [_graphql_item(1000)],
            has_next_page=False,
            end_cursor="cursor-2",
        ),
    }

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        cursor = next(
            (argument.removeprefix("after=") for argument in args if argument.startswith("after=")),
            None,
        )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=pages[cursor], stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    client = GhProjectClient()

    found = client.item_list_paginated(
        owner="ll7",
        project_number=5,
        project_id="project-id",
        limit=1000,
    )

    assert len(found) == 1001
    assert found[-1]["content"]["number"] == 1000
    assert len(calls) == 2
    assert "after=cursor-1" in calls[1]
    assert client.last_item_fetch_stats is not None
    assert client.last_item_fetch_stats.pages == 2
    assert client.last_item_fetch_stats.accumulated_items == 1001


def test_paginated_item_list_rejects_repeated_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cursor loop fails closed before a partial score calculation can start."""

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        index = 1 if not any(argument.startswith("after=") for argument in args) else 2
        payload = _graphql_items_payload(
            [_graphql_item(index)],
            has_next_page=True,
            end_cursor="cursor-loop",
        )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="repeated cursor"):
        GhProjectClient().item_list_paginated(
            owner="ll7",
            project_number=5,
            project_id="project-id",
            limit=1000,
        )


def test_paginated_item_list_blocks_before_a_page_when_quota_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later-page quota failure occurs before another GraphQL request."""

    calls: list[list[str]] = []
    snapshots = iter(
        [
            RateLimitSnapshot(
                status="ok",
                graphql_remaining=1_000,
                graphql_reset_at=1_800_000_000,
                core_remaining=4_000,
                core_reset_at=1_800_000_000,
            ),
            RateLimitSnapshot(
                status="ok",
                graphql_remaining=100,
                graphql_reset_at=1_800_000_000,
                core_remaining=4_000,
                core_reset_at=1_800_000_000,
            ),
        ]
    )
    monkeypatch.setattr(project_priority_score, "read_rate_limit", lambda: next(snapshots))

    page = _graphql_items_payload([_graphql_item(1)], has_next_page=True, end_cursor="cursor-1")

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=page, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(ProjectQuotaBlockedError):
        GhProjectClient().item_list_paginated(
            owner="ll7",
            project_number=5,
            project_id="project-id",
            limit=1000,
        )

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("page_info", "message"),
    [
        ({"hasNextPage": "false", "endCursor": None}, "hasNextPage"),
        ({"hasNextPage": False, "endCursor": 42}, "endCursor"),
        ({"hasNextPage": True, "endCursor": None}, "non-empty endCursor"),
        ({"hasNextPage": False}, "required fields"),
    ],
)
def test_paginated_item_list_rejects_missing_or_malformed_page_metadata(
    monkeypatch: pytest.MonkeyPatch,
    page_info: dict[str, object],
    message: str,
) -> None:
    """A page without valid continuation metadata cannot prove completeness."""

    payload = json.dumps(
        {
            "data": {
                "node": {
                    "__typename": "ProjectV2",
                    "items": {"nodes": [_graphql_item(1)], "pageInfo": page_info},
                }
            }
        }
    )

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match=message):
        GhProjectClient().item_list_paginated(
            owner="ll7",
            project_number=5,
            project_id="project-id",
            limit=1000,
        )


def test_paginated_item_list_normalizes_project_field_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GraphQL field-value unions retain the existing score-sync item keys."""

    item = _graphql_item(7)
    item["fieldValues"] = {
        "nodes": [
            {
                "__typename": "ProjectV2ItemFieldSingleSelectValue",
                "name": "Todo",
                "field": {"name": "Status"},
            },
            {
                "__typename": "ProjectV2ItemFieldNumberValue",
                "number": 5,
                "field": {"name": "Improvement"},
            },
            {
                "__typename": "ProjectV2ItemFieldNumberValue",
                "number": 0.8,
                "field": {"name": "Success Probability"},
            },
            {
                "__typename": "ProjectV2ItemFieldNumberValue",
                "number": 2,
                "field": {"name": "Priority Score"},
            },
        ],
        "pageInfo": {"hasNextPage": False, "endCursor": None},
    }
    payload = _graphql_items_payload([item], has_next_page=False, end_cursor=None)

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    found = GhProjectClient().item_list_paginated(
        owner="ll7",
        project_number=5,
        project_id="project-id",
        limit=1000,
    )

    assert found[0]["status"] == "Todo"
    assert found[0]["improvement"] == 5
    assert found[0]["success Probability"] == 0.8
    assert found[0]["priority Score"] == 2


def test_targeted_lookup_uses_server_query_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supported CLI query locates an exact issue without a full project scan."""

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

    assert found[0]["content"]["number"] == 250
    assert len(calls) == 1
    assert calls[0][calls[0].index("--query") + 1] == "is:issue 250"
    assert calls[0][calls[0].index("--limit") + 1] == "25"


def test_targeted_lookup_falls_back_when_query_flag_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older CLIs use the bounded exact-match fallback without weakening safety.

    Issue #5870: a project with more items than the default cap must not force a
    full untruncated page for a targeted ``--issue-number`` lookup. The helper
    first tries the newer query surface, then uses a portable bounded project
    list when the CLI rejects ``--query``.
    """

    calls: list[list[str]] = []

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if "--query" in args:
            raise subprocess.CalledProcessError(
                1,
                args,
                output="",
                stderr="unknown flag: --query",
            )
        items = [{"id": "item250", "content": {"type": "Issue", "number": 250}}]
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=json.dumps({"items": items}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    client = GhProjectClient()
    found = client.item_list_until_issue(owner="ll7", project_number=5, issue_number=250, limit=25)
    assert len(found) == 1
    assert found[0]["content"]["number"] == 250
    assert len(calls) == 2
    assert "--query" in calls[0]
    assert "--query" not in calls[1]
    assert calls[1][calls[1].index("--limit") + 1] == "25"


def test_targeted_lookup_does_not_hide_non_query_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authentication and other command failures remain fail-closed errors."""

    def _fake_run(
        args: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            1,
            args,
            output="",
            stderr="authentication failed",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="authentication failed"):
        GhProjectClient().item_list_until_issue(
            owner="ll7", project_number=5, issue_number=250, limit=25
        )


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
    """The single-page item-list owner retains its explicit truncation guard.

    This regression guards the compatibility constraint from issue #5870:
    introducing a complete cursor owner must not weaken callers that still use
    the bounded ``item_list`` surface. A capped page must still raise
    ``GhListTruncated`` there.
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
