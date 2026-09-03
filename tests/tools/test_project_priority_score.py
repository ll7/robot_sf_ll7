"""Tests for the Project #5 priority score sync helper."""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import TYPE_CHECKING

import pytest

from scripts.dev.github_quota import RateLimitSnapshot
from scripts.tools import project_priority_score
from scripts.tools.project_priority_score import (
    DEFAULT_ALPHA,
    DEFAULT_SUCCESS_PROBABILITY,
    EFFORT_FIELD,
    PRIORITY_SCORE_FIELD,
    REQUIRED_NUMBER_FIELDS,
    GhProjectClient,
    GhProjectTimeoutError,
    MissingProjectScopeError,
    ProjectItemFetchStats,
    ProjectQuotaBlockedError,
    ScoreInputs,
    SyncOptions,
    SyncPreview,
    _numeric_field_literal,
    build_previews,
    compute_priority_score,
    field_keys,
    load_project_cache,
    main,
    normalize_inputs,
    sync_scores,
    write_summary,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _healthy_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep score-sync unit tests off the live quota endpoint unless overridden."""
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


class FakeGhProjectClient:
    """Deterministic fake gh client for score-sync tests."""

    def __init__(
        self,
        *,
        fields: list[dict],
        items: list[dict],
        issue_snapshots: dict[int, dict] | None = None,
    ) -> None:
        """Store fake project fields, items, and captured side effects."""

        self._fields = fields
        self._items = items
        self.created_fields: list[str] = []
        self.updated_numbers: list[tuple[str, str, str, float]] = []
        self.field_list_calls = 0
        self.targeted_limits: list[int] = []
        self.paginated_limits: list[int] = []
        self.paginated_project_ids: list[str | None] = []
        self.issue_snapshot_calls: list[tuple[str, int]] = []
        self.project_rechecks: list[int] = []
        self.project_recheck_overrides: dict[int, list[dict]] = {}
        self.last_eligibility_plan: dict | None = None
        self._issue_snapshots = issue_snapshots or {
            int(item["content"]["number"]): {
                "number": int(item["content"]["number"]),
                "title": str(item["content"]["title"]),
                "state": "OPEN",
                "updated_at": "2026-08-24T00:00:00Z",
                "labels": ["state:ready"],
            }
            for item in items
            if isinstance(item.get("content"), dict)
            and isinstance(item["content"].get("number"), int)
            and item["content"]["number"] > 0
        }

    def project_id(self, *, owner: str, project_number: int) -> str:
        """Return a stable fake project ID for update calls."""

        return "project-id"

    def field_list(self, *, owner: str, project_number: int) -> list[dict]:
        """Return the current fake project field payload."""

        self.field_list_calls += 1
        return self._fields

    def ensure_number_field(self, *, owner: str, project_number: int, name: str) -> None:
        """Append a missing numeric field to the fake project schema."""

        self.created_fields.append(name)
        self._fields.append({"id": f"field-{name}", "name": name, "type": "ProjectV2Field"})

    def item_list(self, *, owner: str, project_number: int, limit: int) -> list[dict]:
        """Return project items up to the requested limit."""

        return self._items[:limit]

    def item_list_paginated(
        self,
        *,
        owner: str,
        project_number: int,
        project_id: str | None = None,
        limit: int,
        min_graphql_remaining: int = 100,
    ) -> list[dict]:
        """Return the fake's complete cursor-paginated item accumulation."""

        self.paginated_limits.append(limit)
        self.paginated_project_ids.append(project_id)
        return list(self._items)

    def item_list_until_issue(
        self,
        *,
        owner: str,
        project_number: int,
        issue_number: int,
        limit: int = 100,
    ) -> list[dict]:
        """Simulate an exact issue query over the fake project items."""

        self.targeted_limits.append(limit)
        self.project_rechecks.append(issue_number)
        if issue_number in self.project_recheck_overrides:
            return self.project_recheck_overrides[issue_number]
        for item in self._items:
            content = item.get("content") or {}
            if content.get("type") == "Issue" and content.get("number") == issue_number:
                return [item]
        return []

    def issue_snapshot(self, *, repo: str, issue_number: int) -> dict:
        """Return one deterministic REST-style issue snapshot."""

        self.issue_snapshot_calls.append((repo, issue_number))
        snapshot = self._issue_snapshots.get(issue_number)
        if isinstance(snapshot, Exception):
            raise snapshot
        if snapshot is None:
            raise RuntimeError(f"issue #{issue_number} unavailable")
        return dict(snapshot)

    def update_number_field(
        self,
        *,
        item_id: str,
        field_id: str,
        project_id: str,
        number: float,
    ) -> None:
        """Capture numeric project-field updates for assertions."""

        self.updated_numbers.append((item_id, field_id, project_id, number))


def _field(name: str) -> dict:
    """Return a minimal Project field fixture."""
    return {"id": f"field-{name}", "name": name, "type": "ProjectV2Field"}


def _item(issue_number: int, **fields: object) -> dict:
    """Return a minimal Project item fixture with optional field values."""
    payload = {
        "id": f"item-{issue_number}",
        "status": "Todo",
        "content": {
            "type": "Issue",
            "number": issue_number,
            "title": f"Issue {issue_number}",
            "repository": "ll7/robot_sf_ll7",
        },
    }
    payload.update(fields)
    return payload


def test_normalize_inputs_clamps_and_defaults() -> None:
    """Verify score inputs remain stable under missing and invalid field values.

    This matters because project fields are user-maintained and the sync helper
    must not produce absurd rankings from negative effort or out-of-range odds.
    """

    inputs = normalize_inputs(
        {
            "improvement": 11,
            "success probability": 1.7,
            lower_first_key(EFFORT_FIELD): 0,
            lower_first_key("Time Criticality"): 9,
            lower_first_key("Unlock Factor"): 0.1,
        }
    )

    assert inputs == ScoreInputs(
        improvement=10.0,
        success_probability=0.017,
        effort_hours=0.1,
        time_criticality=9.0,
        unlock_factor=0.1,
    )
    lower_bound_inputs = normalize_inputs(
        {
            "improvement": -4,
            lower_first_key("Time Criticality"): 0,
            lower_first_key("Unlock Factor"): -1,
        }
    )
    assert lower_bound_inputs.improvement == 0.1
    assert lower_bound_inputs.time_criticality == 0.1
    assert lower_bound_inputs.unlock_factor == 0.1

    defaulted = normalize_inputs({})
    assert defaulted.success_probability == DEFAULT_SUCCESS_PROBABILITY
    assert defaulted.effort_hours == 1.0


def test_normalize_inputs_accepts_percent_scale_success_probability() -> None:
    """Verify whole-percent inputs normalize without breaking 0-1 probabilities.

    This matters because GitHub project number fields may be entered as either
    fractional probabilities or whole percentages, and the scoring model needs
    one consistent 0-1 representation before ranking work.
    """

    assert normalize_inputs({"success probability": 1.0}).success_probability == 1.0
    assert normalize_inputs({"success probability": 1.7}).success_probability == 0.017
    assert normalize_inputs({"success probability": 5}).success_probability == 0.05
    assert normalize_inputs({"success probability": 60}).success_probability == 0.6


def test_build_previews_skips_done_and_rounds_scores() -> None:
    """Verify preview generation respects project status and score rounding.

    This matters because the workflow should not churn done items and should
    produce a stable numeric field that sorts cleanly in project views.
    """

    previews = build_previews(
        [
            _item(1, improvement=5, **{lower_first_key(EFFORT_FIELD): 8}),
            _item(2, status="Done", improvement=10),
        ],
        alpha=DEFAULT_ALPHA,
        round_digits=4,
        issue_number=None,
        skip_statuses={"Done"},
    )

    assert len(previews) == 1
    assert previews[0].issue_number == 1
    assert previews[0].new_score == round(previews[0].new_score, 4)


def test_build_previews_only_empty_skips_already_scored_items() -> None:
    """only_empty assesses unscored issues and never touches an existing priority.

    This is the autopilot auto-fill contract: fill empty priorities cheaply, but leave human-set or
    previously-computed scores alone so the loop does not churn the project board.
    """

    items = [
        _item(1, improvement=5, **{lower_first_key(EFFORT_FIELD): 8}),
        _item(
            2,
            improvement=5,
            **{lower_first_key(EFFORT_FIELD): 8, lower_first_key(PRIORITY_SCORE_FIELD): 1.23},
        ),
    ]
    kwargs = {
        "alpha": DEFAULT_ALPHA,
        "round_digits": 4,
        "issue_number": None,
        "skip_statuses": set(),
    }

    only_empty = build_previews(items, only_empty=True, **kwargs)
    assert [p.issue_number for p in only_empty] == [1]

    # Without the flag, both items (including the already-scored one) are previewed.
    both = build_previews(items, **kwargs)
    assert {p.issue_number for p in both} == {1, 2}


def test_sync_scores_ensures_fields_and_writes_updates() -> None:
    """Verify sync can create missing fields and update the derived score.

    This matters because the first run should bootstrap the Project #5 schema
    and then write a numeric `Priority Score` field without manual setup.
    """

    client = FakeGhProjectClient(
        fields=[_field(EFFORT_FIELD)],
        items=[_item(699, improvement=5, **{lower_first_key(EFFORT_FIELD): 8})],
    )

    previews = sync_scores(
        client,
        SyncOptions(
            owner="ll7",
            project_number=5,
            ensure_fields=True,
            limit=50,
            alpha=DEFAULT_ALPHA,
            round_digits=6,
            issue_number=699,
            dry_run=False,
            skip_statuses={"Done"},
        ),
    )

    assert PRIORITY_SCORE_FIELD in client.created_fields
    assert len(previews) == 1
    assert client.updated_numbers == [
        ("item-699", f"field-{PRIORITY_SCORE_FIELD}", "project-id", previews[0].new_score)
    ]
    assert client.field_list_calls == 2


def test_sync_scores_unscoped_uses_paginated_item_owner() -> None:
    """Unscoped sync must use explicit cursor completion before any score write."""

    client = FakeGhProjectClient(
        fields=[
            _field(name)
            for name in (
                EFFORT_FIELD,
                *REQUIRED_NUMBER_FIELDS,
            )
        ],
        items=[_item(699, improvement=5, **{lower_first_key(EFFORT_FIELD): 8})],
    )

    previews = sync_scores(
        client,
        SyncOptions(
            owner="ll7",
            project_number=5,
            ensure_fields=False,
            limit=1000,
            alpha=DEFAULT_ALPHA,
            round_digits=6,
            issue_number=None,
            dry_run=False,
            skip_statuses={"Done"},
        ),
    )

    assert [preview.issue_number for preview in previews] == [699]
    assert client.paginated_limits == [1000]
    assert client.paginated_project_ids == ["project-id"]
    assert client.updated_numbers


def test_sync_scores_does_not_write_when_paginated_read_fails() -> None:
    """A later item-read failure happens before score previews or writes."""

    class LatePaginationFailureClient(FakeGhProjectClient):
        """Fail while completing the logical item scan."""

        def item_list_paginated(self, **kwargs: object) -> list[dict]:
            raise RuntimeError("second page failed")

    client = LatePaginationFailureClient(
        fields=[
            _field(name)
            for name in (
                EFFORT_FIELD,
                *REQUIRED_NUMBER_FIELDS,
            )
        ],
        items=[_item(700, improvement=5, **{lower_first_key(EFFORT_FIELD): 8})],
    )

    with pytest.raises(RuntimeError, match="second page failed"):
        sync_scores(
            client,
            SyncOptions(
                owner="ll7",
                project_number=5,
                ensure_fields=False,
                limit=1000,
                alpha=DEFAULT_ALPHA,
                round_digits=6,
                issue_number=None,
                dry_run=False,
                skip_statuses={"Done"},
            ),
        )

    assert client.updated_numbers == []


def test_sync_scores_reserves_write_budget_before_any_score_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A low post-read quota blocks before the first score mutation."""

    monkeypatch.setattr(
        project_priority_score,
        "read_rate_limit",
        lambda: RateLimitSnapshot(
            status="ok",
            graphql_remaining=100,
            graphql_reset_at=1_800_000_123,
            core_remaining=4_000,
            core_reset_at=1_800_000_456,
        ),
    )
    client = FakeGhProjectClient(
        fields=[
            _field(name)
            for name in (
                EFFORT_FIELD,
                *REQUIRED_NUMBER_FIELDS,
            )
        ],
        items=[_item(703, improvement=5, **{lower_first_key(EFFORT_FIELD): 8})],
    )

    with pytest.raises(ProjectQuotaBlockedError):
        sync_scores(
            client,
            SyncOptions(
                owner="ll7",
                project_number=5,
                ensure_fields=False,
                limit=1000,
                alpha=DEFAULT_ALPHA,
                round_digits=6,
                issue_number=None,
                dry_run=False,
                skip_statuses={"Done"},
            ),
        )

    assert client.updated_numbers == []


def test_sync_scores_unscoped_dry_run_does_not_write() -> None:
    """A complete unscoped dry run computes previews without item-edit writes."""

    client = FakeGhProjectClient(
        fields=[
            _field(name)
            for name in (
                EFFORT_FIELD,
                *REQUIRED_NUMBER_FIELDS,
            )
        ],
        items=[_item(701, improvement=5, **{lower_first_key(EFFORT_FIELD): 8})],
    )

    previews = sync_scores(
        client,
        SyncOptions(
            owner="ll7",
            project_number=5,
            ensure_fields=False,
            limit=1000,
            alpha=DEFAULT_ALPHA,
            round_digits=6,
            issue_number=None,
            dry_run=True,
            skip_statuses={"Done"},
        ),
    )

    assert [preview.issue_number for preview in previews] == [701]
    assert client.updated_numbers == []


def test_only_empty_builds_live_eligibility_plan_and_writes_ready_issue() -> None:
    """Auto-fill writes only after a current open state:ready REST snapshot."""

    client = FakeGhProjectClient(
        fields=[_field(name) for name in (EFFORT_FIELD, *REQUIRED_NUMBER_FIELDS)],
        items=[_item(701, improvement=5, **{lower_first_key(EFFORT_FIELD): 8})],
    )

    previews = sync_scores(
        client,
        SyncOptions(
            owner="ll7",
            project_number=5,
            ensure_fields=False,
            limit=1000,
            alpha=DEFAULT_ALPHA,
            round_digits=6,
            issue_number=None,
            dry_run=False,
            skip_statuses={"Done"},
            only_empty=True,
        ),
    )

    assert [preview.issue_number for preview in previews] == [701]
    assert client.issue_snapshot_calls == [("ll7/robot_sf_ll7", 701)] * 2
    assert client.project_rechecks == [701]
    assert client.updated_numbers
    assert client.last_eligibility_plan == {
        "schema": "project_priority_eligibility_plan.v1",
        "status": "applied",
        "counts": {"eligible": 1, "skipped": 0, "blocked": 0},
        "writes_performed": True,
        "attempted_rows": [{"issue_number": 701, "item_id": "item-701", "written": True}],
        "writes_performed_count": 1,
        "items": [
            {
                "issue_number": 701,
                "project_item_id": "item-701",
                "project_status": "Todo",
                "decision": "eligible",
                "reason_code": "open_ready_exact_state",
                "issue_updated_at": "2026-08-24T00:00:00Z",
            }
        ],
    }


@pytest.mark.parametrize(
    ("snapshot", "reason_code", "decision"),
    [
        (
            {
                "number": 701,
                "title": "Issue 701",
                "state": "CLOSED",
                "updated_at": "2026-08-24T00:00:00Z",
                "labels": ["state:done"],
            },
            "issue_terminal",
            "skipped",
        ),
        (
            {
                "number": 701,
                "title": "Issue 701",
                "state": "OPEN",
                "updated_at": "2026-08-24T00:00:00Z",
                "labels": ["state:parked", "deferred"],
            },
            "issue_not_ready",
            "skipped",
        ),
        (
            {
                "number": 701,
                "title": "Issue 701",
                "state": "OPEN",
                "updated_at": "2026-08-24T00:00:00Z",
                "labels": ["state:ready", "decision-required"],
            },
            "issue_ambiguous",
            "blocked",
        ),
    ],
)
def test_only_empty_skips_ineligible_live_issue_states(
    snapshot: dict, reason_code: str, decision: str
) -> None:
    """Terminal, parked, and decision-gated issues never receive default scores."""

    client = FakeGhProjectClient(
        fields=[_field(name) for name in (EFFORT_FIELD, *REQUIRED_NUMBER_FIELDS)],
        items=[_item(701, improvement=5)],
        issue_snapshots={701: snapshot},
    )

    assert (
        sync_scores(
            client,
            SyncOptions(
                owner="ll7",
                project_number=5,
                ensure_fields=False,
                limit=1000,
                alpha=DEFAULT_ALPHA,
                round_digits=6,
                issue_number=None,
                dry_run=False,
                skip_statuses={"Done"},
                only_empty=True,
            ),
        )
        == []
    )
    assert client.updated_numbers == []
    assert client.last_eligibility_plan["items"][0]["reason_code"] == reason_code
    assert client.last_eligibility_plan["items"][0]["decision"] == decision


def test_only_empty_blocks_unavailable_issue_snapshot_without_writes() -> None:
    """REST uncertainty is visible and fail-closed for the affected item."""

    client = FakeGhProjectClient(
        fields=[_field(name) for name in (EFFORT_FIELD, *REQUIRED_NUMBER_FIELDS)],
        items=[_item(701, improvement=5)],
        issue_snapshots={701: RuntimeError("REST unavailable")},
    )

    assert (
        sync_scores(
            client,
            SyncOptions(
                owner="ll7",
                project_number=5,
                ensure_fields=False,
                limit=1000,
                alpha=DEFAULT_ALPHA,
                round_digits=6,
                issue_number=None,
                dry_run=False,
                skip_statuses={"Done"},
                only_empty=True,
            ),
        )
        == []
    )
    assert client.updated_numbers == []
    assert client.last_eligibility_plan["items"][0]["reason_code"] == "issue_state_unavailable"


def test_only_empty_propagates_issue_snapshot_timeout_before_any_write() -> None:
    """A timed-out eligibility read aborts the whole guarded batch."""

    timeout = GhProjectTimeoutError(
        command=("gh", "api", "repos/ll7/robot_sf_ll7/issues/702"),
        timeout_seconds=5.0,
        phase="read",
    )
    client = FakeGhProjectClient(
        fields=[_field(name) for name in (EFFORT_FIELD, *REQUIRED_NUMBER_FIELDS)],
        items=[_item(701, improvement=5), _item(702, improvement=5)],
        issue_snapshots={
            701: {
                "number": 701,
                "title": "Issue 701",
                "state": "OPEN",
                "updated_at": "2026-08-24T00:00:00Z",
                "labels": ["state:ready"],
            },
            702: timeout,
        },
    )

    with pytest.raises(GhProjectTimeoutError) as exc_info:
        sync_scores(
            client,
            SyncOptions(
                owner="ll7",
                project_number=5,
                ensure_fields=False,
                limit=1000,
                alpha=DEFAULT_ALPHA,
                round_digits=6,
                issue_number=None,
                dry_run=False,
                skip_statuses={"Done"},
                only_empty=True,
            ),
        )

    assert exc_info.value is timeout
    assert client.issue_snapshot_calls == [
        ("ll7/robot_sf_ll7", 701),
        ("ll7/robot_sf_ll7", 702),
    ]
    assert client.updated_numbers == []


def test_only_empty_propagates_revalidation_timeout_before_any_write() -> None:
    """A timed-out guarded re-read cannot be downgraded to ordinary drift."""

    timeout = GhProjectTimeoutError(
        command=("gh", "api", "repos/ll7/robot_sf_ll7/issues/701"),
        timeout_seconds=5.0,
        phase="read",
    )

    class RevalidationTimeoutClient(FakeGhProjectClient):
        def issue_snapshot(self, *, repo: str, issue_number: int) -> dict:
            snapshot = super().issue_snapshot(repo=repo, issue_number=issue_number)
            if len(self.issue_snapshot_calls) > 1:
                raise timeout
            return snapshot

    client = RevalidationTimeoutClient(
        fields=[_field(name) for name in (EFFORT_FIELD, *REQUIRED_NUMBER_FIELDS)],
        items=[_item(701, improvement=5)],
    )

    with pytest.raises(GhProjectTimeoutError) as exc_info:
        sync_scores(
            client,
            SyncOptions(
                owner="ll7",
                project_number=5,
                ensure_fields=False,
                limit=1000,
                alpha=DEFAULT_ALPHA,
                round_digits=6,
                issue_number=None,
                dry_run=False,
                skip_statuses={"Done"},
                only_empty=True,
            ),
        )

    assert exc_info.value is timeout
    assert client.updated_numbers == []


def test_only_empty_blocks_cross_repository_project_item() -> None:
    """Issue numbers from another repository cannot alias Robot SF REST state."""

    item = _item(701, improvement=5)
    item["content"]["repository"] = "ll7/another-repo"
    client = FakeGhProjectClient(
        fields=[_field(name) for name in (EFFORT_FIELD, *REQUIRED_NUMBER_FIELDS)],
        items=[item],
    )

    assert (
        sync_scores(
            client,
            SyncOptions(
                owner="ll7",
                project_number=5,
                ensure_fields=False,
                limit=1000,
                alpha=DEFAULT_ALPHA,
                round_digits=6,
                issue_number=None,
                dry_run=False,
                skip_statuses={"Done"},
                only_empty=True,
            ),
        )
        == []
    )
    assert client.issue_snapshot_calls == []
    assert client.last_eligibility_plan["items"][0]["reason_code"] == "project_repo_mismatch"


def test_only_empty_plan_reports_terminal_project_status_and_malformed_issue() -> None:
    """The complete plan explains stale terminal and malformed Project rows."""

    malformed = {
        "id": "item-malformed",
        "status": "Todo",
        "content": {"type": "Issue", "number": "701", "title": "Malformed"},
    }
    client = FakeGhProjectClient(
        fields=[_field(name) for name in (EFFORT_FIELD, *REQUIRED_NUMBER_FIELDS)],
        items=[_item(701, status="Done", improvement=5), malformed],
    )

    assert (
        sync_scores(
            client,
            SyncOptions(
                owner="ll7",
                project_number=5,
                ensure_fields=False,
                limit=1000,
                alpha=DEFAULT_ALPHA,
                round_digits=6,
                issue_number=None,
                dry_run=False,
                skip_statuses={"Done"},
                only_empty=True,
            ),
        )
        == []
    )
    assert client.issue_snapshot_calls == []
    assert client.updated_numbers == []
    assert [row["reason_code"] for row in client.last_eligibility_plan["items"]] == [
        "malformed_issue_number",
        "project_status_terminal",
    ]


def test_only_empty_aborts_batch_when_issue_drifts_before_write() -> None:
    """No Project write occurs when an issue changes after eligibility planning."""

    class DriftingIssueClient(FakeGhProjectClient):
        def issue_snapshot(self, *, repo: str, issue_number: int) -> dict:
            snapshot = super().issue_snapshot(repo=repo, issue_number=issue_number)
            if len(self.issue_snapshot_calls) > 1:
                snapshot["labels"] = ["state:running"]
            return snapshot

    client = DriftingIssueClient(
        fields=[_field(name) for name in (EFFORT_FIELD, *REQUIRED_NUMBER_FIELDS)],
        items=[_item(701, improvement=5)],
    )

    assert (
        sync_scores(
            client,
            SyncOptions(
                owner="ll7",
                project_number=5,
                ensure_fields=False,
                limit=1000,
                alpha=DEFAULT_ALPHA,
                round_digits=6,
                issue_number=None,
                dry_run=False,
                skip_statuses={"Done"},
                only_empty=True,
            ),
        )
        == []
    )
    assert client.updated_numbers == []
    assert client.last_eligibility_plan["status"] == "blocked_drift"
    assert client.last_eligibility_plan["items"][0]["reason_code"] == "issue_state_drift"


def test_only_empty_aborts_batch_when_project_item_drifts_before_write() -> None:
    """Project item/status changes are checked before the first field mutation."""

    client = FakeGhProjectClient(
        fields=[_field(name) for name in (EFFORT_FIELD, *REQUIRED_NUMBER_FIELDS)],
        items=[_item(701, improvement=5)],
    )
    client.project_recheck_overrides[701] = [_item(701, status="Done", improvement=5)]

    assert (
        sync_scores(
            client,
            SyncOptions(
                owner="ll7",
                project_number=5,
                ensure_fields=False,
                limit=1000,
                alpha=DEFAULT_ALPHA,
                round_digits=6,
                issue_number=None,
                dry_run=False,
                skip_statuses={"Done"},
                only_empty=True,
            ),
        )
        == []
    )
    assert client.updated_numbers == []
    assert client.last_eligibility_plan["status"] == "blocked_drift"
    assert client.last_eligibility_plan["items"][0]["reason_code"] == "project_item_drift"


def test_sync_scores_skips_malformed_issue_numbers_when_indexing_items() -> None:
    """Verify malformed project items cannot collide in the issue lookup map.

    This matters because the sync pass should ignore broken issue payloads
    rather than folding multiple malformed items onto one synthetic key.
    """

    client = FakeGhProjectClient(
        fields=[
            _field(name)
            for name in (
                EFFORT_FIELD,
                *(
                    "Improvement",
                    "Success Probability",
                    "Time Criticality",
                    "Unlock Factor",
                    PRIORITY_SCORE_FIELD,
                ),
            )
        ],
        items=[
            _item(699, improvement=5, **{lower_first_key(EFFORT_FIELD): 8}),
            {"id": "broken-a", "status": "Todo", "content": {"type": "Issue", "title": "Broken A"}},
            {
                "id": "broken-b",
                "status": "Todo",
                "content": {"type": "Issue", "number": -1, "title": "Broken B"},
            },
        ],
    )

    previews = sync_scores(
        client,
        SyncOptions(
            owner="ll7",
            project_number=5,
            ensure_fields=False,
            limit=50,
            alpha=DEFAULT_ALPHA,
            round_digits=6,
            issue_number=699,
            dry_run=False,
            skip_statuses={"Done"},
        ),
    )

    assert [preview.issue_number for preview in previews] == [699]
    assert client.updated_numbers == [
        ("item-699", f"field-{PRIORITY_SCORE_FIELD}", "project-id", previews[0].new_score)
    ]


def test_sync_scores_targeted_finds_issue_beyond_default_limit() -> None:
    """`--issue-number N` locates N past the default 400-item boundary cheaply.

    Issue #5870: the targeted sync must not call the full fail-closed
    ``item_list`` (which raises at the cap) before applying the issue filter.
    Here the project holds 2,300+ items; the targeted pass must find issue
    2299 using a bounded server-side query without needing a full project list.
    """

    items = [
        _item(number, improvement=5, **{lower_first_key(EFFORT_FIELD): 8})
        for number in range(1, 2301)
    ]
    client = FakeGhProjectClient(
        fields=[
            _field(name)
            for name in (
                EFFORT_FIELD,
                *REQUIRED_NUMBER_FIELDS,
            )
        ],
        items=items,
    )

    previews = sync_scores(
        client,
        SyncOptions(
            owner="ll7",
            project_number=5,
            ensure_fields=False,
            limit=400,
            alpha=DEFAULT_ALPHA,
            round_digits=6,
            issue_number=2299,
            dry_run=False,
            skip_statuses={"Done"},
        ),
    )

    assert [p.issue_number for p in previews] == [2299]
    assert client.targeted_limits == [400]
    assert client.updated_numbers == [
        ("item-2299", f"field-{PRIORITY_SCORE_FIELD}", "project-id", previews[0].new_score)
    ]


def test_sync_scores_targeted_does_not_update_others() -> None:
    """Targeted mode updates exactly the requested issue and no other item."""

    client = FakeGhProjectClient(
        fields=[
            _field(name)
            for name in (
                EFFORT_FIELD,
                *REQUIRED_NUMBER_FIELDS,
            )
        ],
        items=[
            _item(1, improvement=5, **{lower_first_key(EFFORT_FIELD): 8}),
            _item(2, improvement=5, **{lower_first_key(EFFORT_FIELD): 8}),
            _item(3, improvement=5, **{lower_first_key(EFFORT_FIELD): 8}),
        ],
    )

    previews = sync_scores(
        client,
        SyncOptions(
            owner="ll7",
            project_number=5,
            ensure_fields=False,
            limit=400,
            alpha=DEFAULT_ALPHA,
            round_digits=6,
            issue_number=2,
            dry_run=False,
            skip_statuses={"Done"},
        ),
    )

    assert [p.issue_number for p in previews] == [2]
    assert len(client.updated_numbers) == 1
    assert client.updated_numbers[0][0] == "item-2"


def test_sync_scores_targeted_missing_issue_returns_no_updates() -> None:
    """A missing issue number yields a bounded empty result, not an ambiguous scan."""

    client = FakeGhProjectClient(
        fields=[
            _field(name)
            for name in (
                EFFORT_FIELD,
                *REQUIRED_NUMBER_FIELDS,
            )
        ],
        items=[_item(1, improvement=5, **{lower_first_key(EFFORT_FIELD): 8})],
    )

    previews = sync_scores(
        client,
        SyncOptions(
            owner="ll7",
            project_number=5,
            ensure_fields=False,
            limit=400,
            alpha=DEFAULT_ALPHA,
            round_digits=6,
            issue_number=999,
            dry_run=False,
            skip_statuses={"Done"},
        ),
    )

    assert previews == []
    assert client.updated_numbers == []


def test_write_summary_persists_machine_readable_payload(tmp_path: Path) -> None:
    """Verify the optional summary artifact is reproducible JSON.

    This matters because the scheduled workflow should be able to upload a sync
    report artifact that explains what score changes were computed.
    """

    preview = build_previews(
        [_item(42, improvement=3, **{lower_first_key(EFFORT_FIELD): 2})],
        alpha=DEFAULT_ALPHA,
        round_digits=6,
        issue_number=None,
        skip_statuses={"Done"},
    )[0]
    summary = tmp_path / "priority-score-summary.json"

    write_summary(summary, [preview])

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["items"][0]["issue_number"] == 42
    assert payload["items"][0]["inputs"]["improvement"] == 3


def test_load_project_cache_accepts_matching_field_id_hints(tmp_path: Path) -> None:
    """A matching local cache can support a no-write targeted read without Project metadata calls."""
    cache = tmp_path / "project5.json"
    fields = {
        name: {"id": f"field-{name}", "type": "number"}
        for name in (EFFORT_FIELD, *REQUIRED_NUMBER_FIELDS)
    }
    cache.write_text(
        json.dumps(
            {
                "owner": "ll7",
                "project_number": 5,
                "project_id": "project-id",
                "fields": fields,
            }
        ),
        encoding="utf-8",
    )

    metadata = load_project_cache(cache, owner="ll7", project_number=5)

    assert metadata is not None
    assert metadata.project_id == "project-id"
    assert metadata.fields[PRIORITY_SCORE_FIELD]["name"] == PRIORITY_SCORE_FIELD
    assert load_project_cache(cache, owner="other", project_number=5) is None

    client = FakeGhProjectClient(
        fields=[],
        items=[_item(42, improvement=3, **{lower_first_key(EFFORT_FIELD): 2})],
    )
    previews = sync_scores(
        client,
        SyncOptions(
            owner="ll7",
            project_number=5,
            ensure_fields=False,
            limit=400,
            alpha=DEFAULT_ALPHA,
            round_digits=6,
            issue_number=42,
            dry_run=True,
            skip_statuses={"Done"},
            cache_file=cache,
        ),
    )
    assert [preview.issue_number for preview in previews] == [42]
    assert client.field_list_calls == 0
    assert client.updated_numbers == []


def lower_first_key(name: str) -> str:
    """Mirror the current gh item-list key shape for multi-word fields."""

    return name[:1].lower() + name[1:]


def test_field_keys_accept_both_known_cli_variants() -> None:
    """Verify field lookup handles both lower-first and fully-lowercase keys.

    This matters because the live gh CLI output uses lower-first keys like
    `expected Duration in Hours`, while local fixtures often use lowercase.
    """

    assert field_keys("Improvement") == ("improvement",)
    assert field_keys("Priority Score") == ("priority Score", "priority score")


def test_compute_priority_score_matches_expected_value_extension() -> None:
    """Verify the score follows the documented extended EV-per-effort formula.

    This matters because the prioritization model is the public contract of the
    workflow, so its implementation must match the documented equation exactly.
    """

    inputs = ScoreInputs(
        improvement=5.0,
        success_probability=0.7,
        effort_hours=8.0,
        time_criticality=1.5,
        unlock_factor=2.0,
    )

    expected = (5.0 * 0.7 * 1.5 * 2.0) / (8.0**DEFAULT_ALPHA)
    assert compute_priority_score(inputs, alpha=DEFAULT_ALPHA) == expected


def test_gh_project_client_surfaces_actionable_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify gh failures include MCP-first and auth-scope guidance.

    This matters because the score-sync helper intentionally remains the
    deterministic `gh` fallback, so command failures should explain how to
    recover instead of surfacing an opaque subprocess error.
    """

    def _raise(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Raise an authentication failure from the gh CLI fallback."""
        raise subprocess.CalledProcessError(
            1,
            ["gh", "project", "item-list"],
            output="",
            stderr="authentication failed",
        )

    client = GhProjectClient()
    monkeypatch.setattr(subprocess, "run", _raise)

    with pytest.raises(RuntimeError, match="prefer the GitHub MCP/app tools"):
        client.item_list(owner="ll7", project_number=5, limit=1)


def test_gh_project_client_classifies_missing_read_project_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing read:project is distinct from an arbitrary gh authentication failure."""

    def _raise(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            1,
            ["gh", "project", "field-list"],
            output="",
            stderr="error: your authentication token is missing required scopes [read:project]",
        )

    monkeypatch.setattr(subprocess, "run", _raise)

    with pytest.raises(MissingProjectScopeError) as exc_info:
        GhProjectClient().field_list(owner="ll7", project_number=5)

    assert exc_info.value.required_scopes == ("read:project",)
    assert exc_info.value.command[:3] == ("gh", "project", "field-list")


def test_main_only_empty_missing_scope_is_non_fatal_json_without_writes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The autopilot auto-fill path reports a blocker and leaves score writes untouched."""

    def _raise(*args: object, **kwargs: object) -> list[dict]:
        raise MissingProjectScopeError(
            command=("gh", "project", "field-list", "5"),
            details="error: your authentication token is missing required scopes [read:project]",
            required_scopes=("read:project",),
        )

    updates: list[float] = []
    monkeypatch.setattr(GhProjectClient, "field_list", _raise)
    monkeypatch.setattr(
        GhProjectClient,
        "update_number_field",
        lambda self, **kwargs: updates.append(float(kwargs["number"])),
    )

    assert main(["sync", "--only-empty", "--ensure-fields"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["reason"] == "missing_project_scope"
    assert payload["required_scopes"] == ["read:project"]
    assert payload["fallback"] == "live-label ordering"
    assert payload["non_fatal"] is True
    assert payload["writes_performed"] is False
    assert payload["items"] == []
    assert updates == []


def test_main_non_empty_scope_failure_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the auto-fill mode converts a missing Project scope into a success status."""

    monkeypatch.setattr(
        GhProjectClient,
        "field_list",
        lambda self, **kwargs: (_ for _ in ()).throw(
            MissingProjectScopeError(
                command=("gh", "project", "field-list", "5"),
                details="missing required scopes [read:project]",
                required_scopes=("read:project",),
            )
        ),
    )

    with pytest.raises(MissingProjectScopeError):
        main(["sync"])


def test_gh_project_client_timeout_raises_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung `gh` subprocess must surface command, budget, and phase (issue #8263)."""

    seen: dict[str, object] = {}

    def _hang(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(cmd=["gh", "project", "item-list"], timeout=120.0)

    monkeypatch.setattr(subprocess, "run", _hang)

    with pytest.raises(GhProjectTimeoutError) as exc_info:
        GhProjectClient().item_list(owner="ll7", project_number=5, limit=1)

    assert seen["timeout"] == 120.0
    assert exc_info.value.command[:3] == ("gh", "project", "item-list")
    assert exc_info.value.timeout_seconds == 120.0
    assert exc_info.value.phase == "read"


def test_gh_project_client_timeout_records_write_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeouts during the apply stage must name the write phase (issue #8263)."""

    def _hang(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["gh", "project", "item-edit"], timeout=5.0)

    monkeypatch.setattr(subprocess, "run", _hang)
    client = GhProjectClient(timeout_seconds=5.0)
    client.current_phase = "write"

    with pytest.raises(GhProjectTimeoutError) as exc_info:
        client.run("project", "item-edit", "5")

    assert exc_info.value.phase == "write"
    assert exc_info.value.timeout_seconds == 5.0


def test_gh_project_client_rejects_non_positive_or_non_finite_timeout() -> None:
    """Invalid subprocess budgets are programming errors, not slow API calls."""

    for value in (0, -1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="must be positive and finite"):
            GhProjectClient(timeout_seconds=value)
        with pytest.raises(ValueError, match="must be positive and finite"):
            GhProjectClient(total_budget_seconds=value)


def test_main_only_empty_timeout_is_non_fatal_json_without_writes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The autopilot auto-fill path reports a timeout and leaves score writes untouched."""

    def _raise(*args: object, **kwargs: object) -> list[dict]:
        raise GhProjectTimeoutError(
            command=("gh", "project", "field-list", "5"),
            timeout_seconds=120.0,
            phase="read",
        )

    updates: list[float] = []
    monkeypatch.setattr(GhProjectClient, "field_list", _raise)
    monkeypatch.setattr(
        GhProjectClient,
        "update_number_field",
        lambda self, **kwargs: updates.append(float(kwargs["number"])),
    )

    assert main(["sync", "--only-empty", "--ensure-fields"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "timeout_blocked"
    assert payload["reason"] == "gh_subprocess_timeout"
    assert payload["command"][:3] == ["gh", "project", "field-list"]
    assert payload["timeout_seconds"] == 120.0
    assert payload["phase"] == "read"
    assert payload["retryable"] is True
    assert payload["fallback"] == "live-label ordering"
    assert payload["non_fatal"] is True
    assert payload["writes_performed"] is False
    assert payload["write_ambiguity"] is False
    assert payload["items"] == []
    assert updates == []


def test_main_non_empty_timeout_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the auto-fill mode converts a `gh` timeout into a success status."""

    monkeypatch.setattr(
        GhProjectClient,
        "field_list",
        lambda self, **kwargs: (_ for _ in ()).throw(
            GhProjectTimeoutError(
                command=("gh", "project", "field-list", "5"),
                timeout_seconds=120.0,
                phase="read",
            )
        ),
    )

    with pytest.raises(GhProjectTimeoutError):
        main(["sync"])


def test_main_gh_timeout_flag_reaches_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--gh-timeout configures the per-command subprocess budget."""

    seen: dict[str, float] = {}

    def _fake_sync_scores(client: GhProjectClient, options: SyncOptions) -> list[SyncPreview]:
        seen["timeout"] = client.timeout_seconds
        seen["options_timeout"] = options.gh_timeout_seconds
        seen["total_budget"] = client.total_budget_seconds
        seen["options_total_budget"] = options.gh_total_budget_seconds
        return []

    monkeypatch.setattr(project_priority_score, "sync_scores", _fake_sync_scores)

    assert main(["sync", "--only-empty", "--gh-timeout", "5", "--gh-total-budget", "60"]) == 0
    assert seen["timeout"] == 5.0
    assert seen["options_timeout"] == 5.0
    assert seen["total_budget"] == 60.0
    assert seen["options_total_budget"] == 60.0


def test_main_gh_timeout_rejects_non_positive_or_non_finite() -> None:
    """Invalid CLI budgets fail closed before quota reads or Project writes."""

    for flag in ("--gh-timeout", "--gh-total-budget"):
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with pytest.raises(SystemExit) as exc_info:
                main(["sync", f"{flag}={value}"])
            assert exc_info.value.code == 2


def test_gh_project_client_total_budget_breach_skips_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exhausted operation budget fails closed before starting `gh` (issue #8263)."""
    calls: list[list[str]] = []

    def _record(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args[0]))  # type: ignore[arg-type]
        raise AssertionError("subprocess must not start after budget exhaustion")

    monkeypatch.setattr(subprocess, "run", _record)
    client = GhProjectClient(timeout_seconds=120.0, total_budget_seconds=60.0)
    client._budget_start -= 61.0

    with pytest.raises(GhProjectTimeoutError) as exc_info:
        client.run("project", "item-list", "5")

    assert calls == []
    assert exc_info.value.budget_exhausted is True
    assert exc_info.value.timeout_seconds == 60.0
    assert exc_info.value.phase == "read"
    assert "was not started" in str(exc_info.value)


def test_gh_project_client_caps_command_at_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each command is capped at the remaining operation budget (issue #8263)."""
    seen: dict[str, float] = {}

    def _record(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["timeout"] = float(kwargs["timeout"])
        return subprocess.CompletedProcess(args=list(args[0]), returncode=0, stdout="{}")

    monkeypatch.setattr(subprocess, "run", _record)
    client = GhProjectClient(timeout_seconds=120.0, total_budget_seconds=60.0)
    client._budget_start -= 55.0

    client.run_json("project", "view", "5")

    assert 0 < seen["timeout"] <= 5.5


def test_ensure_required_fields_marks_field_creation_as_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A field-create timeout reports the attempted field and ambiguous outcome."""

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = list(args[0])  # type: ignore[arg-type]
        if command[1:3] == ["project", "field-list"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({"fields": []}),
            )
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", _run)

    with pytest.raises(GhProjectTimeoutError) as exc_info:
        project_priority_score.ensure_required_fields(
            GhProjectClient(timeout_seconds=5.0),
            owner="ll7",
            project_number=5,
        )

    error = exc_info.value
    assert error.phase == "write"
    assert error.attempted_field_names == [REQUIRED_NUMBER_FIELDS[0]]
    assert error.created_field_names == []
    assert error.completed_write_count == 0
    assert error.writes_performed_count == 0
    assert error.write_ambiguity is True
    payload = project_priority_score._blocked_project_timeout_payload(
        owner="ll7", project_number=5, error=error
    )
    assert payload["attempted_field_names"] == [REQUIRED_NUMBER_FIELDS[0]]
    assert payload["created_field_names"] == []
    assert payload["completed_field_names"] == []
    assert payload["completed_write_count"] == 0
    assert payload["writes_performed"] is False
    assert payload["write_ambiguity"] is True


def test_ensure_required_fields_reports_partial_field_creation_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later field-create timeout retains earlier schema mutations and ambiguity."""

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = list(args[0])  # type: ignore[arg-type]
        if command[1:3] == ["project", "field-list"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({"fields": []}),
            )
        field_name = command[command.index("--name") + 1]
        if field_name == REQUIRED_NUMBER_FIELDS[1]:
            raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs["timeout"])
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)

    with pytest.raises(GhProjectTimeoutError) as exc_info:
        project_priority_score.ensure_required_fields(
            GhProjectClient(timeout_seconds=5.0),
            owner="ll7",
            project_number=5,
        )

    error = exc_info.value
    assert error.phase == "write"
    assert error.attempted_field_names == list(REQUIRED_NUMBER_FIELDS[:2])
    assert error.created_field_names == [REQUIRED_NUMBER_FIELDS[0]]
    assert error.completed_write_count == 1
    assert error.writes_performed_count == 1
    assert error.write_ambiguity is True
    payload = project_priority_score._blocked_project_timeout_payload(
        owner="ll7", project_number=5, error=error
    )
    assert payload["completed_field_names"] == [REQUIRED_NUMBER_FIELDS[0]]
    assert payload["completed_write_count"] == 1
    assert payload["writes_performed"] is True
    assert payload["write_ambiguity"] is True


def test_ensure_required_fields_reports_completed_fields_when_final_read_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-creation read timeout retains completed schema writes without ambiguity."""

    field_list_calls = 0

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal field_list_calls
        command = list(args[0])  # type: ignore[arg-type]
        if command[1:3] == ["project", "field-list"]:
            field_list_calls += 1
            if field_list_calls == 2:
                raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs["timeout"])
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({"fields": []}),
            )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)

    with pytest.raises(GhProjectTimeoutError) as exc_info:
        project_priority_score.ensure_required_fields(
            GhProjectClient(timeout_seconds=5.0),
            owner="ll7",
            project_number=5,
        )

    error = exc_info.value
    assert error.phase == "read"
    assert error.attempted_field_names == list(REQUIRED_NUMBER_FIELDS)
    assert error.created_field_names == list(REQUIRED_NUMBER_FIELDS)
    assert error.completed_write_count == len(REQUIRED_NUMBER_FIELDS)
    assert error.writes_performed_count == len(REQUIRED_NUMBER_FIELDS)
    assert error.write_ambiguity is False
    payload = project_priority_score._blocked_project_timeout_payload(
        owner="ll7", project_number=5, error=error
    )
    assert payload["writes_performed"] is True
    assert payload["completed_write_count"] == len(REQUIRED_NUMBER_FIELDS)
    assert payload["write_ambiguity"] is False


def test_ensure_required_fields_reports_unstarted_field_creation_budget_breach() -> None:
    """A field-create budget breach reports a planned field without ambiguity."""

    client = GhProjectClient(timeout_seconds=5.0, total_budget_seconds=60.0)
    client.field_list = lambda **kwargs: []

    def _budget_breach(**kwargs: object) -> None:
        raise GhProjectTimeoutError(
            command=("gh", "project", "field-create", "5"),
            timeout_seconds=60.0,
            phase=client.current_phase,
            budget_exhausted=True,
        )

    client.ensure_number_field = _budget_breach

    with pytest.raises(GhProjectTimeoutError) as exc_info:
        project_priority_score.ensure_required_fields(
            client,
            owner="ll7",
            project_number=5,
        )

    error = exc_info.value
    assert error.attempted_field_names == [REQUIRED_NUMBER_FIELDS[0]]
    assert error.created_field_names == []
    assert error.completed_write_count == 0
    assert error.write_ambiguity is False
    payload = project_priority_score._blocked_project_timeout_payload(
        owner="ll7", project_number=5, error=error
    )
    assert payload["writes_performed"] is False
    assert payload["completed_write_count"] == 0
    assert payload["write_ambiguity"] is False


def test_main_only_empty_budget_breach_reports_unstarted_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The auto-fill path reports a budget breach without claiming writes."""

    def _raise(*args: object, **kwargs: object) -> list[dict]:
        raise GhProjectTimeoutError(
            command=("gh", "project", "item-list", "5"),
            timeout_seconds=600.0,
            phase="read",
            budget_exhausted=True,
        )

    monkeypatch.setattr(GhProjectClient, "field_list", _raise)

    assert main(["sync", "--only-empty"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "timeout_blocked"
    assert payload["budget_exhausted"] is True
    assert payload["timeout_seconds"] == 600.0
    assert payload["writes_performed"] is False
    assert "was not started" in payload["message"]


def test_write_budget_breach_is_not_reported_as_ambiguous(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A write-phase budget exhaustion names an unstarted command, not an ambiguous write."""

    error = GhProjectTimeoutError(
        command=("gh", "api", "graphql"),
        timeout_seconds=60.0,
        phase="write",
        budget_exhausted=True,
    )

    assert (
        project_priority_score._handle_only_empty_failure(
            args=argparse.Namespace(owner="ll7", project_number=5, only_empty=True),
            error=error,
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["write_ambiguity"] is False
    assert "may still have landed" not in payload["message"]


def test_apply_score_updates_reraises_write_phase_timeout() -> None:
    """A partial timed-out write reports known progress and remains ambiguous."""
    from types import SimpleNamespace

    calls = 0

    def _update(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise GhProjectTimeoutError(
                command=("gh", "api", "graphql"),
                timeout_seconds=5.0,
                phase="write",
            )

    plan: dict[str, object] = {}
    client = SimpleNamespace(
        last_eligibility_plan=plan,
        update_number_field=_update,
    )
    options = SyncOptions(
        owner="ll7",
        project_number=5,
        ensure_fields=False,
        limit=10,
        alpha=0.8,
        round_digits=6,
        issue_number=None,
        dry_run=False,
        skip_statuses=set(),
    )
    previews = [
        SimpleNamespace(issue_number=1, new_score=3.0),
        SimpleNamespace(issue_number=2, new_score=4.0),
    ]

    with pytest.raises(GhProjectTimeoutError) as exc_info:
        project_priority_score._apply_score_updates(
            client,
            options,
            [(previews[0], {"id": "item-1"}), (previews[1], {"id": "item-2"})],
            issue_snapshots={},
            score_field_id="field-1",
            project_id="project-1",
        )
    assert exc_info.value.phase == "write"
    assert exc_info.value.attempted_rows == [
        {"issue_number": 1, "item_id": "item-1", "written": True},
        {"issue_number": 2, "item_id": "item-2", "written": False},
    ]
    assert exc_info.value.writes_performed_count == 1
    payload = project_priority_score._blocked_project_timeout_payload(
        owner="ll7",
        project_number=5,
        error=exc_info.value,
    )
    assert payload["attempted_rows"] == exc_info.value.attempted_rows
    assert payload["writes_performed_count"] == 1
    assert payload["writes_performed"] is True
    assert payload["write_ambiguity"] is True
    assert "1 score write(s) completed" in payload["message"]
    assert plan["status"] == "timeout_blocked"
    assert plan["writes_performed_count"] == 1


def test_main_reports_complete_item_fetch_stats(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Successful sync output exposes the cursor-scan completeness evidence."""

    client = FakeGhProjectClient(
        fields=[
            _field(name)
            for name in (
                EFFORT_FIELD,
                *REQUIRED_NUMBER_FIELDS,
            )
        ],
        items=[_item(702, improvement=5, **{lower_first_key(EFFORT_FIELD): 8})],
    )
    client.last_item_fetch_stats = ProjectItemFetchStats(pages=32, accumulated_items=3137)
    monkeypatch.setattr(project_priority_score, "GhProjectClient", lambda **kwargs: client)

    assert main(["sync", "--dry-run", "--limit", "1000"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["item_fetch"] == {"pages": 32, "accumulated_items": 3137}
    assert payload["items"][0]["issue_number"] == 702


def test_main_quota_block_is_explicit_and_performs_no_project_writes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A low quota blocks before schema/item reads or score writes and can be resumed later."""
    monkeypatch.setattr(
        project_priority_score,
        "read_rate_limit",
        lambda: RateLimitSnapshot(
            status="ok",
            graphql_remaining=3,
            graphql_reset_at=1_800_000_123,
            core_remaining=4_000,
            core_reset_at=1_800_000_456,
        ),
    )
    project_reads: list[str] = []
    monkeypatch.setattr(
        GhProjectClient,
        "field_list",
        lambda self, **kwargs: project_reads.append("field-list") or [],
    )

    assert main(["sync", "--only-empty"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "quota_blocked"
    assert payload["writes_performed"] is False
    assert payload["non_fatal"] is True
    assert payload["resume_after"] == 1_800_000_123
    assert project_reads == []

    assert main(["sync"]) == 2
    second_payload = json.loads(capsys.readouterr().out)
    assert second_payload["status"] == "quota_blocked"
    assert second_payload["non_fatal"] is False
    assert project_reads == []


def test_gh_project_client_retries_user_owned_project_commands_with_at_me(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify project commands retry with `@me` after the owner-type gh quirk.

    This matters because the scripted score-sync path should stay reliable on
    gh builds that reject an explicit user login such as `ll7` for some
    `gh project` subcommands.
    """

    from scripts.tools.project_priority_score import GhProjectClient

    calls: list[list[str]] = []

    def _fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        """Fail for explicit user ownership and succeed for @me retry."""
        calls.append(args)
        if "--owner" in args and args[args.index("--owner") + 1] == "ll7":
            raise subprocess.CalledProcessError(
                1,
                args,
                output="",
                stderr="unknown owner type",
            )
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"fields": []}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    client = GhProjectClient()
    assert client.field_list(owner="ll7", project_number=5) == []
    assert calls[0][0:4] == ["gh", "project", "field-list", "5"]
    assert calls[0][calls[0].index("--owner") + 1] == "ll7"
    assert calls[1][calls[1].index("--owner") + 1] == "@me"


def test_gh_project_client_does_not_retry_unknown_owner_with_at_me(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify only the known ll7 owner quirk can trigger the `@me` fallback.

    This matters because silently retrying arbitrary owners could read or write
    the wrong project if the helper grows beyond the current ll7-only usage.
    """

    from scripts.tools.project_priority_score import GhProjectClient

    calls: list[list[str]] = []

    def _fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        """Always raise an owner-type failure for non-retriable owners."""
        calls.append(args)
        raise subprocess.CalledProcessError(
            1,
            args,
            output="",
            stderr="unknown owner type",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    client = GhProjectClient()
    with pytest.raises(RuntimeError, match="unknown owner type"):
        client.field_list(owner="octocat", project_number=5)

    assert len(calls) == 1
    assert calls[0][calls[0].index("--owner") + 1] == "octocat"


def test_terminal_project_row_never_receives_an_item_edit() -> None:
    """A row the plan skips as terminal must never produce an item-edit (#8152)."""
    client = FakeGhProjectClient(
        fields=[_field(name) for name in (EFFORT_FIELD, *REQUIRED_NUMBER_FIELDS)],
        items=[
            _item(8021, status="Done"),
            _item(701),
        ],
    )
    previews = sync_scores(
        client,
        SyncOptions(
            owner="ll7",
            project_number=5,
            ensure_fields=False,
            limit=100,
            alpha=DEFAULT_ALPHA,
            round_digits=6,
            issue_number=None,
            dry_run=False,
            skip_statuses=set(),
            only_empty=True,
        ),
    )

    # The terminal row is excluded from the eligible write set entirely.
    assert sorted(preview.issue_number for preview in previews) == [701]
    written_items = [item_id for item_id, _f, _p, _n in client.updated_numbers]
    assert written_items == ["item-701"]
    assert client.last_eligibility_plan is not None
    assert client.last_eligibility_plan["counts"]["skipped"] == 1
    assert client.last_eligibility_plan["writes_performed"] is True


def test_numeric_field_literal_stays_plain_and_within_eight_decimals() -> None:
    """Written literals are plain decimals with at most 8 decimal places."""
    assert _numeric_field_literal(0.055072) == "0.055072"
    assert _numeric_field_literal(0.000012345) == "0.00001234"
    assert _numeric_field_literal(12.055072123) == "12.05507212"
    assert _numeric_field_literal(3.5) == "3.5"
    assert "e" not in _numeric_field_literal(0.00001)
    with pytest.raises(ValueError, match="8-decimal limit"):
        _numeric_field_literal(float("nan"))


def test_update_number_field_uses_exact_graphql_decimal_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid gh's float32 item-edit path when writing a Project Number field."""

    calls: list[list[str]] = []

    def _fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"data":{"updateProjectV2ItemFieldValue":{"projectV2Item":{"id":"item-1"}}}}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    GhProjectClient().update_number_field(
        item_id="item-1",
        field_id="field-1",
        project_id="project-1",
        number=0.7,
    )

    assert len(calls) == 1
    command = calls[0]
    assert command[:3] == ["gh", "api", "graphql"]
    assert "--number" not in command
    query = command[command.index("-f") + 1]
    assert "number: 0.7" in query
    assert "number: $number" not in query
    assert "projectId=project-1" in command
    assert "itemId=item-1" in command
    assert "fieldId=field-1" in command


@pytest.mark.parametrize(
    "payload",
    [
        {"data": None},
        {"data": {"updateProjectV2ItemFieldValue": {"projectV2Item": {}}}},
    ],
)
def test_update_number_field_rejects_malformed_graphql_response(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    """Malformed successful responses never masquerade as a completed write."""

    def _fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="returned no updated project item"):
        GhProjectClient().update_number_field(
            item_id="item-1",
            field_id="field-1",
            project_id="project-1",
            number=0.7,
        )


class _RejectingGhProjectClient(FakeGhProjectClient):
    """Fake client whose numeric writes are rejected like a live gh failure."""

    def update_number_field(self, **_: object) -> None:
        """Raise the gh error shape observed on numeric-shape rejection."""

        raise RuntimeError("gh: Number values cannot exceed 8 decimal places")


def test_apply_failure_writes_structured_summary_and_never_raises(tmp_path: Path) -> None:
    """A rejected write is recorded with attempted rows and retryability (#8152)."""
    client = _RejectingGhProjectClient(
        fields=[_field(name) for name in (EFFORT_FIELD, *REQUIRED_NUMBER_FIELDS)],
        items=[_item(701)],
    )
    previews = sync_scores(
        client,
        SyncOptions(
            owner="ll7",
            project_number=5,
            ensure_fields=False,
            limit=100,
            alpha=DEFAULT_ALPHA,
            round_digits=6,
            issue_number=None,
            dry_run=False,
            skip_statuses=set(),
            only_empty=True,
        ),
    )

    assert previews == []
    plan = client.last_eligibility_plan
    assert plan is not None
    assert plan["status"] == "apply_failed"
    failure = plan["failure"]
    assert failure["retryable"] is True
    assert failure["writes_performed"] == 0
    assert failure["attempted_rows"] == [
        {"issue_number": 701, "item_id": "item-701", "written": False}
    ]
    assert "Number values cannot exceed 8 decimal places" in failure["error"]

    summary_path = tmp_path / "summary.json"
    write_summary(summary_path, previews, plan)
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["eligibility_plan"]["status"] == "apply_failed"
    assert persisted["eligibility_plan"]["failure"]["retryable"] is True
