"""Tests for the Project #5 priority score sync helper."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.tools.project_priority_score import (
    DEFAULT_ALPHA,
    DEFAULT_SUCCESS_PROBABILITY,
    EFFORT_FIELD,
    PRIORITY_SCORE_FIELD,
    ScoreInputs,
    SyncOptions,
    build_previews,
    compute_priority_score,
    field_keys,
    normalize_inputs,
    sync_scores,
    write_summary,
)

if TYPE_CHECKING:
    from pathlib import Path


class FakeGhProjectClient:
    """Deterministic fake gh client for score-sync tests."""

    def __init__(self, *, fields: list[dict], items: list[dict]) -> None:
        """Store fake project fields, items, and captured side effects."""

        self._fields = fields
        self._items = items
        self.created_fields: list[str] = []
        self.updated_numbers: list[tuple[str, str, str, float]] = []
        self.field_list_calls = 0

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
    return {"id": f"field-{name}", "name": name, "type": "ProjectV2Field"}


def _item(issue_number: int, **fields: object) -> dict:
    payload = {
        "id": f"item-{issue_number}",
        "status": "Todo",
        "content": {
            "type": "Issue",
            "number": issue_number,
            "title": f"Issue {issue_number}",
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
            "improvement": -4,
            "success probability": 1.7,
            lower_first_key(EFFORT_FIELD): 0,
            lower_first_key("Time Criticality"): 9,
            lower_first_key("Unlock Factor"): 0.1,
        }
    )

    assert inputs == ScoreInputs(
        improvement=0.0,
        success_probability=1.0,
        effort_hours=0.1,
        time_criticality=2.0,
        unlock_factor=1.0,
    )

    defaulted = normalize_inputs({})
    assert defaulted.success_probability == DEFAULT_SUCCESS_PROBABILITY
    assert defaulted.effort_hours == 1.0


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
