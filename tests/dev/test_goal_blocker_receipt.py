"""Tests for deterministic goal-autopilot blocker receipts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev import autopilot_state_snapshot, snapshot_issue_batch
from scripts.dev.github_quota import RateLimitSnapshot
from scripts.dev.goal_blocker_receipt import (
    BLOCKER_CLASSES,
    build_fingerprint_inputs,
    build_receipt,
    evaluate_redispatch,
    fingerprint,
    load_receipt,
    missing,
    not_applicable,
    unavailable,
    validate_receipt,
    write_receipt,
)

if TYPE_CHECKING:
    from pathlib import Path


def _inputs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "issue_body_digest": "body-digest-1",
        "issue_labels": ["state:blocked", "type:workflow"],
        "origin_main_sha": "a" * 40,
        "pr_number": not_applicable("no relevant PR"),
        "base_sha": not_applicable("no relevant PR"),
        "head_sha": not_applicable("no relevant PR"),
        "dependency_state": {"closure": "open"},
        "required_inputs": {"artifact": "missing"},
    }
    values.update(overrides)
    return build_fingerprint_inputs(**values)


def _receipt(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "repository": "ll7/robot_sf_ll7",
        "issue_number": 7612,
        "blocker_class": "dependency",
        "required_input": "dependency closure",
        "unblock_condition": "dependency is merged and visible",
        "evidence": [{"source": "issue:#7612", "detail": "dependency remains open"}],
        "safe_work_completed": ["mapped the canonical dependency owner"],
        "next_owner": "maintainer",
        "retryable": True,
        **{
            "issue_body_digest": "body-digest-1",
            "issue_labels": ["state:blocked", "type:workflow"],
            "origin_main_sha": "a" * 40,
            "pr_number": not_applicable("no relevant PR"),
            "base_sha": not_applicable("no relevant PR"),
            "head_sha": not_applicable("no relevant PR"),
            "dependency_state": {"closure": "open"},
            "required_inputs": {"artifact": "missing"},
        },
    }
    values.update(overrides)
    return build_receipt(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("blocker_class", sorted(BLOCKER_CLASSES))
def test_builder_supports_every_blocker_class(blocker_class: str) -> None:
    """Every issue-defined blocker class should produce a valid receipt."""
    receipt = _receipt(blocker_class=blocker_class)

    assert receipt["schema"] == "goal_blocker_receipt.v1"
    assert validate_receipt(receipt) == {"ok": True, "errors": []}
    assert receipt["blocker"]["class"] == blocker_class


def test_identical_inputs_have_stable_fingerprint_even_when_mapping_order_changes() -> None:
    """Equivalent issue, dependency, and input mappings must hash identically."""
    first = _inputs(
        issue_labels=["type:workflow", "state:blocked"],
        dependency_state={"b": 2, "a": 1},
        required_inputs={"z": False, "a": []},
    )
    second = _inputs(
        issue_labels=["state:blocked", "type:workflow"],
        dependency_state={"a": 1, "b": 2},
        required_inputs={"a": [], "z": False},
    )

    assert fingerprint(first) == fingerprint(second)


def test_empty_false_missing_and_unavailable_values_are_distinct() -> None:
    """Field-state tags prevent absent data from colliding with valid empty values."""
    empty = _inputs(dependency_state=[])
    false = _inputs(dependency_state=False)
    absent = _inputs(dependency_state=missing("dependency response omitted"))
    unavailable_state = _inputs(dependency_state=unavailable("GitHub API unavailable"))

    assert (
        len(
            {
                fingerprint(empty),
                fingerprint(false),
                fingerprint(absent),
                fingerprint(unavailable_state),
            }
        )
        == 4
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_path"),
    [
        ("issue_body_digest", "body-digest-2", "issue.body_digest"),
        ("issue_labels", ["state:blocked", "type:workflow", "priority:2"], "issue.labels"),
        ("origin_main_sha", "b" * 40, "repository.origin_main_sha"),
        ("base_sha", "b" * 40, "pull_request.base_sha"),
        ("head_sha", "c" * 40, "pull_request.head_sha"),
        ("dependency_state", {"closure": "merged"}, "dependency_state"),
        ("required_inputs", {"artifact": "available"}, "required_inputs"),
    ],
)
def test_changed_invalidating_field_reopens_redispatch(
    field: str, value: object, expected_path: str
) -> None:
    """Every material issue, head, dependency, or required-input change re-evaluates."""
    receipt = _receipt()
    current = _inputs(**{field: value})

    decision = evaluate_redispatch(
        receipt,
        current_inputs=current,
        repository="ll7/robot_sf_ll7",
        issue_number=7612,
    )

    assert decision["decision"] == "blocker_changed"
    assert decision["action"] == "re_evaluate"
    assert expected_path in decision["changed_fields"]


def test_matching_receipt_suppresses_new_worker_dispatch() -> None:
    """An exact current match is the sole suppressing redispatch result."""
    receipt = _receipt()

    decision = evaluate_redispatch(
        receipt,
        current_inputs=_inputs(),
        repository="ll7/robot_sf_ll7",
        issue_number=7612,
    )

    assert decision == {
        "decision": "blocked_unchanged",
        "action": "no_action",
        "reason": "fingerprint_matches",
        "changed_fields": [],
        "errors": [],
    }


def test_missing_or_unavailable_current_fields_fail_open() -> None:
    """The loop must re-evaluate instead of suppressing when current state is incomplete."""
    receipt = _receipt()

    missing_decision = evaluate_redispatch(
        receipt,
        current_inputs=build_fingerprint_inputs(),
        repository="ll7/robot_sf_ll7",
        issue_number=7612,
    )
    unavailable_decision = evaluate_redispatch(
        receipt,
        current_inputs=_inputs(origin_main_sha=unavailable("origin/main read failed")),
        repository="ll7/robot_sf_ll7",
        issue_number=7612,
    )

    assert missing_decision["decision"] == "current_state_unavailable"
    assert unavailable_decision["decision"] == "current_state_unavailable"
    assert missing_decision["action"] == unavailable_decision["action"] == "re_evaluate"


def test_malformed_or_stale_receipts_fail_open() -> None:
    """Tampered fingerprints and identity mismatches never park an issue silently."""
    receipt = _receipt()
    receipt["blocker"]["fingerprint"] = "sha256:" + "0" * 64

    invalid = evaluate_redispatch(
        receipt,
        current_inputs=_inputs(),
        repository="ll7/robot_sf_ll7",
        issue_number=7612,
    )
    stale = evaluate_redispatch(
        _receipt(),
        current_inputs=_inputs(),
        repository="other/repository",
        issue_number=7612,
    )

    assert invalid["decision"] == "invalid_receipt"
    assert stale["decision"] == "stale_receipt"
    assert invalid["action"] == stale["action"] == "re_evaluate"


def test_receipt_round_trip_uses_external_directory_and_snapshot_reports_decision(
    tmp_path: Path,
) -> None:
    """The snapshot owner should read the external receipt and expose its decision."""
    receipt = _receipt()
    path = write_receipt(receipt, path=tmp_path / "issue-7612.json")
    loaded = load_receipt(issue_number=7612, directory=tmp_path)

    assert path == tmp_path / "issue-7612.json"
    assert loaded["status"] == "available"
    row = snapshot_issue_batch._blocker_receipt_snapshot(
        issue_number=7612,
        repository="ll7/robot_sf_ll7",
        labels=["type:workflow", "state:blocked"],
        body_digest="body-digest-1",
        receipts_dir=str(tmp_path),
        current_inputs=_inputs(),
    )
    assert row["fingerprint"] == receipt["blocker"]["fingerprint"]
    assert row["decision"] == "blocked_unchanged"
    assert row["action"] == "no_action"


def test_claimable_snapshot_fences_unchanged_blocker_from_dispatch(tmp_path: Path) -> None:
    """A matching receipt changes the queue classification before worker selection."""
    write_receipt(_receipt(), path=tmp_path / "issue-7612.json")
    issue_list = [
        {
            "number": 7612,
            "title": "workflow blocker",
            "state": "OPEN",
            "url": "https://github.test/issues/7612",
            "labels": [{"name": "state:ready"}],
            "assignees": [],
        }
    ]
    with (
        patch("scripts.dev.snapshot_issue_batch._gh") as gh,
        patch(
            "scripts.dev.snapshot_issue_batch._rate_limit_snapshot",
            return_value=RateLimitSnapshot(
                status="ok",
                graphql_remaining=4_000,
                graphql_reset_at=1_800_000_000,
                core_remaining=4_000,
                core_reset_at=1_800_000_000,
            ),
        ),
        patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claims,
    ):
        gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        claims.return_value = {7612: {"ok": True, "claimed": False, "sha": None}}
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=1,
            blocker_receipts_dir=str(tmp_path),
            blocker_current_inputs=_inputs(),
        )

    row = payload["issues"][0]
    assert row["classification"] == "blocked_receipt"
    assert row["dispatch_allowed"] is False
    assert row["reason"] == "unchanged blocker fingerprint; skip autonomous claim"
    assert payload["redispatch"]["suppressed_redispatch_count"] == 1


def test_snapshot_summary_counts_suppression_and_reason() -> None:
    """Queue summaries must make suppressed and re-evaluated rows visible."""
    summary = snapshot_issue_batch.summarize_redispatch(
        [
            {"blocker_receipt": {"decision": "blocked_unchanged"}},
            {"blocker_receipt": {"decision": "blocker_changed"}},
            {"blocker_receipt": {"decision": "no_receipt"}},
        ]
    )

    assert summary["suppressed_redispatch_count"] == 1
    assert summary["re_evaluation_count"] == 2
    assert summary["decision_counts"] == {
        "blocker_changed": 1,
        "blocked_unchanged": 1,
        "no_receipt": 1,
    }


def test_autopilot_snapshot_exposes_receipt_summary(tmp_path: Path) -> None:
    """The full goal-loop checkpoint should carry suppression counts and reasons."""
    write_receipt(_receipt(), path=tmp_path / "issue-7612.json")

    summary = autopilot_state_snapshot.blocker_receipt_snapshot(
        [7612],
        repository="ll7/robot_sf_ll7",
        receipts_dir=str(tmp_path),
        current_inputs=_inputs(),
    )

    assert summary["suppressed_redispatch_count"] == 1
    assert summary["issues"][0]["blocker_receipt"]["decision"] == "blocked_unchanged"


def test_load_receipt_classifies_malformed_json(tmp_path: Path) -> None:
    """A malformed external artifact must be visible to the next loop."""
    path = tmp_path / "issue-7612.json"
    path.write_text("{not-json", encoding="utf-8")

    loaded = load_receipt(issue_number=7612, directory=tmp_path)

    assert loaded["status"] == "malformed"
    assert loaded["receipt"] is None
    assert loaded["errors"]


def test_current_input_json_round_trip_preserves_wrapped_states(tmp_path: Path) -> None:
    """Receipts retain explicit unavailable and not-applicable field states."""
    receipt = _receipt(
        dependency_state=unavailable("dependency API unavailable"),
        pr_number=not_applicable("no PR exists"),
        base_sha=not_applicable("no PR exists"),
        head_sha=not_applicable("no PR exists"),
    )
    path = write_receipt(receipt, path=tmp_path / "issue-7612.json")
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert raw["fingerprint_inputs"]["dependency_state"] == {
        "status": "unavailable",
        "reason": "dependency API unavailable",
    }
    assert raw["fingerprint_inputs"]["pull_request.number"]["status"] == "not_applicable"
