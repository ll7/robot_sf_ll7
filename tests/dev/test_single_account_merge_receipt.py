"""Contract tests for the single-account merge receipt (issue #7669)."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from scripts.dev import single_account_merge_receipt as receipt_module
from scripts.dev.single_account_merge_receipt import (
    EVIDENCE_STATES,
    HOLD_KEYS,
    apply_guarded_merge,
    build_live_evidence,
    build_receipt,
    classify_implementation_review,
    detect_post_merge_incident,
    normalize_required_checks,
    receipt_digest,
    record_merge_result,
    validate_merge_authority_fixture,
    validate_receipt,
    verify_receipt,
)

HEAD_SHA = "a" * 40
BASE_SHA = "b" * 40
CURRENT_BASE_SHA = "c" * 40
METADATA_DIGEST = "d" * 64
REVIEW_DIGEST = "e" * 64
OBSERVED_AT = "2026-08-21T00:00:00Z"


def _clear_holds() -> dict[str, dict[str, Any]]:
    return {key: {"status": "clear", "reason_codes": [], "source": "fixture"} for key in HOLD_KEYS}


def _receipt(*, holds: dict[str, dict[str, Any]] | None = None, **overrides: Any) -> dict[str, Any]:
    """Build a complete ready receipt fixture with explicit immutable evidence."""
    values: dict[str, Any] = {
        "repository": "owner/repo",
        "pr_number": 42,
        "head_sha": HEAD_SHA,
        "base_sha": BASE_SHA,
        "current_base_sha": CURRENT_BASE_SHA,
        "metadata_digest": METADATA_DIGEST,
        "required_checks": [
            {
                "name": "CI",
                "head_sha": HEAD_SHA,
                "status": "completed",
                "conclusion": "success",
                "identity": "github-actions",
            }
        ],
        "review_source": {
            "status": "accepted",
            "kind": "static_report",
            "identity": "independent-reviewer",
            "head_sha": HEAD_SHA,
            "metadata_digest": METADATA_DIGEST,
            "evidence_digest": REVIEW_DIGEST,
        },
        "thread_resolution": {"status": "resolved", "unresolved": 0},
        "requested_reviewers": {"status": "clear", "count": 0, "identities": []},
        "requested_teams": {"status": "clear", "count": 0, "identities": []},
        "holds": holds or _clear_holds(),
        "observed_at": OBSERVED_AT,
        "gate_audit": {"schema": "merge_queue_gate.v1", "passed": True},
        "pr_state": "OPEN",
        "pr_merged_at": None,
    }
    values.update(overrides)
    return build_receipt(**values)


def _live_evidence(receipt: dict[str, Any]) -> dict[str, Any]:
    """Project a receipt back into the live-evidence shape used for rereads."""
    evidence = {
        "repository": receipt["repository"],
        "pr_number": receipt["pr_number"],
        "head_sha": receipt["head_sha"],
        "base_sha": receipt["base_sha"],
        "current_base_sha": receipt["current_base_sha"],
        "metadata_digest": receipt["metadata_digest"],
        "pr_state": receipt["pr_state"],
        "pr_merged_at": receipt["pr_merged_at"],
        "required_checks": copy.deepcopy(receipt["required_checks"]),
        "review_source": copy.deepcopy(receipt["implementation_review"]),
        "thread_resolution": copy.deepcopy(receipt["thread_resolution"]),
        "requested_reviewers": copy.deepcopy(receipt["requested_reviewers"]),
        "requested_teams": copy.deepcopy(receipt["requested_teams"]),
        "holds": copy.deepcopy(receipt["holds"]),
        "gate_audit": copy.deepcopy(receipt["gate_audit"]),
    }
    if "evidence_provenance" in receipt:
        evidence["evidence_provenance"] = copy.deepcopy(receipt["evidence_provenance"])
    return evidence


def _snapshot_with_provenance() -> dict[str, Any]:
    """Build a live snapshot with the complete source-route contract."""
    return {
        "number": 42,
        "pr_state": "OPEN",
        "pr_merged_at": None,
        "draft": False,
        "head_sha": HEAD_SHA,
        "base_sha": BASE_SHA,
        "metadata_digest": METADATA_DIGEST,
        "body": "final body",
        "labels": ["merge-ready"],
        "checks": {"overall": "success"},
        "changed_coverage": {"status": "success", "head_sha": HEAD_SHA},
        "required_checks": [
            {
                "name": "CI",
                "head_sha": HEAD_SHA,
                "status": "completed",
                "conclusion": "success",
            }
        ],
        "review_evidence": {},
        "requested_reviewers": [],
        "requested_teams": [],
        "reviewers_requested": False,
        "evidence_provenance": {
            "schema": "single_account_merge_evidence_provenance.v1",
            "data_source": "rest_fallback_graphql_quota",
            "ordinary_facts": {
                "pull_request": "rest",
                "labels": "rest",
                "comments": "rest",
                "reviews": "rest",
                "requested_reviewers": "rest",
                "check_rollup": "rest",
                "base_sha": "rest",
                "changed_coverage": "rest",
            },
            "review_threads": {"source": "graphql", "status": "separate_query"},
            "fallback_diagnostic": "GitHub GraphQL quota exhausted",
        },
    }


def test_complete_receipt_is_deterministic_and_verifies() -> None:
    first = _receipt()
    second = _receipt()

    assert first == second
    assert first["status"] == "ready"
    assert first["reason_codes"] == []
    assert first["receipt_digest"] == receipt_digest(first)
    assert validate_receipt(first)["passed"] is True
    assert verify_receipt(first)["passed"] is True


def test_each_hold_dimension_blocks_independently() -> None:
    for key in HOLD_KEYS:
        holds = _clear_holds()
        holds[key] = {"status": "held", "reason_codes": [f"{key}_hold"], "source": "fixture"}
        blocked = _receipt(holds=holds)
        assert blocked["status"] == "blocked"
        assert f"hold_{key}_held" in blocked["reason_codes"]


@pytest.mark.parametrize(
    ("pr_state", "pr_merged_at", "expected_reason"),
    [
        ("CLOSED", None, "pr_not_open"),
        ("MERGED", "2026-08-23T19:28:15Z", "pr_already_merged"),
    ],
)
def test_terminal_pr_state_blocks_ready_receipts(
    pr_state: str, pr_merged_at: str | None, expected_reason: str
) -> None:
    blocked = _receipt(pr_state=pr_state, pr_merged_at=pr_merged_at)

    assert blocked["status"] == "blocked"
    assert "pr_not_open" in blocked["reason_codes"]
    assert expected_reason in blocked["reason_codes"]


def test_non_passing_gate_audit_blocks_ready_receipts() -> None:
    blocked = _receipt(
        gate_audit={
            "schema": "merge_queue_gate.v1",
            "passed": False,
            "reasons": ["stale_merge_base"],
        }
    )

    assert blocked["status"] == "blocked"
    assert "merge_queue_gate_not_passed" in blocked["reason_codes"]
    assert "merge_queue_gate_stale_merge_base" in blocked["reason_codes"]


def test_explicit_non_passing_gate_audit_cannot_be_overridden_by_legacy_status() -> None:
    blocked = _receipt(
        gate_audit={
            "schema": "merge_queue_gate.v1",
            "passed": False,
            "status": "success",
            "reasons": ["stale_merge_base"],
        }
    )

    assert blocked["status"] == "blocked"
    assert "merge_queue_gate_not_passed" in blocked["reason_codes"]
    assert "merge_queue_gate_stale_merge_base" in blocked["reason_codes"]


def test_live_head_metadata_and_check_changes_block_without_reconstructing_receipt() -> None:
    receipt = _receipt()

    changed_metadata = _live_evidence(receipt)
    changed_metadata["metadata_digest"] = "f" * 64
    assert (
        "live_metadata_digest_changed"
        in verify_receipt(receipt, live_evidence=changed_metadata)["reasons"]
    )

    changed_head = _live_evidence(receipt)
    changed_head["head_sha"] = "f" * 40
    assert "live_head_sha_changed" in verify_receipt(receipt, live_evidence=changed_head)["reasons"]

    changed_checks = _live_evidence(receipt)
    changed_checks["required_checks"]["checks"][0]["conclusion"] = "failure"
    changed_checks["required_checks"]["status"] = "failure"
    assert (
        "live_required_checks_changed"
        in verify_receipt(receipt, live_evidence=changed_checks)["reasons"]
    )

    changed_lifecycle = _live_evidence(receipt)
    changed_lifecycle["pr_state"] = "CLOSED"
    assert (
        "live_pr_state_changed"
        in verify_receipt(receipt, live_evidence=changed_lifecycle)["reasons"]
    )

    changed_gate = _live_evidence(receipt)
    changed_gate["gate_audit"] = {
        "schema": "merge_queue_gate.v1",
        "passed": False,
        "reasons": ["stale_merge_base"],
    }
    assert (
        "live_gate_audit_changed" in verify_receipt(receipt, live_evidence=changed_gate)["reasons"]
    )


def test_live_evidence_provenance_is_immutable_for_receipt_validation() -> None:
    """A changed API route or source record must not be hidden during reread."""
    provenance = _snapshot_with_provenance()["evidence_provenance"]
    receipt = _receipt(evidence_provenance=provenance)
    changed = _live_evidence(receipt)
    changed["evidence_provenance"]["data_source"] = "graphql"

    assert (
        "live_evidence_provenance_changed"
        in verify_receipt(receipt, live_evidence=changed)["reasons"]
    )


def test_quota_evidence_provenance_matches_receipt_schema() -> None:
    """The new route evidence remains machine-validatable as part of v1."""
    from pathlib import Path

    import jsonschema

    schema_path = Path(receipt_module.__file__).with_name(
        "single_account_merge_receipt.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(
        _receipt(evidence_provenance=_snapshot_with_provenance()["evidence_provenance"])
    )


def test_build_live_evidence_reports_rest_facts_when_thread_graphql_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quota recovery emits an auditable blocked receipt instead of an opaque error."""
    snapshot = _snapshot_with_provenance()
    monkeypatch.setattr(
        "scripts.dev.merge_queue_gate.fetch_pr_snapshot",
        lambda *args, **kwargs: (snapshot, None),
    )
    monkeypatch.setattr(
        "scripts.dev.merge_queue_gate.fetch_main_sha",
        lambda *args, **kwargs: CURRENT_BASE_SHA,
    )
    monkeypatch.setattr(
        "scripts.dev.merge_queue_gate.fetch_threads_resolved",
        lambda *args, **kwargs: (None, "GitHub GraphQL quota exhausted"),
    )

    evidence, error = build_live_evidence(42, repository="owner/repo")

    assert error is None
    assert evidence is not None
    assert evidence["evidence_provenance"]["data_source"] == "rest_fallback_graphql_quota"
    assert evidence["evidence_provenance"]["review_threads"] == {
        "source": "graphql",
        "status": "unavailable",
        "diagnostic": "GitHub GraphQL quota exhausted",
    }
    assert evidence["thread_resolution"]["status"] == "unavailable"
    assert evidence["gate_audit"]["thread_resolution"] == "not_evaluated"
    assert "review_threads_not_evaluated" in evidence["gate_audit"]["reasons"]

    receipt = build_receipt(**evidence)
    assert receipt["status"] == "blocked"
    assert "review_threads_unavailable" in receipt["reason_codes"]
    assert receipt["evidence_provenance"]["ordinary_facts"]["check_rollup"] == "rest"


def test_build_live_evidence_keeps_rest_snapshot_failure_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed ordinary REST snapshot remains an error and cannot become partial evidence."""
    monkeypatch.setattr(
        "scripts.dev.merge_queue_gate.fetch_pr_snapshot",
        lambda *args, **kwargs: ({}, "REST snapshot fallback failed"),
    )

    evidence, error = build_live_evidence(42, repository="owner/repo")

    assert evidence is None
    assert error == "REST snapshot fallback failed"


def test_review_carrier_precedence_and_fail_closed_states() -> None:
    accepted_check = {
        "identity": "CodeRabbit",
        "approved_source": True,
        "head_sha": HEAD_SHA,
        "metadata_digest": METADATA_DIGEST,
        "status": "completed",
        "conclusion": "success",
        "evidence_digest": REVIEW_DIGEST,
    }
    evidence = {
        "head_sha": HEAD_SHA,
        "metadata_digest": METADATA_DIGEST,
        "check_runs": [accepted_check],
        "comments": [
            {
                "identity": "untrusted",
                "body": "single-account-review: accepted @ " + "f" * 40,
            }
        ],
    }
    classified = classify_implementation_review(evidence)
    assert classified["status"] == "accepted"
    assert classified["carrier"]["kind"] == "check_run"

    pending = classify_implementation_review(
        {
            "head_sha": HEAD_SHA,
            "metadata_digest": METADATA_DIGEST,
            "check_runs": [
                {
                    **accepted_check,
                    "status": "in_progress",
                }
            ],
        }
    )
    assert pending["status"] == "pending"

    self_review = classify_implementation_review(
        {
            "head_sha": HEAD_SHA,
            "metadata_digest": METADATA_DIGEST,
            "waiver_actor": "owner",
            "reviews": [
                {
                    "identity": "owner",
                    "state": "APPROVED",
                    "authorAssociation": "OWNER",
                    "head_sha": HEAD_SHA,
                    "metadata_digest": METADATA_DIGEST,
                }
            ],
        }
    )
    assert self_review["status"] == "conflicting"
    assert "owner_self_review_not_independent" in self_review["reason_codes"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "unavailable"),
        ([{"name": "CI", "head_sha": HEAD_SHA, "status": "in_progress"}], "pending"),
        (
            [{"name": "CI", "head_sha": "f" * 40, "status": "completed", "conclusion": "success"}],
            "stale",
        ),
        (
            [{"name": "CI", "head_sha": HEAD_SHA, "status": "completed", "conclusion": "failure"}],
            "failure",
        ),
        ([{"head_sha": HEAD_SHA, "status": "completed", "conclusion": "success"}], "malformed"),
    ],
)
def test_required_check_states_are_distinct(raw: Any, expected: str) -> None:
    assert normalize_required_checks(raw, head_sha=HEAD_SHA)["status"] == expected
    assert expected in EVIDENCE_STATES or expected in {"success"}


def test_recorded_merge_sha_preserves_digest_and_post_merge_incident_boundary() -> None:
    receipt = _receipt()
    merged_sha = "f" * 40
    merged = record_merge_result(
        receipt,
        status="merged",
        returned_merged_sha=merged_sha,
        response={"merged": True, "sha": merged_sha},
        observed_at=OBSERVED_AT,
    )

    assert merged["receipt_digest"] == receipt["receipt_digest"]
    assert verify_receipt(merged, require_merged=True)["passed"] is True
    assert detect_post_merge_incident(merged, observed_merged_sha=merged_sha)["status"] == "healthy"
    incident = detect_post_merge_incident(merged, observed_merged_sha="1" * 40)
    assert incident["status"] == "incident"
    assert incident["waiver_reuse"] == "blocked"


def test_guarded_apply_is_one_put_then_closed_merged_readback() -> None:
    receipt = _receipt()
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_api(
        method: str, path: str, payload: dict[str, Any] | None
    ) -> tuple[dict[str, Any], None]:
        calls.append((method, path, payload))
        if method == "PUT":
            return {"merged": True, "sha": "f" * 40}, None
        return {"state": "closed", "merged": True}, None

    merged, error = apply_guarded_merge(
        receipt, repository="owner/repo", api=fake_api, observed_at=OBSERVED_AT
    )
    assert error is None
    assert merged is not None
    assert calls == [
        ("PUT", "repos/owner/repo/pulls/42/merge", {"sha": HEAD_SHA, "merge_method": "squash"}),
        ("GET", "repos/owner/repo/pulls/42", None),
    ]
    assert merged["receipt"]["merge_result"]["returned_merged_sha"] == "f" * 40


def test_guarded_apply_failure_returns_the_observed_receipt() -> None:
    receipt = _receipt()

    def failed_api(
        method: str, path: str, payload: dict[str, Any] | None
    ) -> tuple[dict[str, Any], None]:
        return {"merged": False, "message": "head changed"}, None

    failed, error = apply_guarded_merge(receipt, repository="owner/repo", api=failed_api)
    assert error == "head changed"
    assert failed is not None
    assert failed["receipt_digest"] == receipt["receipt_digest"]
    assert failed["merge_result"]["status"] == "failed"


def test_validate_mode_is_read_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    receipt = _receipt()
    receipt_file = tmp_path / "receipt.json"
    receipt_file.write_text(json.dumps(receipt), encoding="utf-8")

    monkeypatch.setattr(
        receipt_module,
        "build_live_evidence",
        lambda *args, **kwargs: (_live_evidence(receipt), None),
    )
    monkeypatch.setattr(
        receipt_module,
        "_run_gh_api",
        lambda *args, **kwargs: pytest.fail("validate mode must not call the merge API"),
    )

    assert (
        receipt_module.main(
            [
                "--pr",
                "42",
                "--repo",
                "owner/repo",
                "--mode",
                "validate",
                "--receipt-file",
                str(receipt_file),
            ]
        )
        == 0
    )


def test_merge_authority_fixture_is_current() -> None:
    result = validate_merge_authority_fixture()
    assert result["passed"] is True, result
