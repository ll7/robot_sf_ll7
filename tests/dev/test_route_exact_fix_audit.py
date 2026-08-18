"""Contract tests for the report-to-exact-fix review route."""

from __future__ import annotations

import json

import pytest

from scripts.dev import route_exact_fix_audit


def _report(*, complete: bool = True) -> dict[str, object]:
    return {
        "schema": "open_state_label_hygiene.v1",
        "repo": "ll7/robot_sf_ll7",
        "read_only": True,
        "issue_writes": False,
        "project_writes": False,
        "complete_for_open_issues": complete,
        "truncated_any": not complete,
        "candidate_count": 1,
        "issues": [
            {
                "number": 123,
                "title": "stale issue",
                "url": "https://github.com/ll7/robot_sf_ll7/issues/123",
                "active_labels": ["state:ready"],
                "classification": "merged_reference_needs_exact_fix_review",
                "merged_prs": [
                    {
                        "number": 456,
                        "title": "Fix #123",
                        "url": "https://github.com/ll7/robot_sf_ll7/pull/456",
                        "merged_at": "2026-08-18T12:00:00Z",
                        "merge_commit_sha": "a" * 40,
                    }
                ],
            }
        ],
    }


def test_build_review_queue_routes_without_authorizing_a_disposition() -> None:
    """A merged reference becomes a pending exact-fix packet, never a write."""
    queue = route_exact_fix_audit.build_review_queue(_report())

    assert queue["schema"] == "exact_fix_review_queue.v1"
    assert queue["route_complete"] is True
    assert queue["disposition_authorized"] is False
    assert queue["counts"] == {
        "candidates": 1,
        "ready_for_manual_review": 0,
        "needs_exact_fix_evidence": 1,
    }
    row = queue["candidates"][0]
    assert row["classification"] == "needs_exact_fix_evidence"
    assert row["covering_prs"][0]["number"] == 456
    assert row["safe_mutations"] == []
    assert row["exact_fix_guard"]["named_symbol"]["status"] == "missing"


def test_build_review_queue_rejects_partial_source_report() -> None:
    """A partial hygiene report cannot be converted into an authoritative queue."""
    with pytest.raises(route_exact_fix_audit.InputContractError, match="incomplete"):
        route_exact_fix_audit.build_review_queue(_report(complete=False))


def test_build_review_queue_marks_explicit_evidence_ready_but_keeps_review_gate() -> None:
    """Complete exact-fix fields make a packet reviewable, not auto-disposable."""
    evidence = {
        "schema": "exact_fix_evidence.v1",
        "issues": [
            {
                "number": 123,
                "covering_pr": 456,
                "named_symbol": "scripts/example.py:apply_fix",
                "failure_signature": "ValueError: stale label",
                "failing_file_line": "scripts/example.py:42",
                "regression_proof": "tests/dev/test_example.py::test_stale_label",
                "current_main_sha": "b" * 40,
            }
        ],
    }

    queue = route_exact_fix_audit.build_review_queue(_report(), evidence=evidence)

    assert queue["counts"] == {
        "candidates": 1,
        "ready_for_manual_review": 1,
        "needs_exact_fix_evidence": 0,
    }
    row = queue["candidates"][0]
    assert row["classification"] == "ready_for_manual_exact_fix_review"
    assert row["exact_fix_guard"]["current_main_sha"]["status"] == "provided"
    assert row["safe_mutations"] == []
    assert queue["disposition_authorized"] is False


def test_build_review_queue_binds_optional_evidence_to_reported_pr() -> None:
    """An evidence manifest cannot substitute an unrelated covering PR."""
    evidence = {
        "schema": "exact_fix_evidence.v1",
        "issues": [{"number": 123, "covering_pr": 999}],
    }

    with pytest.raises(route_exact_fix_audit.InputContractError, match="absent"):
        route_exact_fix_audit.build_review_queue(_report(), evidence=evidence)


def test_build_review_queue_rejects_evidence_for_unknown_issue() -> None:
    """Evidence for an issue outside the complete source report is not silently ignored."""
    evidence = {
        "schema": "exact_fix_evidence.v1",
        "issues": [{"number": 999}],
    }

    with pytest.raises(route_exact_fix_audit.InputContractError, match="absent from"):
        route_exact_fix_audit.build_review_queue(_report(), evidence=evidence)


def test_main_emits_fail_closed_json_for_invalid_report(tmp_path, capsys) -> None:
    """The CLI exposes malformed-input failure without attempting a write."""
    report_path = tmp_path / "partial.json"
    report_path.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")

    assert route_exact_fix_audit.main(["--report", str(report_path)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "exact_fix_review_queue.v1"
    assert payload["route_complete"] is False
    assert payload["disposition_authorized"] is False
