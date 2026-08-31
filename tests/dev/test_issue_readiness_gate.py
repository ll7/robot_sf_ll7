"""Offline tests for the post-create issue readiness gate (issue #8131)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from scripts.dev import issue_readiness_gate
from scripts.dev.issue_implementability import READY_LABEL


def _issue(
    number: int, *, labels: list[str], state: str = "open", updated_at: str = "2026-08-31T00:00:00Z"
):
    return {
        "status": "ok",
        "number": number,
        "title": f"issue {number}",
        "body": "body text",
        "state": state,
        "url": f"https://github.test/issues/{number}",
        "labels": sorted(labels),
        "assignees": [],
        "updated_at": updated_at,
    }


def _ready_admission(number: int):
    return {
        "schema": "goal_issue_admission.v1",
        "ok": True,
        "issue": number,
        "preflight": {"classification": "ready", "ready": True, "reasons": []},
    }


def _not_ready_admission(number: int, classification: str):
    return {
        "schema": "goal_issue_admission.v1",
        "ok": False,
        "issue": number,
        "preflight": {
            "classification": classification,
            "ready": False,
            "reasons": [f"{classification} gate"],
        },
    }


def test_gate_issue_adds_and_verifies_readiness_on_pass() -> None:
    """A passing live admission adds exactly one verified readiness label."""
    reads = [
        _issue(8100, labels=["bug"]),
        _issue(8100, labels=["bug"]),
        _issue(8100, labels=["bug", READY_LABEL]),
    ]
    with (
        patch.object(issue_readiness_gate.gh_issue_rest, "fetch_issue", side_effect=reads) as fetch,
        patch.object(issue_readiness_gate.goal_issue_admission, "admit_issue") as admit,
        patch.object(issue_readiness_gate.gh_pr_label_rest, "add_label") as add_label,
    ):
        admit.return_value = _ready_admission(8100)
        add_label.return_value = {"status": "ok"}
        payload = issue_readiness_gate.gate_issue(8100, repo="ll7/robot_sf_ll7")

    assert payload["outcome"] == "ready"
    assert payload["ready_added"] is True
    assert payload["verified"] is True
    assert fetch.call_count == 3
    add_label.assert_called_once_with(8100, READY_LABEL, repo="ll7/robot_sf_ll7")
    admit.assert_called_once_with(
        8100,
        repo="ll7/robot_sf_ll7",
        remote="origin",
        source_ref="origin/main",
        check_only=True,
    )


def test_gate_issue_never_labels_when_admission_fails() -> None:
    """A failed admission classification must not call the label writer."""
    with (
        patch.object(issue_readiness_gate.gh_issue_rest, "fetch_issue") as fetch,
        patch.object(issue_readiness_gate.goal_issue_admission, "admit_issue") as admit,
        patch.object(issue_readiness_gate.gh_pr_label_rest, "add_label") as add_label,
    ):
        fetch.return_value = _issue(8101, labels=["bug"])
        admit.return_value = _not_ready_admission(8101, "needs_spec")
        payload = issue_readiness_gate.gate_issue(8101, repo="ll7/robot_sf_ll7")

    assert payload["outcome"] == "needs_spec"
    assert payload["ready_added"] is False
    assert payload["reasons"] == ["needs_spec gate"]
    add_label.assert_not_called()


@pytest.mark.parametrize(
    "classification", ["blocked", "parent", "human_decision", "already_claimed"]
)
def test_gate_issue_non_ready_classifications_add_nothing(classification: str) -> None:
    """Every non-ready classification leaves readiness absent without label writes."""
    with (
        patch.object(issue_readiness_gate.gh_issue_rest, "fetch_issue") as fetch,
        patch.object(issue_readiness_gate.goal_issue_admission, "admit_issue") as admit,
        patch.object(issue_readiness_gate.gh_pr_label_rest, "add_label") as add_label,
    ):
        fetch.return_value = _issue(8102, labels=[])
        admit.return_value = _not_ready_admission(8102, classification)
        payload = issue_readiness_gate.gate_issue(8102, repo="ll7/robot_sf_ll7")

    assert payload["outcome"] == classification
    assert payload["ready_added"] is False
    add_label.assert_not_called()


def test_gate_issue_drift_between_reads_blocks_readiness_write() -> None:
    """Body/label/state drift between admission and write causes zero readiness writes."""
    reads = [
        _issue(8103, labels=["bug"]),
        _issue(8103, labels=["bug"], updated_at="2026-08-31T01:00:00Z"),
    ]
    with (
        patch.object(issue_readiness_gate.gh_issue_rest, "fetch_issue", side_effect=reads),
        patch.object(issue_readiness_gate.goal_issue_admission, "admit_issue") as admit,
        patch.object(issue_readiness_gate.gh_pr_label_rest, "add_label") as add_label,
    ):
        admit.return_value = _ready_admission(8103)
        payload = issue_readiness_gate.gate_issue(8103, repo="ll7/robot_sf_ll7")

    assert payload["outcome"] == "drift"
    assert payload["ready_added"] is False
    add_label.assert_not_called()


def test_gate_issue_is_idempotent_when_already_ready() -> None:
    """A retry on an already-ready issue adds nothing and stays verified."""
    issue = _issue(8104, labels=["bug", READY_LABEL])
    with (
        patch.object(
            issue_readiness_gate.gh_issue_rest, "fetch_issue", side_effect=[issue, issue]
        ) as fetch,
        patch.object(issue_readiness_gate.goal_issue_admission, "admit_issue") as admit,
        patch.object(issue_readiness_gate.gh_pr_label_rest, "add_label") as add_label,
    ):
        admit.return_value = _ready_admission(8104)
        payload = issue_readiness_gate.gate_issue(8104, repo="ll7/robot_sf_ll7")

    assert payload["outcome"] == "already_ready"
    assert payload["ready_added"] is False
    assert payload["verified"] is True
    assert fetch.call_count == 2
    add_label.assert_not_called()


def test_gate_issue_fails_closed_on_exact_read_error() -> None:
    """An exact-read error produces an error outcome and no label mutation."""
    with (
        patch.object(issue_readiness_gate.gh_issue_rest, "fetch_issue") as fetch,
        patch.object(issue_readiness_gate.gh_pr_label_rest, "add_label") as add_label,
    ):
        fetch.return_value = {"number": 8105, "status": "error", "error": "transport down"}
        payload = issue_readiness_gate.gate_issue(8105, repo="ll7/robot_sf_ll7")

    assert payload["outcome"] == "error"
    assert payload["phase"] == "initial"
    assert payload["ready_added"] is False
    add_label.assert_not_called()


def test_gate_issue_refuses_closed_issue() -> None:
    """A non-open issue is a state conflict before any admission work."""
    with (
        patch.object(issue_readiness_gate.gh_issue_rest, "fetch_issue") as fetch,
        patch.object(issue_readiness_gate.goal_issue_admission, "admit_issue") as admit,
    ):
        fetch.return_value = _issue(8106, labels=[], state="CLOSED")
        payload = issue_readiness_gate.gate_issue(8106, repo="ll7/robot_sf_ll7")

    assert payload["outcome"] == "state_conflict"
    admit.assert_not_called()


def test_gate_issue_readback_failure_reports_unverified_write() -> None:
    """A missing label at readback fails closed with the write recorded unverified."""
    reads = [
        _issue(8107, labels=["bug"]),
        _issue(8107, labels=["bug"]),
        _issue(8107, labels=["bug"]),
    ]
    with (
        patch.object(issue_readiness_gate.gh_issue_rest, "fetch_issue", side_effect=reads),
        patch.object(issue_readiness_gate.goal_issue_admission, "admit_issue") as admit,
        patch.object(issue_readiness_gate.gh_pr_label_rest, "add_label") as add_label,
    ):
        admit.return_value = _ready_admission(8107)
        add_label.return_value = {"status": "ok"}
        payload = issue_readiness_gate.gate_issue(8107, repo="ll7/robot_sf_ll7")

    assert payload["outcome"] == "error"
    assert payload["phase"] == "readback"
    assert payload["ready_added"] is True
    assert payload["verified"] is False


def test_create_issue_strips_readiness_from_initial_labels() -> None:
    """Creation omits state:ready from the initial label set and parses the new number."""
    created = _issue(8110, labels=["bug"])
    gated_ready = _issue(8110, labels=["bug", READY_LABEL])
    with (
        patch.object(issue_readiness_gate.subprocess, "run") as run,
        patch.object(
            issue_readiness_gate.gh_issue_rest,
            "fetch_issue",
            side_effect=[created, created, gated_ready],
        ),
        patch.object(issue_readiness_gate.goal_issue_admission, "admit_issue") as admit,
        patch.object(issue_readiness_gate.gh_pr_label_rest, "add_label") as add_label,
    ):
        run.return_value = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": "https://github.test/ll7/robot_sf_ll7/issues/8110\n",
                "stderr": "",
            },
        )()
        admit.return_value = _ready_admission(8110)
        add_label.return_value = {"status": "ok"}
        payload = issue_readiness_gate.create_issue(
            title="t",
            body_file="/tmp/body.md",
            labels=["bug", READY_LABEL],
            repo="ll7/robot_sf_ll7",
        )

    assert payload["outcome"] == "ready"
    assert payload["issue"] == 8110
    gh_args = run.call_args.args[0]
    assert READY_LABEL not in gh_args
    assert "bug" in gh_args
    add_label.assert_called_once_with(8110, READY_LABEL, repo="ll7/robot_sf_ll7")


def test_create_issue_fails_closed_when_url_is_unparseable() -> None:
    """An unparseable create output is an error with no issue or label write."""
    with patch.object(issue_readiness_gate.subprocess, "run") as run:
        run.return_value = type(
            "Completed", (), {"returncode": 0, "stdout": "unexpected", "stderr": ""}
        )()
        payload = issue_readiness_gate.create_issue(
            title="t",
            body_file="/tmp/body.md",
            labels=[],
            repo="ll7/robot_sf_ll7",
        )

    assert payload["outcome"] == "error"
    assert payload["phase"] == "create"
    assert payload["issue"] is None


def test_main_gate_json_output_is_stable(capsys) -> None:  # type: ignore[no-untyped-def]
    """The JSON payload carries the schema, outcome, and digest fields."""
    with (
        patch.object(issue_readiness_gate.gh_issue_rest, "fetch_issue") as fetch,
        patch.object(issue_readiness_gate.goal_issue_admission, "admit_issue") as admit,
    ):
        fetch.return_value = _issue(8111, labels=[])
        admit.return_value = _not_ready_admission(8111, "needs_spec")
        rc = issue_readiness_gate.main(["gate", "8111", "--repo", "ll7/robot_sf_ll7", "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "issue_readiness_gate.v1"
    assert payload["outcome"] == "needs_spec"
    assert payload["ready_added"] is False
