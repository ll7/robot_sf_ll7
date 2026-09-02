"""Tests for compact issue-batch snapshots."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev import snapshot_issue_batch
from scripts.dev.github_quota import RateLimitSnapshot
from scripts.dev.snapshot_issue_batch import (
    _batch_claim_statuses,
    _issue_classification,
    expand_issue_numbers,
    main,
    snapshot_active_issue_portfolio,
    snapshot_blocked_external_issues,
    snapshot_claimable_issues,
    snapshot_issues,
)


@pytest.fixture(autouse=True)
def _healthy_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep legacy list fixtures on the healthy GraphQL path unless a test overrides it."""
    monkeypatch.setattr(
        snapshot_issue_batch,
        "_rate_limit_snapshot",
        lambda: RateLimitSnapshot(
            status="ok",
            graphql_remaining=4_000,
            graphql_reset_at=1_800_000_000,
            core_remaining=4_000,
            core_reset_at=1_800_000_000,
        ),
    )


def _claim_status(number: int, *, claimed: bool = False, ok: bool = True, sha: str | None = None):
    """Return a compact claim status payload for snapshot tests."""
    return {
        "ok": ok,
        "claimed": claimed if ok else None,
        "claim_ref": f"agent-claims/issue-{number}",
        "sha": sha if claimed else None,
    }


def test_expand_issue_numbers_treats_two_values_as_range() -> None:
    """Two ascending values should support the concise batch command."""
    assert expand_issue_numbers([2665, 2667], expand_range=True) == [2665, 2666, 2667]
    assert expand_issue_numbers([2665, 2667], expand_range=False) == [2665, 2667]


@pytest.mark.parametrize(
    "label",
    [
        "blocked:needs-maintainer",
        "blocked:needs-campaign",
        "state:review",
        "needs-triage",
        "deferred",
        "state:deferred",
        "state:parked",
        "parent",
    ],
)
def test_explicit_dispatch_stop_labels_are_not_claimable(label: str) -> None:
    """Every explicit dispatch-stop label must fence autonomous claim dispatch."""
    classification, reason = _issue_classification(
        assignees=[],
        claim={"ok": True, "claimed": False},
        labels=[label],
        state="OPEN",
    )

    assert classification == "blocked_label"
    assert label in reason


@pytest.mark.parametrize("dispatch_stop_label", ["state:review", "needs-triage"])
def test_snapshot_claimable_issues_excludes_dispatch_stop_labels_from_dispatch(
    dispatch_stop_label: str,
) -> None:
    """Claimable snapshots must retain dispatch-stop rows as non-claimable audit entries."""
    issue_list = [
        {
            "number": 2710,
            "title": f"{dispatch_stop_label} issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2710",
            "labels": [{"name": dispatch_stop_label}],
            "assignees": [],
        }
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {2710: _claim_status(2710)}
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=1,
            )

    row = payload["issues"][0]
    assert row["classification"] == (
        "review" if dispatch_stop_label == "state:review" else "state_conflict"
    )
    assert row["reason"] == (
        "the issue is already in review"
        if dispatch_stop_label == "state:review"
        else (
            "exactly one execution state label is required unless a known hold qualifier already "
            "blocks dispatch; found none; state qualifiers are none"
        )
    )


@pytest.mark.parametrize("label", ["blocked:needs-maintainer", "blocked:needs-campaign"])
def test_active_portfolio_explicit_blockers_are_not_executable(label: str) -> None:
    """Active-portfolio routing must honor the same explicit blocker namespace."""
    classification, reason = snapshot_issue_batch._portfolio_classification(
        labels=[label],
        title="workflow issue",
        assignees=[],
        claim=_claim_status(1),
    )

    assert classification == "needs_human_decision"
    assert reason == "maintainer decision label blocks autonomous execution"


def test_snapshot_issues_emits_compact_fields() -> None:
    """Snapshot output should include excerpts and claim state without raw bodies."""
    body = " ".join(["detail"] * 100)
    rest_issue = {
        "number": 2665,
        "status": "ok",
        "title": "workflow: compact issue snapshot",
        "body": body,
        "state": "OPEN",
        "url": "https://github.test/issues/2665",
        "labels": ["enhancement", "workflow"],
        "assignees": ["alice"],
    }
    with patch("scripts.dev.snapshot_issue_batch.gh_issue_rest") as mock_rest:
        mock_rest.fetch_issue.return_value = rest_issue
        with patch("scripts.dev.snapshot_issue_batch.status_issue") as claim:
            claim.return_value = {
                "ok": True,
                "claimed": False,
                "claim_ref": "agent-claims/issue-2665",
                "sha": None,
            }
            payload = snapshot_issues(
                [2665], repo="ll7/robot_sf_ll7", body_limit=20, remote="origin"
            )

    issue = payload["issues"][0]
    assert payload["schema"] == "issue_batch_snapshot.v1"
    assert issue["number"] == 2665
    assert issue["labels"] == ["enhancement", "workflow"]
    assert issue["assignees"] == ["alice"]
    assert issue["body_excerpt"] == "detail detail detail"
    assert issue["body_truncated"] is True
    assert issue["claim"]["claimed"] is False
    assert issue["linked_prs"] == []
    assert "goal_driven_agent_loops" in issue["recommended_context_pack"]


def test_main_returns_error_when_any_issue_fails(capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI should keep a JSON payload even when one issue cannot be fetched."""
    with patch("scripts.dev.snapshot_issue_batch.gh_issue_rest") as mock_rest:
        mock_rest.fetch_issue.return_value = {
            "number": 1,
            "status": "error",
            "error": "issue 1 failed: not found",
        }
        rc = main(["1", "--json"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["issues"][0]["status"] == "error"


def test_snapshot_issues_can_write_context_capsules(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Optional capsules should seed workers without broad rediscovery."""
    rest_issue = {
        "number": 2666,
        "status": "ok",
        "title": "docs: claim boundary first",
        "body": "short body",
        "state": "OPEN",
        "url": "https://github.test/issues/2666",
        "labels": ["docs"],
        "assignees": [],
    }
    with patch("scripts.dev.snapshot_issue_batch.gh_issue_rest") as mock_rest:
        mock_rest.fetch_issue.return_value = rest_issue
        with patch("scripts.dev.snapshot_issue_batch.status_issue") as claim:
            claim.return_value = {
                "ok": True,
                "claimed": True,
                "claim_ref": "agent-claims/issue-2666",
                "sha": "abc123",
            }
            payload = snapshot_issues(
                [2666],
                repo="ll7/robot_sf_ll7",
                body_limit=300,
                remote="origin",
                capsule_dir=str(tmp_path),
            )

    path = tmp_path / "issue_2666_context_capsule.json"
    assert payload["issues"][0]["context_capsule_path"] == str(path)
    capsule = json.loads(path.read_text())
    assert capsule["schema"] == "issue_context_capsule.v1"
    assert capsule["issue"]["number"] == 2666
    assert capsule["claim"]["claimed"] is True
    assert capsule["files_to_read"] == ["docs/context/INDEX.md"]


def test_snapshot_claimable_issues_includes_classification_without_body() -> None:
    """Claimable discovery should return compact entries without raw body data."""
    issue_list = [
        {
            "number": 2667,
            "title": "claimable issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2667",
            "labels": [{"name": "workflow"}],
            "assignees": [],
            "body": "secret body that should not appear",
        },
        {
            "number": 2668,
            "title": "blocked issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2668",
            "labels": [{"name": "state:blocked"}],
            "assignees": [],
            "body": "another secret body that should not appear",
        },
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {
                2667: _claim_status(2667),
                2668: _claim_status(2668),
            }
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=2,
            )

    assert payload["mode"] == "candidate_queue"
    assert payload["legacy_mode"] == "claimable"
    assert payload["issues"][0]["classification"] == "state_conflict"
    assert payload["claimable_count"] == 0
    assert payload["claimable_issues"] == []
    assert payload["admission_reason_histogram"] == {
        "blocked": 1,
        "state_label_conflict": 1,
    }
    assert payload["issues"][0]["admission"]["classification"] == "state_conflict"
    assert payload["issues"][0]["admission"]["claim_outcome"] == "unclaimed"
    assert payload["issues"][0]["body_excerpt"] == ""
    assert payload["issues"][0]["body_truncated"] is False
    assert payload["issues"][1]["classification"] == "blocked"
    assert "reason" in payload["issues"][1]
    claim.assert_called_once_with([2667, 2668], remote="origin")


@pytest.mark.parametrize(
    ("field", "value"),
    [("outcome", None), ("outcome", ""), ("classification", None), ("classification", " ")],
)
def test_snapshot_claimable_issues_rejects_malformed_admission_fields(
    field: str, value: object
) -> None:
    """Missing or empty admission fields cannot authorize complete queue evidence."""
    number = 9005
    issue_list = [
        {
            "number": number,
            "title": f"ready issue {number}",
            "state": "OPEN",
            "url": f"https://github.test/issues/{number}",
            "labels": [{"name": "state:ready"}],
            "assignees": [],
        }
    ]
    malformed = {
        "schema": snapshot_issue_batch.goal_issue_admission.SCHEMA,
        "ok": False,
        "outcome": "not_admitted",
        "classification": "needs_spec",
        "claim_outcome": "unclaimed",
        "reasons": ["missing contract section"],
    }
    if value is None:
        malformed.pop(field)
    else:
        malformed[field] = value
    with (
        patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh,
        patch(
            "scripts.dev.snapshot_issue_batch._batch_claim_statuses",
            return_value={number: _claim_status(number)},
        ),
        patch(
            "scripts.dev.snapshot_issue_batch._issue_admission",
            return_value=malformed,
        ),
    ):
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        payload = snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=20,
        )

    assert payload["queue_completeness"] == "unavailable"
    assert payload["zero_work_authoritative"] is False


def test_snapshot_claimable_issues_uses_live_admission_for_ready_candidates() -> None:
    """Ready candidates must use the canonical check-only wrapper, including future gates."""
    issue_list = [
        {
            "number": 2669,
            "title": "ready issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2669",
            "labels": [{"name": "state:ready"}],
            "assignees": [],
        }
    ]
    preflight = {
        "schema": "issue_implementability.v1",
        "classification": "needs_dependency",
        "reasons": ["mandatory dependency is unsatisfied"],
        "ready": False,
        "write_allowed": False,
        "claim": _claim_status(2669),
    }
    with (
        patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh,
        patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim,
        patch("scripts.dev.snapshot_issue_batch.goal_issue_admission.admit_issue") as admit,
    ):
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        claim.return_value = {2669: _claim_status(2669)}
        admit.return_value = {
            "schema": "goal_issue_admission.v1",
            "ok": False,
            "outcome": "not_admitted",
            "write_attempted": False,
            "source_ref": "origin/main",
            "preflight": preflight,
            "claim": preflight["claim"],
        }
        payload = snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7", remote="origin", body_limit=150, limit=1
        )

    admission = payload["issues"][0]["admission"]
    assert admission["classification"] == "needs_dependency"
    assert admission["outcome"] == "not_admitted"
    assert admission["claim_outcome"] == "unclaimed"
    assert payload["claimable_count"] == 0
    admit.assert_called_once_with(
        2669,
        repo="ll7/robot_sf_ll7",
        remote="origin",
        source_ref="origin/main",
        check_only=True,
    )


@pytest.mark.parametrize(
    ("status", "classification", "reason_fragment"),
    [
        (
            "blocked_unchanged",
            "blocked_receipt",
            "blocker receipt unchanged",
        ),
        (
            "blocker_changed",
            "needs_re_evaluation",
            "require fresh evaluation",
        ),
    ],
)
def test_blocker_decision_fences_claimable_snapshot_dispatch(
    tmp_path, status: str, classification: str, reason_fragment: str
) -> None:  # type: ignore[no-untyped-def]
    """External blocker decisions prevent direct worker admission in queue snapshots."""
    decision_path = tmp_path / "decisions.json"
    decision_path.write_text(
        json.dumps(
            [
                {
                    "issue": 2710,
                    "status": status,
                    "reason": "fingerprint decision",
                    "receipt_digest": "a" * 64,
                    "current_fingerprint": "b" * 64,
                }
            ]
        ),
        encoding="utf-8",
    )
    issue_list = [
        {
            "number": 2710,
            "title": "claimable issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2710",
            "labels": [],
            "assignees": [],
        }
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {2710: _claim_status(2710)}
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=1,
                blocker_decision_paths=[str(decision_path)],
            )

    row = payload["issues"][0]
    assert row["classification"] == classification
    assert reason_fragment in row["reason"]
    assert row["blocker_decision"]["status"] == status
    assert row["dispatch_allowed"] is False


def test_malformed_blocker_decision_fails_closed_before_issue_discovery(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A malformed decision artifact cannot silently leave the queue claimable."""
    decision_path = tmp_path / "malformed.json"
    decision_path.write_text(json.dumps({"issue": 2710, "status": "unknown"}), encoding="utf-8")

    with patch("scripts.dev.snapshot_issue_batch._list_open_issues") as listing:
        payload = snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=1,
            blocker_decision_paths=[str(decision_path)],
        )

    listing.assert_not_called()
    assert payload["status"] == "error"
    assert payload["issues"][0]["status"] == "error"


def test_blocked_receipt_without_fingerprint_fails_closed_before_issue_discovery(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    """An incomplete suppression decision cannot fence a claimable issue."""
    decision_path = tmp_path / "incomplete.json"
    decision_path.write_text(
        json.dumps(
            {
                "issue": 2710,
                "status": "blocked_unchanged",
                "reason": "fingerprint decision",
            }
        ),
        encoding="utf-8",
    )

    with patch("scripts.dev.snapshot_issue_batch._list_open_issues") as listing:
        payload = snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=1,
            blocker_decision_paths=[str(decision_path)],
        )

    listing.assert_not_called()
    assert payload["status"] == "error"
    assert "current_fingerprint" in payload["errors"][0]


def test_snapshot_claimable_issues_fences_compute_routed_issue() -> None:
    """Compute-gated issues must not look claimable, even when marked ready."""
    issue_list = [
        {
            "number": 7009,
            "title": "friction: exclude compute-routed issues from claimable snapshots",
            "state": "OPEN",
            "url": "https://github.test/issues/7009",
            "labels": [
                {"name": "routing:needs-compute"},
                {"name": "state:ready"},
            ],
            "assignees": [],
        }
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {7009: _claim_status(7009)}
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=1,
            )

    issue = payload["issues"][0]
    assert issue["classification"] == "needs_compute"
    assert issue["reason"] == "the issue is routed to compute or campaign execution"


def test_snapshot_issues_fail_closed_for_closed_issue_state() -> None:
    """Explicit issue snapshots must not classify closed issues as claimable."""
    rest_issue = {
        "number": 2680,
        "status": "ok",
        "title": "closed but otherwise claimable issue",
        "body": "closed issue body",
        "state": " CLOSED ",
        "url": "https://github.test/issues/2680",
        "labels": ["workflow"],
        "assignees": [],
    }
    with patch("scripts.dev.snapshot_issue_batch.gh_issue_rest") as mock_rest:
        mock_rest.fetch_issue.return_value = rest_issue
        with patch("scripts.dev.snapshot_issue_batch.status_issue") as claim:
            claim.return_value = _claim_status(2680)
            payload = snapshot_issues(
                [2680], repo="ll7/robot_sf_ll7", body_limit=150, remote="origin"
            )

    row = payload["issues"][0]
    assert row["state"] == "CLOSED"
    assert row["classification"] == "closed"
    assert row["reason"] == "issue state is CLOSED; skip autonomous claim"


def test_snapshot_issues_reads_rest_when_graphql_quota_exhausted() -> None:
    """Explicit snapshots must succeed via REST when GraphQL quota is exhausted.

    Regression for issue #6845: the explicit snapshot path used to call
    ``gh issue view --json`` (GraphQL-backed), so every requested issue became
    an error row when GraphQL quota was exhausted even though the REST API was
    healthy. The path now routes through the REST-backed normalized reader
    ``scripts.dev.gh_issue_rest.fetch_issue`` and returns the compact row.
    """
    rest_issue = {
        "number": 6819,
        "status": "ok",
        "title": "workflow: explicit snapshot under quota exhaustion",
        "body": "REST body remains readable when GraphQL quota is exhausted",
        "state": "OPEN",
        "url": "https://github.com/ll7/robot_sf_ll7/issues/6819",
        "labels": ["enhancement", "workflow"],
        "assignees": [],
    }
    with (
        patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh,
        patch("scripts.dev.snapshot_issue_batch.gh_issue_rest") as mock_rest,
        patch("scripts.dev.snapshot_issue_batch.status_issue") as claim,
    ):
        # Pin the GraphQL-backed helper to the observed quota-exhaustion error.
        # If the explicit path ever regressed to `gh issue view --json`, _gh would
        # fire and this error would surface as an error row instead of a snapshot.
        mock_gh.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="GraphQL: API rate limit already exceeded",
        )
        mock_rest.fetch_issue.return_value = rest_issue
        claim.return_value = {
            "ok": True,
            "claimed": False,
            "claim_ref": "agent-claims/issue-6819",
            "sha": None,
        }
        payload = snapshot_issues([6819], repo="ll7/robot_sf_ll7", body_limit=300, remote="origin")

    issue = payload["issues"][0]
    assert issue["status"] == "ok"
    assert issue["number"] == 6819
    assert issue["state"] == "OPEN"
    assert issue["labels"] == ["enhancement", "workflow"]
    assert issue["classification"] == "state_conflict"
    assert issue["reason"] == (
        "exactly one execution state label is required unless a known hold qualifier already "
        "blocks dispatch; found none; state qualifiers are none"
    )
    assert issue["body_excerpt"] == rest_issue["body"]
    assert issue["body_truncated"] is False
    # Explicit reads must go through the REST reader, not the GraphQL CLI.
    mock_rest.fetch_issue.assert_called_once_with(6819, repo="ll7/robot_sf_ll7")
    mock_gh.assert_not_called()


def test_snapshot_issues_fails_closed_on_malformed_rest_success_payload() -> None:
    """A status=ok response must contain the complete normalized issue contract."""
    rest_issue = {
        "status": "ok",
        "title": "malformed explicit response",
        "body": "body",
        "state": "OPEN",
        "url": "https://github.test/issues/6819",
        "labels": [],
        "assignees": [],
    }
    with (
        patch("scripts.dev.snapshot_issue_batch.gh_issue_rest") as mock_rest,
        patch("scripts.dev.snapshot_issue_batch.status_issue") as claim,
    ):
        mock_rest.fetch_issue.return_value = rest_issue
        payload = snapshot_issues([6819], repo="ll7/robot_sf_ll7", body_limit=300, remote="origin")

    row = payload["issues"][0]
    assert row == {
        "number": 6819,
        "status": "error",
        "error": "REST issue read returned malformed data: REST issue response has no positive integer number",
    }
    claim.assert_not_called()


def test_snapshot_issues_fails_closed_when_rest_reader_raises_value_error() -> None:
    """A normalization ValueError must become an error row rather than escape the CLI."""
    with (
        patch("scripts.dev.snapshot_issue_batch.gh_issue_rest") as mock_rest,
        patch("scripts.dev.snapshot_issue_batch.status_issue") as claim,
    ):
        mock_rest.fetch_issue.side_effect = ValueError("invalid issue number")
        payload = snapshot_issues([6819], repo="ll7/robot_sf_ll7", body_limit=300, remote="origin")

    row = payload["issues"][0]
    assert row["status"] == "error"
    assert row["number"] == 6819
    assert "invalid issue number" in row["error"]
    claim.assert_not_called()


def test_snapshot_claimable_issues_fail_closed_for_closed_and_unknown_state() -> None:
    """Claimable list rows must fail closed when gh returns non-open or missing states."""
    issue_list = [
        {
            "number": 2681,
            "title": "closed claimable-looking issue",
            "state": "CLOSED",
            "url": "https://github.test/issues/2681",
            "labels": [],
            "assignees": [],
        },
        {
            "number": 2682,
            "title": "missing state claimable-looking issue",
            "url": "https://github.test/issues/2682",
            "labels": [],
            "assignees": [],
        },
        {
            "number": 2683,
            "title": "malformed state claimable-looking issue",
            "state": None,
            "url": "https://github.test/issues/2683",
            "labels": [],
            "assignees": [],
        },
        {
            "number": 2684,
            "title": "open issue remains claimable",
            "state": "OPEN",
            "url": "https://github.test/issues/2684",
            "labels": [],
            "assignees": [],
        },
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {
                2681: _claim_status(2681),
                2682: _claim_status(2682),
                2683: _claim_status(2683),
                2684: _claim_status(2684),
            }
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=4,
            )

    classifications = [issue["classification"] for issue in payload["issues"]]
    assert classifications == ["closed", "state_unknown", "state_unknown", "state_conflict"]
    assert [issue["state"] for issue in payload["issues"]] == ["CLOSED", "", "", "OPEN"]
    assert payload["issues"][0]["reason"] == "issue state is CLOSED; skip autonomous claim"
    assert payload["issues"][1]["reason"] == "issue state missing or unknown; skip autonomous claim"
    assert payload["issues"][2]["reason"] == "issue state missing or unknown; skip autonomous claim"


def test_snapshot_claimable_issues_uses_one_batch_claim_lookup() -> None:
    """Claimable snapshots should not shell out once per listed issue."""
    issue_list = [
        {
            "number": 2667,
            "title": "claimable issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2667",
            "labels": [],
            "assignees": [],
        },
        {
            "number": 2668,
            "title": "claimed issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2668",
            "labels": [],
            "assignees": [],
        },
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch.status_issue") as per_issue_claim:
            with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as batch_claim:
                batch_claim.return_value = {
                    2667: _claim_status(2667),
                    2668: _claim_status(2668, claimed=True, sha="abc123"),
                }
                payload = snapshot_claimable_issues(
                    repo="ll7/robot_sf_ll7",
                    remote="origin",
                    body_limit=150,
                    limit=2,
                )

    per_issue_claim.assert_not_called()
    batch_claim.assert_called_once_with([2667, 2668], remote="origin")
    assert [issue["classification"] for issue in payload["issues"]] == [
        "state_conflict",
        "already_claimed",
    ]
    assert payload["issues"][1]["claim"] == {
        "ok": True,
        "claimed": True,
        "claim_ref": "agent-claims/issue-2668",
        "sha": "abc123",
    }


def test_snapshot_claimable_issues_uses_bounded_rest_when_graphql_is_near_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low GraphQL quota falls back to one REST page and returns a resume cursor."""
    monkeypatch.setattr(
        snapshot_issue_batch,
        "_rate_limit_snapshot",
        lambda: RateLimitSnapshot(
            status="ok",
            graphql_remaining=50,
            graphql_reset_at=1_800_000_123,
            core_remaining=4_000,
            core_reset_at=1_800_000_456,
        ),
    )
    rest_rows = [
        {
            "number": 2701,
            "title": "first issue",
            "state": "open",
            "html_url": "https://github.test/issues/2701",
            "labels": [],
            "assignees": [],
        },
        {
            "number": 2702,
            "title": "pull request is not an issue",
            "state": "open",
            "html_url": "https://github.test/pull/2702",
            "labels": [],
            "assignees": [],
            "pull_request": {"url": "https://api.github.test/pulls/2702"},
        },
        {
            "number": 2703,
            "title": "second issue",
            "state": "open",
            "html_url": "https://github.test/issues/2703",
            "labels": [],
            "assignees": [],
        },
    ]
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(rest_rows), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {
                2701: _claim_status(2701),
                2703: _claim_status(2703),
            }
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=2,
            )

    assert payload["status"] == "ok"
    assert payload["data_source"] == "rest"
    assert [issue["number"] for issue in payload["issues"]] == [2701, 2703]
    assert payload["resume_cursor"] == {"source": "rest", "page": 2, "limit": 2}
    assert mock_gh.call_args.args[0][:2] == ["api", "repos/ll7/robot_sf_ll7/issues"]
    claim.assert_called_once_with([2701, 2703], remote="origin")


def test_snapshot_claimable_issues_is_empty_and_resumable_when_all_sources_are_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed low-quota fallback returns no partial queue and no claim reads."""
    monkeypatch.setattr(
        snapshot_issue_batch,
        "_rate_limit_snapshot",
        lambda: RateLimitSnapshot(
            status="ok",
            graphql_remaining=0,
            graphql_reset_at=1_800_000_123,
            core_remaining=4_000,
            core_reset_at=1_800_000_456,
        ),
    )
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="REST: API rate limit exceeded",
        )
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=20,
            )

    assert payload["status"] == "quota_blocked"
    assert payload["issues"] == [
        {"status": "quota_blocked", "error": "REST: API rate limit exceeded"}
    ]
    assert payload["resume_cursor"] == {"source": "rest", "page": 1, "limit": 20}
    claim.assert_not_called()


def test_snapshot_claimable_issues_resumes_rest_page_even_with_healthy_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A returned page cursor is honored without reopening an unbounded GraphQL scan."""
    rest_rows = [
        {
            "number": 2710,
            "title": "resumed issue",
            "state": "open",
            "html_url": "https://github.test/issues/2710",
            "labels": [],
            "assignees": [],
        }
    ]
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(rest_rows), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {2710: _claim_status(2710)}
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=20,
                resume_page=7,
            )

    assert payload["data_source"] == "rest"
    assert payload["issues"][0]["number"] == 2710
    assert mock_gh.call_args.args[0][0] == "api"
    assert "--field" in mock_gh.call_args.args[0]
    claim.assert_called_once_with([2710], remote="origin")


def test_batch_claim_statuses_parses_claim_refs_once() -> None:
    """Batch claim lookup should parse matching remote refs and synthesize unclaimed rows."""
    stdout = (
        "abc123\trefs/heads/agent-claims/issue-2668\n"
        "def456\trefs/heads/agent-claims/issue-not-a-number\n"
    )

    with patch("scripts.dev.snapshot_issue_batch.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=stdout,
            stderr="",
        )
        statuses = _batch_claim_statuses([2667, 2668], remote="origin")

    run.assert_called_once_with(
        ["git", "ls-remote", "--heads", "origin", "refs/heads/agent-claims/issue-*"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert statuses[2667]["claimed"] is False
    assert statuses[2667]["sha"] is None
    assert statuses[2668]["claimed"] is True
    assert statuses[2668]["sha"] == "abc123"


def test_batch_claim_status_failure_fails_closed() -> None:
    """Batch lookup failures should not falsely mark listed issues unclaimed."""
    with patch("scripts.dev.snapshot_issue_batch.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=128,
            stdout="",
            stderr="network unavailable",
        )
        statuses = _batch_claim_statuses([2667, 2668], remote="origin")

    assert statuses[2667]["ok"] is False
    assert statuses[2667]["claimed"] is None
    assert statuses[2667]["error"] == "network unavailable"
    assert statuses[2668]["ok"] is False
    assert statuses[2668]["claimed"] is None


def test_snapshot_claimable_issues_excludes_blocked_external_by_default() -> None:
    """Default claim routing should quarantine external-data blockers."""
    issue_list = [
        {
            "number": 2962,
            "title": "workflow issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2962",
            "labels": [{"name": "workflow"}],
            "assignees": [],
        },
        {
            "number": 2415,
            "title": "data: stage external asset",
            "state": "OPEN",
            "url": "https://github.test/issues/2415",
            "labels": [{"name": "resource:external-data"}, {"name": "state:blocked"}],
            "assignees": [],
        },
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {
                2962: _claim_status(2962),
                2415: _claim_status(2415),
            }
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=2,
            )

    assert [issue["number"] for issue in payload["issues"]] == [2962]
    assert payload["excluded_counts"] == {"blocked_external": 1}
    assert payload["include_blocked_external"] is False


def test_snapshot_claimable_issues_can_include_blocked_external() -> None:
    """Explicit routing should expose quarantined external blockers."""
    issue_list = [
        {
            "number": 2415,
            "title": "data: stage external asset",
            "state": "OPEN",
            "url": "https://github.test/issues/2415",
            "labels": [{"name": "resource:external-data"}, {"name": "state:blocked"}],
            "assignees": [],
        }
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {2415: _claim_status(2415)}
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=1,
                include_blocked_external=True,
            )

    assert payload["include_blocked_external"] is True
    assert payload["issues"][0]["classification"] == "blocked"
    assert payload["excluded_counts"] == {"blocked_external": 1}


def test_snapshot_blocked_external_issues_writes_human_report(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Blocked external report should provide one action and monthly review date per row."""
    from datetime import UTC, datetime

    report_path = tmp_path / "blocked-assets.md"
    issue_list = [
        {
            "number": 2415,
            "title": "data: stage external asset",
            "state": "OPEN",
            "url": "https://github.test/issues/2415",
            "labels": [
                {"name": "resource:external-data"},
                {"name": "state:blocked"},
                {"name": "state:ready"},
            ],
            "assignees": [],
        },
        {
            "number": 2962,
            "title": "workflow: executable now",
            "state": "OPEN",
            "url": "https://github.test/issues/2962",
            "labels": [{"name": "resource:external-data"}, {"name": "state:ready"}],
            "assignees": [],
        },
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        payload = snapshot_blocked_external_issues(
            repo="ll7/robot_sf_ll7",
            report_path=str(report_path),
            limit=10,
            now=datetime(2026, 6, 15, tzinfo=UTC),
        )

    assert payload["schema"] == "blocked_external_assets_report.v1"
    assert payload["recommended_state_label"] == "state:blocked-external-input"
    assert payload["row_count"] == 1
    row = payload["rows"][0]
    assert row["number"] == 2415
    assert row["human_action"] == (
        "Stage or document the required external data/asset/license before agent execution."
    )
    assert row["monthly_review_date"] == "2026-07-01"
    assert "add `state:blocked-external-input`" in row["label_recommendation"]
    assert "remove `state:ready`" in row["label_recommendation"]
    assert "#2415 data: stage external asset" in report_path.read_text(encoding="utf-8")


def test_snapshot_blocked_external_issues_reports_unexpected_json_shape() -> None:
    """Blocked external report should fail closed on unexpected gh JSON shape."""
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"message": "not an issue list"}),
            stderr="",
        )
        payload = snapshot_blocked_external_issues(
            repo="ll7/robot_sf_ll7",
            limit=10,
        )

    assert payload["row_count"] == 0
    assert payload["errors"] == [{"status": "error", "error": "expected gh issue list JSON array"}]


def test_main_rejects_include_blocked_external_without_claimable(capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI should not silently ignore --include-blocked-external outside claimable mode."""
    rc = main(["--include-blocked-external", "--json"])

    assert rc == 1
    assert "--include-blocked-external requires --claimable" in capsys.readouterr().err


def test_snapshot_active_issue_portfolio_classifies_and_writes_report(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Active portfolio report should classify rows and recommend label changes only."""
    report_path = tmp_path / "portfolio.md"
    issue_list = [
        {
            "number": 2967,
            "title": "workflow: generate active issue portfolio",
            "state": "OPEN",
            "url": "https://github.test/issues/2967",
            "labels": [{"name": "workflow"}, {"name": "state:ready"}],
            "assignees": [],
        },
        {
            "number": 2415,
            "title": "data: stage external asset",
            "state": "OPEN",
            "url": "https://github.test/issues/2415",
            "labels": [
                {"name": "resource:external-data"},
                {"name": "state:blocked"},
                {"name": "state:ready"},
            ],
            "assignees": [],
        },
        {
            "number": 1134,
            "title": "map: needs decision",
            "state": "OPEN",
            "url": "https://github.test/issues/1134",
            "labels": [{"name": "decision-required"}, {"name": "state:ready"}],
            "assignees": [],
        },
        {
            "number": 2946,
            "title": "analysis: diagnostic figure pack",
            "state": "OPEN",
            "url": "https://github.test/issues/2946",
            "labels": [{"name": "research"}],
            "assignees": [],
        },
        {
            "number": 2910,
            "title": "epic: validation benchmark",
            "state": "OPEN",
            "url": "https://github.test/issues/2910",
            "labels": [{"name": "type:synthesis"}, {"name": "priority: high"}],
            "assignees": [],
        },
        {
            "number": 2965,
            "title": "benchmark: release readiness dashboard",
            "state": "OPEN",
            "url": "https://github.test/issues/2965",
            "labels": [{"name": "benchmark"}, {"name": "priority: high"}, {"name": "state:ready"}],
            "assignees": [],
        },
        {
            "number": 2845,
            "title": "prediction: blocked local study",
            "state": "OPEN",
            "url": "https://github.test/issues/2845",
            "labels": [{"name": "state:blocked"}],
            "assignees": [],
        },
        {
            "number": 2441,
            "title": "slurm: finalize trace collection",
            "state": "OPEN",
            "url": "https://github.test/issues/2441",
            "labels": [{"name": "resource:slurm"}, {"name": "state:ready"}],
            "assignees": [],
        },
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {
                int(issue["number"]): _claim_status(int(issue["number"])) for issue in issue_list
            }
            payload = snapshot_active_issue_portfolio(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                report_path=str(report_path),
                limit=10,
            )

    assert payload["schema"] == "active_issue_portfolio.v1"
    assert payload["row_count"] == 8
    rows = {row["number"]: row for row in payload["rows"]}
    assert rows[2967]["classification"] == "executable_now"
    assert rows[2967]["owner_type"] == "agent"
    assert rows[2415]["classification"] == "blocked_external_asset"
    assert rows[2415]["owner_type"] == "external data"
    assert rows[2415]["label_recommendation"] == (
        "add `state:blocked-external-input`; remove `state:ready`"
    )
    assert rows[1134]["classification"] == "needs_human_decision"
    assert rows[1134]["owner_type"] == "maintainer"
    assert rows[2946]["classification"] == "diagnostic_only"
    assert rows[2946]["label_recommendation"] == "add `evidence:analysis-only`"
    assert rows[2910]["classification"] == "stale_synthesis"
    assert rows[2965]["classification"] == "paper_critical"
    assert rows[2965]["label_recommendation"] == "add `paper-critical`"
    assert rows[2845]["classification"] == "needs_human_decision"
    assert rows[2845]["owner_type"] == "maintainer"
    assert rows[2441]["classification"] == "executable_now"
    assert rows[2441]["owner_type"] == "Slurm"
    assert payload["classification_counts"]["blocked_external_asset"] == 1
    claim.assert_called_once_with([2967, 2415, 1134, 2946, 2910, 2965, 2845, 2441], remote="origin")
    markdown = report_path.read_text(encoding="utf-8")
    assert "# Active Issue Portfolio" in markdown
    assert (
        "| #2967 workflow: generate active issue portfolio | executable_now | agent |" in markdown
    )


def test_snapshot_active_issue_portfolio_reports_unexpected_json_shape() -> None:
    """Active portfolio should fail closed when gh returns an unexpected JSON shape."""
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"message": "not an issue list"}),
            stderr="",
        )
        payload = snapshot_active_issue_portfolio(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            limit=10,
        )

    assert payload["row_count"] == 0
    assert payload["errors"] == [{"status": "error", "error": "expected gh issue list JSON array"}]


def test_snapshot_active_issue_portfolio_fails_closed_on_claim_lookup_error() -> None:
    """Portfolio rows should avoid executable_now when claim state cannot be read."""
    issue_list = [
        {
            "number": 3001,
            "title": "workflow issue",
            "state": "OPEN",
            "url": "https://github.test/issues/3001",
            "labels": [{"name": "workflow"}],
            "assignees": [],
        }
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {3001: _claim_status(3001, ok=False)}
            payload = snapshot_active_issue_portfolio(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                limit=1,
            )

    row = payload["rows"][0]
    assert row["classification"] == "needs_human_decision"
    assert row["reason"] == "unable to read claim state; skip autonomous claim"
    assert row["owner_type"] == "maintainer"


def test_snapshot_active_issue_portfolio_fails_closed_on_malformed_claim_status() -> None:
    """Malformed batched claim values should not classify rows as executable."""
    issue_list = [
        {
            "number": 3002,
            "title": "workflow issue",
            "state": "OPEN",
            "url": "https://github.test/issues/3002",
            "labels": [{"name": "workflow"}],
            "assignees": [],
        }
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {3002: None}
            payload = snapshot_active_issue_portfolio(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                limit=1,
            )

    row = payload["rows"][0]
    assert row["classification"] == "needs_human_decision"
    assert row["reason"] == "unable to read claim state; skip autonomous claim"
    assert row["owner_type"] == "maintainer"


def test_snapshot_active_issue_portfolio_preserves_null_optional_fields() -> None:
    """Explicit null title or URL fields should not leak as literal None strings."""
    issue_list = [
        {
            "number": 3000,
            "title": None,
            "state": "OPEN",
            "url": None,
            "labels": [],
            "assignees": [],
        }
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {3000: _claim_status(3000)}
            payload = snapshot_active_issue_portfolio(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                limit=1,
            )

    row = payload["rows"][0]
    assert row["title"] == ""
    assert row["url"] == ""
    assert "None" not in payload["markdown"]
    assert "#3000 " in payload["markdown"]


def test_main_rejects_active_portfolio_with_claimable(capsys) -> None:  # type: ignore[no-untyped-def]
    """Portfolio mode should be a distinct bounded report mode."""
    rc = main(["--active-portfolio", "--claimable", "--json"])

    assert rc == 1
    assert "--active-portfolio cannot be combined" in capsys.readouterr().err


def test_main_claimable_mode_can_be_called_without_issue_numbers() -> None:  # type: ignore[no-untyped-def]
    """CLI should run compact claim discovery when issue numbers are intentionally omitted."""
    issue_list = [
        {
            "number": 2669,
            "title": "no-arg mode issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2669",
            "labels": [],
            "assignees": [],
        }
    ]
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {2669: _claim_status(2669)}
            rc = main(["--claimable", "--json", "--limit", "1"])

    assert rc == 0


def test_snapshot_claimable_issues_filters_ready_label_before_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claimable discovery must constrain listing to state:ready before the page limit."""
    monkeypatch.setattr(
        snapshot_issue_batch,
        "_load_blocker_decisions",
        lambda paths: ({}, []),
    )
    listing = snapshot_issue_batch._listing_result(
        status="ok",
        listed=[],
        error="",
        data_source="graphql",
        rate_limit=RateLimitSnapshot(
            status="ok",
            graphql_remaining=4_000,
            graphql_reset_at=1_800_000_000,
            core_remaining=4_000,
            core_reset_at=1_800_000_000,
        ),
        quota={},
        resume_cursor=None,
    )
    with patch("scripts.dev.snapshot_issue_batch._list_open_issues") as list_issues:
        list_issues.return_value = listing
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=3,
        )

    assert payload["status"] == "ok"
    assert payload["queue_completeness"] == "complete"
    assert payload["candidate_scope"] == "state:ready"
    assert payload["zero_work_authoritative"] is True
    assert payload["claimable_count"] == 0
    list_issues.assert_called_once_with(
        repo="ll7/robot_sf_ll7",
        limit=3,
        min_graphql_remaining=snapshot_issue_batch.DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
        resume_page=1,
        label=snapshot_issue_batch.issue_implementability.READY_LABEL,
    )


def test_list_open_issues_propagates_label_to_graphql_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GraphQL discovery path must receive the ready-label candidate filter."""
    seen: dict[str, object] = {}

    def fake_graphql(*, repo: str, limit: int, label: str | None) -> dict[str, object]:
        seen["label"] = label
        return {"status": "ok", "listed": []}

    monkeypatch.setattr(snapshot_issue_batch, "_graphql_open_issue_list", fake_graphql)
    listing = snapshot_issue_batch._list_open_issues(
        repo="ll7/robot_sf_ll7",
        limit=5,
        min_graphql_remaining=snapshot_issue_batch.DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
        label=snapshot_issue_batch.issue_implementability.READY_LABEL,
    )

    assert listing["status"] == "ok"
    assert seen["label"] == snapshot_issue_batch.issue_implementability.READY_LABEL


def test_list_open_issues_propagates_label_to_rest_resume_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The REST resume discovery path must receive the ready-label candidate filter."""
    seen: dict[str, object] = {}

    def fake_rest(*, repo: str, page: int, limit: int, label: str | None) -> dict[str, object]:
        seen["label"] = label
        seen["page"] = page
        return {"status": "ok", "listed": [], "resume_cursor": None}

    monkeypatch.setattr(snapshot_issue_batch, "_rest_open_issue_list", fake_rest)
    listing = snapshot_issue_batch._list_open_issues(
        repo="ll7/robot_sf_ll7",
        limit=5,
        min_graphql_remaining=snapshot_issue_batch.DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
        resume_page=2,
        label=snapshot_issue_batch.issue_implementability.READY_LABEL,
    )

    assert listing["status"] == "ok"
    assert listing["data_source"] == "rest"
    assert seen["label"] == snapshot_issue_batch.issue_implementability.READY_LABEL
    assert seen["page"] == 2


def test_snapshot_claimable_issues_marks_truncated_scan_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumable ready-candidate page must forbid an authoritative zero-work verdict."""
    monkeypatch.setattr(
        snapshot_issue_batch,
        "_rate_limit_snapshot",
        lambda: RateLimitSnapshot(
            status="ok",
            graphql_remaining=50,
            graphql_reset_at=1_800_000_123,
            core_remaining=4_000,
            core_reset_at=1_800_000_456,
        ),
    )
    rows = [
        {
            "number": 2801 + offset,
            "title": f"ready issue {offset}",
            "state": "open",
            "html_url": f"https://github.test/issues/{2801 + offset}",
            "labels": [{"name": "state:ready"}],
            "assignees": [],
        }
        for offset in range(2)
    ]
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(rows), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses") as claim:
            claim.return_value = {
                2801: _claim_status(2801),
                2802: _claim_status(2802),
            }
            payload = snapshot_issue_batch.snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=2,
            )

    assert payload["resume_cursor"] == {"source": "rest", "page": 2, "limit": 2}
    assert payload["queue_completeness"] == "incomplete"
    assert payload["zero_work_authoritative"] is False
    assert payload["truncated"] is True


def test_snapshot_claimable_issues_marks_failed_scan_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed or quota-blocked discovery must be unavailable, never zero-work evidence."""
    monkeypatch.setattr(
        snapshot_issue_batch,
        "_load_blocker_decisions",
        lambda paths: ({}, []),
    )
    listing = snapshot_issue_batch._listing_result(
        status="quota_blocked",
        listed=[],
        error="rate limit exhausted",
        data_source="none",
        rate_limit=RateLimitSnapshot(
            status="error",
            graphql_remaining=0,
            graphql_reset_at=1_800_000_000,
            core_remaining=0,
            core_reset_at=1_800_000_000,
        ),
        quota={"status": "quota_blocked"},
        resume_cursor={"source": "rest", "page": 1, "limit": 3},
    )
    with patch("scripts.dev.snapshot_issue_batch._list_open_issues") as list_issues:
        list_issues.return_value = listing
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=3,
        )

    assert payload["status"] == "quota_blocked"
    assert payload["queue_completeness"] == "unavailable"
    assert payload["zero_work_authoritative"] is False
    assert payload["claimable_count"] == 0
