"""Regression tests for repository-wide WIP admission (issue #7520)."""

from __future__ import annotations

from datetime import UTC, datetime

from scripts.dev import issue_claim, wip_capacity

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
HEAD = "a" * 40


def _policy() -> dict[str, object]:
    return wip_capacity.load_policy()


def _pr(
    number: int,
    issue: int,
    *,
    labels: list[str] | None = None,
    overall: str = "pending",
    body: str | None = None,
    title: str | None = None,
) -> dict[str, object]:
    return {
        "number": number,
        "status": "ok",
        "draft": False,
        "title": title or f"fix: lane for issue #{issue}",
        "body": body or f"Refs #{issue}",
        "head_sha": HEAD,
        "labels": labels or ["priority:1"],
        "checks": {"overall": overall},
        "base_freshness": {
            "verdict": "fresh",
            "base_sha": "base",
            "current_main_sha": "base",
        },
    }


def _snapshot(*prs: dict[str, object], truncated: bool = False) -> dict[str, object]:
    return {
        "schema": "pr_queue_snapshot.v2",
        "repo": "ll7/robot_sf_ll7",
        "mode": "active",
        "truncated": truncated,
        "prs": list(prs),
    }


def _evaluate(
    snapshot: dict[str, object],
    *,
    claims: list[dict[str, object]] | None = None,
    proposed: dict[str, object] | None = None,
    mode: str = "enforce",
) -> dict[str, object]:
    return wip_capacity.evaluate_capacity(
        snapshot,
        claims or [],
        _policy(),
        proposed=proposed,
        mode=mode,
        now=NOW,
    )


def test_under_limit_allows_a_new_implementation_lane() -> None:
    """A complete under-limit queue admits a distinct ordinary issue."""
    result = _evaluate(
        _snapshot(_pr(1, 101)),
        proposed={"issue": 102, "labels": ["priority:1"]},
    )

    assert result["decision"] == "allow"
    assert result["counts"] == {"implementation": 1, "campaign_operations": 0}
    assert result["remaining"] == {"implementation": 2, "campaign_operations": 2}
    assert result["raw_open_pr_count_diagnostic"] == 1


def test_exact_limit_blocks_both_applicable_lanes() -> None:
    """The configured three/f two lane limits are enforced independently."""
    implementation = [_pr(index, 100 + index) for index in range(1, 4)]
    campaigns = [
        _pr(10, 110, labels=["campaign"]),
        _pr(11, 111, labels=["slurm"]),
    ]
    snapshot = _snapshot(*implementation, *campaigns)

    implementation_result = _evaluate(
        snapshot,
        proposed={"issue": 120, "labels": ["priority:1"]},
    )
    campaign_result = _evaluate(
        snapshot,
        proposed={"issue": 121, "labels": ["campaign"]},
    )

    assert implementation_result["decision"] == "block"
    assert campaign_result["decision"] == "block"
    assert {item["lane"] for item in implementation_result["blockers"] if "lane" in item} == {
        "implementation"
    }
    assert {item["lane"] for item in campaign_result["blockers"] if "lane" in item} == {
        "campaign_operations"
    }


def test_report_only_is_observable_but_not_claimed_as_available() -> None:
    """The rollout switch reports a full queue without turning it into capacity."""
    result = _evaluate(
        _snapshot(*[_pr(index, 200 + index) for index in range(1, 4)]),
        proposed={"issue": 250, "labels": ["priority:1"]},
        mode="report-only",
    )

    assert result["decision"] == "report_only"
    assert result["allowed"] is False
    assert result["available_capacity_proven"] is True
    assert any(item["reason"] == "wip_limit_full" for item in result["blockers"])


def test_author_decision_is_visible_but_not_counted() -> None:
    """A live decision packet parks a PR outside productive WIP capacity."""
    row = _pr(2, 202, labels=["priority:1", "decision-required"])
    row["comments"] = [
        {
            "author": "maintainer",
            "authorAssociation": "OWNER",
            "body": "### Decision packet\nAwaiting author ruling.",
        }
    ]

    result = _evaluate(_snapshot(row))

    assert result["counts"]["implementation"] == 0
    assert result["excluded_items"][0]["state"] == "author_decision"
    assert result["excluded_items"][0]["reason"] == "author_decision_parked"


def test_active_writer_and_review_are_counted_as_one_implementation_lane() -> None:
    """Writer ownership and reviewer activity remain active until terminal disposition."""
    writer = _pr(3, 203)
    writer["comments"] = [
        {
            "author": "worker",
            "authorAssociation": "COLLABORATOR",
            "body": f"review-claim: implementation @ {HEAD} until 2026-08-18T13:00:00Z",
        }
    ]
    review = _pr(4, 204)
    review["reviews"] = [{"state": "COMMENTED", "authorAssociation": "COLLABORATOR"}]

    result = _evaluate(_snapshot(writer, review))

    assert result["counts"]["implementation"] == 2
    assert {item["state"] for item in result["counted_lanes"]} == {"active_writer", "pending_ci"}
    assert any(item["reason"] == "review_in_progress" for item in result["counted_lanes"])


def test_common_current_main_baseline_is_explicitly_blocked_and_excluded() -> None:
    """Common-baseline work remains visible without consuming productive capacity."""
    row = _pr(5, 205, body="Blocked by the common current-main baseline.")
    row["base_freshness"] = {
        "verdict": "stale",
        "base_sha": "old",
        "current_main_sha": "new",
    }

    result = _evaluate(_snapshot(row))

    assert result["counts"]["implementation"] == 0
    assert result["excluded_items"][0]["category"] == "blocked"
    assert result["excluded_items"][0]["reason"] == "common_current_main_baseline_blocked"


def test_stale_claim_is_explicitly_excluded() -> None:
    """Stale/expired claim evidence releases capacity only in the projection."""
    result = _evaluate(
        _snapshot(),
        claims=[{"issue": 206, "claim_ref": "agent-claims/issue-206", "stale": True}],
    )

    assert result["counts"]["implementation"] == 0
    assert result["excluded_items"][0]["category"] == "stale_or_expired_claim"


def test_competing_prs_are_one_lane_but_fail_coordination_closed() -> None:
    """Duplicate PRs cannot inflate capacity or silently pass ownership admission."""
    result = _evaluate(_snapshot(_pr(7, 207), _pr(8, 207)))

    assert result["counts"]["implementation"] == 1
    assert result["coordination_blockers"][0]["reason"] == "competing_pr_same_issue"
    assert any(item["reason"] == "competing_pr_same_issue" for item in result["excluded_items"])
    assert result["decision"] == "block"


def test_owned_claim_can_continue_to_worktree_or_pr_without_new_lane() -> None:
    """An already-admitted owner may continue its claim even when new capacity is full."""
    result = _evaluate(
        _snapshot(*[_pr(index, 600 + index) for index in range(1, 4)]),
        claims=[{"issue": 610, "claim_ref": "agent-claims/issue-610", "state": "active"}],
        proposed={"issue": 610, "labels": ["priority:1"]},
    )

    assert result["decision"] == "allow"
    assert result["continuation_of_owned_claim"] is True
    assert not any(item["reason"] == "wip_limit_full" for item in result["blockers"])


def test_unknown_queue_evidence_never_proves_capacity() -> None:
    """A truncated queue is a blocker even when the visible portion is empty."""
    result = _evaluate(
        _snapshot(truncated=True),
        proposed={"issue": 208, "labels": ["priority:1"]},
    )

    assert result["decision"] == "block"
    assert result["available_capacity_proven"] is False
    assert any(item["reason"] == "capacity_evidence_unavailable" for item in result["blockers"])


def _audited_exemption(
    kind: str, scope: str, *, expires: str = "2026-08-18T13:00:00Z"
) -> dict[str, str]:
    return {
        "kind": kind,
        "actor": "maintainer",
        "reason": "bounded incident repair",
        "scope": scope,
        "issued_at": "2026-08-18T11:00:00Z",
        "expires_at": expires,
    }


def test_p0_red_main_exemption_is_narrow_and_audited() -> None:
    """Only an audited priority-zero red-main proposal bypasses a full implementation lane."""
    full = _snapshot(*[_pr(index, 300 + index) for index in range(1, 4)])
    result = _evaluate(
        full,
        proposed={
            "issue": 310,
            "labels": ["priority:0", "red-main"],
            "exemption": _audited_exemption("p0_red_main", "issue:310"),
        },
    )

    assert result["decision"] == "allow"
    assert result["exemption"]["kind"] == "p0_red_main"


def test_security_exemption_requires_security_label() -> None:
    """A security exemption cannot be used for an ordinary issue."""
    full = _snapshot(*[_pr(index, 400 + index) for index in range(1, 4)])
    result = _evaluate(
        full,
        proposed={
            "issue": 410,
            "labels": ["priority:1"],
            "exemption": _audited_exemption("security_incident", "issue:410"),
        },
    )

    assert result["decision"] == "block"
    assert any(
        item["reason"] == "security_exemption_requires_security_label"
        for item in result["blockers"]
    )


def test_expired_maintainer_override_cannot_bypass_full_lane() -> None:
    """Overrides are time-bounded and never become permanent implicit bypasses."""
    full = _snapshot(*[_pr(index, 500 + index) for index in range(1, 4)])
    result = _evaluate(
        full,
        proposed={
            "issue": 510,
            "labels": ["priority:1"],
            "exemption": _audited_exemption(
                "maintainer_override", "issue:510", expires="2026-08-18T11:59:00Z"
            ),
        },
    )

    assert result["decision"] == "block"
    assert any(
        item["reason"] == "exemption_expired_or_timestamp_invalid" for item in result["blockers"]
    )
    assert any(item["reason"] == "wip_limit_full" for item in result["blockers"])


def test_shared_command_contracts_use_the_capacity_gate() -> None:
    """Claim and PR publication commands point at the one shared evaluator."""
    claim_command = issue_claim.build_wip_capacity_command(
        520, repo="ll7/robot_sf_ll7", remote="origin"
    )
    assert claim_command[-3:] == ["--proposed-issue", "520", "--json"]
    assert claim_command[-4] == "policy"
