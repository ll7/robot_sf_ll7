"""Offline tests for the merge-ready carrier binding gate (issue #7610)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from scripts.dev.pr_carrier_gate import (
    check_merge_ready_carriers,
    review_comment_covers,
    stale_carrier_sentinels,
)

HEAD_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
BASE_SHA = "b1c2d3e4f5061728394a5b6c7d8e9f0011121314"
OTHER_SHA = "deadbeef00000000000000000000000000000001"


def _proc(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess:
    """Build a fake ``gh api`` response."""
    return subprocess.CompletedProcess(["gh", "api"], returncode, stdout=stdout, stderr=stderr)


def _pr_payload(
    *,
    title: str = "fix(benchmark): gate carriers",
    body: str = "",
    head: str = HEAD_SHA,
    base: str = BASE_SHA,
) -> str:
    return json.dumps({"title": title, "body": body, "head": {"sha": head}, "base": {"sha": base}})


def _comments_payload(*bodies: str) -> str:
    return json.dumps(
        [
            {"user": {"login": "ll7"}, "authorAssociation": "COLLABORATOR", "body": body}
            if not body.startswith("bot:")
            else {
                "user": {"login": "github-actions[bot]"},
                "authorAssociation": "MEMBER",
                "body": body[4:],
            }
            for body in bodies
        ]
    )


def _reviews_payload(
    *bodies: str,
    commit_id: str = HEAD_SHA,
    state: str = "COMMENTED",
) -> str:
    """Build pull-request review payloads returned by the REST endpoint."""
    return json.dumps(
        [
            {
                "user": {"login": "ll7"},
                "authorAssociation": "COLLABORATOR",
                "body": body,
                "commit_id": commit_id,
                "state": state,
            }
            for body in bodies
        ]
    )


def _review_comment() -> str:
    return (
        "## Exact-head self-review\n\n"
        f"- Exact base: `{BASE_SHA}`\n"
        f"- Exact head reviewed: `{HEAD_SHA}`\n"
        "- Findings: none"
    )


def _body_with_carrier(body: str, sha: str) -> str:
    return body + f"\n\nExact head: {sha}"


def test_pure_review_carrier_matches_live_state() -> None:
    """A review naming the live head and live base covers the live state."""
    assert review_comment_covers(
        _review_comment(),
        live_head=HEAD_SHA,
        live_base=BASE_SHA,
    )


def test_pure_review_carrier_accepts_exact_head_implementation_review() -> None:
    """Explicit implementation-review wording is equivalent to the canonical self-review."""
    comment = _review_comment().replace(
        "Exact-head self-review", "Exact-head implementation review"
    )

    assert review_comment_covers(comment, live_head=HEAD_SHA, live_base=BASE_SHA)


def test_pure_review_carrier_rejects_ambiguous_exact_head_review() -> None:
    """A generic review heading cannot become a merge-ready carrier by naming one SHA."""
    comment = _review_comment().replace("Exact-head self-review", "Exact-head review")

    assert not review_comment_covers(comment, live_head=HEAD_SHA, live_base=BASE_SHA)


def test_pure_review_carrier_rejects_implementation_review_prefixes() -> None:
    """A longer unrelated phrase must not match the implementation-review contract."""
    comment = _review_comment().replace(
        "Exact-head self-review", "Exact-head implementation reviewability"
    )

    assert not review_comment_covers(comment, live_head=HEAD_SHA, live_base=BASE_SHA)


def test_pure_review_carrier_rejects_foreign_head() -> None:
    """A review of a different head never covers the live state."""
    assert not review_comment_covers(
        _review_comment(),
        live_head=OTHER_SHA,
        live_base=BASE_SHA,
    )


def test_pure_review_carrier_rejects_stale_declared_base() -> None:
    """A review declaring an old base is not current-base evidence."""
    comment = _review_comment().replace(BASE_SHA, OTHER_SHA)
    assert not review_comment_covers(comment, live_head=HEAD_SHA, live_base=BASE_SHA)


def test_pure_review_carrier_rejects_stale_reviewed_base() -> None:
    """Common ``Base reviewed:`` wording remains bound to the live base."""
    comment = _review_comment().replace(
        f"Exact base: `{BASE_SHA}`", f"Base reviewed: `{OTHER_SHA}`"
    )
    assert not review_comment_covers(comment, live_head=HEAD_SHA, live_base=BASE_SHA)


def test_pure_review_carrier_ignores_bot_comments() -> None:
    """A bot comment is not a human review carrier."""
    assert not review_comment_covers(
        "Exact-head self-review by coderabbitai[bot]",
        live_head=HEAD_SHA,
        live_base=BASE_SHA,
    )


def test_pure_stale_sentinels_detected() -> None:
    """Observed #7547 markers must be recognized as stale narratives."""
    assert stale_carrier_sentinels(
        "the earlier focused self-review is not current-base merge evidence"
    )
    assert stale_carrier_sentinels("domain-aware approval remained pending")
    assert stale_carrier_sentinels("pending-domain-review disposition")
    assert not stale_carrier_sentinels("all domain-aware approvals are recorded as approved")


def test_gate_admits_bound_body_and_review() -> None:
    """A body naming the live head plus a live-bound review comment pass the gate."""
    title = "fix(benchmark): gate carriers"
    body = _body_with_carrier("### Summary\n\nBounded gate repair.", HEAD_SHA)
    with (
        patch(
            "scripts.dev.pr_carrier_gate._gh_api_get",
            side_effect=[
                _proc(stdout=_pr_payload(title=title, body=body)),
                _proc(stdout=_comments_payload(_review_comment())),
                _proc(stdout=_reviews_payload()),
            ],
        ),
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "ok"
    assert result["live_head_sha"] == HEAD_SHA


def test_gate_admits_review_endpoint_carrier_without_compatibility_comment() -> None:
    """A live-head COMMENTED review from the REST endpoint is a valid carrier."""
    title = "fix(benchmark): gate carriers"
    body = _body_with_carrier("### Summary\n\nBounded gate repair.", HEAD_SHA)
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload()),
            _proc(stdout=_reviews_payload(_review_comment())),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "ok"
    assert result["carrier_source"] == "pull_request_review"


def test_gate_admits_implementation_review_compatibility_comment() -> None:
    """The documented implementation-review wording passes the real gate path."""
    title = "fix(benchmark): gate carriers"
    body = _body_with_carrier("### Summary\n\nBounded gate repair.", HEAD_SHA)
    implementation_review = _review_comment().replace(
        "Exact-head self-review", "Exact-head implementation review"
    )
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload(implementation_review)),
            _proc(stdout=_reviews_payload()),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "ok"
    assert result["carrier_source"] == "issue_comment"


def test_gate_withholds_merge_ready_while_exact_head_review_claim_is_active() -> None:
    """An in-flight trusted review claim cannot authorize the live head."""
    title = "fix(benchmark): gate carriers"
    body = _body_with_carrier("### Summary\n\nBounded gate repair.", HEAD_SHA)
    claim = f"review-claim: lane-a @ {HEAD_SHA} until 2099-01-01T00:00:00Z"
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload(_review_comment(), claim)),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "active exact-head review claim" in result["error"]
    assert "review worker is still running" in result["error"]


def test_gate_allows_current_carrier_after_exact_head_review_claim_release() -> None:
    """A released claim leaves the existing exact-head carrier gate authoritative."""
    title = "fix(benchmark): gate carriers"
    body = _body_with_carrier("### Summary\n\nBounded gate repair.", HEAD_SHA)
    claim = f"review-claim: lane-a @ {HEAD_SHA} until 2099-01-01T00:00:00Z"
    release = f"review-claim: released @ {HEAD_SHA}"
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload(_review_comment(), claim, release)),
            _proc(stdout=_reviews_payload()),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "ok"


def test_gate_withholds_merge_ready_for_active_review_endpoint_claim() -> None:
    """A claim in the canonical review endpoint is also an admission hold."""
    title = "fix(benchmark): gate carriers"
    body = _body_with_carrier("### Summary\n\nBounded gate repair.", HEAD_SHA)
    claim = f"review-claim: lane-a @ {HEAD_SHA} until 2099-01-01T00:00:00Z"
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload()),
            _proc(stdout=_reviews_payload(claim)),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "active exact-head review claim" in result["error"]


def test_gate_withholds_mixed_carriers_for_active_review_endpoint_claim() -> None:
    """A compatibility carrier cannot hide an active claim in the review endpoint."""
    title = "fix(benchmark): gate carriers"
    body = _body_with_carrier("### Summary\n\nBounded gate repair.", HEAD_SHA)
    claim = f"review-claim: lane-a @ {HEAD_SHA} until 2099-01-01T00:00:00Z"
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload(_review_comment())),
            _proc(stdout=_reviews_payload(claim)),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "active exact-head review claim" in result["error"]


def test_gate_rejects_review_endpoint_foreign_commit() -> None:
    """A review body cannot override a review object bound to an older commit."""
    title = "fix(benchmark): gate carriers"
    body = _body_with_carrier("### Summary\n\nBounded gate repair.", HEAD_SHA)
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload()),
            _proc(stdout=_reviews_payload(_review_comment(), commit_id=OTHER_SHA)),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "no exact-head review carrier" in result["error"]


def test_gate_rejects_bot_review_endpoint_carrier() -> None:
    """A bot review cannot satisfy the canonical review carrier path."""
    title = "fix(benchmark): gate carriers"
    body = _body_with_carrier("### Summary\n\nBounded gate repair.", HEAD_SHA)
    bot_review = json.loads(_reviews_payload(_review_comment()))
    bot_review[0]["user"] = {"login": "github-actions[bot]"}
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload()),
            _proc(stdout=json.dumps(bot_review)),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "no exact-head review carrier" in result["error"]


def test_gate_rejects_malformed_review_endpoint_payload() -> None:
    """Without a compatibility comment, malformed review JSON fails closed."""
    title = "fix(benchmark): gate carriers"
    body = _body_with_carrier("### Summary\n\nBounded gate repair.", HEAD_SHA)
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload()),
            _proc(stdout=json.dumps({"review": "not-a-list"})),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "payload was not a list" in result["error"]


def test_gate_rejects_stale_narrative_in_review_endpoint() -> None:
    """A stale review narrative blocks a review-endpoint carrier."""
    title = "fix(benchmark): gate carriers"
    body = _body_with_carrier("### Summary\n\nBounded gate repair.", HEAD_SHA)
    stale_review = "## Review note\n\nThe prior evidence is not current-base merge evidence."
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload()),
            _proc(stdout=_reviews_payload(stale_review)),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "not current-base merge evidence" in result["error"]


def test_gate_fails_closed_on_stale_body_carrier() -> None:
    """A body carrying exact-head carriers for an older head must withhold merge-ready."""
    title = "fix(benchmark): gate carriers"
    stale_body = _body_with_carrier("### Summary\n\nOriginal body.", OTHER_SHA)
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=stale_body)),
            _proc(stdout=_comments_payload(_review_comment())),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "do not match the live head" in result["error"]


def test_gate_fails_closed_on_not_ready_body_narrative() -> None:
    """A body declaring not-merge-ready must withhold merge-ready."""
    title = "fix(benchmark): gate carriers"
    body = "### Summary\n\nThis PR is not merge-ready yet."
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload(_review_comment())),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "not-ready sentinels" in result["error"]


def test_gate_fails_closed_without_review_carrier() -> None:
    """No human review for the live head means no merge-ready disposition."""
    title = "fix(benchmark): gate carriers"
    body = "### Summary\n\nBounded gate repair."
    stale_review = _review_comment().replace(HEAD_SHA, OTHER_SHA)
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload(stale_review)),
            _proc(stdout=_reviews_payload(stale_review)),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "no exact-head review carrier" in result["error"]


def test_gate_fails_closed_on_stale_carrier_comment() -> None:
    """A refresh note declaring not-current-base evidence blocks merge-ready."""
    title = "fix(benchmark): gate carriers"
    body = "### Summary\n\nBounded gate repair."
    refresh_note = (
        "## Current-base refresh note\n\n"
        "this earlier focused self-review is not current-base merge evidence"
    )
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload(_review_comment(), refresh_note)),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "not current-base merge evidence" in result["error"]


def test_gate_fails_closed_on_pending_domain_comment() -> None:
    """A pending-domain-review disposition comment must withhold merge-ready."""
    title = "fix(benchmark): gate carriers"
    body = "### Summary\n\nBounded gate repair."
    pending = "Independent domain-aware review: domain-aware approval remained pending."
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(title=title, body=body)),
            _proc(stdout=_comments_payload(_review_comment(), pending)),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "domain" in result["error"]


def test_gate_fails_closed_on_transport_error() -> None:
    """An unreadable live PR is an error, never permission to label."""
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        return_value=_proc(returncode=1, stderr="HTTP 403: forbidden"),
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "HTTP 403" in result["error"]


def test_gate_fails_closed_when_pr_head_changes_between_guard_reads() -> None:
    """The carrier read must re-check head/base to close the guard-to-write race."""
    body = _body_with_carrier("### Summary\n\nBounded gate repair.", HEAD_SHA)
    with patch(
        "scripts.dev.pr_carrier_gate._gh_api_get",
        side_effect=[
            _proc(stdout=_pr_payload(body=body, head=OTHER_SHA)),
            _proc(stdout=_comments_payload(_review_comment())),
        ],
    ):
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA,
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "head changed during carrier read" in result["error"]


def test_gate_rejects_malformed_shas() -> None:
    """Abbreviated live SHAs fail closed without any network read."""
    with patch("scripts.dev.pr_carrier_gate._gh_api_get") as mock_get:
        result = check_merge_ready_carriers(
            7610,
            live_head=HEAD_SHA[:12],
            live_base=BASE_SHA,
        )

    assert result["status"] == "error"
    assert "full 40-character" in result["error"]
    mock_get.assert_not_called()
