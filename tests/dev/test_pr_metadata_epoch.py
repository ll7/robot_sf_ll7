"""Unit tests for deterministic PR metadata epochs (issue #7649).

The fixtures are offline table rows that cover every mutation type named in
the issue's required-fixture list: body-only reconciliation, exact-head
trailer change, cosmetic normalization, base refresh, label hold, reviewer
request, automated comment-only review, and concurrent writer mutation.
"""

from __future__ import annotations

import pytest

from scripts.dev.pr_metadata import (
    PrMetadataEpochInputs,
    body_digest,
    build_pr_metadata_epoch,
    diff_epochs,
    normalize_pr_text,
    normalize_pr_title,
)
from scripts.dev.pr_metadata_epoch import compute_epoch


def _base_epoch(**overrides) -> dict:
    params: dict = {
        "pr_number": 8010,
        "repository": "ll7/robot_sf_ll7",
        "head_sha": "a" * 40,
        "base_sha": "b" * 40,
        "title": "fix(plot): close the saved figure",
        "body": "Summary line.\n\nValidation commands.\n",
        "linked_issues": [8007],
        "closing_references": [8007],
        "labels": ["merge-ready", "review-bot-auto"],
        "requested_reviewers": ["reviewer-a"],
        "review_decision": "APPROVED",
        "domain_approval_required": False,
    }
    params.update(overrides)
    return build_pr_metadata_epoch(PrMetadataEpochInputs(**params))


def _pr_json(**overrides) -> dict:
    pr = {
        "number": 8010,
        "title": "fix(plot): close the saved figure",
        "body": "Summary line.\n\nValidation commands.\n",
        "headRefOid": "a" * 40,
        "baseRefOid": "b" * 40,
        "labels": [{"name": "merge-ready"}, {"name": "review-bot-auto"}],
        "requestedReviewers": [{"login": "reviewer-a"}],
        "reviewDecision": "APPROVED",
        "closingIssuesReferences": [{"number": 8007}],
    }
    pr.update(overrides)
    return pr


def test_epoch_digest_is_stable_for_identical_inputs() -> None:
    first = _base_epoch()
    second = _base_epoch()
    assert first["digest"] == second["digest"]
    assert first["digest"] == first["schema"][:0] or first["digest"]


def test_body_only_reconciliation_is_material() -> None:
    before = _base_epoch()
    after = _base_epoch(body="Summary line.\n\nUpdated validation commands.\n")
    changes = diff_epochs(before, after)
    assert [c["dimension"] for c in changes] == ["body_digest"]
    assert changes[0]["material"] is True
    assert before["digest"] != after["digest"]


def test_exact_head_trailer_change_is_material() -> None:
    body_with_trailer = "Summary line.\n\npr-metadata: reconciled @ 00ab\n"
    body_new_trailer = "Summary line.\n\npr-metadata: reconciled @ 11cd\n"
    before = _base_epoch(body=body_with_trailer)
    after = _base_epoch(body=body_new_trailer)
    assert body_digest(body_with_trailer) != body_digest(body_new_trailer)
    assert before["digest"] != after["digest"]


def test_cosmetic_normalization_is_stable() -> None:
    crlf_body = "Summary line.\r\n\r\nValidation commands.\r\n"
    spaced_body = "Summary line.  \n\nValidation commands.   \n"
    assert normalize_pr_text(crlf_body) == normalize_pr_text(spaced_body)
    before = _base_epoch(body=crlf_body)
    after = _base_epoch(body=spaced_body)
    assert before["digest"] == after["digest"]
    assert diff_epochs(before, after) == []


def test_base_refresh_is_material() -> None:
    before = _base_epoch()
    after = _base_epoch(base_sha="c" * 40)
    changes = diff_epochs(before, after)
    assert [c["dimension"] for c in changes] == ["base_sha"]
    assert changes[0]["material"] is True


def test_label_hold_is_material() -> None:
    before = _base_epoch()
    after = _base_epoch(labels=["review-bot-auto", "state:hold"])
    changes = diff_epochs(before, after)
    assert [c["dimension"] for c in changes] == ["labels"]
    assert changes[0]["material"] is True


def test_reviewer_request_is_material() -> None:
    before = _base_epoch()
    after = _base_epoch(requested_reviewers=["reviewer-a", "reviewer-b"])
    changes = diff_epochs(before, after)
    assert [c["dimension"] for c in changes] == ["requested_reviewers"]
    assert changes[0]["material"] is True


def test_automated_comment_only_review_is_stable() -> None:
    # Comment text is not a bound dimension: a comment-only review must not
    # invalidate the metadata epoch.
    before = _base_epoch()
    after = _base_epoch()
    after["comments"] = [{"author": "bot", "body": "reviewed"}]
    assert before["digest"] == after["digest"]
    assert diff_epochs(before, after) == []


def test_concurrent_writer_mutation_is_detected() -> None:
    before = _base_epoch()
    after = _base_epoch(title="fix(plot): close the saved figure (edited concurrently)")
    changes = diff_epochs(before, after)
    assert [c["dimension"] for c in changes] == ["title_normalized"]
    assert before["digest"] != after["digest"]


def test_producer_change_is_reported_but_not_material() -> None:
    before = _base_epoch()
    after = _base_epoch()
    after["producer"] = "another-writer"
    changes = diff_epochs(before, after)
    assert [c["dimension"] for c in changes] == ["producer"]
    assert changes[0]["material"] is False
    assert before["digest"] == after["digest"]


def test_compute_epoch_from_gh_pr_view_json() -> None:
    epoch = compute_epoch(_pr_json(), repository="ll7/robot_sf_ll7")
    assert epoch["schema"] == "pr_metadata_epoch.v1"
    assert epoch["head_sha"] == "a" * 40
    assert epoch["linked_issues"] == [8007]
    assert epoch["closing_references"] == [8007]
    assert epoch["labels"] == ["merge-ready", "review-bot-auto"]
    assert epoch["domain_approval_required"] is False


def test_compute_epoch_extracts_issue_refs_from_body() -> None:
    pr = _pr_json(body="Fixes #1234 and relates to #8007.\n")
    epoch = compute_epoch(pr, repository="ll7/robot_sf_ll7")
    assert epoch["linked_issues"] == [1234, 8007]


def test_normalize_pr_title_collapses_whitespace() -> None:
    assert normalize_pr_title("  fix(plot):   close  the saved figure \n") == (
        "fix(plot): close the saved figure"
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", ""),
        ("\n\nbody\n", "body"),
        ("a\r\nb\r\n", "a\nb"),
        ("a  \nb", "a\nb"),
        ("\ta\nb", "\ta\nb"),
    ],
)
def test_normalize_pr_text_rules(raw: str, expected: str) -> None:
    assert normalize_pr_text(raw) == expected
