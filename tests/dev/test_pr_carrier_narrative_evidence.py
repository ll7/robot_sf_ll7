"""Reviewer regressions for warnings that still supply required merge evidence."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from scripts.dev import gh_pr_label_rest as label
from scripts.dev import merge_queue_gate as queue
from scripts.dev import single_account_merge_receipt as receipt
from scripts.dev.check_pr_ci_status import _classify_required_checks
from scripts.dev.pr_metadata import metadata_digest

HEAD = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
BASE = "b1c2d3e4f5061728394a5b6c7d8e9f0011121314"
TITLE = "fix: bounded tooling change"
BODY = "Bounded tooling repair."
DIGEST = metadata_digest(TITLE, BODY)


def _accepted() -> str:
    """Build the reviewer's complete acceptance."""
    return (
        f"## Exact-head independent review\nExact head reviewed: `{HEAD}`\n"
        f"Exact base: `{BASE}`\ngate-verdict: accepted @ {HEAD}\n"
        f"base-policy: ordinary-cas @ {HEAD}\npr-metadata: reconciled @ {DIGEST}\n"
    )


def _record(body: str, timestamp: str, source: str) -> dict:
    """Represent either endpoint with its real publication and review bindings."""
    item = {"body": body, "user": {"login": "reviewer"}, "author_association": "OWNER"}
    item["submitted_at" if source == "reviews" else "created_at"] = timestamp
    if source == "reviews":
        item.update(state="COMMENTED", commit_id=HEAD)
    return item


def _response(payload: dict | list) -> subprocess.CompletedProcess:
    """Mock only the existing REST transport boundary."""
    return subprocess.CompletedProcess(["gh", "api"], 0, stdout=json.dumps(payload), stderr="")


def _identity() -> dict:
    """Return a clean, open PR at the exact guarded head and base."""
    return {
        "title": TITLE,
        "body": BODY,
        "state": "open",
        "merged_at": None,
        "head": {"sha": HEAD},
        "base": {"sha": BASE},
    }


@pytest.mark.parametrize("new_source", ["comments", "reviews"])
def test_required_approved_warning_review_is_not_superseded(new_source: str) -> None:
    """A complete acceptance cannot erase the sole eligible receipt review's warning."""
    old = _record(_accepted() + "\npending domain approval", "2026-10-19T08:00:00Z", "reviews")
    old.update(
        id=101,
        state="APPROVED",
        author_association="MEMBER",
        user={"login": "external-reviewer"},
    )
    new = _record(_accepted(), "2026-10-20T08:00:00Z", new_source)
    new.update(id=102, user={"login": "ll7"})
    comments = [new] if new_source == "comments" else []
    reviews = [old, new] if new_source == "reviews" else [old]
    ci = _classify_required_checks([{"name": "ci", "status": "COMPLETED", "conclusion": "SUCCESS"}])
    assert not ci["missing"] and not ci["not_green"], ci
    evidence = queue._to_receipt_review_evidence(
        reviews, head_sha=HEAD, expected_metadata_digest=DIGEST
    )
    selected = receipt.classify_implementation_review(
        {"head_sha": HEAD, "metadata_digest": DIGEST, "reviews": evidence}
    )
    assert selected["status"] == "accepted", selected
    assert selected["carrier"]["identity"] == "external-reviewer", selected
    replacement = receipt.classify_implementation_review(
        {
            "head_sha": HEAD,
            "metadata_digest": DIGEST,
            "reviews": queue._to_receipt_review_evidence(
                [new] if new_source == "reviews" else [],
                head_sha=HEAD,
                expected_metadata_digest=DIGEST,
            ),
        }
    )
    assert replacement["status"] != "accepted", replacement
    if new_source == "reviews":
        assert replacement["reason_codes"] == ["review_event_not_approved"], replacement
    with (
        patch("scripts.dev.pr_write_guard._gh_api_get", return_value=_response(_identity())),
        patch(
            "scripts.dev.pr_carrier_gate._gh_api_get",
            side_effect=[_response(_identity()), _response(comments), _response(reviews)],
        ),
        patch(
            "scripts.dev.gh_pr_label_rest._add_label_once",
            return_value={"status": "ok", "simulated_write": True},
        ) as write,
    ):
        result = label.add_label(
            10459, "merge-ready", target="pr", expected_head_sha=HEAD, expected_base_sha=BASE
        )
    assert result["status"] == "error", (
        "The warning review remains the sole accepted receipt review source, but label admission succeeds",
        result,
    )
    assert "pending domain approval" in result["error"]
    write.assert_not_called()
