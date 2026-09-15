"""Conditional readiness must never skip CI or the guarded merge-ready write."""

from unittest.mock import patch

from scripts.dev.promote_merge_if_ci_green import promote

HEAD = "a" * 40
BASE = "b" * 40


def _labels(*names: str) -> dict[str, object]:
    return {"status": "ok", "labels": list(names)}


def _ci(overall: str, head: str = HEAD) -> dict[str, object]:
    return {"status": "ok", "head_sha": head, "checks": {"overall": overall}}


def test_pending_ci_keeps_conditional_label_without_merge_ready_write() -> None:
    with (
        patch(
            "scripts.dev.promote_merge_if_ci_green.get_label_names",
            return_value=_labels("merge-if-ci-green"),
        ),
        patch("scripts.dev.promote_merge_if_ci_green.read_ci", return_value=_ci("pending")),
        patch("scripts.dev.promote_merge_if_ci_green.add_label") as add,
    ):
        result = promote(42, repo="owner/repo", head_sha=HEAD, base_sha=BASE)
    assert result["status"] == "waiting"
    add.assert_not_called()


def test_green_ci_promotes_same_head_through_guarded_label_helper() -> None:
    with (
        patch(
            "scripts.dev.promote_merge_if_ci_green.get_label_names",
            side_effect=[
                _labels("merge-if-ci-green"),
                _labels("merge-if-ci-green"),
                _labels("merge-ready"),
            ],
        ),
        patch("scripts.dev.promote_merge_if_ci_green.read_ci", return_value=_ci("success")),
        patch(
            "scripts.dev.promote_merge_if_ci_green.add_label", return_value={"status": "ok"}
        ) as add,
        patch(
            "scripts.dev.promote_merge_if_ci_green.remove_label", return_value={"status": "ok"}
        ) as remove,
    ):
        result = promote(42, repo="owner/repo", head_sha=HEAD, base_sha=BASE)
    assert result["status"] == "promoted"
    assert add.call_args.kwargs == {
        "repo": "owner/repo",
        "target": "pr",
        "expected_head_sha": HEAD,
        "expected_base_sha": BASE,
    }
    assert remove.call_args.kwargs["expected_head_sha"] == HEAD


def test_moved_head_or_removed_conditional_label_prevents_promotion() -> None:
    with (
        patch(
            "scripts.dev.promote_merge_if_ci_green.get_label_names",
            return_value=_labels("merge-if-ci-green"),
        ),
        patch(
            "scripts.dev.promote_merge_if_ci_green.read_ci", return_value=_ci("success", "c" * 40)
        ),
        patch("scripts.dev.promote_merge_if_ci_green.add_label") as add,
    ):
        result = promote(42, repo="owner/repo", head_sha=HEAD, base_sha=BASE)
    assert result["reason"] == "head_moved"
    add.assert_not_called()

    with (
        patch(
            "scripts.dev.promote_merge_if_ci_green.get_label_names",
            side_effect=[
                _labels("merge-if-ci-green"),
                _labels(),
            ],
        ),
        patch("scripts.dev.promote_merge_if_ci_green.read_ci", return_value=_ci("success")),
        patch("scripts.dev.promote_merge_if_ci_green.add_label") as add,
    ):
        result = promote(42, repo="owner/repo", head_sha=HEAD, base_sha=BASE)
    assert result["reason"] == "conditional_label_removed"
    add.assert_not_called()


def test_newer_review_hold_prevents_promotion_after_green_ci() -> None:
    """A maintainer's newer hold must outweigh the older accepted review."""
    with (
        patch(
            "scripts.dev.promote_merge_if_ci_green.get_label_names",
            side_effect=[
                _labels("merge-if-ci-green"),
                _labels("merge-if-ci-green", "needs-review"),
            ],
        ),
        patch("scripts.dev.promote_merge_if_ci_green.read_ci", return_value=_ci("success")),
        patch("scripts.dev.promote_merge_if_ci_green.add_label") as add,
    ):
        result = promote(42, repo="owner/repo", head_sha=HEAD, base_sha=BASE)
    assert result == {
        "status": "blocked",
        "reason": "newer_hold_label",
        "labels": ["needs-review"],
        "number": 42,
    }
    add.assert_not_called()
