"""Contract checks for label-driven multi-machine factory skills."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / ".agents" / "skills"


def _skill(name: str) -> str:
    return (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")


def test_issue_factory_uses_labels_and_authorized_isolated_lanes() -> None:
    """Projects must not control admission, and parallel lanes must remain isolated."""
    text = _skill("goal-issue-implementation")

    assert "GitHub labels are authoritative workflow state" in text
    assert "Never use Project membership, columns, or numeric fields as the queue source" in text
    assert "Missing Project access or tooling is non-blocking" in text
    assert "orchestrator-authorized bounded batch" in text
    assert "different issue claim, branch, linked worktree" in text
    assert "Workers may not expand concurrency" in text


def test_issue_preparation_and_partial_pr_contract_are_autonomous() -> None:
    """Preparation may classify work, while partial delivery preserves the parent contract."""
    text = _skill("goal-issue-implementation")

    for marker in (
        "mark a clear, bounded issue `state:ready`",
        "split a broad issue",
        "mark an issue covered",
        "apply `state:blocked`",
        "add `decision-required`",
        "close an exact duplicate",
    ):
        assert marker in text
    assert "must use `Refs #<parent>` rather than `Closes`/`Fixes`" in text
    assert "create and link a successor issue" in text


def test_bounded_run_authorizes_factory_github_mutations() -> None:
    """A started run may mutate its workflow surfaces without confirmation loops."""
    issue = _skill("goal-issue-implementation")
    review = _skill("goal-pr-review")
    merger = _skill("gh-pr-merger")

    for marker in ("add or remove labels", "comments", "branches", "open or update PRs", "threads"):
        assert marker in issue
    for marker in ("apply labels", "post comments", "push safe fixes", "resolve threads"):
        assert marker in review
    assert "execute the guarded squash merge without another per-PR confirmation" in merger


def test_route_artifacts_stay_outside_worktrees_and_durable_deletion_needs_owner() -> None:
    """Control-plane reports are never commits, and scientific deletion stays owner-gated."""
    for name in ("goal-issue-implementation", "goal-pr-review", "gh-pr-merger"):
        text = _skill(name)
        prose = " ".join(text.split())
        assert "outside" in text and "worktree" in text
        assert "RESULT.md" in text and "REVIEW.json" in text
        assert "never" in text and "commit" in text
        assert "Owner approval is required before deleting durable scientific artifacts" in prose


def test_single_account_review_waiver_never_waives_exact_head_evidence() -> None:
    """A single account can self-review, but every readiness decision remains SHA-bound."""
    review = _skill("goal-pr-review")
    merger = _skill("gh-pr-merger")

    assert "Single-Account Internal-Review Waiver" in review
    assert "post an exact-head review-evidence comment" in review
    assert "Exact-head evidence remains mandatory" in review
    assert "any head change invalidates" in review
    assert "single-account internal-review waiver" in merger
    assert "single-account waiver never waives exact-head evidence" in merger
