"""Tests for conditional delivery triggers and proportional friction handling.

Issues #8944 and #8936: worktree/PR ceremony is triggered by mutation and custody risk, and
incidental friction gets a proportional response instead of an unconditional fix-or-file rule.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_MD = REPO_ROOT / "AGENTS.md"
LIFECYCLE = REPO_ROOT / "docs" / "dev" / "worktree_lifecycle.md"

DELIVERY_TRIGGERS = (
    "No mutation and no durable artifact",
    "Bounded mutation without collision or custody risk",
    "Concurrent, multi-step, or artifact-bearing mutation",
    "Requested delivery, review, or merge",
)

FRICTION_RESPONSES = (
    "Threatens correctness",
    "cheaper to fix than to work around",
    "Repeated and materially valuable",
    "Do not widen the task",
)


def test_delivery_trigger_table_covers_risk_situations() -> None:
    """Root guidance states the four mutation/delivery situations and the read-only exemption."""
    text = AGENTS_MD.read_text(encoding="utf-8")
    assert "## Mutation And Delivery Triggers" in text
    for phrase in DELIVERY_TRIGGERS:
        assert phrase in text, phrase
    assert "No branch, worktree, PR, or landing procedure" in text
    assert "explicit handoff" in text


def test_root_does_not_restate_worktree_procedure() -> None:
    """Bootstrap, teardown, and stash procedure lives once in the lifecycle owner."""
    agents = AGENTS_MD.read_text(encoding="utf-8")
    for fragment in (
        "uv venv .venv && uv sync --all-extras",
        "scripts/dev/create_worktree.sh",
        "safe_stash_pop.sh",
        "stash namespace",
    ):
        assert fragment not in agents, fragment
    lifecycle = LIFECYCLE.read_text(encoding="utf-8")
    assert "scripts/dev/create_worktree.sh" in lifecycle
    assert "safe_stash_pop.sh" in lifecycle
    assert "stash namespace" in lifecycle


def test_custody_behavior_tests_exist() -> None:
    """Dirty, concurrent, ignored-artifact, and abandoned-work custody stays covered by tests."""
    for relative in (
        "tests/dev/test_stale_worktree_reaper.py",
        "tests/dev/test_worktree_hygiene_snapshot.py",
        "tests/dev/test_worktree_lifecycle_lease_race.py",
        "tests/dev/test_review_worktree_guard.py",
    ):
        assert (REPO_ROOT / relative).is_file(), relative


def test_friction_response_is_proportional() -> None:
    """Friction handling is a decision rule, not an unconditional obligation."""
    text = AGENTS_MD.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "## Friction And Follow-Up" in text
    for phrase in FRICTION_RESPONSES:
        assert phrase in normalized, phrase
    assert "before moving on" not in normalized
    assert "runs twice" not in normalized
    for example in (
        "missing validation script",
        "repeated manual release step",
        "harmless one-off inconvenience",
        "unrelated code smell",
        "stale documentation",
    ):
        assert example in normalized, example
