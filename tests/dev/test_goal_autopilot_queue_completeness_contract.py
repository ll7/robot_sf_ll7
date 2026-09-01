"""Contract tests for goal-autopilot queue-completeness authority."""

from pathlib import Path


def test_goal_autopilot_requires_page_one_fully_evaluated_queue() -> None:
    """A zero-work stop must require page-one, fully evaluated queue evidence."""
    skill = Path(".agents/skills/goal-autopilot/SKILL.md").read_text(encoding="utf-8")

    assert "`queue_completeness: complete` is authoritative only for a page-one" in skill
    assert "unavailable claim read is never complete queue evidence" in skill


def test_goal_autopilot_zero_work_is_parent_only_and_head_bound() -> None:
    """Lane exhaustion must not be promoted to a global terminal result."""
    skill = Path(".agents/skills/goal-autopilot/SKILL.md").read_text(encoding="utf-8")

    assert "`genuine_zero_work` is a controller-only terminal state" in skill
    assert "No delegated" in skill
    assert "goal_autopilot_zero_work_proof.v1" in skill
    assert "readiness gating" in skill
    assert (
        "A zero" in skill
        and "claimable issue count proves only implementation-lane exhaustion" in skill
    )
