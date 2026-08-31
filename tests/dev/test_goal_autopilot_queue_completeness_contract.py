"""Contract tests for goal-autopilot queue-completeness authority."""

from pathlib import Path


def test_goal_autopilot_requires_page_one_fully_evaluated_queue() -> None:
    """A zero-work stop must require page-one, fully evaluated queue evidence."""
    skill = Path(".agents/skills/goal-autopilot/SKILL.md").read_text(encoding="utf-8")

    assert "`queue_completeness: complete` is authoritative only for a page-one" in skill
    assert "unavailable claim read is never complete queue evidence" in skill
