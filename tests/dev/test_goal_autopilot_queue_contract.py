"""Tests for the goal-autopilot zero-work authority contract."""

from pathlib import Path


def test_goal_autopilot_requires_complete_authoritative_zero() -> None:
    """Autopilot must not translate a partial numeric zero into genuine zero work."""
    skill = Path(".agents/skills/goal-autopilot/SKILL.md").read_text(encoding="utf-8")

    assert "Never infer `genuine_zero_work` from `claimable_count` alone." in skill
    assert "`candidate_scope: state:ready`" in skill
    assert "`zero_work_authoritative: true`" in skill
    assert "`queue_status: incomplete`" in skill
    assert "`queue_status: unavailable`" in skill
