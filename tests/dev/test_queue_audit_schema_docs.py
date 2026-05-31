"""Checks for the queue_audit.v1 exhausted-queue output contract."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / ".agents" / "skills" / "schemas" / "queue_audit.v1.yaml"
SKILL_PATH = REPO_ROOT / ".agents" / "skills" / "goal-issue-implementation" / "SKILL.md"

CLASSIFICATIONS = {
    "parent_or_epic",
    "analysis_only",
    "blocked_external",
    "blocked_slurm",
    "covered_by_pr",
    "ready_local",
    "ambiguous",
    "too_broad",
    "blocked_other",
}

REQUIRED_EXAMPLE_CLASSIFICATIONS = {
    "parent_or_epic",
    "analysis_only",
    "blocked_external",
    "blocked_slurm",
    "covered_by_pr",
    "ready_local",
}

REQUIRED_FIELDS = {
    "issue:",
    "classification:",
    "implementable_now:",
    "recommended_action:",
}


def test_queue_audit_schema_names_required_fields_and_classifications() -> None:
    """The schema contract should expose the required row fields and allowed classes."""
    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    schema = yaml.safe_load(schema_text)

    assert schema["queue_audit"]["schema"] == "queue_audit.v1"
    assert REQUIRED_FIELDS <= set(schema_text.split())
    for classification in CLASSIFICATIONS:
        assert classification in schema_text


def test_goal_issue_implementation_documents_required_queue_audit_examples() -> None:
    """The exhausted-queue skill docs should show the classes required by issue #1719."""
    skill_text = SKILL_PATH.read_text(encoding="utf-8")

    assert "queue_audit.v1" in skill_text
    for field in REQUIRED_FIELDS:
        assert field.rstrip(":").replace("_", " ") in skill_text or field in skill_text
    for classification in REQUIRED_EXAMPLE_CLASSIFICATIONS:
        assert f"classification: {classification}" in skill_text


def test_queue_audit_routing_blocks_nonimplementation_classes() -> None:
    """Blocked and non-implementation classes must not be treated as ready work."""
    skill_text = SKILL_PATH.read_text(encoding="utf-8")

    assert "must not be reported as ready implementation work" in skill_text
    assert "Blocked-external and blocked-SLURM" in skill_text
    assert "issues must route to `mark_blocked`, `clarify`, or `skip`" in skill_text
