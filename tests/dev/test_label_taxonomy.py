"""Keep the shared decision-cockpit label reference tied to local authorities."""

from __future__ import annotations

from pathlib import Path

from scripts.dev import issue_audit_core

REPO_ROOT = Path(__file__).parents[2]
TAXONOMY = REPO_ROOT / "docs/ai/label-taxonomy.md"


def test_decision_cockpit_taxonomy_exists_and_links_authorities() -> None:
    """The shared skill's repository-local taxonomy must remain discoverable."""
    text = TAXONOMY.read_text(encoding="utf-8")

    assert "../../CONTRIBUTING.md#issue-state-labels-and-dispatch" in text
    assert "../../scripts/dev/issue_audit_core.py" in text
    for heading in (
        "## Decision flow",
        "## Lifecycle and origin",
        "## Execution state",
        "## Resource",
        "## Type",
        "## Evidence",
    ):
        assert heading in text


def test_taxonomy_covers_classifier_execution_states() -> None:
    """The prose reference must not drift behind the fail-closed classifier."""
    text = TAXONOMY.read_text(encoding="utf-8")

    for label in issue_audit_core.EXECUTION_STATE_LABELS:
        assert f"`{label}`" in text
