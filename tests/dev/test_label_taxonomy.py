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


def test_taxonomy_covers_every_classifier_qualifier() -> None:
    """The prose qualifier list must not drift behind STATE_QUALIFIER_LABELS."""
    from scripts.dev import issue_state_taxonomy

    text = TAXONOMY.read_text(encoding="utf-8")

    missing = [
        label
        for label in sorted(issue_state_taxonomy.STATE_QUALIFIER_LABELS)
        if f"`{label}`" not in text
    ]
    assert missing == [], f"undocumented qualifiers: {missing}"


def test_state_label_inventory_check_fails_closed_on_unclassified_label() -> None:
    """An unknown state:* label must be reported, never silently tolerated."""
    from scripts.dev import check_state_label_inventory as check

    result = check.classify_labels(["state:ready", "state:typo-invented"])

    assert result["ok"] is False
    assert result["unclassified"] == ["state:typo-invented"]
    assert result["deliberately_unclassified"] == []


def test_state_label_inventory_check_allows_only_named_exclusions() -> None:
    """state:done is a named exclusion; a near-miss typo must still fail."""
    from scripts.dev import check_state_label_inventory as check
    from scripts.dev import issue_state_taxonomy

    assert issue_state_taxonomy.DELIBERATELY_UNCLASSIFIED_STATE_LABELS == {"state:done"}

    ok = check.classify_labels(["state:ready", "state:done"])
    assert ok["ok"] is True
    assert ok["deliberately_unclassified"] == ["state:done"]

    near_miss = check.classify_labels(["state:done ", "state:done-ish"])
    assert near_miss["ok"] is False
    assert "state:done-ish" in near_miss["unclassified"]


def test_state_label_inventory_check_accepts_current_qualifiers() -> None:
    from scripts.dev import check_state_label_inventory as check
    from scripts.dev import issue_state_taxonomy

    labels = sorted(issue_state_taxonomy.KNOWN_STATE_LABELS) + ["state:done"]
    result = check.classify_labels(labels)

    assert result["ok"] is True
    assert result["unclassified"] == []


def test_state_label_inventory_check_reads_a_labels_file(tmp_path) -> None:
    """The offline input keeps the unit test hermetic (no network)."""
    from scripts.dev import check_state_label_inventory as check

    path = tmp_path / "labels.txt"
    path.write_text("state:ready\n# a comment\nstate:reviewing\n", encoding="utf-8")

    result = check.classify_labels(check._read_labels_file(str(path)))
    assert result["ok"] is True
