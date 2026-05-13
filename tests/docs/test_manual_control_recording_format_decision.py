"""Tests for the manual-control recording format decision note."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTE = ROOT / "docs" / "context" / "issue_1163_manual_control_recording_format.md"
DOCS_README = ROOT / "docs" / "README.md"
CONTEXT_README = ROOT / "docs" / "context" / "README.md"


def test_manual_control_recording_format_decision_is_documented() -> None:
    """The #1163 no-change decision should preserve thresholds and provenance requirements."""
    text = NOTE.read_text(encoding="utf-8")
    normalized_text = " ".join(text.split())

    assert (
        "Keep append-only JSONL as the canonical manual-control recording format" in normalized_text
    )
    assert "Do not add a general compact recording format yet" in normalized_text
    assert "estimated horizon-500 size is at or below 10 MiB per attempt" in text
    assert "sustained write throughput is at least 1,000 records/second" in text
    assert "read throughput is at least 1,000 records/second" in text
    assert "source record schema and source record index" in text
    assert "manual_control_bc_v1" in text


def test_manual_control_recording_format_decision_is_linked() -> None:
    """Normal docs entry points should expose the #1163 decision note."""
    note_name = "issue_1163_manual_control_recording_format.md"

    assert note_name in DOCS_README.read_text(encoding="utf-8")
    assert note_name in CONTEXT_README.read_text(encoding="utf-8")
