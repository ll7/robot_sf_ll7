"""Tests for research reproducibility metadata parsing."""

from pathlib import Path

import pytest

from robot_sf.research.exceptions import ValidationError
from robot_sf.research.metadata import parse_tracker_manifest


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[]", "JSON object"),
        ('{"steps": "not-a-list"}', "steps must be a list"),
        ('{"steps": ["not-an-object"]}', "steps must contain objects"),
        ('{"summary": []}', "summary must be an object"),
        ('{"seeds": "not-a-list"}', "seeds must be a list"),
    ],
)
def test_parse_tracker_manifest_rejects_invalid_structural_payloads(
    tmp_path: Path, payload: str, message: str
) -> None:
    """Malformed manifest shapes fail through the documented validation boundary."""
    manifest_path = tmp_path / "tracker.json"
    manifest_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        parse_tracker_manifest(manifest_path)


def test_parse_tracker_manifest_rejects_invalid_utf8(tmp_path: Path) -> None:
    """Unreadable manifest bytes are reported as validation failures."""
    manifest_path = tmp_path / "tracker.json"
    manifest_path.write_bytes(b"{\xff")

    with pytest.raises(ValidationError, match="Failed to parse tracker manifest"):
        parse_tracker_manifest(manifest_path)


def test_parse_tracker_manifest_preserves_valid_summary_and_seeds(tmp_path: Path) -> None:
    """Valid mapping-shaped manifests retain their existing normalized output."""
    manifest_path = tmp_path / "tracker.json"
    manifest_path.write_text(
        '{"run_id": "run-1", "summary": {"seeds": [4, 5]}, "steps": []}',
        encoding="utf-8",
    )

    parsed = parse_tracker_manifest(manifest_path)

    assert parsed["run_id"] == "run-1"
    assert parsed["seeds"] == [4, 5]
    assert parsed["summary"] == {"seeds": [4, 5]}
