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
        ('{"steps": null}', "steps must be a list"),
        ('{"steps": ["not-an-object"]}', "steps must contain objects"),
        ('{"enabled_steps": "not-a-list"}', "enabled_steps must be a list"),
        ('{"enabled_steps": null}', "enabled_steps must be a list"),
        ('{"summary": []}', "summary must be an object"),
        ('{"summary": null}', "summary must be an object"),
        ('{"summary": {"seeds": "not-a-list"}}', "summary seeds must be a list"),
        ('{"summary": {"metrics": []}}', "summary metrics must be an object"),
        ('{"seeds": "not-a-list"}', "seeds must be a list"),
        ('{"seeds": null}', "seeds must be a list"),
        ('{"metrics": "not-an-object"}', "metrics must be an object"),
        ('{"metrics": null}', "metrics must be an object"),
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


def test_report_loader_rejects_malformed_manifest(tmp_path: Path, monkeypatch) -> None:
    """The report CLI loader shares the fail-closed tracker validation boundary."""
    from scripts.research.generate_report import load_tracker_manifest

    manifest_dir = tmp_path / "output" / "run-tracker" / "run-1"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text("[]", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        load_tracker_manifest("run-1")

    assert exc_info.value.code == 1
