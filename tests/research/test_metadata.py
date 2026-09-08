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


def test_parse_tracker_manifest_rejects_malformed_earlier_jsonl_shape(tmp_path: Path) -> None:
    """A valid final JSONL record cannot hide an invalid earlier record."""
    manifest_path = tmp_path / "tracker.jsonl"
    manifest_path.write_text(
        '{"summary": {"seeds": "not-a-list"}}\n{"summary": {"seeds": [4]}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="summary seeds must be a list"):
        parse_tracker_manifest(manifest_path)


def test_report_loader_rejects_oversized_json_integer(tmp_path: Path, monkeypatch) -> None:
    """Oversized JSON integers fail through the report loader boundary."""
    from robot_sf.research.metadata import load_tracker_manifest_payload
    from scripts.research.generate_report import load_tracker_manifest

    manifest_dir = tmp_path / "output" / "run-tracker" / "run-1"
    manifest_dir.mkdir(parents=True)
    oversized_integer = "9" * 5000
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(f'{{"seed": {oversized_integer}}}', encoding="utf-8")

    with pytest.raises(ValidationError, match="Failed to parse tracker manifest"):
        load_tracker_manifest_payload(manifest_path)

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        load_tracker_manifest("run-1")
    assert exc_info.value.code == 1


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


def test_report_summary_numeric_coercion_fails_closed() -> None:
    """Report summary seeds and timesteps reject mixed or unrepresentable values."""
    from scripts.research.generate_report import (
        _coerce_summary_float_list,
        _extract_report_inputs,
        extract_metric_records_from_manifest,
    )

    with pytest.raises(ValidationError, match="summary.seeds must contain an integer"):
        extract_metric_records_from_manifest({"summary": {"seeds": [1, None]}})

    assert _coerce_summary_float_list({"baseline_timesteps": [12.5]}, "baseline_timesteps") == [
        12.5
    ]
    with pytest.raises(ValidationError, match="summary.baseline_timesteps must contain"):
        _coerce_summary_float_list({"baseline_timesteps": [int("9" * 1000)]}, "baseline_timesteps")

    with pytest.raises(SystemExit) as exc_info:
        _extract_report_inputs({"summary": {"seeds": [1, None]}})
    assert exc_info.value.code == 1

    summary, records, seeds, baseline, pretrained = _extract_report_inputs(
        {"summary": {"seeds": [2], "baseline_timesteps": [10.0]}}
    )
    assert summary["seeds"] == [2]
    assert records == []
    assert seeds == [2]
    assert baseline == [10.0]
    assert pretrained == []


def test_report_summary_numeric_coercion_preserves_zero_and_ignores_null() -> None:
    """Provenance summary lists retain zero and omit unavailable null entries."""
    from scripts.research.generate_report import _coerce_summary_float_list

    assert _coerce_summary_float_list({"baseline_timesteps": [0, None]}, "baseline_timesteps") == [
        0.0
    ]
    assert _coerce_summary_float_list({"pretrained_timesteps": [0.0]}, "pretrained_timesteps") == [
        0.0
    ]


def test_report_summary_numeric_coercion_rejects_negative_timesteps() -> None:
    """Report provenance summaries reject negative convergence values."""
    from scripts.research.generate_report import _coerce_summary_float_list

    with pytest.raises(ValidationError, match="must be non-negative"):
        _coerce_summary_float_list({"baseline_timesteps": [-1]}, "baseline_timesteps")


def test_comparison_fallback_preserves_zero_timesteps() -> None:
    """Comparison-summary fallback retains scalar zero convergence values."""
    from scripts.research.generate_report import _apply_comparison_fallback

    _, _, baseline, pretrained = _apply_comparison_fallback(
        {"comparison": {"timesteps_to_convergence": {"baseline": 0, "pretrained": 0}}},
        [],
        [],
        [],
        [],
    )

    assert baseline == [0.0]
    assert pretrained == [0.0]


def test_tracker_float_coercion_rejects_nonfinite_values() -> None:
    """Shared metric coercion rejects NaN before report aggregation."""
    from robot_sf.research.tracker_manifest import coerce_tracker_float

    with pytest.raises(ValidationError, match="finite number"):
        coerce_tracker_float(float("nan"), "metrics.success_rate")
    with pytest.raises(ValidationError, match="finite number"):
        coerce_tracker_float(True, "metrics.success_rate")


def test_tracker_numeric_coercion_rejects_lossy_types() -> None:
    """Shared coercion rejects fractional seeds, booleans, and incomplete comparisons."""
    from robot_sf.research.tracker_manifest import coerce_tracker_float, coerce_tracker_int
    from scripts.research.generate_report import _comparison_metric_records

    with pytest.raises(ValidationError, match="must contain an integer"):
        coerce_tracker_int(42.5, "summary.seeds")
    with pytest.raises(ValidationError, match="must contain an integer"):
        coerce_tracker_int(True, "summary.seeds")
    with pytest.raises(ValidationError, match="must contain an integer"):
        coerce_tracker_int("not-an-integer", "summary.seeds")

    with pytest.raises(ValidationError, match="finite number"):
        coerce_tracker_float("not-a-number", "metrics.success_rate")

    records, seeds = _comparison_metric_records(
        {"success_rate": {"baseline": 0.5, "pretrained": 0.7}}
    )
    assert len(records) == 2
    assert seeds == [0, 1]
    with pytest.raises(ValidationError, match="require baseline and pretrained"):
        _comparison_metric_records({"success_rate": {"baseline": 0.5}})
