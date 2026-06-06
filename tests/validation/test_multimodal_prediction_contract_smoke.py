"""Tests for the issue #2496 multimodal prediction contract smoke."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.validation import run_multimodal_prediction_contract_smoke as smoke


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL file into dictionaries."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_default_smoke_writes_required_rows(tmp_path: Path) -> None:
    """Default smoke emits native comparator rows and expected fail-closed rows."""
    output_root = tmp_path / "smoke"
    exit_code = smoke.main(["--output-root", str(output_root)])

    assert exit_code == 0
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(output_root / "rows.jsonl")

    assert summary["passed"] is True
    assert summary["benchmark_evidence"] is False
    assert summary["comparator_keys"] == [
        "reactive_no_prediction",
        "single_trajectory_prediction",
        "multimodal_equal_weight",
        "multimodal_confidence_weighted",
    ]
    assert summary["fail_closed_keys"] == [
        "missing_prediction_input",
        "degraded_prediction_input",
    ]

    required_fields = set(summary["required_trace_fields"])
    for row in rows:
        assert required_fields.issubset(row)

    native_rows = [row for row in rows if not row["expected_fail_closed"]]
    assert {row["readiness_status"] for row in native_rows} == {"native"}
    multimodal_row = next(row for row in rows if row["planner_key"] == "multimodal_equal_weight")
    assert multimodal_row["hypothesis_count_per_pedestrian"] == {"7": 3}
    assert multimodal_row["prediction_sample_count"] == 3
    assert sum(multimodal_row["hypothesis_confidence_vector"]["7"]) == 1.0

    fail_closed_rows = [row for row in rows if row["expected_fail_closed"]]
    assert {row["readiness_status"] for row in fail_closed_rows} == {"not_available", "failed"}
    assert all(row["fallback_or_degraded_reason"] for row in fail_closed_rows)


def test_smoke_fails_on_non_normalized_confidence(tmp_path: Path) -> None:
    """Confidence vectors must sum to one for native prediction rows."""
    config = yaml.safe_load(smoke.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["comparators"][2]["hypothesis_confidences"] = [0.5, 0.5, 0.5]
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output_root = tmp_path / "out"
    exit_code = smoke.main(["--config", str(config_path), "--output-root", str(output_root)])

    assert exit_code == 1
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] is False
    assert any("confidence sum" in failure for failure in summary["failures"])


def test_smoke_fails_on_sample_count_mismatch(tmp_path: Path) -> None:
    """Sample count must match the per-pedestrian hypothesis count."""
    config = yaml.safe_load(smoke.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    rows = smoke.build_rows(config)
    rows[2]["prediction_sample_count"] = 2

    failures = smoke.validate_rows(config, rows)

    assert any("prediction_sample_count=2" in failure for failure in failures)


def test_smoke_config_points_to_existing_parent_contract() -> None:
    """Smoke config remains tied to the proposal contract it executes."""
    repo_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(smoke.DEFAULT_CONFIG.read_text(encoding="utf-8"))

    assert config["issue"] == 2496
    assert config["parent_issue"] == 2476
    assert (repo_root / config["parent_contract"]).is_file()
    assert config["benchmark_evidence"] is False
