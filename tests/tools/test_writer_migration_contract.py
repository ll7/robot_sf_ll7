"""Focused writer-migration contract tests for issue #6511 (slice 2 of #6497).

Pins the shared ``robot_sf.evidence.writers`` contract as adopted by the
migrated non-benchmark evidence writers under ``scripts/tools/``,
``scripts/dev/``, ``scripts/training/``, and ``hooks/``:

- JSON writes are marker-additive (``review_marker``) and preserve every
  original schema field;
- text writes prepend the canonical marker and preserve the body;
- CSV writes emit a comment marker line that a comment-aware reader skips
  without losing rows;
- repeated writes for fixed inputs are byte-deterministic.
"""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from robot_sf.evidence.writers import (
    review_marker,
    review_marker_json,
    write_csv,
    write_json,
    write_text,
)
from scripts.tools import generate_mixed_scenario_matrix as gen

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Shared-writer contract
# ---------------------------------------------------------------------------


def test_write_json_marker_is_additive_and_schema_preserving(tmp_path: Path) -> None:
    """Review marker is added without altering any original schema field."""
    payload = {
        "schema_version": "mixed_scenario_matrix.v1",
        "issue": 2766,
        "scenario_count": 3,
        "status_counts": {"available": 1, "diagnostic_only": 2},
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_json(first, payload)
    write_json(second, payload)

    parsed = json.loads(first.read_text(encoding="utf-8"))
    assert parsed["review_marker"] == review_marker_json()
    for key, value in payload.items():
        assert parsed[key] == value

    assert json.loads(second.read_text(encoding="utf-8")) == parsed


def test_write_json_is_byte_deterministic(tmp_path: Path) -> None:
    """Identical inputs produce byte-identical JSON output."""
    payload = {"schema_version": "test.v1", "rows": [{"a": 1, "b": [2, 3]}]}
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    write_json(first, payload)
    write_json(second, payload)
    assert first.read_bytes() == second.read_bytes()


def test_write_text_prepends_marker_and_preserves_body(tmp_path: Path) -> None:
    """Canonical marker is prepended and the generated body is retained."""
    body = "# Issue #2766 Mixed Scenario Matrix\n\nGenerated evidence."
    path = tmp_path / "report.md"
    write_text(path, body, issue_ref="robot_sf#2766")
    text = path.read_text(encoding="utf-8")
    assert text.startswith(review_marker("robot_sf#2766") + "\n")
    assert body in text


def test_write_csv_marker_is_skippable_by_comment_aware_reader(tmp_path: Path) -> None:
    """CSV marker line is a comment a reader can skip without losing rows."""
    rows = [{"scenario": "a", "value": "1"}, {"scenario": "b", "value": "2"}]
    path = tmp_path / "table.csv"
    write_csv(path, rows)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# AI-GENERATED NEEDS-REVIEW"

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        assert list(reader) == rows


# ---------------------------------------------------------------------------
# Migrated generator end-to-end schema preservation
# ---------------------------------------------------------------------------


def _sources(tmp_path: Path) -> gen.SourceInputs:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "dissertation_evidence_ledger.v2",
                "rows": [
                    {
                        "area": "prediction",
                        "artifact_status": "current",
                        "evidence_tier": "diagnostic",
                        "allowed_wording": "test",
                        "caveat": "test",
                    }
                ],
                "stale_artifact_summary": [],
            }
        ),
        encoding="utf-8",
    )
    signal = tmp_path / "signal.json"
    signal.write_text(json.dumps({"eligible_rows": [], "excluded_rows": []}), encoding="utf-8")
    obs = tmp_path / "obs.json"
    obs.write_text(json.dumps({"summary": {}, "conditions": []}), encoding="utf-8")
    cv = tmp_path / "cv.json"
    cv.write_text(json.dumps({"results_by_trace": []}), encoding="utf-8")
    gap = tmp_path / "gap.json"
    gap.write_text("{}", encoding="utf-8")
    neg = tmp_path / "neg"
    neg.mkdir()
    return gen.SourceInputs(
        ledger=ledger,
        signal_summary=signal,
        obs_noise_summary=obs,
        cv_forecast=cv,
        gap_report=gap,
        negative_result_dir=neg,
    )


def test_generate_matrix_emits_marker_and_preserves_schema(tmp_path: Path) -> None:
    """Regenerated matrix JSON carries the marker and its original schema keys."""
    out_dir = tmp_path / "output"
    md_path, json_path = gen.generate_matrix(sources=_sources(tmp_path), output_dir=out_dir)

    assert md_path.read_text(encoding="utf-8").startswith(review_marker("robot_sf#2766") + "\n")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["review_marker"] == review_marker_json()
    assert payload["schema_version"] == "mixed_scenario_matrix.v1"
    assert payload["issue"] == 2766
    assert payload["scenario_count"] == len(gen.SCENARIO_SLICES)
    assert "status_counts" in payload
