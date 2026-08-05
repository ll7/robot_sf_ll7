"""Tests for retained h600 mechanism/exposure sidecar builder."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from scripts.validation.build_issue_4242_h600_mechanism_exposure_sidecars import build_sidecars

if TYPE_CHECKING:
    from pathlib import Path

REVIEW_MARKER_LINE = "# AI-GENERATED NEEDS-REVIEW"


def _read_marked_csv(path: Path) -> list[dict[str, str]]:
    """Read a shared-writer CSV, asserting and skipping its review-marker first line."""
    with path.open(encoding="utf-8", newline="") as handle:
        assert handle.readline().strip() == REVIEW_MARKER_LINE
        return list(csv.DictReader(handle))


def _write_seed_rows(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_id",
                "scenario_id",
                "planner_key",
                "seed",
                "repeat_index",
                "success",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "episode_id": "ep-1",
                "scenario_id": "classic_static_deadlock",
                "planner_key": "goal",
                "seed": "111",
                "repeat_index": "0",
                "success": "1.0",
            }
        )


def test_retained_rows_without_traces_write_not_derivable_sidecars(tmp_path: Path) -> None:
    """Retained h600 episode rows without traces become explicit blocked sidecar rows."""

    reports_dir = tmp_path / "output" / "run" / "reports"
    _write_seed_rows(reports_dir / "seed_episode_rows.csv")
    manifest_path = tmp_path / "evidence" / "source_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "issue_4195_h600_aggregation.v1.source_manifest",
                "runs": [
                    {
                        "job_id": "13268",
                        "run_label": "confirm",
                        "reports_dir": str(reports_dir.relative_to(tmp_path)),
                        "campaign": {"campaign_id": "fixture-campaign"},
                    }
                ],
                "generated_outputs": ["source_manifest.json"],
            }
        ),
        encoding="utf-8",
    )

    summary = build_sidecars(
        source_manifest=manifest_path.relative_to(tmp_path),
        output_dir=manifest_path.parent,
        generated_at="2026-07-03T00:00:00Z",
        exposure_radius_m=2.0,
        low_exposure_success_threshold=0.1,
        repo_root=tmp_path,
    )

    assert summary["retained_episode_rows"] == 1
    assert summary["mechanism_status_counts"] == {"not_derivable_missing_trace": 1}
    assert summary["interaction_exposure_status_counts"] == {"not_derivable_missing_trace": 1}

    mechanism_rows = _read_marked_csv(manifest_path.parent / "h600_mechanism_labels_sidecar.csv")
    exposure_rows = _read_marked_csv(manifest_path.parent / "h600_interaction_exposure_sidecar.csv")
    assert mechanism_rows[0]["mechanism_label"] == "unknown"
    assert mechanism_rows[0]["mechanism_backfill_status"] == "not_derivable_missing_trace"
    assert exposure_rows[0]["interaction_exposure_share"] == ""
    assert exposure_rows[0]["interaction_exposure_backfill_status"] == "not_derivable_missing_trace"
    refreshed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "h600_mechanism_labels_sidecar.csv" in refreshed_manifest["generated_outputs"]
    assert refreshed_manifest["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    backfill_manifest = json.loads(
        (manifest_path.parent / "h600_mechanism_exposure_backfill_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert backfill_manifest["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    report_first_line = (
        (manifest_path.parent / "h600_mechanism_exposure_backfill_report.md")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert report_first_line == "<!-- AI-GENERATED (robot_sf#4242) - NEEDS-REVIEW -->"
    checksum_lines = (manifest_path.parent / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert checksum_lines[0] == REVIEW_MARKER_LINE
    assert checksum_lines[1:]


def test_retained_trace_json_computes_exposure_sidecar(tmp_path: Path) -> None:
    """Trace JSON columns are enough to compute exposure without aggregate imputation."""

    reports_dir = tmp_path / "output" / "run" / "reports"
    reports_dir.mkdir(parents=True)
    with (reports_dir / "seed_episode_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_id",
                "scenario_id",
                "planner_key",
                "seed",
                "repeat_index",
                "success",
                "robot_positions_json",
                "pedestrian_positions_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "episode_id": "ep-1",
                "scenario_id": "classic_static_deadlock",
                "planner_key": "goal",
                "seed": "111",
                "repeat_index": "0",
                "success": "true",
                "robot_positions_json": json.dumps([[0, 0], [2, 0]]),
                "pedestrian_positions_json": json.dumps([[[0.5, 0]], [[10, 0]]]),
            }
        )
    manifest_path = tmp_path / "evidence" / "source_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "job_id": "13268",
                        "run_label": "confirm",
                        "reports_dir": str(reports_dir.relative_to(tmp_path)),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = build_sidecars(
        source_manifest=manifest_path.relative_to(tmp_path),
        output_dir=manifest_path.parent,
        generated_at="2026-07-03T00:00:00Z",
        exposure_radius_m=1.0,
        low_exposure_success_threshold=0.75,
        repo_root=tmp_path,
    )

    assert summary["interaction_exposure_status_counts"] == {"computed_from_retained_trace": 1}
    exposure_rows = _read_marked_csv(manifest_path.parent / "h600_interaction_exposure_sidecar.csv")
    assert exposure_rows[0]["interaction_exposure_status"] == "computed"
    assert (
        exposure_rows[0]["interaction_exposure_backfill_status"] == "computed_from_retained_trace"
    )


def test_sidecar_bundle_is_byte_deterministic(tmp_path: Path) -> None:
    """Re-running the builder with a pinned generated_at yields identical artifact bytes."""

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root in (first_root, second_root):
        reports_dir = root / "output" / "run" / "reports"
        _write_seed_rows(reports_dir / "seed_episode_rows.csv")
        manifest_path = root / "evidence" / "source_manifest.json"
        manifest_path.parent.mkdir()
        manifest_path.write_text(
            json.dumps(
                {
                    "runs": [
                        {
                            "job_id": "13268",
                            "run_label": "confirm",
                            "reports_dir": str(reports_dir.relative_to(root)),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        build_sidecars(
            source_manifest=manifest_path.relative_to(root),
            output_dir=manifest_path.parent,
            generated_at="2026-07-03T00:00:00Z",
            exposure_radius_m=2.0,
            low_exposure_success_threshold=0.1,
            repo_root=root,
        )

    first_names = {path.name for path in (first_root / "evidence").iterdir()}
    assert first_names == {path.name for path in (second_root / "evidence").iterdir()}
    for name in sorted(first_names):
        assert (first_root / "evidence" / name).read_bytes() == (
            second_root / "evidence" / name
        ).read_bytes(), name
