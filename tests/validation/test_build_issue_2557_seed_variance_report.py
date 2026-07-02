"""Tests for the issue #2557 seed-variance report builder."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validation import build_issue_2557_seed_variance_report as report_builder

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = (
    REPO_ROOT / "docs/context/evidence/issue_2557_replica_readiness_packet_2026-06-29/packet.json"
)


def test_report_admits_rescued_topups_but_summarizes_available_metrics() -> None:
    """The report keeps 17 admitted rows and recovered metrics."""
    report = report_builder.build_report(PACKET)

    assert report["counts"] == {
        "admitted_runs": 17,
        "clean_metric_runs": 14,
        "manifest_incomplete_admitted_runs": 3,
        "metrics_available_runs": 17,
        "metrics_pending_runs": 0,
    }
    assert report["maintainer_ruling"]["excluded_jobs"] == [12916]
    assert report["maintainer_ruling"]["no_reruns"] is True

    rescued = {
        row["job_id"]: row
        for row in report["provenance_rows"]
        if row["lineage"] == "manifest-incomplete"
    }
    assert set(rescued) == {12917, 12931, 12932}
    assert {row["seed"] for row in rescued.values()} == {502, 503, 504}
    assert all(row["admission_status"] == "admitted_metrics_available" for row in rescued.values())
    assert all("manifest serialization failed" in row["caveat"] for row in rescued.values())
    assert rescued[12932]["wandb_url"] == "https://wandb.ai/ll7/robot_sf/runs/klqes0h1"
    assert rescued[12917]["best_success_rate"] == pytest.approx(0.8857142857142857)
    assert rescued[12931]["best_eval_step"] == 7_864_320

    snqi = report["seed_variance_summary"]["snqi"]
    assert snqi["count"] == 17
    assert snqi["mean"] == pytest.approx(0.13274234383632)
    assert snqi["std"] == pytest.approx(0.10936920977202573)
    assert snqi["range"] == pytest.approx(0.3473526164311878)
    assert snqi["bootstrap_mean_ci95"] == pytest.approx([0.08134226702286855, 0.18196850974667875])


def test_artifact_writer_emits_provenance_and_checksums(tmp_path: Path) -> None:
    """Artifact writing includes reviewable provenance rows and valid SHA256 sums."""
    report = report_builder.build_report(PACKET)
    report_builder.write_artifact(report, tmp_path)

    report_path = tmp_path / "report.json"
    provenance_path = tmp_path / "per_run_provenance.csv"
    readme_path = tmp_path / "README.md"
    checksums_path = tmp_path / "SHA256SUMS"
    assert report_path.is_file()
    assert provenance_path.is_file()
    assert readme_path.is_file()
    assert checksums_path.is_file()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["claim_boundary"].startswith("Diagnostic seed-variance evidence only")

    with provenance_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 17
    assert {row["lineage"] for row in rows} == {"clean", "manifest-incomplete"}
    assert all(row["best_snqi"] != "na" for row in rows if row["lineage"] == "manifest-incomplete")

    checksum_lines = checksums_path.read_text(encoding="utf-8").splitlines()
    expected = {
        f"{hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}  {name}"
        for name in ("README.md", "per_run_provenance.csv", "report.json")
    }
    assert set(checksum_lines) == expected


def test_completed_job_rows_are_validated_before_sorting(tmp_path) -> None:
    """Malformed packet rows fail with a clear validation error."""
    packet = tmp_path / "packet.json"
    packet.write_text(
        json.dumps(
            {
                "schema_version": "test",
                "completed_jobs": ["not-an-object"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="entries must be objects"):
        report_builder.build_report(packet)

    packet.write_text(
        json.dumps(
            {
                "schema_version": "test",
                "completed_jobs": [{"seed": 1}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="entries must contain"):
        report_builder.build_report(packet)


def test_metric_summary_handles_no_available_metrics() -> None:
    """Unavailable rows produce explicit zero-count metric summaries."""
    summary = report_builder._metric_summary(
        [
            {
                "metric_status": "pending",
                "snqi": 0.0,
                "success_rate": 0.0,
                "collision_rate": 0.0,
            }
        ]
    )

    assert summary["snqi"]["count"] == 0
    assert summary["success_rate"]["bootstrap_samples"] == report_builder.BOOTSTRAP_SAMPLES
