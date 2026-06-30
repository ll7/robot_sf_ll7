"""Tests for the issue #3798 post-job S20/S30 evidence-gap packet extractor."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from typing import TYPE_CHECKING

from scripts.validation import extract_s20_s30_evidence_gap_packet as extractor

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_complete_artifact_root(root: Path) -> Path:
    _write_json(
        root / "campaign_manifest.json",
        {
            "campaign_id": "issue1554_s20_h500_l40s_mem180_20260628",
            "name": "paper_experiment_matrix_v1_scenario_horizons_h500_s20",
            "started_at_utc": "2026-06-28T18:25:45Z",
            "finished_at_utc": "2026-06-28T20:03:01Z",
            "scenario_matrix": "configs/scenarios/classic_interactions_francis2023.yaml",
            "scenario_matrix_hash": "8609d0192098",
            "git": {
                "commit": "38f921fe374bc954ccc8932bfb055fc021c5b528",
                "branch": "slurm-issue-1554-s20-h500-l40s-mem180-20260628",
            },
            "seed_policy": {
                "seed_set": "paper_eval_s20",
                "resolved_seeds": [111, 112],
            },
        },
    )
    _write_json(root / "run_meta.json", {"campaign_id": "issue1554_s20_h500_l40s_mem180_20260628"})
    _write_csv(
        root / "reports/campaign_table.csv",
        [
            {
                "planner_key": "goal",
                "planner_group": "core",
                "readiness_tier": "baseline-ready",
                "status": "ok",
                "episodes": 2,
                "success_mean": 0.5,
                "collisions_mean": 0.0,
                "snqi_mean": "'-0.1",
            },
            {
                "planner_key": "orca",
                "planner_group": "core",
                "readiness_tier": "baseline-ready",
                "status": "ok",
                "episodes": 2,
                "success_mean": 1.0,
                "collisions_mean": 0.0,
                "snqi_mean": "'0.2",
            },
        ],
    )
    _write_csv(
        root / "reports/seed_episode_rows.csv",
        [
            {"episode_id": "goal-111", "planner_key": "goal", "seed": 111},
            {"episode_id": "orca-112", "planner_key": "orca", "seed": 112},
        ],
    )
    _write_json(root / "reports/statistical_sufficiency.json", {"status": "diagnostic"})
    (root / "reports/campaign_table_core.md").write_text(
        "| planner_key |\n|---|\n", encoding="utf-8"
    )
    (root / "reports/campaign_table_experimental.md").write_text(
        "| planner_key |\n|---|\n", encoding="utf-8"
    )
    (root / "reports/snqi_diagnostics.md").write_text("diagnostic only\n", encoding="utf-8")
    return root


def test_present_job_artifact_metadata_builds_diagnostic_packet(tmp_path: Path) -> None:
    """Complete retrieved metadata prints a diagnostic packet, not a claim upgrade."""

    root = _write_complete_artifact_root(tmp_path / "13175")
    packet = extractor.build_packet(root, job_id="13175")

    assert packet["status"] == "diagnostic_only"
    assert packet["job_id"] == "13175"
    assert packet["campaign"]["seed_set"] == "paper_eval_s20"
    assert packet["coverage_snapshot"]["seed_count"] == 2
    assert packet["coverage_snapshot"]["planners"] == ["goal", "orca"]
    assert packet["retrieved_artifacts"]["missing_required_metadata_files"] == []
    assert packet["next_slurm_go_no_go"]["claim_promotion"] == "no_go"
    assert packet["next_slurm_go_no_go"]["s30_submission_from_issue_3798"] == "not_authorized_here"
    assert packet["next_slurm_go_no_go"]["s30_escalation_status"] == "not_ready_incomplete_s20"
    assert "no paper/dissertation claim edits" in packet["claim_boundary"]
    assert any(
        item["path"] == "campaign_manifest.json" for item in packet["promotable_review_files"]
    )


def test_missing_job_artifact_metadata_fails_closed(tmp_path: Path) -> None:
    """Missing retrieved metadata remains blocked and names the missing files."""

    root = tmp_path / "missing-13175"
    root.mkdir()
    packet = extractor.build_packet(root, job_id="13175")

    assert packet["status"] == "blocked_missing_retrieved_metadata"
    assert (
        "campaign_manifest.json" in packet["retrieved_artifacts"]["missing_required_metadata_files"]
    )
    assert packet["coverage_snapshot"]["episode_rows"] == 0
    assert (
        packet["next_slurm_go_no_go"]["s20_archive_readiness"]
        == "blocked_missing_retrieved_metadata"
    )
    assert (
        packet["next_slurm_go_no_go"]["s30_escalation_status"] == "not_ready_missing_s20_metadata"
    )


def test_dry_run_command_prints_markdown_packet(tmp_path: Path, capsys) -> None:
    """The dry-run CLI prints the packet without writing outputs or submitting work."""

    root = _write_complete_artifact_root(tmp_path / "13175")

    exit_code = extractor.main(["--artifact-root", str(root), "--job-id", "13175", "--markdown"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "# Job 13175 S20/S30 Evidence-Gap Packet" in captured.out
    assert "diagnostic-only evidence-gap packet" in captured.out
    assert (
        "uv run python scripts/validation/check_s20_s30_archive_readiness.py --json" in captured.out
    )


def test_tracked_packet_fixture_renders_without_raw_artifacts(capsys) -> None:
    """Tracked packet fixture keeps public proof independent of ignored output."""

    exit_code = extractor.main(
        [
            "--packet-fixture",
            "docs/context/evidence/issue_3798_post_13175_s20_s30_evidence_gap_packet.json",
            "--markdown",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Status: `diagnostic_only`" in captured.out
    assert "claim_promotion: `no_go`" in captured.out
    assert "s30_escalation_status: `defer_until_claim_owner_authorizes_escalation`" in captured.out
    assert "File count: 56" in captured.out
    assert "requiring ignored output" in captured.out
    assert "no paper/dissertation claim edits" in captured.out


def test_tracked_packet_dry_run_command_executes_without_submission() -> None:
    """Published dry-run command prints the tracked packet without raw output artifacts."""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validation/extract_s20_s30_evidence_gap_packet.py",
            "--packet-fixture",
            "docs/context/evidence/issue_3798_post_13175_s20_s30_evidence_gap_packet.json",
            "--markdown",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Status: `diagnostic_only`" in completed.stdout
    assert "claim_promotion: `no_go`" in completed.stdout
    assert "without submitting Slurm" in completed.stdout
    assert "no paper/dissertation claim edits" in completed.stdout
    assert "Traceback" not in completed.stderr


def test_tracked_packet_rejects_claim_promotion_go(tmp_path: Path) -> None:
    """Tracked packet fixture must fail closed if it authorizes claim promotion."""

    packet = extractor.load_packet_fixture(extractor.DEFAULT_PACKET_FIXTURE)
    packet["next_slurm_go_no_go"]["claim_promotion"] = "go"
    fixture = tmp_path / "claim-go.json"
    fixture.write_text(json.dumps(packet), encoding="utf-8")

    try:
        extractor.load_packet_fixture(fixture)
    except ValueError as exc:
        assert "claim_promotion must remain no_go" in str(exc)
    else:
        raise AssertionError("claim promotion drift should fail closed")


def test_tracked_packet_rejects_issue_3798_s30_submit_authorization(tmp_path: Path) -> None:
    """Issue #3798 packet must not silently become a Slurm submission authorization."""

    packet = extractor.load_packet_fixture(extractor.DEFAULT_PACKET_FIXTURE)
    packet["next_slurm_go_no_go"]["s30_submission_from_issue_3798"] = "authorized"
    fixture = tmp_path / "s30-authorized.json"
    fixture.write_text(json.dumps(packet), encoding="utf-8")

    try:
        extractor.load_packet_fixture(fixture)
    except ValueError as exc:
        assert "must not authorize S30 submission" in str(exc)
    else:
        raise AssertionError("S30 authorization drift should fail closed")


def test_tracked_packet_rejects_stale_planner_count(tmp_path: Path) -> None:
    """Tracked packet fixture must fail closed if summary counts drift."""

    packet = extractor.load_packet_fixture(extractor.DEFAULT_PACKET_FIXTURE)
    packet["coverage_snapshot"]["planner_count"] += 1
    fixture = tmp_path / "stale-planner-count.json"
    fixture.write_text(json.dumps(packet), encoding="utf-8")

    try:
        extractor.load_packet_fixture(fixture)
    except ValueError as exc:
        assert "planner_count must match listed planners" in str(exc)
    else:
        raise AssertionError("stale planner count should fail closed")


def test_tracked_packet_rejects_claim_promoting_review_file(tmp_path: Path) -> None:
    """Promotable review files stay diagnostic and checksum-backed."""

    packet = extractor.load_packet_fixture(extractor.DEFAULT_PACKET_FIXTURE)
    packet["promotable_review_files"][0]["promotion_scope"] = "paper claim upgrade"
    fixture = tmp_path / "claim-promoting-review-file.json"
    fixture.write_text(json.dumps(packet), encoding="utf-8")

    try:
        extractor.load_packet_fixture(fixture)
    except ValueError as exc:
        assert "promotable review file promoted a claim" in str(exc)
    else:
        raise AssertionError("claim-promoting review file should fail closed")
