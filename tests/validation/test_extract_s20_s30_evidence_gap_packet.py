"""Tests for the issue #3798 post-job S20/S30 evidence-gap packet extractor."""

from __future__ import annotations

import csv
import json
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
