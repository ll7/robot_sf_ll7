"""Durable-evidence contract for issue #5579's bounded MPC tuning sensitivity."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_CONFIG = ROOT / "configs/analysis/issue_5579_mpc_tuning_sensitivity.yaml"
PACKET_CONFIG = ROOT / "configs/analysis/issue_5579_mpc_tuning_sensitivity_v2.yaml"
EVIDENCE = ROOT / "docs/context/evidence/issue_5579_mpc_tuning_budget_sensitivity_2026-07-14"
REPORT_JSON = EVIDENCE / "sensitivity_report.json"
REPORT_MD = EVIDENCE / "sensitivity_report.md"
CANDIDATE_CSV = EVIDENCE / "sensitivity_candidate_rows.csv"


def test_compact_report_preserves_the_preregistered_claim_boundary() -> None:
    """The promoted report remains complete, diagnostic-only, and bound to the tracked packet."""
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))

    assert report["schema_version"] == "issue_5579_mpc_tuning_sensitivity_report.v1"
    assert report["issue"] == 5579
    assert report["status"] in {"complete_diagnostic", "blocked"}
    assert report["evidence_tier"] == "diagnostic-only"
    assert report["benchmark_evidence"] is False
    assert "benchmark ranking" in report["claim_boundary"]
    assert report["config_path"] == str(HISTORICAL_CONFIG.relative_to(ROOT))
    assert report["config_sha256"] == _sha256(HISTORICAL_CONFIG)
    packet = yaml.safe_load(PACKET_CONFIG.read_text(encoding="utf-8"))
    assert packet["schema_version"] == "issue_5579_mpc_tuning_sensitivity.v2"
    assert packet["study_id"] == "issue_5579_mpc_tuning_budget_sensitivity_v2"
    assert report["study_id"] != packet["study_id"]
    assert report["config_path"] != str(PACKET_CONFIG.relative_to(ROOT))
    assert report["config_sha256"] != _sha256(PACKET_CONFIG)
    assert packet["artifacts"]["durable_evidence_root"] != str(EVIDENCE.relative_to(ROOT))
    assert report["candidate_count"] == 20
    assert report["target_arm_count"] == 2
    assert report["total_episode_rows"] == 396
    assert report["eligible_episode_rows"] == 295
    assert report["excluded_episode_rows"] == 101
    assert report["read"]["decision"] == "blocked"
    assert report["read"]["detail"] == (
        "Complete native/adapter rows are required before the pre-registered read."
    )
    assert not Path(report["raw_artifact_root"]).is_absolute()

    exclusion_reasons = {
        reason for row in report["candidate_rows"] for reason in row["exclusion_reasons"]
    }
    assert exclusion_reasons == {"fallback", "solver_failure"}


def test_candidate_table_matches_the_compact_json_report() -> None:
    """The reviewable CSV represents all 40 target points and four unchanged incumbents."""
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    with CANDIDATE_CSV.open(encoding="utf-8", newline="") as handle:
        assert handle.readline().strip() == "# AI-GENERATED NEEDS-REVIEW"
        csv_rows = list(csv.DictReader(handle))

    report_rows = report["candidate_rows"]
    assert len(csv_rows) == len(report_rows) == 44
    assert sum(row["target"] is True for row in report_rows) == 40
    assert sum(row["target"] is False for row in report_rows) == 4
    assert all(row["episodes"] == 9 for row in report_rows)
    assert [(row["arm_key"], row["candidate_id"]) for row in csv_rows] == [
        (row["arm_key"], row["candidate_id"]) for row in report_rows
    ]


def test_markdown_leads_with_status_and_claim_boundary_before_results() -> None:
    """Readers see the diagnostic boundary before best-found configurations or interpretation."""
    markdown = REPORT_MD.read_text(encoding="utf-8")
    status_position = markdown.index("- Status:")
    boundary_position = markdown.index("- Claim boundary:")
    results_position = markdown.index("## Best-found target configurations")

    assert status_position < results_position
    assert boundary_position < results_position
    assert (
        "Fallback, degraded, failed, and unavailable rows are never treated as success evidence."
        in markdown
    )
    assert "does not change benchmark metrics, roster status, or paper-facing claims" in markdown
    assert "295 eligible" in markdown
    assert "101 excluded" in markdown


def _sha256(path: Path) -> str:
    """Return the byte-level SHA-256 binding for a tracked config file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
