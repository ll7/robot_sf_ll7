"""Manifest-bound offline campaign audit integration, using dev-seed row bytes."""

import hashlib
import json
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.release_row_anomalies import (
    _release_0_0_8_roster,
    analyze_release_rows,
)
from scripts.analysis.scan_release_audit import main, scan_campaign


def test_scanner_keeps_manifest_denominator_when_rows_or_arms_are_missing(tmp_path: Path):
    root = tmp_path / "campaign"
    (root / "reports").mkdir(parents=True)
    (root / "preflight").mkdir()
    (root / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "planners": [{"key": "goal"}, {"key": "hybrid_v4"}],
                "seed_policy": {"resolved_seeds": [1001, 1002]},
            }
        )
    )
    (root / "preflight/preview_scenarios.json").write_text(
        json.dumps(
            {
                "truncated": False,
                "scenarios": [{"name": "smoke", "seeds": [1001, 1002]}],
            }
        )
    )
    runs = []
    for arm in ["goal", "hybrid_v4"]:
        path = root / "runs" / arm / "episodes.jsonl"
        path.parent.mkdir(parents=True)
        rows = [
            {
                "scenario_id": "smoke",
                "seed": seed,
                "integrity": {"effective_view": {"observation_ped_count": 1}},
                "steps": 30,
                "outcome": {
                    "route_complete": True,
                    "collision_event": False,
                    "timeout_event": False,
                },
                "event_ledger": {"exact_events": {"invalid_run": False}},
                "metrics": {"success": 1},
            }
            for seed in [1001, 1002]
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        # Keep a stale absolute producing-host reference to exercise relocation.
        runs.append(
            {"planner": {"key": arm}, "episodes_path": f"/producer/runs/{arm}/episodes.jsonl"}
        )
    summary = root / "reports/campaign_summary.json"
    summary.write_text(json.dumps({"runs": runs}))
    report = scan_campaign(root)
    assert report["coverage"]["complete_cells"] == 2
    assert report["coverage"]["incomplete_cells"] == 0
    assert report["preflight_accounting"]["expected_cells"] == 2
    summary.write_text(json.dumps({"runs": runs[:1]}))
    missing_arm = scan_campaign(root)
    assert missing_arm["coverage"]["incomplete_cells"] == 2
    assert missing_arm["gate"]["blocked"]
    summary.write_text(json.dumps({"runs": runs}))
    for arm in ["goal", "hybrid_v4"]:
        path = root / "runs" / arm / "episodes.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n")
    missing_cell = scan_campaign(root)
    assert missing_cell["preflight_accounting"]["status"] == "incomplete"
    assert missing_cell["gate"]["blocked"]


@pytest.fixture
def release_campaign(tmp_path):
    root = tmp_path / "release"
    (root / "reports").mkdir(parents=True)
    (root / "preflight").mkdir()
    roster = sorted(_release_0_0_8_roster())
    (root / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "planners": [{"key": arm} for arm in roster],
                "seed_policy": {"resolved_seeds": [1001]},
            }
        )
    )
    (root / "preflight/preview_scenarios.json").write_text(
        json.dumps({"truncated": False, "scenarios": [{"name": "smoke", "seeds": [1001]}]})
    )
    runs = []
    for arm in roster:
        path = root / "runs" / (arm + "__differential_drive") / "episodes.jsonl"
        path.parent.mkdir(parents=True)
        row = {
            "episode_id": "smoke-1001",
            "scenario_id": "smoke",
            "seed": 1001,
            "algo": arm,
            "config_hash": "episode-config",
            "git_hash": "source-commit",
            "algorithm_metadata": {"config_hash": "planner-config"},
            "integrity": {"effective_view": {"observation_ped_count": 1}},
            "steps": 30,
            "outcome": {"route_complete": True, "collision_event": False, "timeout_event": False},
            "event_ledger": {"exact_events": {"invalid_run": False}},
            "metrics": {"success": 1},
        }
        path.write_text(json.dumps(row) + "\n")
        runs.append({"planner": {"key": arm}, "episodes_path": str(path)})
    (root / "reports/campaign_summary.json").write_text(json.dumps({"runs": runs}))
    return root


@pytest.mark.parametrize("retain_run", [False, True])
def test_release_gate_blocks_manifest_missing_risk_dwa(release_campaign, tmp_path, retain_run):
    manifest_path = release_campaign / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["planners"] = [arm for arm in manifest["planners"] if arm["key"] != "risk_dwa"]
    manifest_path.write_text(json.dumps(manifest))
    summary_path = release_campaign / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text())
    if not retain_run:
        summary["runs"] = [run for run in summary["runs"] if run["planner"]["key"] != "risk_dwa"]
    summary_path.write_text(json.dumps(summary))
    if not retain_run:
        diagnostic = scan_campaign(release_campaign)
        assert diagnostic["gate"]["blocked"] is False
    output = tmp_path / "blocked"
    assert (
        main(
            [
                "--campaign-root",
                str(release_campaign),
                "--output-dir",
                str(output),
                "--release-gate",
            ]
        )
        == 1
    )
    report_path = output / "release-row-gate.json"
    if not report_path.exists():
        report_path = output / "release_audit.json"
    report = json.loads(report_path.read_text())
    assert report["gate"]["blocked"]
    assert "release_roster_source_mismatch" in report["gate"]["reasons"]
    assert "risk_dwa" in report["release_roster_binding"]["expected_planner_ids"]


def test_manifest_order_keeps_registry_and_findings_stable(release_campaign):
    first = scan_campaign(release_campaign)
    manifest_path = release_campaign / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["planners"].reverse()
    manifest_path.write_text(json.dumps(manifest))
    second = scan_campaign(release_campaign)
    assert first["detector_registry_digest"] == second["detector_registry_digest"]
    assert first["findings"] == second["findings"]
    assert second["source"]["cohort_source"] == "manifest_planner_ids"


def test_scanner_crash_returns_two(tmp_path):
    assert (
        main(
            [
                "--bundle",
                str(tmp_path / "missing.tar.gz"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "out" / "receipt.json").exists()


def test_bundle_cli_preserves_queue_receipt(release_campaign, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifacts = []
    for path in release_campaign.glob("runs/*/episodes.jsonl"):
        target = bundle / "payload" / path.relative_to(release_campaign)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        artifacts.append(
            {
                "path": str(target.relative_to(bundle)),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "size_bytes": target.stat().st_size,
            }
        )
    (bundle / "publication_manifest.json").write_text(
        json.dumps({"schema_version": "benchmark-publication-bundle.v2", "files": artifacts})
    )
    output = tmp_path / "bundle-audit"
    assert main(["--bundle", str(bundle), "--output-dir", str(output)]) == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["rows"] == 14
    assert (output / "queue.json").is_file()
    assert (output / "release-row-gate.json").is_file()


def test_manifest_roster_order_keeps_nonempty_report_bytes_stable(release_campaign):
    rows = []
    for path in release_campaign.glob("runs/*/episodes.jsonl"):
        row = json.loads(path.read_text())
        row["_release_arm"] = path.parent.name.removesuffix("__differential_drive")
        row["outcome"] = {"route_complete": False, "collision_event": False, "timeout_event": True}
        row["metrics"]["success"] = 0
        rows.append(row)
    roster = sorted(_release_0_0_8_roster())
    preflight = [{"scenario_id": "smoke", "seed": 1001, "invalid_run": False}]
    first = analyze_release_rows(rows, source={"planner_ids": roster}, preflight=preflight)
    second = analyze_release_rows(rows, source={"planner_ids": roster[::-1]}, preflight=preflight)
    assert first["findings"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
