"""Manifest-bound offline campaign audit integration, using dev-seed row bytes."""

import json
from pathlib import Path

from scripts.analysis.scan_release_audit import scan_campaign


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
