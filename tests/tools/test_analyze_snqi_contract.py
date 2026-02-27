"""Tests for SNQI contract analysis CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools import analyze_snqi_contract


def test_analyze_snqi_contract_writes_expected_outputs(tmp_path: Path, capsys) -> None:
    """CLI should emit JSON/MD/CSV artifacts with expected top-level keys."""
    campaign_root = tmp_path / "campaign"
    reports_dir = campaign_root / "reports"
    runs_dir = campaign_root / "runs" / "goal__differential_drive"
    reports_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    episodes_path = runs_dir / "episodes.jsonl"
    episodes_path.write_text(
        json.dumps(
            {
                "episode_id": "e-1",
                "scenario_id": "scn",
                "seed": 1,
                "metrics": {
                    "success": 1.0,
                    "time_to_goal_norm": 0.2,
                    "collisions": 0.0,
                    "near_misses": 0.0,
                    "comfort_exposure": 0.1,
                    "force_exceed_events": 0.0,
                    "jerk_mean": 0.1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary_payload = {
        "planner_rows": [
            {
                "planner_key": "goal",
                "kinematics": "differential_drive",
                "success_mean": "1.0",
                "collisions_mean": "0.0",
                "near_misses_mean": "0.0",
                "comfort_exposure_mean": "0.1",
            }
        ],
        "runs": [
            {
                "planner": {"key": "goal", "kinematics": "differential_drive"},
                "episodes_path": str(episodes_path),
            }
        ],
    }
    (reports_dir / "campaign_summary.json").write_text(
        json.dumps(summary_payload), encoding="utf-8"
    )

    exit_code = analyze_snqi_contract.main(
        [
            "--campaign-root",
            str(campaign_root),
            "--seed",
            "123",
            "--trials",
            "50",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert "snqi_diagnostics_json" in payload
    assert "snqi_diagnostics_md" in payload
    assert "snqi_sensitivity_csv" in payload
    assert "contract_status" in payload

    diagnostics_json = Path(payload["snqi_diagnostics_json"])
    diagnostics_md = Path(payload["snqi_diagnostics_md"])
    sensitivity_csv = Path(payload["snqi_sensitivity_csv"])
    assert diagnostics_json.exists()
    assert diagnostics_md.exists()
    assert sensitivity_csv.exists()

    diagnostics = json.loads(diagnostics_json.read_text(encoding="utf-8"))
    assert diagnostics["schema_version"] == "benchmark-snqi-diagnostics.v1"
    assert "configured_weights" in diagnostics
    assert "calibrated_weights" in diagnostics


def test_analyze_snqi_contract_rejects_inverted_thresholds(tmp_path: Path) -> None:
    """CLI should reject warn thresholds that are not above fail thresholds."""
    campaign_root = tmp_path / "campaign"
    reports_dir = campaign_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "campaign_summary.json").write_text(
        json.dumps({"planner_rows": [], "runs": []}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        analyze_snqi_contract.main(
            [
                "--campaign-root",
                str(campaign_root),
                "--rank-warn-threshold",
                "0.1",
                "--rank-fail-threshold",
                "0.2",
            ]
        )
