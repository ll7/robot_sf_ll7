"""Tests for the verified-harvest #3216 analysis runner."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analysis/run_issue3216_rank_stability.py"
_SPEC = importlib.util.spec_from_file_location("issue5247_rank_stability", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules["issue5247_rank_stability"] = runner
_SPEC.loader.exec_module(runner)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _campaign(root: Path) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True)
    _write_json(
        reports / "campaign_summary.json",
        {
            "campaign": {
                "finished_at_utc": "2026-06-30T12:34:56+00:00",
                "snqi_contract_status": "fail",
            },
            "warnings": [
                "SNQI contract status=fail with snqi_contract.enforcement=warn; "
                "campaign marked with soft contract warning."
            ],
        },
    )
    _write_json(
        reports / "snqi_diagnostics.json",
        {
            "contract_status": "fail",
            "contract_enforcement": "warn",
            "rank_alignment_spearman": 0.2,
            "outcome_separation": 0.1,
            "dominant_component_mean_abs": 0.1,
            "thresholds": {
                "rank_alignment_fail": 0.3,
                "outcome_separation_fail": 0.0,
                "max_component_dominance_fail": 0.27,
            },
        },
    )
    (reports / "seed_episode_rows.csv").write_text(
        "planner_key,scenario_id,seed,success,collision,near_miss,snqi\n"
        "orca,crossing,1,1,0,0,0.8\n"
        "orca,crossing,2,1,0,0,0.9\n"
        "ppo,crossing,1,0,1,0,0.2\n"
        "ppo,crossing,2,0,1,0,0.3\n",
        encoding="utf-8",
    )
    (reports / "scenario_family_breakdown.csv").write_text(
        "scenario_family\ncrossing\n", encoding="utf-8"
    )
    (reports / "campaign_table.csv").write_text(
        "planner_key,execution_mode,status\norca,nominal,successful_evidence\n"
        "ppo,nominal,successful_evidence\n",
        encoding="utf-8",
    )


def test_verified_harvest_runner_writes_reproducible_provenance(tmp_path: Path) -> None:
    """A verified complete campaign delegates analysis and records exact SNQI failure."""

    campaign = tmp_path / "campaign"
    _campaign(campaign)
    harvest_log = tmp_path / "harvest.log"
    harvest_log.write_text("copy done\nVERIFIED_COMPLETE\n", encoding="utf-8")
    planner_config = tmp_path / "planners.yaml"
    planner_config.write_text("planners:\n  - key: orca\n  - key: ppo\n", encoding="utf-8")
    output = tmp_path / "analysis"
    command = [
        sys.executable,
        str(_SCRIPT),
        "--campaign-root",
        str(campaign),
        "--harvest-log",
        str(harvest_log),
        "--planner-config",
        str(planner_config),
        "--output-dir",
        str(output),
        "--bootstrap-samples",
        "12",
        "--rank-resamples",
        "8",
    ]

    first = subprocess.run(command, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    first_hashes = json.loads((output / "analysis_provenance.json").read_text())["output_sha256"]
    second = subprocess.run(command, text=True, capture_output=True, check=False)
    assert second.returncode == 0, second.stderr
    second_payload = json.loads((output / "analysis_provenance.json").read_text())
    assert second_payload["output_sha256"] == first_hashes
    failure = second_payload["snqi_contract_failure"]
    assert failure["campaign_finished_at_utc"] == "2026-06-30T12:34:56+00:00"
    assert failure["enforcement"] == "warn"
    assert failure["failed_checks"] == [
        {
            "check": "rank_alignment_spearman",
            "direction": "below",
            "fail_threshold": 0.3,
            "value": 0.2,
        }
    ]


def test_runner_rejects_harvest_without_completion_marker(tmp_path: Path) -> None:
    """The runner never analyzes a partial or unverified harvest."""

    campaign = tmp_path / "campaign"
    _campaign(campaign)
    harvest_log = tmp_path / "harvest.log"
    harvest_log.write_text("copy done\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--campaign-root",
            str(campaign),
            "--harvest-log",
            str(harvest_log),
            "--output-dir",
            str(tmp_path / "analysis"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "VERIFIED_COMPLETE" in result.stderr
    assert not (tmp_path / "analysis").exists()
