"""Tests for the packet-backed issue #3637 campaign launcher.

The tests mock benchmark execution. They verify the frozen launch packet is the
single source of truth for campaign arguments without running the campaign.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark" / "run_reactivity_replay_rank_study_issue_3637.py"
PACKET = REPO_ROOT / "configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml"

SPEC = importlib.util.spec_from_file_location("_issue_3637_runner", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_runner_uses_packet_plan_and_writes_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The launcher should pass frozen packet values to the existing campaign runner."""
    calls = []

    def fake_run_campaign(*args: object, **kwargs: object) -> dict[str, object]:
        set_path, seeds, planners, out_dir = args
        calls.append(
            {
                "set_path": set_path,
                "seeds": seeds,
                "planners": planners,
                "out_dir": out_dir,
                "horizon": kwargs["horizon"],
                "dt": kwargs["dt"],
                "workers": kwargs["workers"],
                "study_issue": kwargs["study_issue"],
                "evidence_tier": kwargs["evidence_tier"],
            }
        )
        return {"schema_version": "reactivity-ablation-campaign.v1", "issue": kwargs["study_issue"]}

    monkeypatch.setattr(runner, "run_campaign", fake_run_campaign)
    out_dir = tmp_path / "campaign"
    report_json = tmp_path / "report.json"

    exit_code = runner.main(
        [
            "--packet",
            str(PACKET),
            "--out-dir",
            str(out_dir),
            "--report-json",
            str(report_json),
            "--dt",
            "0.2",
            "--workers",
            "2",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["set_path"] == Path("configs/scenarios/sets/classic_crossing_subset.yaml")
    assert call["seeds"] == list(range(101, 121))
    assert call["planners"] == ("goal", "orca", "social_force")
    assert call["out_dir"] == out_dir
    assert call["horizon"] == 300
    assert call["dt"] == 0.2
    assert call["workers"] == 2
    assert call["study_issue"] == 3637
    assert call["evidence_tier"] == "seed_sufficient_candidate"

    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["preflight_status"] == "ready"
    assert report["launch_packet"] == str(PACKET)
    assert "analyze_reactivity_replay_rank_study_issue_3637.py" in report["post_run_analyzer"]
    integration = report["integration_report"]
    assert (
        integration["schema_version"] == "issue-3637-reactivity-replay-rank-integration-report.v1"
    )
    assert integration["status"] == "campaign_ran_analysis_required"
    assert integration["closure_decision"] == "keep_open_until_analysis_artifacts_exist"
    assert integration["delivered_contract"] == {
        "launch_packet": str(PACKET),
        "campaign_dir": str(out_dir),
        "campaign_report": str(report_json),
        "preflight_status": "ready",
        "post_run_analyzer": report["post_run_analyzer"],
    }
    assert integration["required_post_run_artifacts"] == [
        str(out_dir / "analysis" / "README.md"),
        str(out_dir / "analysis" / "analysis.json"),
        str(out_dir / "analysis" / "frozen_gate_input.json"),
        str(out_dir / "analysis" / "rank_bootstrap_summary.json"),
        str(out_dir / "analysis" / "per_planner_condition_metrics.csv"),
    ]
    assert integration["next_empirical_action"] == report["post_run_analyzer"]
    assert integration["forbidden_actions_confirmed"] == {
        "slurm_gpu_submission": False,
        "paper_dissertation_claim_edit": False,
    }


def test_runner_fails_before_campaign_when_packet_preflight_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A weakened launch packet should fail closed without starting the campaign."""
    packet = yaml.safe_load(PACKET.read_text(encoding="utf-8"))
    packet["planners"] = ["goal", "orca"]
    blocked_packet = tmp_path / "blocked.yaml"
    blocked_packet.write_text(yaml.safe_dump(packet), encoding="utf-8")

    def fail_run_campaign(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("campaign must not run for blocked packet")

    monkeypatch.setattr(runner, "run_campaign", fail_run_campaign)

    exit_code = runner.main(["--packet", str(blocked_packet), "--out-dir", str(tmp_path)])

    assert exit_code == 1
