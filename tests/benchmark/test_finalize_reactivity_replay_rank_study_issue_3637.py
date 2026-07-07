"""Tests issue #3637 post-run finalizer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "benchmark" / "finalize_reactivity_replay_rank_study_issue_3637.py"
)

SPEC = importlib.util.spec_from_file_location("_issue_3637_finalizer", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
FINALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZER)

PLANNERS = ("goal", "orca", "social_force")
CONDITIONS = ("reactive", "replay")
SCENARIOS = ("classic_crossing_low", "classic_crossing_high")


def _write_packet(tmp_path: Path, *, seeds: list[int]) -> Path:
    scenario_set = tmp_path / "scenario_set.yaml"
    scenario_set.write_text(
        yaml.safe_dump({"select_scenarios": list(SCENARIOS)}),
        encoding="utf-8",
    )
    packet = {
        "schema_version": "reactivity_replay_rank_study_preflight.v1",
        "issue": 3637,
        "planners": list(PLANNERS),
        "scenario_set": str(scenario_set),
        "horizon": 300,
        "seeds": seeds,
        "replay": {
            "is_trajectory_playback": False,
            "limitation": (
                "'replay' = robot->pedestrian social-force term disabled in live "
                "social-force sim (peds_have_robot_repulsion=false); NOT "
                "pre-recorded trajectory playback."
            ),
        },
        "min_planners": 3,
        "min_seeds": 20,
        "rank_stability_analysis": {
            "paired_seed_resampling": True,
            "required_metrics": ["collision_rate", "near_miss_rate", "min_separation_m"],
            "rank_metric": "collision_rate",
            "bootstrap_resamples": 5000,
            "schedule": "s20",
            "target_ci_half_width": 0.10,
            "rank_effect_stability_threshold": 0.95,
            "seed_sufficiency_gate_command": (
                "uv run python scripts/tools/seed_sufficiency_gate.py "
                "--input-json <frozen_gate_input.json>"
            ),
            "replay_limitation_required": True,
            "claim_boundary": (
                "No paper-facing claim until post-run seed-sufficiency gate and "
                "claim-card review confirm rank-stability evidence with the replay "
                "force-off limitation."
            ),
        },
        "out_of_scope": [
            "no_full_benchmark_campaign",
            "no_slurm_gpu_submission",
            "no_paper_dissertation_claim_edits",
        ],
    }
    packet_path = tmp_path / "packet.yaml"
    packet_path.write_text(yaml.safe_dump(packet), encoding="utf-8")
    return packet_path


def _record(scenario: str, seed: int, *, collision: int) -> dict[str, object]:
    return {
        "scenario_id": scenario,
        "seed": seed,
        "metrics": {
            "total_collision_count": collision,
            "near_misses": collision,
            "min_clearance": 1.0,
        },
    }


def _write_campaign(tmp_path: Path, *, seeds: list[int]) -> tuple[Path, Path]:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    collision_plan = {
        "goal": {"reactive": 0, "replay": 1},
        "orca": {"reactive": 1, "replay": 0},
        "social_force": {"reactive": 0, "replay": 0},
    }
    for planner in PLANNERS:
        for condition in CONDITIONS:
            rows = [
                _record(
                    scenario,
                    seed,
                    collision=collision_plan[planner][condition],
                )
                for seed in seeds
                for scenario in SCENARIOS
            ]
            (campaign_dir / f"episodes_{planner}_{condition}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
    report_path = campaign_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "reactivity-ablation-campaign.v1",
                "replay_limitation": {"is_trajectory_playback": False},
            }
        ),
        encoding="utf-8",
    )
    return campaign_dir, report_path


def test_finalize_runs_analyzer_and_promotes_evidence(tmp_path: Path) -> None:
    """A complete campaign bundle becomes compact durable evidence in one command."""
    seeds = list(range(101, 121))
    packet = _write_packet(tmp_path, seeds=seeds)
    campaign_dir, campaign_report = _write_campaign(tmp_path, seeds=seeds)
    analysis_dir = tmp_path / "analysis"
    evidence_dir = tmp_path / "evidence"

    result = FINALIZER.finalize(
        packet_path=packet,
        campaign_dir=campaign_dir,
        campaign_report=campaign_report,
        analysis_dir=analysis_dir,
        evidence_dir=evidence_dir,
    )

    assert result["status"] == "finalized"
    assert result["issue"] == 3637
    assert result["analysis"]["episode_count"] == 240
    assert result["analysis"]["expected_episode_count"] == 240
    assert result["promotion"]["status"] == "promoted"
    assert (analysis_dir / "seed_gate_decision.json").exists()
    assert (evidence_dir / "seed_gate_decision.json").exists()
    assert (evidence_dir / "manifest.sha256").exists()
    assert result["forbidden_actions_confirmed"] == {
        "benchmark_campaign_execution": False,
        "slurm_gpu_submission": False,
        "paper_dissertation_claim_edit": False,
    }


def test_finalize_fails_closed_when_campaign_outputs_missing(tmp_path: Path) -> None:
    """The finalizer reports ordinary missing inputs as blocked, not partial evidence."""
    seeds = list(range(101, 121))
    packet = _write_packet(tmp_path, seeds=seeds)
    with pytest.raises(FINALIZER.FinalizationError, match="missing campaign report"):
        FINALIZER.finalize(
            packet_path=packet,
            campaign_dir=tmp_path / "missing-campaign",
            campaign_report=tmp_path / "missing-campaign" / "report.json",
            analysis_dir=tmp_path / "analysis",
            evidence_dir=tmp_path / "evidence",
        )


def test_cli_reports_blocked_for_missing_campaign(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI emits a compact blocked JSON object for absent campaign outputs."""
    seeds = list(range(101, 121))
    packet = _write_packet(tmp_path, seeds=seeds)

    exit_code = FINALIZER.main(
        [
            "--packet",
            str(packet),
            "--campaign-dir",
            str(tmp_path / "missing-campaign"),
            "--campaign-report",
            str(tmp_path / "missing-campaign" / "report.json"),
            "--analysis-dir",
            str(tmp_path / "analysis"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["issue"] == 3637
    assert "missing campaign report" in payload["reason"]
