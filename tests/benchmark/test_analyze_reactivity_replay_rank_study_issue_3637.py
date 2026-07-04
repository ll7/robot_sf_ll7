"""Tests for the issue #3637 reactivity-vs-replay rank-study analyzer."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from scripts.tools.seed_sufficiency_gate import SeedGateInput

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "benchmark" / "analyze_reactivity_replay_rank_study_issue_3637.py"
)
SPEC = importlib.util.spec_from_file_location("_issue_3637_analyzer", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


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
            "limitation": "'replay' = robot->pedestrian social-force term disabled in live social-force sim (peds_have_robot_repulsion=false); NOT pre-recorded trajectory playback.",
        },
        "min_planners": 3,
        "min_seeds": len(seeds),
        "rank_stability_analysis": {
            "paired_seed_resampling": True,
            "required_metrics": ["collision_rate", "near_miss_rate", "min_separation_m"],
            "rank_metric": "collision_rate",
            "bootstrap_resamples": 5000,
            "schedule": "s20",
            "target_ci_half_width": 0.10,
            "rank_effect_stability_threshold": 0.95,
            "seed_sufficiency_gate_command": "uv run python scripts/tools/seed_sufficiency_gate.py --input-json <frozen_gate_input.json>",
            "replay_limitation_required": True,
            "claim_boundary": "No paper-facing claim until post-run seed-sufficiency gate claim-card review confirm rank-stability evidence replay force-off limitation.",
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


def _record(
    scenario: str, seed: int, *, collision: int, near: int = 0, clearance: float = 1.0
) -> dict:
    return {
        "scenario_id": scenario,
        "seed": seed,
        "metrics": {
            "total_collision_count": collision,
            "near_misses": near,
            "min_clearance": clearance,
        },
    }


def _write_campaign(
    tmp_path: Path,
    *,
    seeds: list[int],
    missing: tuple[str, str] | None = None,
    duplicate: bool = False,
    non_finite: bool = False,
    unpaired_seed: bool = False,
    stable: bool = True,
) -> tuple[Path, Path]:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    collision_plan = {
        "goal": {"reactive": 0, "replay": 1 if stable else 0},
        "orca": {"reactive": 1, "replay": 0 if stable else 1},
        "social_force": {"reactive": 0, "replay": 0},
    }
    for planner in PLANNERS:
        for condition in CONDITIONS:
            if missing == (planner, condition):
                continue
            rows = []
            condition_seeds = list(seeds)
            if unpaired_seed and planner == "goal" and condition == "replay":
                condition_seeds = seeds[:-1]
            for seed in condition_seeds:
                for scenario in SCENARIOS:
                    collision = collision_plan[planner][condition]
                    if not stable:
                        collision = (
                            seed + PLANNERS.index(planner) + CONDITIONS.index(condition)
                        ) % 2
                    clearance = (
                        float("inf")
                        if non_finite and planner == "goal" and condition == "reactive"
                        else 1.0
                    )
                    rows.append(
                        _record(
                            scenario, seed, collision=collision, near=collision, clearance=clearance
                        )
                    )
            if duplicate and planner == "goal" and condition == "reactive":
                rows.append(dict(rows[0]))
            (campaign_dir / f"episodes_{planner}_{condition}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
    report = {
        "schema_version": "reactivity-ablation-campaign.v1",
        "replay_limitation": {"is_trajectory_playback": False},
    }
    report_path = campaign_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return campaign_dir, report_path


def test_complete_synthetic_matrix_writes_analysis_and_gate_input(tmp_path: Path) -> None:
    """A complete paired matrix produces analysis artifacts and seed-gate input."""
    seeds = list(range(101, 121))
    packet = _write_packet(tmp_path, seeds=seeds)
    campaign_dir, report = _write_campaign(tmp_path, seeds=seeds)
    output_dir = tmp_path / "analysis"

    analysis = ANALYZER.analyze(
        packet_path=packet,
        campaign_dir=campaign_dir,
        campaign_report=report,
        output_dir=output_dir,
    )

    assert analysis["episode_count"] == 3 * 2 * 20 * 2
    assert analysis["per_planner"]["goal"]["replay"]["collision_rate"] == pytest.approx(1.0)
    assert analysis["per_planner"]["goal"]["reactive"]["collision_rate"] == pytest.approx(0.0)
    assert analysis["rank_effect"]["ranking_is_reactivity_sensitive"] is True
    assert analysis["seed_sufficiency_gate_input"]["schedule"] == "s20"
    assert analysis["replay_limitation"]["is_trajectory_playback"] is False
    assert (output_dir / "analysis.json").exists()
    assert (output_dir / "frozen_gate_input.json").exists()
    assert (output_dir / "rank_bootstrap_summary.json").exists()
    assert (output_dir / "per_planner_condition_metrics.csv").exists()


def test_missing_replay_arm_fails_closed(tmp_path: Path) -> None:
    """Missing campaign arm output is not accepted as partial evidence."""
    seeds = list(range(101, 121))
    packet = _write_packet(tmp_path, seeds=seeds)
    campaign_dir, report = _write_campaign(tmp_path, seeds=seeds, missing=("goal", "replay"))

    with pytest.raises(ANALYZER.AnalysisInputError, match="missing campaign episode file"):
        ANALYZER.analyze(packet_path=packet, campaign_dir=campaign_dir, campaign_report=report)


def test_unpaired_seed_sets_fail_closed(tmp_path: Path) -> None:
    """Every planner arm must cover the same paired seed set."""
    seeds = list(range(101, 121))
    packet = _write_packet(tmp_path, seeds=seeds)
    campaign_dir, report = _write_campaign(tmp_path, seeds=seeds, unpaired_seed=True)

    with pytest.raises(ANALYZER.AnalysisInputError, match="missing episode row"):
        ANALYZER.analyze(packet_path=packet, campaign_dir=campaign_dir, campaign_report=report)


def test_duplicate_episode_row_fails_closed(tmp_path: Path) -> None:
    """Duplicate planner/arm/scenario/seed rows would bias aggregate metrics."""
    seeds = list(range(101, 121))
    packet = _write_packet(tmp_path, seeds=seeds)
    campaign_dir, report = _write_campaign(tmp_path, seeds=seeds, duplicate=True)

    with pytest.raises(ANALYZER.AnalysisInputError, match="duplicate episode row"):
        ANALYZER.analyze(packet_path=packet, campaign_dir=campaign_dir, campaign_report=report)


def test_non_finite_metric_fails_closed(tmp_path: Path) -> None:
    """Non-finite metrics are rejected before any rank analysis is emitted."""
    seeds = list(range(101, 121))
    packet = _write_packet(tmp_path, seeds=seeds)
    campaign_dir, report = _write_campaign(tmp_path, seeds=seeds, non_finite=True)

    with pytest.raises(ANALYZER.AnalysisInputError, match="must be finite"):
        ANALYZER.analyze(packet_path=packet, campaign_dir=campaign_dir, campaign_report=report)


def test_frozen_gate_input_only_uses_seed_gate_fields(tmp_path: Path) -> None:
    """Frozen seed-gate input remains compatible with SeedGateInput."""
    seeds = list(range(101, 121))
    packet = _write_packet(tmp_path, seeds=seeds)
    campaign_dir, report = _write_campaign(tmp_path, seeds=seeds)
    analysis = ANALYZER.analyze(
        packet_path=packet, campaign_dir=campaign_dir, campaign_report=report
    )

    gate_input = analysis["seed_sufficiency_gate_input"]
    assert set(gate_input) == {field.name for field in fields(SeedGateInput)}
    SeedGateInput(**gate_input)


def test_campaign_report_trajectory_playback_mismatch_fails_closed(tmp_path: Path) -> None:
    """Campaign reports cannot contradict the packet's force-off replay limitation."""
    seeds = list(range(101, 121))
    packet = _write_packet(tmp_path, seeds=seeds)
    campaign_dir, report = _write_campaign(tmp_path, seeds=seeds)
    report.write_text(
        json.dumps({"replay_limitation": {"is_trajectory_playback": True}}),
        encoding="utf-8",
    )

    with pytest.raises(ANALYZER.AnalysisInputError, match="is_trajectory_playback=false"):
        ANALYZER.analyze(packet_path=packet, campaign_dir=campaign_dir, campaign_report=report)


def test_cli_writes_expected_artifacts(tmp_path: Path) -> None:
    """The command-line interface writes the reviewable analysis files."""
    seeds = list(range(101, 121))
    packet = _write_packet(tmp_path, seeds=seeds)
    campaign_dir, report = _write_campaign(tmp_path, seeds=seeds)
    output_dir = tmp_path / "cli-analysis"

    assert (
        ANALYZER.main(
            [
                "--packet",
                str(packet),
                "--campaign-dir",
                str(campaign_dir),
                "--campaign-report",
                str(report),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert (output_dir / "analysis.json").exists()
    assert (output_dir / "frozen_gate_input.json").exists()
    assert (output_dir / "rank_bootstrap_summary.json").exists()


def test_bootstrap_unstable_synthetic_ranks_sets_rank_flip(tmp_path: Path) -> None:
    """An unstable synthetic contrast is routed through rank_flip_observed."""
    seeds = list(range(101, 121))
    packet = _write_packet(tmp_path, seeds=seeds)
    campaign_dir, report = _write_campaign(tmp_path, seeds=seeds, stable=False)

    analysis = ANALYZER.analyze(
        packet_path=packet, campaign_dir=campaign_dir, campaign_report=report
    )

    assert analysis["paired_seed_bootstrap"]["rank_flip_observed"] is True
