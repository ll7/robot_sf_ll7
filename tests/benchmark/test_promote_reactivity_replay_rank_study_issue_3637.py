"""Tests issue #3637 analyzer evidence promotion."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "benchmark" / "promote_reactivity_replay_rank_study_issue_3637.py"
)
SPEC = importlib.util.spec_from_file_location("_issue_3637_promoter", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
PROMOTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROMOTER)


def _write_analysis_dir(tmp_path: Path, *, replay_is_trajectory: bool = False) -> Path:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    analysis = {
        "schema_version": "reactivity-replay-rank-study-analysis.v1",
        "issue": 3637,
        "source_campaign_issue": 3573,
        "campaign_report": "output/issue_3637_reactivity_rank_study/report.json",
        "planners": ["goal", "orca", "social_force"],
        "conditions": ["reactive", "replay"],
        "seeds": list(range(101, 121)),
        "scenario_set": "configs/scenarios/sets/classic_crossing_subset.yaml",
        "episode_count": 240,
        "expected_episode_count": 240,
        "rank_effect": {"ranking_is_reactivity_sensitive": True},
        "paired_seed_bootstrap": {"rank_flip_observed": False},
        "seed_sufficiency_gate_input": {
            "schedule": "s20",
            "ci_half_width": 0.08,
            "target_ci_half_width": 0.1,
            "rank_flip_observed": False,
            "heldout_delta_abs": None,
            "heldout_delta_threshold": None,
            "invalid_row_count": 0,
        },
        "replay_limitation": {
            "is_trajectory_playback": replay_is_trajectory,
            "note": "'replay' = robot->pedestrian force disabled; not trajectory playback.",
        },
        "claim_boundary": "No paper-facing claim until seed-sufficiency gate review.",
        "claim_decision": "post_run_gate_input_ready",
    }
    (analysis_dir / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    (analysis_dir / "frozen_gate_input.json").write_text(
        json.dumps(analysis["seed_sufficiency_gate_input"]),
        encoding="utf-8",
    )
    (analysis_dir / "rank_bootstrap_summary.json").write_text(
        json.dumps(analysis["paired_seed_bootstrap"]),
        encoding="utf-8",
    )
    (analysis_dir / "per_planner_condition_metrics.csv").write_text(
        "planner,condition,collision_rate\ngoal,reactive,0.0\ngoal,replay,1.0\n",
        encoding="utf-8",
    )
    (analysis_dir / "README.md").write_text("analyzer readme\n", encoding="utf-8")
    return analysis_dir


def test_promote_writes_durable_bundle_and_manifest(tmp_path: Path) -> None:
    """Completed analyzer outputs become a compact docs evidence bundle."""

    analysis_dir = _write_analysis_dir(tmp_path)
    evidence_dir = tmp_path / "evidence"

    result = PROMOTER.promote(analysis_dir, evidence_dir)

    assert result["status"] == "promoted"
    assert result["issue"] == 3637
    assert (evidence_dir / "analysis.json").exists()
    assert (evidence_dir / "frozen_gate_input.json").exists()
    assert (evidence_dir / "rank_bootstrap_summary.json").exists()
    assert (evidence_dir / "per_planner_condition_metrics.csv").exists()
    summary = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "issue-3637-reactivity-replay-rank-promotion.v1"
    assert summary["claim_decision"] == "post_run_gate_input_ready"
    assert summary["out_of_scope"] == [
        "no benchmark campaign execution",
        "no Slurm/GPU submission",
        "no paper/dissertation claim edit",
    ]
    readme = (evidence_dir / "README.md").read_text(encoding="utf-8")
    assert "not a paper-facing claim by itself" in readme
    manifest = (evidence_dir / "manifest.sha256").read_text(encoding="utf-8")
    assert "analysis.json" in manifest
    assert "summary.json" in manifest


def test_missing_required_artifact_fails_closed(tmp_path: Path) -> None:
    """Promotion refuses incomplete analyzer output directories."""

    analysis_dir = _write_analysis_dir(tmp_path)
    (analysis_dir / "rank_bootstrap_summary.json").unlink()

    with pytest.raises(PROMOTER.PromotionError, match="missing rank_bootstrap_summary.json"):
        PROMOTER.promote(analysis_dir, tmp_path / "evidence")


def test_replay_trajectory_contradiction_fails_closed(tmp_path: Path) -> None:
    """Promotion refuses outputs that contradict the force-off replay limitation."""

    analysis_dir = _write_analysis_dir(tmp_path, replay_is_trajectory=True)

    with pytest.raises(PROMOTER.PromotionError, match="is_trajectory_playback=false"):
        PROMOTER.promote(analysis_dir, tmp_path / "evidence")


def test_cli_reports_blocked_for_missing_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI reports blocked when ordinary analyzer inputs are missing."""

    exit_code = PROMOTER.main(
        [
            "--analysis-dir",
            str(tmp_path / "missing"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["issue"] == 3637
