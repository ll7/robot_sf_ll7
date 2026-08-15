"""Focused tests for the issue #6095 discriminability report."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark.build_issue_6095_discriminability_report import (
    EXPECTED_KINEMATICS,
    EpisodeRow,
    RegimeData,
    ReportContractError,
    _episode_row,
    _provenance_limitation_lines,
    _validate_campaign_receipts,
    _validate_episode_row,
    bootstrap_mean_ci,
    classify_stress_floor,
)


def _episode(
    planner_key: str,
    scenario_id: str,
    seed: int,
    *,
    success: float = 0.0,
    collision: float = 0.0,
    near_misses: float = 0.0,
) -> EpisodeRow:
    """Build a minimal validated episode row for pure classification tests."""
    return EpisodeRow(
        planner_key=planner_key,
        scenario_id=scenario_id,
        seed=seed,
        success=success,
        collision=collision,
        near_misses=near_misses,
        near_miss_any=float(near_misses > 0.0),
        execution_mode="native",
        observation_level="tracked_agents_no_noise",
        model_id=None,
        horizon=100,
        dt=0.1,
    )


def _regime() -> RegimeData:
    """Build a four-scenario stress fixture covering every floor class."""
    scenarios = ("both_some", "one_some", "collision_only", "near_miss_only")
    seeds = (111, 112, 113)
    rows: dict[tuple[str, str, int], EpisodeRow] = {}
    for planner in ("orca", "ppo"):
        for scenario in scenarios:
            for seed in seeds:
                kwargs: dict[str, float] = {}
                if scenario == "both_some" and seed == 111:
                    kwargs["success"] = 1.0
                if scenario == "one_some" and planner == "ppo" and seed == 111:
                    kwargs["success"] = 1.0
                if scenario == "collision_only" and planner == "orca" and seed == 111:
                    kwargs["collision"] = 1.0
                if scenario == "near_miss_only" and planner == "ppo" and seed == 111:
                    kwargs["near_misses"] = 2.0
                rows[(planner, scenario, seed)] = _episode(planner, scenario, seed, **kwargs)
    return RegimeData(
        name="stress",
        root=Path("."),
        campaign_id="fixture",
        scenario_matrix="fixture",
        scenario_matrix_hash="fixture",
        git_commit="fixture",
        scenario_ids=scenarios,
        seeds=seeds,
        rows=rows,
        blockers=[],
        warnings=[],
        checkpoint={},
        metadata={"kinematics": EXPECTED_KINEMATICS},
    )


def test_bootstrap_mean_ci_is_deterministic_and_seed_scenario_aware() -> None:
    """The declared bootstrap returns repeatable finite bounds for a 2-D matrix."""
    matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    first = bootstrap_mean_ci(matrix, bootstrap_seed=6095, bootstrap_samples=200)
    second = bootstrap_mean_ci(matrix, bootstrap_seed=6095, bootstrap_samples=200)

    assert first == second
    assert first[0] == 0.5
    assert 0.0 <= first[1] <= first[2] <= 1.0


def test_classify_stress_floor_counts_both_zero_metric_discriminability() -> None:
    """Both-zero scenarios are separated from one-planner and shared successes."""
    result = classify_stress_floor(_regime(), seeds=(111, 112, 113))

    assert result["class_counts"] == {
        "both_planners_some_success": 1,
        "both_planners_zero_success": 2,
        "exactly_one_planner_some_success": 1,
    }
    assert result["both_zero_count"] == 2
    assert result["both_zero_distinguished_count"] == 2
    assert result["both_zero_distinguished_by_collision_count"] == 1
    assert result["both_zero_distinguished_by_near_miss_count"] == 1


def test_provenance_markdown_tracks_staged_receipts() -> None:
    """Human-readable provenance caveats must reflect staged receipt status."""
    receipt = {
        "status": "staged_receipt",
        "identity_matches_expected": True,
        "hash_source": "computed_file",
        "submit_safe": True,
        "load_status": "not_run",
    }

    lines = _provenance_limitation_lines({"nominal": receipt, "stress": receipt})

    rendered = "\n".join(lines)
    assert "staged" in rendered
    assert "metadata-only" not in rendered
    assert "nominal=not_run, stress=not_run" in rendered


def test_episode_row_rejects_missing_or_unknown_termination_reason(tmp_path: Path) -> None:
    """Malformed terminal metadata must not silently become a zero outcome."""
    record = {
        "scenario_id": "scenario",
        "seed": 111,
        "metrics": {"near_misses": 0.0},
    }

    with pytest.raises(ReportContractError, match="termination_reason"):
        _episode_row(record, planner_key="orca", source=tmp_path / "episodes.jsonl")

    record["termination_reason"] = "unknown"
    with pytest.raises(ReportContractError, match="termination_reason"):
        _episode_row(record, planner_key="orca", source=tmp_path / "episodes.jsonl")


def test_episode_row_validates_planner_execution_and_observation_contract() -> None:
    """The report must reject rows that do not match the frozen runtime contract."""
    row = _episode(
        "orca",
        "scenario",
        111,
        success=0.0,
    )
    blockers = _validate_episode_row(
        name="nominal",
        key=("orca", "scenario", 111),
        record={},
        row=replace(row, execution_mode="native", observation_level="oracle_full_state"),
        scenario_ids=("scenario",),
        expected_seeds=(111,),
        expected_commit="fixture",
        expected_model_id="model",
    )

    assert any("execution mode" in blocker for blocker in blockers)
    assert any("observation level" in blocker for blocker in blockers)


def test_episode_row_rejects_failed_or_degraded_statuses() -> None:
    """Raw failed or degraded rows must not become diagnostic metric values."""
    row = _episode("orca", "scenario", 111)
    record = {
        "termination_reason": "error",
        "status": "failure",
        "algorithm_metadata": {"status": "fallback"},
    }

    blockers = _validate_episode_row(
        name="nominal",
        key=("orca", "scenario", 111),
        record=record,
        row=row,
        scenario_ids=("scenario",),
        expected_seeds=(111,),
        expected_commit="fixture",
        expected_model_id="model",
    )

    assert any("failed episode termination" in blocker for blocker in blockers)
    assert any("algorithm status 'fallback'" in blocker for blocker in blockers)


def test_campaign_receipts_require_complete_zero_failure_row_summary() -> None:
    """Missing or non-zero row-status receipts must fail closed."""
    summary = {
        "campaign": {
            "scenario_matrix": "matrix",
            "git_hash": "fixture",
            "benchmark_success": True,
            "evidence_status": "valid",
            "campaign_execution_status": "completed",
            "row_status_summary": {
                "successful_evidence_rows": 1,
                "accepted_unavailable_rows": 1,
                "fallback_or_degraded_rows": 0,
                "unexpected_failed_rows": 0,
            },
        }
    }
    manifest = {
        "scenario_matrix": "matrix",
        "git": {"commit": "fixture"},
        "seed_policy": {"resolved_seeds": [111]},
    }
    integrity = {"status": "valid", "benchmark_success_allowed": True}

    _campaign, blockers = _validate_campaign_receipts(
        name="nominal",
        summary=summary,
        manifest=manifest,
        integrity=integrity,
        expected_matrix="matrix",
        expected_seeds=(111,),
        expected_commit="fixture",
    )

    assert any("accepted unavailable rows are present" in blocker for blocker in blockers)

    summary["campaign"].pop("row_status_summary")
    _campaign, blockers = _validate_campaign_receipts(
        name="nominal",
        summary=summary,
        manifest=manifest,
        integrity=integrity,
        expected_matrix="matrix",
        expected_seeds=(111,),
        expected_commit="fixture",
    )

    assert any("row_status_summary is missing or invalid" in blocker for blocker in blockers)
