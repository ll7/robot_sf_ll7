"""Tests for artifact-driven scenario difficulty analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.scenario_difficulty import (
    _build_seed_index,
    _difficulty_weighted_score,
    _is_benchmark_success,
    _is_consensus_planner,
    _load_verified_simple_ids,
    _max,
    _mean,
    _metric_range,
    _normalized_ranks,
    _percentile,
    _planner_quality_rows,
    _planner_row_index,
    _planner_selection_rows,
    _preview_metadata_lookup,
    _safe_float,
    _scenario_family,
    _seed_field,
    _spearman_rank_correlation,
    _verified_simple_assessment,
    build_scenario_difficulty_analysis,
)

if TYPE_CHECKING:
    from pathlib import Path


def _planner_rows() -> list[dict[str, str]]:
    return [
        {
            "planner_key": "goal",
            "algo": "goal",
            "planner_group": "core",
            "status": "ok",
            "preflight_status": "ok",
            "benchmark_success": "true",
        },
        {
            "planner_key": "orca",
            "algo": "orca",
            "planner_group": "core",
            "status": "ok",
            "preflight_status": "ok",
            "benchmark_success": "true",
        },
        {
            "planner_key": "stream_gap",
            "algo": "stream_gap",
            "planner_group": "experimental",
            "status": "ok",
            "preflight_status": "ok",
            "benchmark_success": "true",
        },
    ]


def _seed_payload() -> dict[str, object]:
    return {
        "rows": [
            {
                "scenario_id": "easy_case",
                "planner_key": "goal",
                "seed_count": 3,
                "summary": {
                    "success": {"ci_half_width": 0.02, "cv": 0.01},
                    "time_to_goal_norm": {"ci_half_width": 0.03, "cv": 0.02},
                    "snqi": {"ci_half_width": 0.04, "cv": 0.03},
                },
            },
            {
                "scenario_id": "easy_case",
                "planner_key": "orca",
                "seed_count": 3,
                "summary": {
                    "success": {"ci_half_width": 0.03, "cv": 0.02},
                    "time_to_goal_norm": {"ci_half_width": 0.04, "cv": 0.03},
                    "snqi": {"ci_half_width": 0.04, "cv": 0.03},
                },
            },
            {
                "scenario_id": "hard_case",
                "planner_key": "goal",
                "seed_count": 3,
                "summary": {
                    "success": {"ci_half_width": 0.11, "cv": 0.20},
                    "time_to_goal_norm": {"ci_half_width": 0.10, "cv": 0.14},
                    "snqi": {"ci_half_width": 0.08, "cv": 0.12},
                },
            },
            {
                "scenario_id": "hard_case",
                "planner_key": "orca",
                "seed_count": 3,
                "summary": {
                    "success": {"ci_half_width": 0.10, "cv": 0.18},
                    "time_to_goal_norm": {"ci_half_width": 0.09, "cv": 0.12},
                    "snqi": {"ci_half_width": 0.08, "cv": 0.11},
                },
            },
        ]
    }


def _preview_payload() -> dict[str, object]:
    return {
        "truncated": False,
        "route_clearance_warnings": [
            {
                "scenario": "hard_case",
                "warning_scope": "route",
                "min_clearance_margin_m": 0.2,
            }
        ],
        "scenarios": [
            {
                "name": "easy_case",
                "simulation_config": {"ped_density": 0.0},
                "metadata": {
                    "archetype": "easy_family",
                    "flow": "none",
                    "behavior": "none",
                    "primary_capability": "frame_consistency",
                    "target_failure_mode": "coordinate_transform",
                    "determinism": "deterministic",
                },
            },
            {
                "name": "hard_case",
                "simulation_config": {"ped_density": 0.5},
                "metadata": {
                    "archetype": "hard_family",
                    "flow": "perpendicular",
                    "behavior": "crowd",
                    "primary_capability": "dynamic_interaction",
                    "target_failure_mode": "social_collision",
                    "determinism": "stochastic",
                },
            },
        ],
    }


def test_build_scenario_difficulty_ranks_harder_scenarios_first() -> None:
    """Difficulty ranking should place scenarios that are weak across core planners above easier control cases because the proxy is meant to separate scenario hardness from planner quality."""
    analysis = build_scenario_difficulty_analysis(
        planner_rows=_planner_rows()[:2],
        scenario_breakdown_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.95",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.05",
                "time_to_goal_norm_mean": "0.30",
                "snqi_mean": "0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.90",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.10",
                "time_to_goal_norm_mean": "0.35",
                "snqi_mean": "0.20",
            },
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.35",
                "collisions_mean": "0.45",
                "near_misses_mean": "0.70",
                "time_to_goal_norm_mean": "0.85",
                "snqi_mean": "-0.40",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.40",
                "collisions_mean": "0.35",
                "near_misses_mean": "0.65",
                "time_to_goal_norm_mean": "0.80",
                "snqi_mean": "-0.30",
            },
        ],
        seed_variability_payload=_seed_payload(),
        preview_payload=_preview_payload(),
    )

    scenario_rows = {row["scenario_id"]: row for row in analysis["scenario_rows"]}
    assert analysis["scenario_rows"][0]["scenario_id"] == "hard_case"
    assert (
        scenario_rows["hard_case"]["difficulty_score"]
        > scenario_rows["easy_case"]["difficulty_score"]
    )
    assert scenario_rows["hard_case"]["route_clearance_warning"] is True
    assert scenario_rows["hard_case"]["seed_success_ci_half_width_mean"] == pytest.approx(0.105)


def test_build_scenario_difficulty_flags_easy_scenario_underperformance() -> None:
    """Residual summaries should flag planners that underperform on easier scenarios because that is the main signal for separating planner mismatch from globally hard scenarios."""
    analysis = build_scenario_difficulty_analysis(
        planner_rows=_planner_rows(),
        scenario_breakdown_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.96",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.05",
                "time_to_goal_norm_mean": "0.30",
                "snqi_mean": "0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.92",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.08",
                "time_to_goal_norm_mean": "0.32",
                "snqi_mean": "0.25",
            },
            {
                "planner_key": "stream_gap",
                "algo": "stream_gap",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.20",
                "collisions_mean": "0.60",
                "near_misses_mean": "0.70",
                "time_to_goal_norm_mean": "0.95",
                "snqi_mean": "-0.50",
            },
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.45",
                "collisions_mean": "0.40",
                "near_misses_mean": "0.65",
                "time_to_goal_norm_mean": "0.80",
                "snqi_mean": "-0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.40",
                "collisions_mean": "0.35",
                "near_misses_mean": "0.60",
                "time_to_goal_norm_mean": "0.78",
                "snqi_mean": "-0.25",
            },
            {
                "planner_key": "stream_gap",
                "algo": "stream_gap",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.38",
                "collisions_mean": "0.38",
                "near_misses_mean": "0.62",
                "time_to_goal_norm_mean": "0.81",
                "snqi_mean": "-0.27",
            },
        ],
        seed_variability_payload=_seed_payload(),
        preview_payload=_preview_payload(),
    )

    planner_summary_rows = {row["planner_key"]: row for row in analysis["planner_summary_rows"]}
    stream_gap = planner_summary_rows["stream_gap"]
    assert stream_gap["easy_scenario_underperformance_count"] == 1
    assert stream_gap["worst_scenarios"][0] == "easy_case"


def test_build_scenario_difficulty_requests_verified_simple_rerun_without_overlap(
    tmp_path: Path,
) -> None:
    """Verified-simple assessment should ask for a bounded pilot when the current campaign has no overlap with the candidate subset because the recommendation would otherwise overclaim."""
    verified_simple_manifest = tmp_path / "verified_simple_subset.yaml"
    verified_simple_manifest.write_text(
        "scenarios:\n  - name: different_case\n    metadata:\n      archetype: control\n",
        encoding="utf-8",
    )

    analysis = build_scenario_difficulty_analysis(
        planner_rows=_planner_rows()[:2],
        scenario_breakdown_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.95",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.05",
                "time_to_goal_norm_mean": "0.30",
                "snqi_mean": "0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.90",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.10",
                "time_to_goal_norm_mean": "0.35",
                "snqi_mean": "0.20",
            },
        ],
        seed_variability_payload=_seed_payload(),
        preview_payload=_preview_payload(),
        verified_simple_manifest_path=verified_simple_manifest,
    )

    assessment = analysis["verified_simple_assessment"]
    assert assessment["status"] == "rerun_required"
    assert assessment["worth_adding"] is None
    assert "bounded pilot" in assessment["recommendation"]


def test_build_scenario_difficulty_reports_fallback_selection_when_core_set_missing() -> None:
    """Fallback consensus metadata should stay internally consistent when a non-paper campaign only exposes non-core planners."""
    analysis = build_scenario_difficulty_analysis(
        planner_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "planner_group": "experimental",
                "status": "ok",
                "preflight_status": "ok",
                "benchmark_success": "true",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "planner_group": "experimental",
                "status": "ok",
                "preflight_status": "ok",
                "benchmark_success": "true",
            },
        ],
        scenario_breakdown_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.95",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.05",
                "time_to_goal_norm_mean": "0.30",
                "snqi_mean": "0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.90",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.10",
                "time_to_goal_norm_mean": "0.35",
                "snqi_mean": "0.20",
            },
        ],
    )

    assert (
        analysis["primary_proxy"]["eligible_planner_selection"]
        == "all planners (fallback: no eligible core set)"
    )
    assert analysis["primary_proxy"]["eligible_planner_count"] == 2
    assert any("fell back to all planners" in finding for finding in analysis["findings"])


def test_scenario_difficulty_helper_edge_cases(tmp_path: Path) -> None:
    """Low-level helpers should handle empty, malformed, and fallback inputs without inventing signal."""
    assert _safe_float("") is None
    assert _safe_float("abc") is None
    assert _safe_float("1.5") == pytest.approx(1.5)
    assert _mean([None, float("nan")]) is None
    assert _max([None]) is None
    assert _metric_range([None]) is None
    assert _percentile([], 0.5) is None
    assert _percentile([3.0], 0.5) == pytest.approx(3.0)
    assert _scenario_family({}) == "unknown"
    assert _scenario_family({"scenario_id": "fallback_case"}) == "fallback_case"
    assert _normalized_ranks({}, higher_is_harder=True) == {}
    assert _normalized_ranks({"only": 1.0}, higher_is_harder=True) == {"only": 0.0}
    assert _difficulty_weighted_score("missing", {}) == (None, {})

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("scenarios: invalid\n", encoding="utf-8")
    assert _load_verified_simple_ids(manifest) == (set(), str(manifest))

    seed_index = _build_seed_index(
        {
            "rows": [
                {"scenario_id": "", "planner_key": "goal"},
                {"scenario_id": "case", "planner_key": ""},
                {
                    "scenario_id": "case",
                    "planner_key": "goal",
                    "seed_count": 3,
                    "summary": {"success": {"ci_half_width": 0.2}},
                },
            ]
        }
    )
    assert list(seed_index) == [("case", "goal")]
    assert _seed_field(seed_index[("case", "goal")], "success", "ci_half_width") == pytest.approx(
        0.2
    )
    assert _seed_field({"summary": []}, "success", "ci_half_width") is None


def test_preview_metadata_lookup_handles_truncation_and_missing_dicts() -> None:
    """Preview metadata parsing should degrade safely when the preflight payload is partial or malformed."""
    lookup, truncated = _preview_metadata_lookup(
        {
            "truncated": True,
            "route_clearance_warnings": [{"scenario": "case_a", "warning_scope": "route"}],
            "scenarios": [
                "bad-row",
                {
                    "id": "case_a",
                    "metadata": [],
                    "simulation_config": [],
                },
            ],
        }
    )

    assert truncated is True
    assert lookup["case_a"]["route_clearance_warning"] is True
    assert lookup["case_a"]["route_clearance_scope"] == "route"
    assert lookup["case_a"]["ped_density"] is None


def test_planner_quality_and_selection_helpers_cover_sparse_inputs() -> None:
    """Planner helper summaries should stay well-defined for sparse rows and fallback selections."""
    quality_rows = _planner_quality_rows(
        [
            {"planner_key": "goal", "planner_group": "core", "success_mean": "0.8"},
            {"planner_key": "goal", "planner_group": "core", "success_mean": "0.6"},
            {"planner_key": "orca", "planner_group": "experimental", "collisions_mean": "0.1"},
        ]
    )
    assert quality_rows[0]["planner_key"] == "goal"
    assert quality_rows[0]["success_mean"] == pytest.approx(0.7)

    selected, reason = _planner_selection_rows(
        [{"planner_key": "goal", "scenario_id": "case"}],
        {"goal": {"planner_group": "experimental", "status": "ok", "benchmark_success": "true"}},
    )
    assert reason == "all planners (fallback: no eligible core set)"
    assert selected[0]["planner_key"] == "goal"

    assert (
        _spearman_rank_correlation(
            [{"planner_key": "goal"}],
            [{"planner_key": "goal"}],
        )
        is None
    )


def test_benchmark_success_and_consensus_filters_cover_failure_modes() -> None:
    """Consensus filtering should exclude fallback/degraded/non-success rows and keep plain successful rows."""
    assert _is_benchmark_success({}) is True
    assert _is_benchmark_success({"benchmark_success": True}) is True
    assert _is_benchmark_success({"benchmark_success": "false"}) is False

    assert _is_consensus_planner({}) is False
    assert _is_consensus_planner({"planner_group": "experimental"}) is False
    assert _is_consensus_planner({"planner_group": "core", "readiness_status": "fallback"}) is False
    assert _is_consensus_planner({"planner_group": "core", "preflight_status": "fallback"}) is False
    assert _is_consensus_planner({"planner_group": "core", "status": "not_available"}) is False
    assert _is_consensus_planner({"planner_group": "core", "benchmark_success": "true"}) is True


def test_build_scenario_difficulty_reports_preview_truncation_and_manifest_missing() -> None:
    """Unavailable artifacts should remain explicit instead of silently fabricating difficulty output."""
    analysis = build_scenario_difficulty_analysis(
        planner_rows=_planner_rows()[:1],
        scenario_breakdown_rows=[],
        preview_payload={"truncated": True},
        verified_simple_manifest_path=None,
    )

    assert analysis["status"] == "unavailable"
    assert analysis["verified_simple_assessment"]["status"] == "manifest_missing"


def test_verified_simple_assessment_supports_candidate_when_order_is_preserved(
    tmp_path: Path,
) -> None:
    """A simple subset should be marked as supported when ordering and seed noise stay aligned with the full campaign."""
    manifest = tmp_path / "verified_simple_subset.yaml"
    manifest.write_text(
        "scenarios:\n  - name: easy_case\n  - name: hard_case\n",
        encoding="utf-8",
    )

    assessment = _verified_simple_assessment(
        [
            {
                "planner_key": "goal",
                "scenario_id": "easy_case",
                "success_mean": "0.95",
                "seed_success_ci_half_width": "0.02",
            },
            {
                "planner_key": "orca",
                "scenario_id": "easy_case",
                "success_mean": "0.90",
                "seed_success_ci_half_width": "0.03",
            },
            {
                "planner_key": "goal",
                "scenario_id": "hard_case",
                "success_mean": "0.45",
                "seed_success_ci_half_width": "0.10",
            },
            {
                "planner_key": "orca",
                "scenario_id": "hard_case",
                "success_mean": "0.40",
                "seed_success_ci_half_width": "0.09",
            },
        ],
        _planner_row_index(_planner_rows()[:2]),
        manifest_path=manifest,
    )

    assert assessment["status"] == "candidate_supported"
    assert assessment["worth_adding"] is True
    assert assessment["comparison_planner_selection"] == "core benchmark-success planners"
    assert assessment["full_seed_success_ci_half_width_mean"] == pytest.approx(0.06)
    assert assessment["subset_seed_success_ci_half_width_mean"] == pytest.approx(0.06)


def test_verified_simple_assessment_marks_candidate_noisy_when_subset_reorders(
    tmp_path: Path,
) -> None:
    """A reordered or noisier subset should stay a debugging gate rather than a calibration aid."""
    manifest = tmp_path / "verified_simple_subset.yaml"
    manifest.write_text(
        "scenarios:\n  - name: hard_case\n",
        encoding="utf-8",
    )

    assessment = _verified_simple_assessment(
        [
            {
                "planner_key": "goal",
                "scenario_id": "easy_case",
                "success_mean": "0.99",
                "seed_success_ci_half_width": "0.01",
            },
            {
                "planner_key": "orca",
                "scenario_id": "easy_case",
                "success_mean": "0.80",
                "seed_success_ci_half_width": "0.01",
            },
            {
                "planner_key": "goal",
                "scenario_id": "hard_case",
                "success_mean": "0.10",
                "seed_success_ci_half_width": "0.30",
            },
            {
                "planner_key": "orca",
                "scenario_id": "hard_case",
                "success_mean": "0.70",
                "seed_success_ci_half_width": "0.30",
            },
        ],
        _planner_row_index(_planner_rows()[:2]),
        manifest_path=manifest,
    )

    assert assessment["status"] == "candidate_noisy"
    assert assessment["worth_adding"] is False


def test_build_scenario_difficulty_supports_verified_simple_candidate_with_overlap(
    tmp_path: Path,
) -> None:
    """Build-level analysis should preserve planner overlap for verified-simple assessment."""
    manifest = tmp_path / "verified_simple_subset.yaml"
    manifest.write_text(
        "scenarios:\n  - name: easy_case\n  - name: hard_case\n",
        encoding="utf-8",
    )

    analysis = build_scenario_difficulty_analysis(
        planner_rows=_planner_rows()[:2],
        scenario_breakdown_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.95",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.05",
                "time_to_goal_norm_mean": "0.30",
                "snqi_mean": "0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.90",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.10",
                "time_to_goal_norm_mean": "0.35",
                "snqi_mean": "0.20",
            },
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.35",
                "collisions_mean": "0.45",
                "near_misses_mean": "0.70",
                "time_to_goal_norm_mean": "0.85",
                "snqi_mean": "-0.40",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.40",
                "collisions_mean": "0.35",
                "near_misses_mean": "0.65",
                "time_to_goal_norm_mean": "0.80",
                "snqi_mean": "-0.30",
            },
        ],
        seed_variability_payload=_seed_payload(),
        preview_payload=_preview_payload(),
        verified_simple_manifest_path=manifest,
    )

    assessment = analysis["verified_simple_assessment"]
    assert assessment["status"] == "candidate_supported"
    assert assessment["worth_adding"] is True


def test_build_scenario_difficulty_marks_verified_simple_candidate_noisy_when_reordered(
    tmp_path: Path,
) -> None:
    """Build-level analysis should surface noisy verified-simple candidates from planner rows."""
    manifest = tmp_path / "verified_simple_subset.yaml"
    manifest.write_text(
        "scenarios:\n  - name: hard_case\n",
        encoding="utf-8",
    )

    seed_payload = {
        "rows": [
            {
                "scenario_id": "easy_case",
                "planner_key": "goal",
                "seed_count": 3,
                "summary": {"success": {"ci_half_width": 0.01}},
            },
            {
                "scenario_id": "easy_case",
                "planner_key": "orca",
                "seed_count": 3,
                "summary": {"success": {"ci_half_width": 0.01}},
            },
            {
                "scenario_id": "hard_case",
                "planner_key": "goal",
                "seed_count": 3,
                "summary": {"success": {"ci_half_width": 0.30}},
            },
            {
                "scenario_id": "hard_case",
                "planner_key": "orca",
                "seed_count": 3,
                "summary": {"success": {"ci_half_width": 0.30}},
            },
        ]
    }
    scenario_rows = [
        {
            "planner_key": "goal",
            "algo": "goal",
            "scenario_id": "easy_case",
            "scenario_name": "easy_case",
            "episodes": 10,
            "success_mean": 0.99,
            "collisions_mean": 0.00,
            "near_misses_mean": 0.01,
            "time_to_goal_norm_mean": 0.90,
            "snqi_mean": 0.70,
        },
        {
            "planner_key": "orca",
            "algo": "orca",
            "scenario_id": "easy_case",
            "scenario_name": "easy_case",
            "episodes": 10,
            "success_mean": 0.10,
            "collisions_mean": 0.10,
            "near_misses_mean": 0.10,
            "time_to_goal_norm_mean": 1.20,
            "snqi_mean": 0.50,
        },
        {
            "planner_key": "goal",
            "algo": "goal",
            "scenario_id": "hard_case",
            "scenario_name": "hard_case",
            "episodes": 10,
            "success_mean": 0.10,
            "collisions_mean": 0.60,
            "near_misses_mean": 0.40,
            "time_to_goal_norm_mean": 1.80,
            "snqi_mean": 0.10,
        },
        {
            "planner_key": "orca",
            "algo": "orca",
            "scenario_id": "hard_case",
            "scenario_name": "hard_case",
            "episodes": 10,
            "success_mean": 0.70,
            "collisions_mean": 0.20,
            "near_misses_mean": 0.20,
            "time_to_goal_norm_mean": 1.10,
            "snqi_mean": 0.60,
        },
    ]

    analysis = build_scenario_difficulty_analysis(
        planner_rows=_planner_rows()[:2],
        scenario_breakdown_rows=scenario_rows,
        seed_variability_payload=seed_payload,
        preview_payload=_preview_payload(),
        verified_simple_manifest_path=manifest,
    )

    assessment = analysis["verified_simple_assessment"]
    assert assessment["status"] == "candidate_noisy"
    assert assessment["worth_adding"] is False
    assert assessment["full_seed_success_ci_half_width_mean"] == pytest.approx(0.155)
    assert assessment["subset_seed_success_ci_half_width_mean"] == pytest.approx(0.30)


def test_verified_simple_assessment_does_not_claim_support_without_noise_evidence(
    tmp_path: Path,
) -> None:
    """High rank correlation without seed-noise evidence should remain mixed rather than supported."""
    manifest = tmp_path / "verified_simple_subset.yaml"
    manifest.write_text(
        "scenarios:\n  - name: easy_case\n  - name: hard_case\n",
        encoding="utf-8",
    )

    assessment = _verified_simple_assessment(
        [
            {"planner_key": "goal", "scenario_id": "easy_case", "success_mean": "0.95"},
            {"planner_key": "orca", "scenario_id": "easy_case", "success_mean": "0.90"},
            {"planner_key": "goal", "scenario_id": "hard_case", "success_mean": "0.45"},
            {"planner_key": "orca", "scenario_id": "hard_case", "success_mean": "0.40"},
        ],
        _planner_row_index(_planner_rows()[:2]),
        manifest_path=manifest,
    )

    assert assessment["status"] == "mixed_signal"
    assert assessment["worth_adding"] is None


def test_build_scenario_difficulty_keeps_missing_planner_group_on_all_planner_fallback() -> None:
    """Campaigns without explicit planner-group metadata should use the all-planners fallback label."""
    analysis = build_scenario_difficulty_analysis(
        planner_rows=[
            {"planner_key": "goal", "algo": "goal", "status": "ok", "benchmark_success": "true"},
            {"planner_key": "orca", "algo": "orca", "status": "ok", "benchmark_success": "true"},
        ],
        scenario_breakdown_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.95",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.05",
                "time_to_goal_norm_mean": "0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.90",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.10",
                "time_to_goal_norm_mean": "0.35",
            },
        ],
    )

    assert (
        analysis["primary_proxy"]["eligible_planner_selection"]
        == "all planners (fallback: no eligible core set)"
    )


def test_build_scenario_difficulty_does_not_flag_easy_underperformance_on_zero_baseline() -> None:
    """Zero residual baselines should not fabricate easy-scenario underperformance flags."""
    analysis = build_scenario_difficulty_analysis(
        planner_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "planner_group": "core",
                "status": "ok",
                "preflight_status": "ok",
                "benchmark_success": "true",
            }
        ],
        scenario_breakdown_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.95",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.05",
                "time_to_goal_norm_mean": "0.30",
            },
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.45",
                "collisions_mean": "0.30",
                "near_misses_mean": "0.40",
                "time_to_goal_norm_mean": "0.80",
            },
        ],
    )

    assert all(
        row["easy_scenario_underperformance"] is False for row in analysis["planner_residual_rows"]
    )
