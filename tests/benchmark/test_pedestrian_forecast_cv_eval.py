"""Tests for the CV forecast evaluation script."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

from robot_sf.benchmark.pedestrian_forecast import (
    goal_aware_cv_baseline,
    interaction_aware_cv_baseline,
    semantic_cv_baseline,
    signal_aware_cv_baseline,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/run_cv_forecast_eval.py"


def _load_script_module():
    """Load the script as a module for testing."""
    spec = importlib.util.spec_from_file_location("run_cv_forecast_eval", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_cv_forecast_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_script_module()

BASELINE_FUNCTIONS = _mod.BASELINE_FUNCTIONS if hasattr(_mod, "BASELINE_FUNCTIONS") else {}
MISSING_FAMILIES = _mod.MISSING_FAMILIES
TRACE_CANDIDATES = _mod.TRACE_CANDIDATES
_actor_class_counts = _mod._actor_class_counts
_build_failure_cases = _mod._build_failure_cases
_compute_dt_s = _mod._compute_dt_s
_extract_trace_steps = _mod._extract_trace_steps
_generate_markdown = _mod._generate_markdown
_get_actor_classes = _mod._get_actor_classes
_pedestrian_count = _mod._pedestrian_count
_summarize_interaction_effect = _mod._summarize_interaction_effect
_trace_has_motion = _mod._trace_has_motion
_evaluate_single_trace = _mod.evaluate_single_trace


def test_extract_trace_steps_converts_frames_key() -> None:
    """Extraction renames 'frames' to a step list with integer pedestrian IDs."""
    trace = {
        "frames": [
            {
                "step": 0,
                "time_s": 0.0,
                "robot": {"position": [0.0, 0.0]},
                "pedestrians": [{"id": "ped_1", "position": [1.0, 0.5], "velocity": [0.0, 0.0]}],
            },
            {
                "step": 1,
                "time_s": 0.1,
                "robot": {"position": [0.01, 0.0]},
                "pedestrians": [{"id": "ped_1", "position": [1.0, 0.5], "velocity": [0.0, 0.0]}],
            },
        ]
    }
    steps = _extract_trace_steps(trace)
    assert len(steps) == 2
    assert steps[0]["pedestrians"][0]["id"] == "ped_1" or isinstance(
        steps[0]["pedestrians"][0]["id"], int
    )
    assert "robot" in steps[0]


def test_extract_trace_steps_converts_string_ids_to_int() -> None:
    """Numeric string IDs are converted to integers."""
    trace = {
        "frames": [
            {
                "step": 0,
                "time_s": 0.0,
                "pedestrians": [{"id": "42", "position": [0, 0], "velocity": [1, 0]}],
            }
        ]
    }
    steps = _extract_trace_steps(trace)
    assert steps[0]["pedestrians"][0]["id"] == 42


def test_extract_trace_steps_hashes_non_numeric_string_ids() -> None:
    """Non-numeric string IDs are deterministically hashed to integers."""
    trace = {
        "frames": [
            {
                "step": 0,
                "time_s": 0.0,
                "pedestrians": [{"id": "ped-abc", "position": [0, 0], "velocity": [0, 0]}],
            }
        ]
    }
    steps = _extract_trace_steps(trace)
    assert isinstance(steps[0]["pedestrians"][0]["id"], int)
    assert (
        steps[0]["pedestrians"][0]["id"] == _extract_trace_steps(trace)[0]["pedestrians"][0]["id"]
    )


def test_compute_dt_s_infers_from_timestamps() -> None:
    """dt_s is inferred from the first two frames' time_s values."""
    trace = {
        "frames": [
            {"time_s": 0.0},
            {"time_s": 0.5},
            {"time_s": 1.0},
        ]
    }
    assert _compute_dt_s(trace) == pytest.approx(0.5)


def test_compute_dt_s_defaults_for_single_frame() -> None:
    """Single-frame traces default to 0.1s dt."""
    trace = {"frames": [{"time_s": 0.0}]}
    assert _compute_dt_s(trace) == 0.1


def test_trace_has_motion_detects_velocity() -> None:
    """Motion detection finds non-zero pedestrian velocity."""
    moving = {
        "frames": [
            {"pedestrians": [{"velocity": [0.1, 0.0]}]},
        ]
    }
    assert _trace_has_motion(moving) is True

    still = {
        "frames": [
            {"pedestrians": [{"velocity": [0.0, 0.0]}]},
        ]
    }
    assert _trace_has_motion(still) is False


def test_pedestrian_count_returns_max() -> None:
    """Pedestrian count returns the maximum across frames."""
    trace = {
        "frames": [
            {"pedestrians": [{"id": 1}]},
            {"pedestrians": [{"id": 1}, {"id": 2}, {"id": 3}]},
            {"pedestrians": [{"id": 1}, {"id": 2}]},
        ]
    }
    assert _pedestrian_count(trace) == 3


def test_actor_class_counts_default_legacy_pedestrians_and_count_cyclists() -> None:
    """Trace metadata exposes actor classes for denominator treatment."""
    trace = {
        "frames": [
            {
                "pedestrians": [
                    {"id": 1},
                    {"id": 2, "actor_type": "cyclist_like_vru"},
                ]
            },
            {
                "pedestrians": [
                    {"id": 1, "actor_type": "pedestrian"},
                    {"id": 2, "actor_type": " cyclist_like_vru "},
                    {"id": 3, "actor_type": "Pedestrian"},
                    {"id": 4, "actor_type": "Bicycle/VRU"},
                ]
            },
        ]
    }

    assert _get_actor_classes(trace) == ["bicycle_vru", "cyclist_like_vru", "pedestrian"]
    assert _actor_class_counts(trace) == {
        "bicycle_vru": 1,
        "cyclist_like_vru": 2,
        "pedestrian": 3,
    }


def test_evaluate_single_trace_missing_file() -> None:
    """Missing trace file produces trace_file_missing status."""
    result = _evaluate_single_trace(
        {
            "family": "test",
            "label": "missing",
            "path": "nonexistent/path.json",
            "scenario_id": "test",
            "planner_id": "test",
            "seed": 0,
        }
    )
    assert result["status"] == "trace_file_missing"
    assert result["metrics"]["forecast_evaluable_samples"] == 0.0


def test_evaluate_single_trace_with_motion() -> None:
    """Trace with motion produces evaluated status."""
    result = _evaluate_single_trace(
        {
            "family": "corridor_interaction",
            "label": "test_default_sf",
            "path": "docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/default_social_force_trace_export.json",
            "scenario_id": "classic_head_on_corridor_low",
            "planner_id": "default_social_force",
            "seed": 111,
        }
    )
    assert result["status"] in ("evaluated", "limited_no_pedestrian_motion")
    assert "metrics" in result


def test_evaluate_all_candidate_traces() -> None:
    """All defined trace candidates can be evaluated without crashing."""
    for candidate in TRACE_CANDIDATES:
        result = _evaluate_single_trace(candidate)
        assert "status" in result
        assert "metrics" in result
        assert result["status"] != "evaluation_error", (
            f"Trace {candidate['label']} failed: {result.get('error', 'unknown')}"
        )


def test_build_failure_cases_extracts_high_miss_rate() -> None:
    """Failure cases are extracted for traces with high miss rate."""
    results = [
        {
            "label": "test_trace",
            "family": "test_family",
            "trace_path": "test/path.json",
            "status": "evaluated",
            "metrics": {
                "forecast_evaluable_samples": 10.0,
                "mean_miss_rate_2s": 0.8,
                "mean_ade_2s": 0.2,
                "mean_negative_log_likelihood_2s": 2.0,
            },
        }
    ]
    cases = _build_failure_cases(results)
    assert len(cases) >= 1
    assert any(c["metric"] == "mean_miss_rate_2s" for c in cases)


def test_build_failure_cases_empty_for_good_results() -> None:
    """No failure cases for traces with low miss rate and ADE."""
    results = [
        {
            "label": "good_trace",
            "family": "test",
            "trace_path": "test/path.json",
            "status": "evaluated",
            "metrics": {
                "forecast_evaluable_samples": 10.0,
                "mean_miss_rate_2s": 0.05,
                "mean_ade_2s": 0.1,
                "mean_negative_log_likelihood_2s": 1.0,
            },
        }
    ]
    cases = _build_failure_cases(results)
    assert len(cases) == 0


def test_generate_markdown_includes_claim_boundary() -> None:
    """Markdown report includes the diagnostic-only claim boundary."""
    md = _generate_markdown(
        [],
        [],
        {
            "issue": 2757,
            "generated_at_utc": "test",
            "command": "test",
            "repo_head": "abc123",
            "horizons_s": [0.5, 1.0, 2.0],
        },
    )
    assert "Diagnostic-only" in md
    assert "2757" in md


def test_generate_markdown_lists_missing_families() -> None:
    """Markdown report lists missing trace families."""
    md = _generate_markdown(
        [],
        [],
        {
            "issue": 2757,
            "generated_at_utc": "test",
            "command": "test",
            "repo_head": "abc",
            "horizons_s": [0.5, 1.0, 2.0],
        },
    )
    for mf in MISSING_FAMILIES:
        assert mf["family"] in md


def test_generate_markdown_lists_evaluated_traces() -> None:
    """Markdown report includes evaluated traces in the table."""
    results = [
        {
            "family": "corridor_interaction",
            "label": "test",
            "status": "evaluated",
            "has_motion": True,
            "frame_count": 20,
            "pedestrians_per_frame": 2,
            "dt_s": 0.1,
            "scenario_id": "test_scenario",
            "metrics": {"forecast_evaluable_samples": 5.0},
            "horizons_s": [0.5, 1.0, 2.0],
        }
    ]
    md = _generate_markdown(
        results,
        [],
        {
            "issue": 2757,
            "generated_at_utc": "test",
            "command": "test",
            "repo_head": "abc",
            "horizons_s": [0.5, 1.0, 2.0],
        },
    )
    assert "corridor_interaction" in md
    assert "test" in md
    assert "Not available for this trace length" in md


def test_generate_markdown_includes_failure_cases() -> None:
    """Markdown report includes failure case details."""
    failure_cases = [
        {
            "trace": "test_trace",
            "family": "test_family",
            "trace_path": "test/path.json",
            "metric": "mean_miss_rate_2s",
            "value": 0.8,
            "interpretation": "High miss rate.",
        }
    ]
    md = _generate_markdown(
        [],
        failure_cases,
        {
            "issue": 2757,
            "generated_at_utc": "test",
            "command": "test",
            "repo_head": "abc",
            "horizons_s": [0.5, 1.0, 2.0],
        },
    )
    assert "Failure Cases" in md
    assert "test_trace" in md
    assert "test/path.json" in md


def test_occluded_emergence_candidate_is_evaluated_with_samples() -> None:
    """The occluded_emergence trace produces evaluated status with evaluable samples."""
    candidate = next(c for c in TRACE_CANDIDATES if c["family"] == "occluded_emergence")
    result = _evaluate_single_trace(candidate)
    assert result["status"] == "evaluated", (
        f"occluded_emergence expected 'evaluated', got '{result['status']}'"
    )
    assert result["metrics"]["forecast_evaluable_samples"] > 0, (
        "occluded_emergence should have forecast_evaluable_samples > 0"
    )
    assert result["has_motion"] is True


def test_non_corridor_evaluated_families_appear_in_gap_summary() -> None:
    """Evaluated non-corridor families are in the evaluated set, not limited or missing."""
    results = []
    for candidate in TRACE_CANDIDATES:
        results.append(_evaluate_single_trace(candidate))

    evaluated_families = sorted({r["family"] for r in results if r["status"] == "evaluated"})
    limited_families = sorted({r["family"] for r in results if r["status"] != "evaluated"})

    assert "corridor_interaction" in evaluated_families
    assert "occluded_emergence" in evaluated_families
    assert "occluded_emergence" not in limited_families
    assert "occluded_emergence" not in [mf["family"] for mf in MISSING_FAMILIES]


def test_report_json_distinguishes_evaluated_limited_missing() -> None:
    """JSON report separates evaluated, limited, and missing trace families."""
    results = []
    for candidate in TRACE_CANDIDATES:
        results.append(_evaluate_single_trace(candidate))

    evaluated_families = sorted({r["family"] for r in results if r["status"] == "evaluated"})
    limited_families = sorted({r["family"] for r in results if r["status"] != "evaluated"})

    assert "corridor_interaction" in evaluated_families
    assert "occluded_emergence" in evaluated_families

    for fam in ["crossing_proxy", "bottleneck"]:
        assert fam in limited_families, (
            f"{fam} should be in limited families, got {limited_families}"
        )

    assert "occluded_emergence" not in limited_families
    assert "occluded_emergence" not in [mf["family"] for mf in MISSING_FAMILIES]


def test_evaluate_single_trace_with_signal_aware_baseline() -> None:
    """Trace can be evaluated with signal_aware baseline."""
    result = _evaluate_single_trace(
        TRACE_CANDIDATES[0],
        baseline_function=signal_aware_cv_baseline,
    )
    assert "status" in result
    assert "metrics" in result
    assert result["status"] != "evaluation_error"


def test_evaluate_single_trace_with_goal_aware_baseline() -> None:
    """Trace can be evaluated with goal_aware baseline."""
    result = _evaluate_single_trace(
        TRACE_CANDIDATES[0],
        baseline_function=goal_aware_cv_baseline,
    )
    assert "status" in result
    assert "metrics" in result
    assert result["status"] != "evaluation_error"


def test_evaluate_single_trace_with_semantic_baseline() -> None:
    """Trace can be evaluated with semantic baseline."""
    result = _evaluate_single_trace(
        TRACE_CANDIDATES[0],
        baseline_function=semantic_cv_baseline,
    )
    assert "status" in result
    assert "metrics" in result
    assert result["status"] != "evaluation_error"


def test_baseline_functions_dict_has_all_keys() -> None:
    """BASELINE_FUNCTIONS dict contains all expected baseline names."""
    assert "cv" in BASELINE_FUNCTIONS
    assert "signal_aware" in BASELINE_FUNCTIONS
    assert "goal_aware" in BASELINE_FUNCTIONS
    assert "semantic" in BASELINE_FUNCTIONS
    assert "interaction_aware" in BASELINE_FUNCTIONS


def test_evaluate_single_trace_with_interaction_aware_baseline() -> None:
    """Trace can be evaluated with interaction_aware baseline."""
    result = _evaluate_single_trace(
        TRACE_CANDIDATES[0],
        baseline_function=interaction_aware_cv_baseline,
    )
    assert "status" in result
    assert "metrics" in result
    assert result["status"] != "evaluation_error"


def test_summarize_interaction_effect_reports_matched_deltas() -> None:
    """Interaction summary compares matched interaction-aware and CV rows."""
    summary = _summarize_interaction_effect(
        [
            {
                "baseline": "cv",
                "family": "corridor_interaction",
                "label": "default",
                "status": "evaluated",
                "mean_ade_1s": 0.1,
                "mean_negative_log_likelihood_1s": 2.0,
            },
            {
                "baseline": "interaction_aware",
                "family": "corridor_interaction",
                "label": "default",
                "status": "evaluated",
                "mean_ade_1s": 0.2,
                "mean_negative_log_likelihood_1s": 1.5,
            },
        ]
    )

    assert summary is not None
    assert summary["matched_rows"] == 1
    assert summary["mean_ade_1s_delta_vs_cv"] == pytest.approx(0.1)
    assert summary["mean_nll_1s_delta_vs_cv"] == pytest.approx(-0.5)
    assert "improved Gaussian likelihood" in summary["conclusion"]
