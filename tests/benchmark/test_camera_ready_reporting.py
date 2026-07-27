"""Focused unit tests for camera-ready reporting helpers and builders.

These tests target the pure helpers and report builders in
``robot_sf.benchmark.camera_ready._reporting``, emphasizing corner cases
(NaN/None/empty handling, fail-closed credibility factors, identifier
fallbacks, empty-vs-absent distinctions, and conditional comparisons) that are
not already asserted directly by the broader campaign tests in
``test_camera_ready_campaign.py``.

Inputs are minimal in-memory dicts/lists plus a ``tmp_path`` for the one writer
that touches disk; no real benchmark artifacts are read.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.camera_ready._reporting import (
    _build_scenario_amv_lookup,
    _campaign_scenario_id,
    _metric_ci,
    _metric_mean,
    _normalized_algorithm_metadata_contract,
    _safe_float,
    _scenario_family,
    _strict_vs_fallback_comparisons,
    build_campaign_credibility_scorecard,
    write_campaign_report,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

# Number of canonical credibility factors defined in _reporting.
_EXPECTED_CREDIBILITY_FACTOR_IDS = (
    "verification",
    "validation",
    "input_pedigree",
    "uncertainty_characterization",
    "results_robustness",
    "use_history",
)


def _assert_nan(value: float) -> None:
    """Assert a value is a NaN (``nan != nan``)."""
    assert math.isnan(value), f"expected NaN, got {value!r}"


# ---------------------------------------------------------------------------
# _metric_mean
# ---------------------------------------------------------------------------


class TestMetricMean:
    """``_metric_mean`` extracts a mean or fails closed with NaN."""

    def test_returns_mean_from_dict_block(self) -> None:
        assert _metric_mean({"success": {"mean": 0.75}}, "success") == 0.75

    def test_coerces_int_mean_to_float(self) -> None:
        result = _metric_mean({"collisions": {"mean": 3}}, "collisions")
        assert result == 3.0
        assert isinstance(result, float)

    def test_coerces_numeric_string_mean(self) -> None:
        assert _metric_mean({"snqi": {"mean": "0.5"}}, "snqi") == 0.5

    def test_metric_block_not_a_dict_returns_nan(self) -> None:
        _assert_nan(_metric_mean({"success": 0.5}, "success"))

    def test_missing_metric_returns_nan(self) -> None:
        _assert_nan(_metric_mean({"other": {"mean": 1.0}}, "success"))

    def test_empty_block_returns_nan(self) -> None:
        _assert_nan(_metric_mean({}, "success"))

    def test_missing_mean_key_returns_nan(self) -> None:
        _assert_nan(_metric_mean({"success": {"ci": [0.1, 0.9]}}, "success"))

    def test_none_mean_returns_nan(self) -> None:
        _assert_nan(_metric_mean({"success": {"mean": None}}, "success"))

    def test_non_numeric_mean_returns_nan(self) -> None:
        _assert_nan(_metric_mean({"success": {"mean": "high"}}, "success"))

    def test_nan_mean_passes_through_as_nan(self) -> None:
        _assert_nan(_metric_mean({"success": {"mean": float("nan")}}, "success"))

    def test_metric_block_none_returns_nan(self) -> None:
        _assert_nan(_metric_mean({"success": None}, "success"))


# ---------------------------------------------------------------------------
# _metric_ci
# ---------------------------------------------------------------------------


class TestMetricCi:
    """``_metric_ci`` extracts a ``(low, high)`` tuple or fails closed with NaNs."""

    def test_returns_ci_tuple_from_block(self) -> None:
        assert _metric_ci({"success": {"mean_ci": [0.6, 0.9]}}, "success") == (0.6, 0.9)

    def test_coerces_int_ci_values_to_float(self) -> None:
        low, high = _metric_ci({"success": {"mean_ci": [0, 1]}}, "success")
        assert (low, high) == (0.0, 1.0)
        assert isinstance(low, float) and isinstance(high, float)

    def test_metric_block_not_a_dict_returns_nan_pair(self) -> None:
        low, high = _metric_ci({"success": [0.0, 1.0]}, "success")
        _assert_nan(low)
        _assert_nan(high)

    def test_missing_metric_returns_nan_pair(self) -> None:
        low, high = _metric_ci({"other": {"mean_ci": [0.0, 1.0]}}, "success")
        _assert_nan(low)
        _assert_nan(high)

    def test_empty_block_returns_nan_pair(self) -> None:
        low, high = _metric_ci({}, "success")
        _assert_nan(low)
        _assert_nan(high)

    def test_missing_mean_ci_returns_nan_pair(self) -> None:
        low, high = _metric_ci({"success": {"mean": 0.5}}, "success")
        _assert_nan(low)
        _assert_nan(high)

    def test_mean_ci_not_a_list_returns_nan_pair(self) -> None:
        low, high = _metric_ci({"success": {"mean_ci": "0.6-0.9"}}, "success")
        _assert_nan(low)
        _assert_nan(high)

    def test_mean_ci_none_returns_nan_pair(self) -> None:
        low, high = _metric_ci({"success": {"mean_ci": None}}, "success")
        _assert_nan(low)
        _assert_nan(high)

    @pytest.mark.parametrize("bad_ci", [[0.6], [0.6, 0.9, 0.95], []])
    def test_mean_ci_wrong_length_returns_nan_pair(self, bad_ci: list[float]) -> None:
        low, high = _metric_ci({"success": {"mean_ci": bad_ci}}, "success")
        _assert_nan(low)
        _assert_nan(high)

    def test_non_numeric_ci_values_return_nan_pair(self) -> None:
        low, high = _metric_ci({"success": {"mean_ci": ["low", "high"]}}, "success")
        _assert_nan(low)
        _assert_nan(high)

    def test_partially_non_numeric_ci_returns_nan_pair(self) -> None:
        low, high = _metric_ci({"success": {"mean_ci": [0.6, None]}}, "success")
        _assert_nan(low)
        _assert_nan(high)

    def test_nan_ci_values_pass_through(self) -> None:
        low, high = _metric_ci({"success": {"mean_ci": [float("nan"), float("nan")]}}, "success")
        _assert_nan(low)
        _assert_nan(high)


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    """``_safe_float`` formats floats with NaN/None/non-numeric handling."""

    def test_formats_positive_float(self) -> None:
        assert _safe_float(0.75) == "0.7500"

    def test_formats_negative_float(self) -> None:
        assert _safe_float(-1.5) == "-1.5000"

    def test_formats_zero(self) -> None:
        assert _safe_float(0.0) == "0.0000"

    def test_truncates_to_four_decimals(self) -> None:
        assert _safe_float(0.123456789) == "0.1235"

    def test_coerces_int(self) -> None:
        assert _safe_float(3) == "3.0000"

    def test_coerces_numeric_string(self) -> None:
        assert _safe_float("2.5") == "2.5000"

    def test_none_returns_nan_string(self) -> None:
        assert _safe_float(None) == "nan"

    def test_nan_float_returns_nan_string(self) -> None:
        assert _safe_float(float("nan")) == "nan"

    def test_non_numeric_string_returns_nan_string(self) -> None:
        assert _safe_float("not-a-number") == "nan"

    def test_bool_true_is_formatted_as_one(self) -> None:
        # bool is a subclass of int; float(True) == 1.0. This documents the
        # coercion behavior rather than treating it as a NaN edge case.
        assert _safe_float(True) == "1.0000"

    def test_inf_is_not_nan_and_formats_value(self) -> None:
        # math.isnan(inf) is False, so inf is formatted, not collapsed to "nan".
        assert _safe_float(float("inf")) == "inf"


# ---------------------------------------------------------------------------
# build_campaign_credibility_scorecard (fail-closed factors)
# ---------------------------------------------------------------------------


def _scorecard_factor(scorecard: dict[str, Any], factor_id: str) -> dict[str, Any]:
    for factor in scorecard["factors"]:
        if factor["factor_id"] == factor_id:
            return factor
    raise AssertionError(f"factor {factor_id!r} not present in scorecard")


class TestCampaignCredibilityScorecard:
    """``build_campaign_credibility_scorecard`` is fail-closed and score-driven."""

    def test_overall_score_none_when_no_assessed_factors(self) -> None:
        scorecard = build_campaign_credibility_scorecard(
            {"campaign": {"campaign_id": "c1"}, "planner_rows": [], "artifacts": {}}
        )
        assert scorecard["overall_score"] is None
        assert scorecard["overall_status"] == "not_assessed"
        assert scorecard["schema_version"] == "campaign_credibility_scorecard.v1"
        assert scorecard["campaign_id"] == "c1"
        # No evidence supplied -> every factor stays not_assessed with null score.
        assert [f["factor_id"] for f in scorecard["factors"]] == list(
            _EXPECTED_CREDIBILITY_FACTOR_IDS
        )
        for factor in scorecard["factors"]:
            assert factor["status"] == "not_assessed"
            assert factor["score"] is None

    def test_overall_score_none_when_payload_missing_keys(self) -> None:
        scorecard = build_campaign_credibility_scorecard({})
        assert scorecard["overall_score"] is None
        assert scorecard["overall_status"] == "not_assessed"
        # Defaults campaign_id when campaign is absent/not a dict.
        assert scorecard["campaign_id"] == "unknown"

    def test_non_dict_campaign_and_artifacts_are_ignored(self) -> None:
        scorecard = build_campaign_credibility_scorecard(
            {"campaign": "not-a-dict", "artifacts": [], "planner_rows": "x"}
        )
        assert scorecard["overall_score"] is None
        assert scorecard["overall_status"] == "not_assessed"
        for factor in scorecard["factors"]:
            assert factor["status"] == "not_assessed"

    def test_input_pedigree_partial_when_git_and_matrix_hash_present(self) -> None:
        scorecard = build_campaign_credibility_scorecard(
            {
                "campaign": {
                    "campaign_id": "c1",
                    "git_hash": "abc123",
                    "scenario_matrix_hash": "sha256:matrix",
                },
                "artifacts": {"campaign_manifest": "manifest.json"},
            }
        )
        pedigree = _scorecard_factor(scorecard, "input_pedigree")
        assert pedigree["status"] == "partial"
        assert pedigree["score"] == 2
        assert "git_hash=abc123" in pedigree["evidence"]
        assert "scenario_matrix_hash=sha256:matrix" in pedigree["evidence"]

    def test_input_pedigree_stays_not_assessed_with_only_git_hash(self) -> None:
        scorecard = build_campaign_credibility_scorecard({"campaign": {"git_hash": "abc123"}})
        pedigree = _scorecard_factor(scorecard, "input_pedigree")
        assert pedigree["status"] == "not_assessed"
        assert pedigree["score"] is None

    def test_uncertainty_score_two_when_seed_count_above_one(self) -> None:
        scorecard = build_campaign_credibility_scorecard({"campaign": {"seed_count": 4}})
        uncertainty = _scorecard_factor(scorecard, "uncertainty_characterization")
        assert uncertainty["status"] == "partial"
        assert uncertainty["score"] == 2

    def test_uncertainty_score_one_when_single_seed_with_variability_artifact(self) -> None:
        scorecard = build_campaign_credibility_scorecard(
            {
                "campaign": {"seed_count": 1},
                "artifacts": {"seed_variability_json": "seed_var.json"},
            }
        )
        uncertainty = _scorecard_factor(scorecard, "uncertainty_characterization")
        assert uncertainty["status"] == "weak"
        assert uncertainty["score"] == 1

    def test_uncertainty_not_assessed_when_single_seed_without_variability(self) -> None:
        scorecard = build_campaign_credibility_scorecard(
            {"campaign": {"seed_count": 1}, "artifacts": {}}
        )
        uncertainty = _scorecard_factor(scorecard, "uncertainty_characterization")
        assert uncertainty["status"] == "not_assessed"
        assert uncertainty["score"] is None

    def test_uncertainty_treats_non_numeric_seed_count_as_zero(self) -> None:
        scorecard = build_campaign_credibility_scorecard(
            {"campaign": {"seed_count": "many"}, "artifacts": {}}
        )
        uncertainty = _scorecard_factor(scorecard, "uncertainty_characterization")
        assert uncertainty["status"] == "not_assessed"
        assert uncertainty["score"] is None

    def test_verification_weak_when_summary_and_table_artifacts_present(self) -> None:
        scorecard = build_campaign_credibility_scorecard(
            {
                "artifacts": {
                    "campaign_summary_json": "summary.json",
                    "campaign_table_csv": "table.csv",
                }
            }
        )
        verification = _scorecard_factor(scorecard, "verification")
        assert verification["status"] == "weak"
        assert verification["score"] == 1

    def test_verification_not_assessed_with_only_summary(self) -> None:
        scorecard = build_campaign_credibility_scorecard(
            {"artifacts": {"campaign_summary_json": "summary.json"}}
        )
        verification = _scorecard_factor(scorecard, "verification")
        assert verification["status"] == "not_assessed"
        assert verification["score"] is None

    def test_results_robustness_weak_when_at_least_one_ok_row(self) -> None:
        scorecard = build_campaign_credibility_scorecard(
            {
                "planner_rows": [
                    {"status": "ok"},
                    {"status": "failed"},
                ]
            }
        )
        robustness = _scorecard_factor(scorecard, "results_robustness")
        assert robustness["status"] == "weak"
        assert robustness["score"] == 1

    def test_results_robustness_zero_when_all_rows_failed(self) -> None:
        scorecard = build_campaign_credibility_scorecard(
            {"planner_rows": [{"status": "failed"}, {"status": "failed"}]}
        )
        robustness = _scorecard_factor(scorecard, "results_robustness")
        assert robustness["status"] == "weak"
        assert robustness["score"] == 0

    def test_overall_status_reflects_mean_of_assessed_scores(self) -> None:
        # input_pedigree (2) + uncertainty (2) -> mean 2.0 -> "partial".
        scorecard = build_campaign_credibility_scorecard(
            {
                "campaign": {
                    "git_hash": "abc123",
                    "scenario_matrix_hash": "sha256:matrix",
                    "seed_count": 5,
                }
            }
        )
        assert scorecard["overall_score"] == 2.0
        assert scorecard["overall_status"] == "partial"

    def test_overall_status_partial_for_mean_between_one_and_two(self) -> None:
        # input_pedigree (2) + verification (1) -> mean 1.5 -> round 2 -> partial.
        scorecard = build_campaign_credibility_scorecard(
            {
                "campaign": {
                    "git_hash": "abc123",
                    "scenario_matrix_hash": "sha256:matrix",
                },
                "artifacts": {
                    "campaign_summary_json": "summary.json",
                    "campaign_table_csv": "table.csv",
                },
            }
        )
        assert scorecard["overall_score"] == 1.5
        assert scorecard["overall_status"] == "partial"

    def test_overall_status_weak_for_mean_below_one_point_five(self) -> None:
        # verification (1) + results_robustness (1) -> mean 1.0 -> "weak".
        scorecard = build_campaign_credibility_scorecard(
            {
                "artifacts": {
                    "campaign_summary_json": "summary.json",
                    "campaign_table_csv": "table.csv",
                },
                "planner_rows": [{"status": "ok"}],
            }
        )
        assert scorecard["overall_score"] == 1.0
        assert scorecard["overall_status"] == "weak"

    def test_claim_boundary_is_present_and_explicit(self) -> None:
        scorecard = build_campaign_credibility_scorecard({})
        assert "benchmark proof" in scorecard["claim_boundary"]
        assert "not" in scorecard["claim_boundary"].lower()

    def test_use_history_factor_always_not_assessed(self) -> None:
        # No campaign evidence path upgrades use_history; it stays fail-closed.
        scorecard = build_campaign_credibility_scorecard(
            {
                "campaign": {
                    "git_hash": "abc123",
                    "scenario_matrix_hash": "sha256:matrix",
                    "seed_count": 5,
                },
                "artifacts": {
                    "campaign_summary_json": "summary.json",
                    "campaign_table_csv": "table.csv",
                },
                "planner_rows": [{"status": "ok"}],
            }
        )
        use_history = _scorecard_factor(scorecard, "use_history")
        assert use_history["status"] == "not_assessed"
        assert use_history["score"] is None


# ---------------------------------------------------------------------------
# _scenario_family / _campaign_scenario_id fallbacks
# ---------------------------------------------------------------------------


class TestScenarioFamily:
    """``_scenario_family`` resolves a label from layered metadata with fallbacks."""

    def test_prefers_metadata_archetype(self) -> None:
        record = {"scenario_params": {"metadata": {"archetype": "corridor"}}}
        assert _scenario_family(record) == "corridor"

    def test_uses_scenario_family_key_in_metadata(self) -> None:
        record = {"scenario_params": {"metadata": {"scenario_family": "crossing"}}}
        assert _scenario_family(record) == "crossing"

    def test_uses_family_key_in_metadata(self) -> None:
        record = {"scenario_params": {"metadata": {"family": "overtaking"}}}
        assert _scenario_family(record) == "overtaking"

    def test_archetype_takes_precedence_over_family(self) -> None:
        record = {"scenario_params": {"metadata": {"archetype": "first", "family": "second"}}}
        assert _scenario_family(record) == "first"

    def test_falls_back_to_scenario_params_top_level(self) -> None:
        record = {"scenario_params": {"archetype": "hallway"}}
        assert _scenario_family(record) == "hallway"

    def test_falls_back_to_record_top_level(self) -> None:
        record = {"archetype": "queue"}
        assert _scenario_family(record) == "queue"

    def test_strips_whitespace_from_label(self) -> None:
        record = {"scenario_params": {"metadata": {"archetype": "  corridor  "}}}
        assert _scenario_family(record) == "corridor"

    def test_ignores_whitespace_only_label(self) -> None:
        record = {"scenario_params": {"metadata": {"archetype": "   "}}}
        # Whitespace-only archetype skipped; falls to scenario_id prefix.
        record["scenario_id"] = "corridor_low_density_01"
        assert _scenario_family(record) == "corridor"

    def test_ignores_non_string_label(self) -> None:
        record = {"scenario_params": {"metadata": {"archetype": 42}}}
        record["scenario_id"] = "crossing_02"
        assert _scenario_family(record) == "crossing"

    def test_falls_back_to_scenario_id_prefix(self) -> None:
        record = {"scenario_id": "narrow_doorway_42"}
        assert _scenario_family(record) == "narrow"

    def test_scenario_id_prefix_splits_on_first_underscore(self) -> None:
        record = {"scenario_id": "t_junction_42"}
        assert _scenario_family(record) == "t"

    def test_scenario_id_prefix_is_not_stripped_of_padding(self) -> None:
        # The truthiness ``.strip()`` check only guards empty strings; the
        # returned prefix keeps any surrounding whitespace from scenario_id.
        record = {"scenario_id": "  hallway_7  "}
        assert _scenario_family(record) == "  hallway"

    def test_returns_unknown_when_nothing_resolvable(self) -> None:
        assert _scenario_family({}) == "unknown"

    def test_scenario_params_not_a_dict_is_treated_as_empty(self) -> None:
        record = {"scenario_params": "not-a-dict", "scenario_id": "corridor_1"}
        assert _scenario_family(record) == "corridor"

    def test_metadata_not_a_dict_is_treated_as_empty(self) -> None:
        record = {
            "scenario_params": {"metadata": "nope"},
            "scenario_id": "crossing_2",
        }
        assert _scenario_family(record) == "crossing"


class TestCampaignScenarioId:
    """``_campaign_scenario_id`` resolves a stable identifier with precedence."""

    def test_prefers_name(self) -> None:
        scenario = {"name": "corridor_low", "scenario_id": "s1", "id": "i1"}
        assert _campaign_scenario_id(scenario) == "corridor_low"

    def test_uses_scenario_id_when_name_absent(self) -> None:
        scenario = {"scenario_id": "s1", "id": "i1"}
        assert _campaign_scenario_id(scenario) == "s1"

    def test_uses_id_when_name_and_scenario_id_absent(self) -> None:
        scenario = {"id": "i1"}
        assert _campaign_scenario_id(scenario) == "i1"

    def test_strips_whitespace(self) -> None:
        assert _campaign_scenario_id({"name": "  hallway  "}) == "hallway"

    def test_ignores_whitespace_only_name(self) -> None:
        scenario = {"name": "   ", "scenario_id": "s1"}
        assert _campaign_scenario_id(scenario) == "s1"

    def test_ignores_non_string_name(self) -> None:
        scenario = {"name": 123, "id": "i1"}
        assert _campaign_scenario_id(scenario) == "i1"

    def test_returns_unknown_when_nothing_resolvable(self) -> None:
        assert _campaign_scenario_id({}) == "unknown"

    def test_returns_unknown_when_all_whitespace(self) -> None:
        assert _campaign_scenario_id({"name": "  ", "scenario_id": "  ", "id": "  "}) == "unknown"


# ---------------------------------------------------------------------------
# _build_scenario_amv_lookup (empty-vs-absent distinction)
# ---------------------------------------------------------------------------


class TestBuildScenarioAmvLookup:
    """``_build_scenario_amv_lookup`` distinguishes empty AMV from absent scenarios."""

    def test_scenario_without_amv_maps_to_empty_dict(self) -> None:
        lookup = _build_scenario_amv_lookup([{"name": "bare"}])
        assert lookup == {"bare": {}}

    def test_empty_input_returns_empty_lookup(self) -> None:
        assert _build_scenario_amv_lookup([]) == {}

    def test_scenario_with_amv_maps_to_taxonomy(self) -> None:
        lookup = _build_scenario_amv_lookup(
            [{"name": "corridor_low", "amv": {"use_case": "corridor", "context": "low_density"}}]
        )
        assert lookup["corridor_low"] == {"use_case": "corridor", "context": "low_density"}

    def test_absent_scenario_id_is_not_present_in_lookup(self) -> None:
        lookup = _build_scenario_amv_lookup([{"name": "present"}])
        # A scenario that was never supplied must be absent, not mapped to {}.
        assert "absent" not in lookup
        assert lookup.get("absent") is None

    def test_unknown_scenario_id_resolves_to_unknown_key(self) -> None:
        # Scenarios with no resolvable identifier collapse to "unknown".
        lookup = _build_scenario_amv_lookup([{}])
        assert lookup == {"unknown": {}}

    def test_multiple_scenarios_use_their_resolved_ids(self) -> None:
        lookup = _build_scenario_amv_lookup(
            [
                {"name": "a", "amv": {"use_case": "corridor"}},
                {"scenario_id": "b"},
                {"id": "c", "amv": {"context": "high_density"}},
            ]
        )
        assert lookup["a"] == {"use_case": "corridor"}
        assert lookup["b"] == {}
        assert lookup["c"] == {"context": "high_density"}


# ---------------------------------------------------------------------------
# _strict_vs_fallback_comparisons (only when both modes present)
# ---------------------------------------------------------------------------


class TestStrictVsFallbackComparisons:
    """``_strict_vs_fallback_comparisons`` emits lines only for paired modes."""

    def test_empty_when_no_rows(self) -> None:
        assert _strict_vs_fallback_comparisons([]) == []

    def test_empty_when_only_strict_mode_present(self) -> None:
        rows = [{"algo": "orca", "socnav_prereq_policy": "fail-fast"}]
        assert _strict_vs_fallback_comparisons(rows) == []

    def test_empty_when_only_fallback_mode_present(self) -> None:
        rows = [{"algo": "orca", "socnav_prereq_policy": "fallback"}]
        assert _strict_vs_fallback_comparisons(rows) == []

    def test_emits_line_when_both_modes_present_for_one_algo(self) -> None:
        rows = [
            {
                "algo": "orca",
                "socnav_prereq_policy": "fail-fast",
                "preflight_status": "pass",
                "success_mean": "0.9000",
            },
            {
                "algo": "orca",
                "socnav_prereq_policy": "fallback",
                "preflight_status": "warn",
                "success_mean": "0.8000",
            },
        ]
        lines = _strict_vs_fallback_comparisons(rows)
        assert len(lines) == 1
        line = lines[0]
        assert "`orca`" in line
        assert "strict preflight=pass" in line
        assert "fallback preflight=warn" in line
        assert "strict success=0.9000" in line
        assert "fallback success=0.8000" in line

    def test_uses_unknown_algo_when_missing(self) -> None:
        rows = [
            {"socnav_prereq_policy": "fail-fast", "success_mean": "0.9000"},
            {"socnav_prereq_policy": "fallback", "success_mean": "0.8000"},
        ]
        lines = _strict_vs_fallback_comparisons(rows)
        assert len(lines) == 1
        assert "`unknown`" in lines[0]

    def test_skips_algos_without_a_full_pair(self) -> None:
        rows = [
            {"algo": "orca", "socnav_prereq_policy": "fail-fast", "success_mean": "0.9000"},
            {"algo": "orca", "socnav_prereq_policy": "fallback", "success_mean": "0.8000"},
            # different_algo has only strict -> no comparison.
            {"algo": "different_algo", "socnav_prereq_policy": "fail-fast"},
            # lone_algo has neither -> no comparison.
            {"algo": "lone_algo"},
        ]
        lines = _strict_vs_fallback_comparisons(rows)
        assert len(lines) == 1
        assert all("orca" in line for line in lines)
        assert not any("different_algo" in line for line in lines)
        assert not any("lone_algo" in line for line in lines)

    def test_emits_one_line_per_algo_with_a_pair(self) -> None:
        rows = [
            {"algo": "alpha", "socnav_prereq_policy": "fail-fast", "success_mean": "0.9"},
            {"algo": "alpha", "socnav_prereq_policy": "fallback", "success_mean": "0.8"},
            {"algo": "beta", "socnav_prereq_policy": "fail-fast", "success_mean": "0.7"},
            {"algo": "beta", "socnav_prereq_policy": "fallback", "success_mean": "0.6"},
        ]
        lines = _strict_vs_fallback_comparisons(rows)
        assert len(lines) == 2
        # Lines are sorted alphabetically by algo.
        assert "`alpha`" in lines[0]
        assert "`beta`" in lines[1]

    def test_treats_missing_policy_as_neither_mode(self) -> None:
        rows = [
            {"algo": "orca", "socnav_prereq_policy": "fail-fast"},
            {"algo": "orca", "socnav_prereq_policy": "something-else"},
        ]
        assert _strict_vs_fallback_comparisons(rows) == []


# ---------------------------------------------------------------------------
# _normalized_algorithm_metadata_contract (non-dict fallback)
# ---------------------------------------------------------------------------


class TestNormalizedAlgorithmMetadataContract:
    """``_normalized_algorithm_metadata_contract`` normalizes to a dict."""

    def test_returns_contract_when_dict(self) -> None:
        summary = {"algorithm_metadata_contract": {"planner_kinematics": {"x": 1}}}
        assert _normalized_algorithm_metadata_contract(summary) == {"planner_kinematics": {"x": 1}}

    def test_returns_empty_dict_when_contract_missing(self) -> None:
        assert _normalized_algorithm_metadata_contract({}) == {}

    def test_returns_empty_dict_when_contract_is_none(self) -> None:
        assert _normalized_algorithm_metadata_contract({"algorithm_metadata_contract": None}) == {}

    def test_returns_empty_dict_when_contract_is_a_string(self) -> None:
        assert (
            _normalized_algorithm_metadata_contract({"algorithm_metadata_contract": "kinematics"})
            == {}
        )

    def test_returns_empty_dict_when_contract_is_a_list(self) -> None:
        assert (
            _normalized_algorithm_metadata_contract({"algorithm_metadata_contract": ["a", "b"]})
            == {}
        )

    def test_crashes_when_summary_is_not_a_dict(self) -> None:
        # The non-dict fallback covers the *contract* value, not the summary:
        # passing a non-dict summary is a type violation that raises rather
        # than silently returning an empty contract.
        with pytest.raises(AttributeError):
            _normalized_algorithm_metadata_contract(None)  # type: ignore[arg-type]

    def test_returns_same_dict_identity_for_valid_contract(self) -> None:
        # When the contract is already a dict, the same object is returned
        # (no defensive copy); document this aliasing behavior explicitly.
        contract = {"planner_kinematics": {"execution_detail": "diff"}}
        summary = {"algorithm_metadata_contract": contract}
        result = _normalized_algorithm_metadata_contract(summary)
        assert result is contract
        assert result["planner_kinematics"]["execution_detail"] == "diff"


# ---------------------------------------------------------------------------
# write_campaign_report (minimal in-memory payload, tmp_path writer)
# ---------------------------------------------------------------------------


class TestWriteCampaignReport:
    """``write_campaign_report`` writes a Markdown report from a minimal payload."""

    def test_writes_markdown_file_to_disk(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.md"
        write_campaign_report(report_path, {"campaign": {"campaign_id": "c1"}})
        assert report_path.exists()
        text = report_path.read_text(encoding="utf-8")
        assert text.startswith("# Camera-Ready Benchmark Campaign Report")
        assert "Campaign ID: `c1`" in text

    def test_renders_not_assessed_scorecard_when_no_scorecard_supplied(
        self, tmp_path: Path
    ) -> None:
        report_path = tmp_path / "report.md"
        write_campaign_report(report_path, {})
        text = report_path.read_text(encoding="utf-8")
        assert "Overall status: `not_assessed`" in text
        assert "Overall score: `None`" in text

    def test_renders_supplied_scorecard(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.md"
        scorecard = {
            "schema_version": "campaign_credibility_scorecard.v1",
            "overall_status": "partial",
            "overall_score": 2.0,
            "claim_boundary": "metadata only",
            "factors": [
                {
                    "factor": "Verification",
                    "status": "weak",
                    "score": 1,
                    "justification": "structured artifacts only",
                }
            ],
        }
        write_campaign_report(report_path, {"credibility_scorecard": scorecard})
        text = report_path.read_text(encoding="utf-8")
        assert "Overall status: `partial`" in text
        assert "Overall score: `2.0`" in text
        assert "Verification" in text

    def test_empty_planner_rows_emits_no_rows_message(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.md"
        write_campaign_report(report_path, {"planner_rows": []})
        text = report_path.read_text(encoding="utf-8")
        assert "No planner rows were produced." in text

    def test_records_campaign_warnings(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.md"
        write_campaign_report(
            report_path, {"warnings": ["seed budget exceeded", "a planner degraded"]}
        )
        text = report_path.read_text(encoding="utf-8")
        assert "- seed budget exceeded" in text
        assert "- a planner degraded" in text

    def test_no_warnings_section_states_none(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.md"
        write_campaign_report(report_path, {})
        text = report_path.read_text(encoding="utf-8")
        assert "- No campaign-level warnings." in text

    def test_aggregate_integrity_block_message_when_not_valid(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.md"
        write_campaign_report(report_path, {"campaign_integrity": {"status": "invalid"}})
        text = report_path.read_text(encoding="utf-8")
        assert "Publication is blocked" in text

    def test_strict_vs_fallback_section_when_no_pair(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.md"
        # Single planner, only one policy mode -> no comparison pair available.
        write_campaign_report(
            report_path,
            {
                "planner_rows": [
                    {
                        "planner_key": "p1",
                        "algo": "orca",
                        "planner_group": "core",
                        "socnav_prereq_policy": "fail-fast",
                    }
                ]
            },
        )
        text = report_path.read_text(encoding="utf-8")
        assert "No within-campaign strict-vs-fallback pair available" in text
