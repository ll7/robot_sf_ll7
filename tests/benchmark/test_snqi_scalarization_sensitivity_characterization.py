"""Characterization baseline tests for ``robot_sf/benchmark/snqi_scalarization_sensitivity.py``.

These tests pin the *current observable behavior* of the scalarization-sensitivity
math on tiny synthetic inputs. They are table-driven and assert exact golden
values for the refactor-sensitive core: the metric-coercion helper, the
constraints-first score formula, the pairwise rank-reversal / disagreement
arithmetic, the Pareto-front classification, and the rank-disagreement /
weight-variant summaries. They also pin the module constant tables.

Purpose (issue #4881, wave 2; Refs #4874, #4770): lock a behavioral baseline so
the post-submission refactor wave can prove behavior-preservation. If a test
reveals a genuine bug, do NOT fix it here — document it and file a separate fix
issue.

These tests are additive: they lock the pure math helpers and constant tables
directly with golden values. They do not duplicate the preflight-status,
end-to-end report-export, or CLI coverage in
``test_snqi_scalarization_sensitivity.py`` and
``test_snqi_scalarization_sensitivity_issue_3653.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    BOUNDED_NORMALIZED_SENSITIVITY_METRICS,
    DEFAULT_SWEEP_FACTORS,
    OPTIONAL_SENSITIVITY_METRICS,
    OPTIONAL_WEIGHTED_SENSITIVITY_METRICS,
    REQUIRED_SENSITIVITY_METRICS,
    SCALARIZATION_SENSITIVITY_PREFLIGHT_SCHEMA,
    SCALARIZATION_SENSITIVITY_SCHEMA,
    SENSITIVITY_PREFLIGHT_BLOCKED,
    SENSITIVITY_PREFLIGHT_MALFORMED,
    SENSITIVITY_PREFLIGHT_READY,
    _constraints_first_endpoint,
    _metric_float,
    _pairwise_disagreement_rate,
    _pairwise_reversal_count,
    _pareto_points,
    _rank_disagreement,
    _variant_summary,
)

# ---------------------------------------------------------------------------
# Constant tables
# ---------------------------------------------------------------------------


def test_schema_and_status_constants_are_pinned() -> None:
    """Pin the schema-version strings and the three preflight status literals."""
    assert SCALARIZATION_SENSITIVITY_SCHEMA == "snqi_scalarization_sensitivity.v1"
    assert SCALARIZATION_SENSITIVITY_PREFLIGHT_SCHEMA == (
        "snqi_scalarization_sensitivity_preflight.v1"
    )
    assert (
        SENSITIVITY_PREFLIGHT_READY,
        SENSITIVITY_PREFLIGHT_BLOCKED,
        SENSITIVITY_PREFLIGHT_MALFORMED,
    ) == (
        "ready",
        "blocked",
        "malformed",
    )


def test_default_sweep_factors_table_is_pinned() -> None:
    """The default weight-sweep factor grid is exactly this tuple."""
    assert DEFAULT_SWEEP_FACTORS == (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def test_required_and_optional_metric_tables_are_pinned() -> None:
    """Pin the required/optional metric tuples and the weighted-optional map."""
    assert REQUIRED_SENSITIVITY_METRICS == (
        "success",
        "time_to_goal_norm",
        "collisions",
        "near_misses",
        "comfort_exposure",
    )
    assert OPTIONAL_SENSITIVITY_METRICS == ("force_exceed_events", "jerk_mean")
    assert OPTIONAL_WEIGHTED_SENSITIVITY_METRICS == {
        "force_exceed_events": "w_force_exceed",
        "jerk_mean": "w_jerk",
    }
    # Bounded-normalized terms are a subset of the required terms.
    assert BOUNDED_NORMALIZED_SENSITIVITY_METRICS == (
        "success",
        "time_to_goal_norm",
        "comfort_exposure",
    )
    assert set(BOUNDED_NORMALIZED_SENSITIVITY_METRICS).issubset(REQUIRED_SENSITIVITY_METRICS)


# ---------------------------------------------------------------------------
# _metric_float: coercion + finite guard
# ---------------------------------------------------------------------------


def test_metric_float_coerces_numeric_strings_and_numbers() -> None:
    """Numeric values (incl. strings) coerce to float; the default is unused."""
    assert _metric_float({"x": "1.5"}, "x", 0.0) == 1.5
    assert _metric_float({"x": 2}, "x", 0.0) == 2.0


def test_metric_float_bool_is_special_cased_before_default() -> None:
    """``True`` -> 1.0 and ``False`` -> 0.0; bool never falls through to default."""
    assert _metric_float({"x": True}, "x", 7.0) == 1.0
    assert _metric_float({"x": False}, "x", 7.0) == 0.0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_metric_float_non_finite_returns_default(bad_value: float) -> None:
    """Non-finite values fall through to the provided default."""
    assert _metric_float({"x": bad_value}, "x", 2.0) == 2.0


def test_metric_float_missing_or_non_numeric_returns_default() -> None:
    """Missing keys and non-numeric values both fall through to the default."""
    assert _metric_float({}, "x", 2.0) == 2.0
    assert _metric_float({"x": "bad"}, "x", 2.0) == 2.0


# ---------------------------------------------------------------------------
# _constraints_first_endpoint: exact score formula
# ---------------------------------------------------------------------------


def test_constraints_first_score_single_clean_episode() -> None:
    """One clean episode: score = 1 - 0.01*time = 0.995 for time_to_goal_norm=0.5."""
    row = {
        "success": 1.0,
        "time_to_goal_norm": 0.5,
        "collisions": 0,
        "near_misses": 0,
        "timeout": 0,
        "deadlock": 0,
    }
    out = _constraints_first_endpoint([row])
    assert out["constraints_first_score"] == pytest.approx(0.995)
    assert out["episodes"] == 1


def test_constraints_first_score_two_mixed_episodes() -> None:
    """Two episodes (one clean, one all-events) pin every penalty coefficient."""
    rows = [
        {
            "success": 1.0,
            "collisions": 0,
            "near_misses": 0,
            "timeout": 0,
            "deadlock": 0,
            "time_to_goal_norm": 0.4,
        },
        {
            "success": 0.0,
            "collisions": 2,
            "near_misses": 1,
            "timeout": 1,
            "deadlock": 1,
            "time_to_goal_norm": 0.6,
        },
    ]
    out = _constraints_first_endpoint(rows)
    assert out["success_rate"] == pytest.approx(0.5)
    assert out["collision_event_rate"] == pytest.approx(0.5)
    assert out["near_miss_event_rate"] == pytest.approx(0.5)
    assert out["timeout_rate"] == pytest.approx(0.5)
    assert out["deadlock_rate"] == pytest.approx(0.5)
    assert out["time_to_goal_norm_mean"] == pytest.approx(0.5)
    # 0.5 - 0.5 - 0.5*0.5 - 0.25*0.5 - 0.25*0.5 - 0.01*0.5 = -0.505
    assert out["constraints_first_score"] == pytest.approx(-0.505)


def test_constraints_first_score_missing_time_defaults_to_one() -> None:
    """A missing ``time_to_goal_norm`` defaults to 1.0, contributing -0.01."""
    out = _constraints_first_endpoint([{"success": 1.0}])
    assert out["time_to_goal_norm_mean"] == pytest.approx(1.0)
    assert out["constraints_first_score"] == pytest.approx(0.99)


def test_constraints_first_endpoint_field_set_is_pinned() -> None:
    """The endpoint exposes exactly the documented field set."""
    out = _constraints_first_endpoint([{"success": 1.0}])
    assert set(out) == {
        "episodes",
        "success_rate",
        "collision_event_rate",
        "near_miss_event_rate",
        "timeout_rate",
        "deadlock_rate",
        "time_to_goal_norm_mean",
        "constraints_first_score",
    }


def test_constraints_first_endpoint_empty_raises() -> None:
    """An empty episode set raises the pinned ``ValueError``."""
    with pytest.raises(ValueError, match="planner must have at least one episode"):
        _constraints_first_endpoint([])


# ---------------------------------------------------------------------------
# _pairwise_reversal_count / _pairwise_disagreement_rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (["A", "B", "C"], ["A", "B", "C"], 0),
        (["A", "B", "C"], ["C", "B", "A"], 3),
        (["A", "B", "C"], ["A", "C", "B"], 1),
        (["A", "B"], ["B", "A"], 1),
    ],
)
def test_pairwise_reversal_count_goldens(left: list[str], right: list[str], expected: int) -> None:
    """Pin the inversion count for identical, reversed, and partially-shuffled orders."""
    assert _pairwise_reversal_count(left, right) == expected


def test_pairwise_reversal_count_set_mismatch_raises() -> None:
    """Orders with different planner sets raise the pinned ``ValueError``."""
    with pytest.raises(ValueError, match="rank orders must contain the same planners"):
        _pairwise_reversal_count(["A", "B"], ["A", "C"])


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (["A", "B", "C"], ["C", "B", "A"], 1.0),
        (["A", "B", "C"], ["A", "C", "B"], 1 / 3),
        (["A", "B"], ["B", "A"], 1.0),
        (["A", "B", "C"], ["A", "B", "C"], 0.0),
    ],
)
def test_pairwise_disagreement_rate_goldens(
    left: list[str], right: list[str], expected: float
) -> None:
    """Pin the normalized disagreement rate over the possible pair count."""
    assert _pairwise_disagreement_rate(left, right) == pytest.approx(expected)


def test_pairwise_disagreement_rate_single_or_empty_is_zero() -> None:
    """With fewer than two planners (possible == 0) the rate is defined as 0.0."""
    assert _pairwise_disagreement_rate(["A"], ["A"]) == 0.0
    assert _pairwise_disagreement_rate([], []) == 0.0


# ---------------------------------------------------------------------------
# _pareto_points: front classification
# ---------------------------------------------------------------------------


def _rows(points: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
    """Build planner rows from ``(name, constraints_first_score, snqi_mean)``."""
    return [
        {"planner": name, "constraints_first_score": x, "snqi_mean": y} for name, x, y in points
    ]


def test_pareto_points_two_non_dominated_sorted_by_score() -> None:
    """Two non-dominating planners are both kept, sorted by constraints score."""
    front = _pareto_points(_rows([("A", 0.8, 0.6), ("B", 0.6, 0.8)]))
    assert [p["planner"] for p in front] == ["B", "A"]
    assert front[0] == {"planner": "B", "constraints_first_score": 0.6, "snqi_mean": 0.8}


def test_pareto_points_dominated_point_excluded() -> None:
    """A strictly-dominated planner is dropped from the front."""
    front = _pareto_points(_rows([("A", 0.8, 0.8), ("B", 0.6, 0.6)]))
    assert [p["planner"] for p in front] == ["A"]


def test_pareto_points_equal_points_both_kept() -> None:
    """Two identical points neither dominate the other -> both kept."""
    front = _pareto_points(_rows([("A", 0.5, 0.5), ("B", 0.5, 0.5)]))
    assert [p["planner"] for p in front] == ["A", "B"]


# ---------------------------------------------------------------------------
# _rank_disagreement / _variant_summary: winner + reversal summaries
# ---------------------------------------------------------------------------


def test_rank_disagreement_winner_and_reversal_goldens() -> None:
    """Pin winner disagreement, winners, and reversal count for swapped orders."""
    out = _rank_disagreement(["A", "B"], ["B", "A"])
    assert out["snqi_winner"] == "A"
    assert out["constraints_first_winner"] == "B"
    assert out["winner_disagreement"] is True
    assert out["pairwise_reversal_count"] == 1
    assert out["pairwise_disagreement_rate"] == pytest.approx(1.0)


def test_rank_disagreement_agreement_has_no_winner_change() -> None:
    """Identical orders -> no winner disagreement and zero reversals."""
    out = _rank_disagreement(["A", "B"], ["A", "B"])
    assert out["winner_disagreement"] is False
    assert out["pairwise_reversal_count"] == 0


@pytest.mark.parametrize(
    "base,variant,changed", [(["A", "B"], ["B", "A"], True), (["A", "B"], ["A", "B"], False)]
)
def test_variant_summary_winner_changed_flag(
    base: list[str], variant: list[str], changed: bool
) -> None:
    """``winner_changed`` tracks whether the top-ranked planner differs from base."""
    out = _variant_summary(base, variant)
    assert out["winner_changed"] is changed
    assert out["order"] == list(variant)
