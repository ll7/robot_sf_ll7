"""Characterization baseline tests for ``robot_sf/benchmark/constraints_first_scoring.py``.

These tests pin the *current observable behavior* of the constraints-first
(non-compensatory) scoring layer on small synthetic episode sets. They are
table-driven and assert exact golden values, including edge cases (empty
planner set, single episode, None/non-numeric metric values).

Purpose (issue #4874, Refs #4770): lock a behavioral baseline so the
post-submission refactor wave can prove behavior-preservation. If a test
reveals a genuine bug, do NOT fix it here — document it and file a separate
fix issue.

These tests are additive: they focus on exact golden-value pinning of the
summary/report shape and do not duplicate the property coverage in
``test_constraints_first_scoring.py``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from robot_sf.benchmark.constraints_first_scoring import (
    CONSTRAINTS_FIRST_SCHEMA,
    AdmissibilityGates,
    build_constraints_first_report,
    collision_upper_confidence_bound,
    constraints_first_planner_summary,
    group_episodes_by_planner,
    is_episode_admissible,
    ranking_inversion,
    survivorship_aware_metric,
)

# Shared synthetic episode set: one admissible safe episode, one collision
# episode, one timed-out episode with a None comfort value.
_EPISODES: list[dict[str, Any]] = [
    {
        "comfort": 2.0,
        "efficiency": 0.5,
        "collisions": 0,
        "safe_success": True,
        "near_miss_severity": 0.1,
        "timeout": False,
        "deadlock": False,
    },
    {
        "comfort": 4.0,
        "efficiency": 0.4,
        "collisions": 1,
        "safe_success": False,
        "near_miss_severity": 0.9,
        "timeout": False,
        "deadlock": False,
    },
    {
        "comfort": None,  # non-numeric -> excluded from means
        "efficiency": 0.6,
        "collisions": 0,
        "safe_success": True,
        "near_miss_severity": 0.05,
        "timeout": True,
        "deadlock": False,
    },
]


# ---------------------------------------------------------------------------
# collision_upper_confidence_bound exact values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_events", "n_episodes", "confidence", "expected"),
    [
        (0, 20, 0.95, 0.13910834066826516),  # near rule-of-three, exact beta.ppf
        (1, 4, 0.95, 0.7513953742698181),
        (0, 1, 0.95, 0.95),  # 0 of 1 at 0.95 -> 0.95
        (2, 2, 0.95, 1.0),  # all collide -> 1.0
    ],
)
def test_collision_ucb_exact_values(
    n_events: int, n_episodes: int, confidence: float, expected: float
) -> None:
    """Pin exact Clopper-Pearson upper bounds for representative inputs."""
    assert collision_upper_confidence_bound(n_events, n_episodes, confidence=confidence) == (
        pytest.approx(expected)
    )


def test_collision_ucb_rejects_invalid_arguments() -> None:
    """Out-of-range event/episode counts and confidence raise ``ValueError``."""
    with pytest.raises(ValueError):
        collision_upper_confidence_bound(0, 0)
    with pytest.raises(ValueError):
        collision_upper_confidence_bound(5, 3)  # n_events > n_episodes
    with pytest.raises(ValueError):
        collision_upper_confidence_bound(0, 10, confidence=1.0)


# ---------------------------------------------------------------------------
# survivorship_aware_metric
# ---------------------------------------------------------------------------


def test_survivorship_aware_metric_exposes_conditioning_bias() -> None:
    """Unconditional vs safe-conditioned means and their signed delta."""
    out = survivorship_aware_metric(_EPISODES, "comfort")
    assert out["metric"] == "comfort"
    # unconditional: mean of {2.0, 4.0} (None excluded) = 3.0
    assert out["unconditional_mean"] == pytest.approx(3.0)
    # conditioned on safe_success: only ep1 (comfort=2.0)
    assert out["conditioned_on_safe_success_mean"] == pytest.approx(2.0)
    # delta = conditioned - unconditional = -1.0
    assert out["survivorship_delta"] == pytest.approx(-1.0)
    assert out["n_all"] == 2
    assert out["n_safe_success"] == 1


def test_survivorship_aware_metric_empty_when_all_non_numeric() -> None:
    """When no numeric values are present, both means and delta are None."""
    episodes = [{"comfort": "bad"}, {"comfort": None}]
    out = survivorship_aware_metric(episodes, "comfort")
    assert out["unconditional_mean"] is None
    assert out["conditioned_on_safe_success_mean"] is None
    assert out["survivorship_delta"] is None
    assert out["n_all"] == 0


# ---------------------------------------------------------------------------
# constraints_first_planner_summary
# ---------------------------------------------------------------------------


def test_planner_summary_exact_rates_and_schema() -> None:
    """Admissibility and collision rates on the shared episode set."""
    s = constraints_first_planner_summary(_EPISODES)
    # admissible: ep1 yes; ep2 has collision>0 -> no; ep3 timed out -> no  => 1/3
    assert s["admissible_rate"] == pytest.approx(1 / 3)
    # collision episodes: ep2 only => 1/3
    assert s["collision_rate"] == pytest.approx(1 / 3)
    assert s["schema_version"] == CONSTRAINTS_FIRST_SCHEMA
    assert s["evidence_kind"] == "diagnostic_proxy"
    assert s["n_episodes"] == 3
    assert set(s) == {
        "schema_version",
        "evidence_kind",
        "n_episodes",
        "admissible_rate",
        "collision_rate",
        "collision_upper_confidence_bound",
        "comfort",
        "efficiency",
    }


def test_planner_summary_rejects_empty_input() -> None:
    """An empty episode list is rejected."""
    with pytest.raises(ValueError):
        constraints_first_planner_summary([])


def test_is_episode_admissible_respects_gates() -> None:
    """Collision, timeout, and deadlock gates each gate admissibility."""
    assert is_episode_admissible({"collisions": 0, "timeout": False, "deadlock": False}) is True
    assert is_episode_admissible({"collisions": 1}) is False
    assert is_episode_admissible({"collisions": 0, "timeout": True}) is False
    assert is_episode_admissible({"collisions": 0, "deadlock": True}) is False
    # Near-miss severity cap is opt-in via gates.
    gated = AdmissibilityGates(max_near_miss_severity=0.2)
    assert is_episode_admissible({"collisions": 0, "near_miss_severity": 0.5}, gated) is False
    assert is_episode_admissible({"collisions": 0, "near_miss_severity": 0.1}, gated) is True


# ---------------------------------------------------------------------------
# ranking_inversion
# ---------------------------------------------------------------------------


def test_ranking_inversion_no_change_when_orders_match() -> None:
    """Identical orderings produce rank_delta 0 and no inverted planners."""
    scores = {"p3": 0.9, "p1": 0.5, "p2": 0.5}  # tie p1/p2 broken by name
    out = ranking_inversion(scores, scores)
    assert out["any_inversion"] is False
    assert out["inverted_planners"] == []
    # p3 best (rank 1); p1, p2 tie -> ranks 2, 3 by name
    assert out["per_planner"]["p3"]["compensatory_rank"] == 1
    assert out["per_planner"]["p1"]["compensatory_rank"] == 2


def test_ranking_inversion_detects_swapped_order() -> None:
    """A swapped ordering is detected and both planners are inverted."""
    out = ranking_inversion({"p1": 0.9, "p2": 0.1}, {"p1": 0.1, "p2": 0.9})
    assert out["inverted_planners"] == ["p1", "p2"]
    assert out["any_inversion"] is True
    # p1 compensatory rank 1, constraints-first rank 2 -> delta +1
    assert out["per_planner"]["p1"]["rank_delta"] == 1


def test_ranking_inversion_requires_same_planner_set() -> None:
    """Mismatched planner sets raise ``ValueError``."""
    with pytest.raises(ValueError):
        ranking_inversion({"p1": 1.0}, {"p2": 1.0})


# ---------------------------------------------------------------------------
# build_constraints_first_report
# ---------------------------------------------------------------------------


def test_build_report_shape_and_ranking_inversion_block() -> None:
    """The report carries versioned metadata, per-planner summaries, and ranking inversion."""
    report = build_constraints_first_report(
        {"X": _EPISODES, "Y": [_EPISODES[0]]},
        compensatory_scores={"X": 0.7, "Y": 0.3},
    )
    assert set(report) == {
        "schema_version",
        "metric_layer_schema_version",
        "metric_layer_order",
        "evidence_kind",
        "per_planner",
        "ranking_inversion",
    }
    assert report["schema_version"] == CONSTRAINTS_FIRST_SCHEMA
    assert report["evidence_kind"] == "diagnostic_proxy"
    # metric_layer_order is the locked layered-evaluation order.
    assert report["metric_layer_order"] == [
        "safety_gate",
        "liveness",
        "social_compliance",
        "efficiency",
        "comfort",
        "operational",
    ]
    assert set(report["per_planner"]) == {"X", "Y"}
    assert "ranking_inversion" in report


def test_build_report_omits_ranking_inversion_without_compensatory_scores() -> None:
    """The ranking-inversion block is omitted when no compensatory scores are supplied."""
    report = build_constraints_first_report({"X": [_EPISODES[0]]})
    assert "ranking_inversion" not in report


def test_build_report_rejects_empty_input() -> None:
    """An empty planner map is rejected."""
    with pytest.raises(ValueError):
        build_constraints_first_report({})


def test_build_report_uses_admissible_rate_as_constraints_first_score() -> None:
    """Ranking-inversion contrasts composite vs admissible_rate ordering."""
    # Planner A: all admissible (admissible_rate=1.0); Planner B: none admissible (0.0).
    a = [{"collisions": 0, "timeout": False, "deadlock": False}]
    b = [{"collisions": 1, "timeout": False, "deadlock": False}]
    report = build_constraints_first_report(
        {"A": a, "B": b},
        compensatory_scores={"A": 0.2, "B": 0.8},  # composite favors B
    )
    ri = report["ranking_inversion"]
    # Under constraints-first, A (admissible) ranks above B -> A rank improves.
    assert ri["per_planner"]["A"]["rank_delta"] < 0
    assert ri["per_planner"]["B"]["rank_delta"] > 0


# ---------------------------------------------------------------------------
# group_episodes_by_planner
# ---------------------------------------------------------------------------


def test_group_episodes_by_planner_keys_and_missing_key_error() -> None:
    """Records group by planner key; a record missing the key raises."""
    grouped = group_episodes_by_planner([{"planner": "A", "x": 1}, {"planner": "A", "x": 2}])
    assert list(grouped) == ["A"]
    assert len(grouped["A"]) == 2
    with pytest.raises(ValueError):
        group_episodes_by_planner([{"x": 1}])


# Keep math referenced for analyzers when NaN assertions expand.
_ = math
