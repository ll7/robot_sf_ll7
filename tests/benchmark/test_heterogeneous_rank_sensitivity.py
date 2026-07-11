"""Tests for rank sensitivity and bootstrap comparison (issue #3574)."""

from __future__ import annotations

from robot_sf.benchmark.heterogeneous_rank_sensitivity import compute_bootstrap_rank_sensitivity


def test_compute_bootstrap_rank_sensitivity_basic() -> None:
    """Verify that rankings, probabilities, and reversals are computed correctly."""
    # Create mock records
    records = []
    # Seed 1: A=10, B=5, C=2
    # Seed 2: A=12, B=6, C=3
    # Seed 3: A=11, B=5.5, C=2.5
    # For heterogeneous arm
    for seed, vals in [(1, [10, 5, 2]), (2, [12, 6, 3]), (3, [11, 5.5, 2.5])]:
        for p_idx, planner in enumerate(["A", "B", "C"]):
            records.append(
                {
                    "population_arm": "heterogeneous",
                    "planner": planner,
                    "seed": seed,
                    "metrics": {"clearance_m": vals[p_idx]},
                }
            )

    # For mean_matched_homogeneous arm: reverse order of B and C
    # Seed 1: A=10, B=2, C=5
    # Seed 2: A=12, B=3, C=6
    # Seed 3: A=11, B=2.5, C=5.5
    for seed, vals in [(1, [10, 2, 5]), (2, [12, 3, 6]), (3, [11, 2.5, 5.5])]:
        for p_idx, planner in enumerate(["A", "B", "C"]):
            records.append(
                {
                    "population_arm": "mean_matched_homogeneous",
                    "planner": planner,
                    "seed": seed,
                    "metrics": {"clearance_m": vals[p_idx]},
                }
            )

    result = compute_bootstrap_rank_sensitivity(
        records,
        metric_key="clearance_m",
        planners=["A", "B", "C"],
        higher_is_safer=True,
        num_bootstrap=100,
        seed=42,
    )

    assert result["status"] == "ready"
    assert result["common_seeds_count"] == 3

    # Het ranking should be A, B, C
    assert result["arms"]["heterogeneous"]["ranking"] == ["A", "B", "C"]
    # Hom ranking should be A, C, B
    assert result["arms"]["mean_matched_homogeneous"]["ranking"] == ["A", "C", "B"]

    # Reversal should be detected
    assert len(result["reversals"]) == 1
    assert result["reversals"][0]["type"] == "rank_order_disagreement"

    # Pairwise probability P(A beats B) under heterogeneous should be 1.0 (since A is always > B)
    assert result["arms"]["heterogeneous"]["pairwise_probabilities"]["A_beats_B"] == 1.0
    assert result["arms"]["heterogeneous"]["pairwise_probabilities"]["B_beats_A"] == 0.0


def test_rank_sensitivity_keeps_response_law_fractions_separate() -> None:
    """Response-law sweep rows cannot overwrite one another by shared seed."""

    records = []
    for fraction, values in [(0.0, (2.0, 1.0)), (0.5, (1.0, 2.0))]:
        for arm in ("heterogeneous", "mean_matched_homogeneous"):
            for seed in (1, 2):
                for planner, value in zip(("A", "B"), values, strict=True):
                    records.append(
                        {
                            "population_arm": arm,
                            "response_law_fraction": fraction,
                            "planner": planner,
                            "seed": seed,
                            "metrics": {"clearance_m": value},
                        }
                    )

    result = compute_bootstrap_rank_sensitivity(
        records,
        metric_key="clearance_m",
        planners=["A", "B"],
        num_bootstrap=20,
        seed=42,
    )

    assert result["status"] == "ready"
    assert result["arms"]["heterogeneous/response_law_fraction_0"]["ranking"] == ["A", "B"]
    assert result["arms"]["heterogeneous/response_law_fraction_0.5"]["ranking"] == ["B", "A"]


def test_rank_sensitivity_identifies_a_reversal_within_one_response_fraction() -> None:
    """Sweep-arm reversals retain the fraction-qualified arm names."""

    records = []
    for arm, values in [
        ("heterogeneous", (2.0, 1.0)),
        ("mean_matched_homogeneous", (1.0, 2.0)),
    ]:
        for seed in (1, 2):
            for planner, value in zip(("A", "B"), values, strict=True):
                records.append(
                    {
                        "population_arm": arm,
                        "response_law_fraction": 0.25,
                        "planner": planner,
                        "seed": seed,
                        "metrics": {"clearance_m": value},
                    }
                )

    result = compute_bootstrap_rank_sensitivity(
        records,
        metric_key="clearance_m",
        planners=["A", "B"],
        num_bootstrap=20,
        seed=42,
    )

    assert result["reversals"] == [
        {
            "type": "rank_order_disagreement",
            "heterogeneous_arm": "heterogeneous/response_law_fraction_0.25",
            "mean_matched_homogeneous_arm": "mean_matched_homogeneous/response_law_fraction_0.25",
            "heterogeneous_ranking": ["A", "B"],
            "mean_matched_homogeneous_ranking": ["B", "A"],
            "description": "Heterogeneous ranking ['A', 'B'] differs from homogeneous ['B', 'A']",
        }
    ]


def test_insufficient_seeds_blocks() -> None:
    """Verify that result shows blocked when seed count is less than 3."""
    records = [
        {
            "population_arm": "heterogeneous",
            "planner": "A",
            "seed": 1,
            "metrics": {"clearance_m": 5.0},
        },
        {
            "population_arm": "heterogeneous",
            "planner": "B",
            "seed": 1,
            "metrics": {"clearance_m": 4.0},
        },
        {
            "population_arm": "mean_matched_homogeneous",
            "planner": "A",
            "seed": 1,
            "metrics": {"clearance_m": 5.0},
        },
        {
            "population_arm": "mean_matched_homogeneous",
            "planner": "B",
            "seed": 1,
            "metrics": {"clearance_m": 4.0},
        },
    ]
    result = compute_bootstrap_rank_sensitivity(
        records,
        metric_key="clearance_m",
        planners=["A", "B"],
        higher_is_safer=True,
    )
    assert result["status"] == "blocked"
    assert "Insufficient paired seeds" in result["blockers"][0]


def test_rank_sensitivity_preserves_falsy_fields_and_none_scenario_params() -> None:
    """Regression for issue #4618 R1: keep seed=0 and do not crash on null params."""
    records = []
    for seed, metric in [(0, 1.0), (1, 2.0)]:
        records.extend(
            [
                {
                    "population_arm": "",
                    "planner": "A",
                    "seed": seed,
                    "scenario_params": None,
                    "metrics": {"clearance_m": metric},
                },
                {
                    "population_arm": "",
                    "planner": "B",
                    "seed": seed,
                    "scenario_params": {
                        "population_arm": "fallback_arm",
                        "planner": "fallback_planner",
                        "seed": 99,
                    },
                    "metrics": {"clearance_m": metric - 0.5},
                },
            ]
        )

    result = compute_bootstrap_rank_sensitivity(
        records,
        metric_key="clearance_m",
        planners=["A", "B"],
        higher_is_safer=True,
        num_bootstrap=10,
        seed=42,
    )

    assert result["status"] == "ready"
    assert result["common_seeds_count"] == 2
    assert list(result["arms"]) == [""]
