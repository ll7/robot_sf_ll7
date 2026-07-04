"""Tests for Chapter 8 statistics reproducibility helpers."""

from __future__ import annotations

import pytest

from robot_sf.research.ch8_statistics import (
    StatisticResult,
    bootstrap_mean_ci,
    evaluate_statistic,
    partial_eta_squared,
    spearman_rho,
)


def test_partial_eta_squared_one_way_groups() -> None:
    """One-way partial eta squared is recomputed from group observations."""

    assert partial_eta_squared({"a": [1.0, 2.0], "b": [4.0, 5.0]}) == pytest.approx(0.9)


def test_spearman_rho_rejects_mismatched_pairs() -> None:
    """Paired rank source rows must stay aligned."""

    with pytest.raises(ValueError, match="same length"):
        spearman_rho([1.0, 2.0], [1.0])


def test_spearman_rho_recomputes_rank_correlation() -> None:
    """Spearman rho recomputes finite paired rank samples."""

    assert spearman_rho([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_bootstrap_mean_ci_is_deterministic() -> None:
    """The bootstrap path records a fixed seed and sample count."""

    first = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], samples=10000, confidence_level=0.95, seed=7)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], samples=10000, confidence_level=0.95, seed=7)

    assert first == second
    assert first["samples"] == 10000
    assert first["mean"] == pytest.approx(2.5)
    assert first["ci"] == pytest.approx([1.5, 3.5])


def test_bootstrap_mean_ci_rejects_invalid_parameters() -> None:
    """Bootstrap inputs must carry positive samples and valid confidence."""

    with pytest.raises(ValueError, match="samples"):
        bootstrap_mean_ci([1.0, 2.0], samples=0, confidence_level=0.95, seed=7)
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_mean_ci([1.0, 2.0], samples=10, confidence_level=1.5, seed=7)


def test_statistic_result_serializes_blockers() -> None:
    """Packet rows expose a stable JSON-ready shape."""

    result = StatisticResult(
        statistic_id="row",
        statistic_kind="spearman_rho",
        status="computed_mismatch",
        computed={"value": 0.0},
        expected={"value": 1.0},
        blockers=("value mismatch",),
    )

    assert result.to_json() == {
        "id": "row",
        "kind": "spearman_rho",
        "status": "computed_mismatch",
        "computed": {"value": 0.0},
        "expected": {"value": 1.0},
        "blockers": ["value mismatch"],
    }


def test_evaluate_statistic_fails_closed_without_source_data() -> None:
    """Missing Chapter 8 source rows become a blocked packet row, not a claim."""

    result = evaluate_statistic(
        {
            "id": "ch8_spearman_rho_minus_0_998",
            "statistic_kind": "spearman_rho",
            "expected": {"value": -0.998, "tolerance": 0.0005},
            "data": {},
        }
    )

    assert result.status == "blocked_missing_source_data"
    assert result.computed == {}
    assert result.expected["value"] == -0.998


def test_evaluate_statistic_matches_expected_eta_squared() -> None:
    """Manifest evaluation routes eta-squared rows through the shared helper."""

    result = evaluate_statistic(
        {
            "id": "eta",
            "statistic_kind": "partial_eta_squared",
            "expected": {"value": 0.9, "tolerance": 1e-12},
            "data": {"groups": {"a": [1.0, 2.0], "b": [4.0, 5.0]}},
        }
    )

    assert result.status == "matches_expected"
    assert result.computed["value"] == pytest.approx(0.9)


def test_evaluate_statistic_reports_mismatches() -> None:
    """Expected-value disagreements stay visible in packet blockers."""

    result = evaluate_statistic(
        {
            "id": "rho",
            "statistic_kind": "spearman_rho",
            "expected": {"value": -0.998, "tolerance": 1e-12},
            "data": {"x_values": [1.0, 2.0, 3.0], "y_values": [30.0, 20.0, 10.0]},
        }
    )

    assert result.status == "computed_mismatch"
    assert "does not match expected" in result.blockers[0]


def test_evaluate_statistic_reports_bootstrap_ci_and_sample_mismatch() -> None:
    """Bootstrap packet rows compare CI and sample count contract fields."""

    result = evaluate_statistic(
        {
            "id": "boot",
            "statistic_kind": "bootstrap_mean_ci",
            "expected": {"samples": 9, "ci": [0.0, 1.0], "tolerance": 1e-12},
            "data": {
                "values": [1.0, 2.0, 3.0, 4.0],
                "samples": 10,
                "confidence_level": 0.95,
                "seed": 7,
            },
        }
    )

    assert result.status == "computed_mismatch"
    assert any("samples" in blocker for blocker in result.blockers)
    assert any("ci[" in blocker for blocker in result.blockers)


def test_evaluate_statistic_fails_closed_for_invalid_or_unknown_inputs() -> None:
    """Invalid manifests produce blocked rows instead of exceptions."""

    invalid = evaluate_statistic(
        {
            "id": "bad",
            "statistic_kind": "partial_eta_squared",
            "expected": {"value": 0.0},
            "data": {"groups": {"only": [1.0]}},
        }
    )
    unknown = evaluate_statistic(
        {
            "id": "unknown",
            "statistic_kind": "not_supported",
            "expected": {"value": 0.0},
            "data": {"values": [1.0, 2.0]},
        }
    )

    assert invalid.status == "blocked_invalid_source_data"
    assert unknown.status == "blocked_invalid_source_data"


def test_evaluate_statistic_records_missing_expected_value() -> None:
    """Computed rows without expected targets do not become reproducible claims."""

    result = evaluate_statistic(
        {
            "id": "rho",
            "statistic_kind": "spearman_rho",
            "data": {"x_values": [1.0, 2.0, 3.0], "y_values": [30.0, 20.0, 10.0]},
        }
    )

    assert result.status == "computed_expected_value_missing"
    assert result.blockers == ("expected value block is missing",)


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ({"only": [1.0, 2.0]}, "at least two"),
        ({"a": [1.0], "b": [1.0]}, "residual"),
        ({"a": [1.0, 1.0], "b": [1.0, 1.0]}, "denominator"),
    ],
)
def test_partial_eta_squared_rejects_invalid_groups(
    groups: dict[str, list[float]], message: str
) -> None:
    """Eta-squared source groups must have enough finite variation."""

    with pytest.raises(ValueError, match=message):
        partial_eta_squared(groups)


def test_finite_sequence_validation_rejects_non_finite_values() -> None:
    """Public helpers reject non-finite source values."""

    with pytest.raises(ValueError, match="non-finite"):
        spearman_rho([1.0, float("nan")], [1.0, 2.0])
