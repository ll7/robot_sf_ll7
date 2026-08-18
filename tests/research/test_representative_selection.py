"""Tests for the shared representative-run selection rule.

The rule is the anti-cherry-picking guarantee behind every single-run exhibit
(campaign majority verdicts, replay videos, trace dossiers), so it is pinned
here once rather than at each call site.
"""

from __future__ import annotations

import pytest

from robot_sf.research.representative_selection import (
    PRIMARY_ORDER_PARAMETER,
    VERDICT_SEVERITY,
    RepresentativeCandidate,
    majority_verdict,
    primary_order_parameter,
    select_representative_index,
)


def _candidate(verdict: str, primary_order: float, seed: int) -> RepresentativeCandidate:
    """Build one candidate run."""
    return RepresentativeCandidate(verdict=verdict, primary_order=primary_order, seed=seed)


def test_verdict_severity_is_ordered_weakest_first() -> None:
    assert VERDICT_SEVERITY == ("absent_or_negligible", "weak_partial", "clearly_present")


def test_majority_verdict_picks_the_most_common_label() -> None:
    verdicts = ["clearly_present", "clearly_present", "weak_partial"]
    assert majority_verdict(verdicts) == "clearly_present"


def test_majority_verdict_tie_breaks_toward_the_weaker_label() -> None:
    assert majority_verdict(["clearly_present", "absent_or_negligible"]) == "absent_or_negligible"
    assert majority_verdict(["clearly_present", "weak_partial"]) == "weak_partial"


def test_majority_verdict_accepts_precounted_labels() -> None:
    assert majority_verdict({"clearly_present": 2, "weak_partial": 1}) == "clearly_present"
    assert majority_verdict({"clearly_present": 1, "weak_partial": 1}) == "weak_partial"


def test_majority_verdict_ignores_zero_counts() -> None:
    assert majority_verdict({"clearly_present": 2, "absent_or_negligible": 0}) == "clearly_present"


def test_majority_verdict_ranks_unknown_labels_last() -> None:
    # An unrecognized label must never win the "weaker label" tie-break.
    assert majority_verdict(["mystery_label", "weak_partial"]) == "weak_partial"


def test_majority_verdict_between_unknown_labels_is_order_independent() -> None:
    """The pinned campaign fallback orders equally unknown labels lexically."""

    forward = majority_verdict(["zeta_label", "alpha_label"])
    reverse = majority_verdict(["alpha_label", "zeta_label"])
    assert forward == reverse == "alpha_label"


def test_majority_verdict_requires_input() -> None:
    with pytest.raises(ValueError, match="at least one verdict"):
        majority_verdict([])


def test_select_representative_index_takes_median_of_majority_pool() -> None:
    candidates = [
        _candidate("clearly_present", 5.0, 1),
        _candidate("clearly_present", 3.0, 2),
        _candidate("clearly_present", 4.0, 3),
        _candidate("absent_or_negligible", 0.0, 4),
    ]
    # Majority pool is the three clearly_present runs; their median value is 4.0.
    assert select_representative_index(candidates) == 2


def test_select_representative_index_excludes_minority_verdicts() -> None:
    candidates = [
        _candidate("absent_or_negligible", 0.0, 1),
        _candidate("weak_partial", 1.0, 2),
        _candidate("weak_partial", 2.0, 3),
    ]
    # The lone absent run is outside the pool even though it is the middle value.
    assert select_representative_index(candidates) == 1


def test_select_representative_index_even_pool_takes_the_lower_median() -> None:
    candidates = [
        _candidate("weak_partial", 0.1, 10),
        _candidate("weak_partial", 0.2, 11),
    ]
    assert select_representative_index(candidates) == 0


def test_select_representative_index_even_pool_lower_median_ignores_seed_order() -> None:
    # The lower-median run wins even when the upper-median run has a lower seed:
    # rank decides first, and the seed only breaks an exact value tie.
    candidates = [
        _candidate("weak_partial", 0.2, 5),
        _candidate("weak_partial", 0.1, 20),
    ]
    assert select_representative_index(candidates) == 1


def test_select_representative_index_breaks_exact_ties_on_lower_seed() -> None:
    candidates = [
        _candidate("weak_partial", 1.0, 9),
        _candidate("weak_partial", 1.0, 4),
        _candidate("weak_partial", 2.0, 7),
    ]
    # Sorted by (value, seed) the pool is seed 4, seed 9, seed 7; median is seed 9.
    assert select_representative_index(candidates) == 0


def test_select_representative_index_single_candidate() -> None:
    assert select_representative_index([_candidate("weak_partial", 1.0, 1)]) == 0


def test_select_representative_index_requires_input() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        select_representative_index([])


def test_select_representative_index_is_deterministic_under_input_reordering() -> None:
    candidates = [
        _candidate("clearly_present", 5.0, 1),
        _candidate("clearly_present", 3.0, 2),
        _candidate("clearly_present", 4.0, 3),
        _candidate("absent_or_negligible", 0.0, 4),
    ]
    chosen = candidates[select_representative_index(candidates)]
    shuffled = list(reversed(candidates))
    assert shuffled[select_representative_index(shuffled)] == chosen


def test_primary_order_parameter_covers_the_canonical_scenarios() -> None:
    assert primary_order_parameter("bidirectional_corridor") == "lane_segregation_index"
    assert primary_order_parameter("narrow_doorway") == "oscillation_flips"
    assert primary_order_parameter("high_density_exit") == "exit_density_ratio"


def test_primary_order_parameter_fails_closed_for_unknown_scenarios() -> None:
    with pytest.raises(KeyError, match="no primary order parameter"):
        primary_order_parameter("not_a_scenario")


def test_primary_order_parameter_map_is_read_only() -> None:
    with pytest.raises(TypeError):
        PRIMARY_ORDER_PARAMETER["narrow_doorway"] = "something_else"  # type: ignore[index]
