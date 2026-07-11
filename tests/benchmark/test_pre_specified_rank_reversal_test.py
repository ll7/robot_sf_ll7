"""Tests for the pre-specified (preregistered) rank-reversal test (issue #3574).

``pre_specified_rank_reversal_test`` is the *pre-specified rank-reversal test* named in the
issue Definition of Done. It complements ``compute_bootstrap_rank_sensitivity`` (which reports
bootstrap P(A beats B) and a descriptive ranking-list disagreement flag) by adding a formal
hypothesis test that only declares a reversal when a planner pair is bootstrap-determined in
both arms with opposite signs. These tests pin that contract on synthetic fixture records so
the behavior is preserved when real campaign episode records eventually feed it.
"""

from __future__ import annotations

from typing import Any

import pytest

from robot_sf.benchmark.heterogeneous_rank_sensitivity import (
    RANK_REVERSAL_TEST_SCHEMA,
    pre_specified_rank_reversal_test,
)

_METRIC = "clearance_m"
_HET = "heterogeneous"
_HOM = "mean_matched_homogeneous"


def _rec(arm: str, planner: str, seed: Any, value: float) -> dict[str, Any]:
    """Build a minimal episode record for arm/planner/seed/metric."""
    return {
        "population_arm": arm,
        "planner": planner,
        "seed": seed,
        "metrics": {_METRIC: value},
    }


def _reversal_fixture() -> list[dict[str, Any]]:
    """Heterogeneous ranks A>B; mean-matched ranks B>A at every seed (clean reversal)."""
    records: list[dict[str, Any]] = []
    for seed in (1, 2, 3, 4, 5):
        records.append(_rec(_HET, "A", seed, 10.0 + seed))
        records.append(_rec(_HET, "B", seed, 2.0 + seed))
        records.append(_rec(_HOM, "A", seed, 2.0 + seed))
        records.append(_rec(_HOM, "B", seed, 10.0 + seed))
    return records


def _stable_fixture() -> list[dict[str, Any]]:
    """Both arms rank A>B at every seed (determined and consistent)."""
    records: list[dict[str, Any]] = []
    for seed in (1, 2, 3, 4, 5):
        records.append(_rec(_HET, "A", seed, 10.0 + seed))
        records.append(_rec(_HET, "B", seed, 2.0 + seed))
        records.append(_rec(_HOM, "A", seed, 10.0 + seed))
        records.append(_rec(_HOM, "B", seed, 2.0 + seed))
    return records


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_fewer_than_two_planners_raises_value_error() -> None:
    """The guard requires >=2 planners and raises a pinned message."""
    with pytest.raises(ValueError, match="At least 2 planners are required to compare ranks"):
        pre_specified_rank_reversal_test([], metric_key=_METRIC, planners=["A"])


def test_identical_arm_names_raise_value_error() -> None:
    """Identical arm names would compare an arm to itself; reject fail-closed."""
    with pytest.raises(ValueError, match="heterogeneous_arm and mean_matched_arm must differ"):
        pre_specified_rank_reversal_test(
            _stable_fixture(),
            metric_key=_METRIC,
            planners=["A", "B"],
            arms=("same", "same"),
        )


@pytest.mark.parametrize("alpha", [0.0, -0.01, 1.0, 1.5])
def test_invalid_alpha_raises_value_error(alpha: float) -> None:
    """alpha must lie in the open interval (0, 1)."""
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        pre_specified_rank_reversal_test(
            _stable_fixture(), metric_key=_METRIC, planners=["A", "B"], alpha=alpha
        )


def test_invalid_num_bootstrap_raises_value_error() -> None:
    """At least one bootstrap resample is required."""
    with pytest.raises(ValueError, match="num_bootstrap must be >= 1"):
        pre_specified_rank_reversal_test(
            _stable_fixture(), metric_key=_METRIC, planners=["A", "B"], num_bootstrap=0
        )


# ---------------------------------------------------------------------------
# Preregistration block (declared-before-results)
# ---------------------------------------------------------------------------


def test_pre_registration_block_is_declared_before_results() -> None:
    """The ready result echoes the ex-ante decision specification verbatim."""
    result = pre_specified_rank_reversal_test(
        _stable_fixture(), metric_key=_METRIC, planners=["A", "B"], num_bootstrap=100, seed=42
    )
    pre = result["pre_registration"]
    assert pre["declared_before_results"] is True
    assert pre["significance_level_alpha"] == pytest.approx(0.05)
    assert pre["ci_method"] == "percentile_bootstrap"
    assert pre["ci_level"] == pytest.approx(0.95)
    assert pre["num_bootstrap"] == 100
    assert pre["higher_is_safer"] is True
    assert pre["heterogeneous_arm"] == _HET
    assert pre["mean_matched_arm"] == _HOM
    assert "stable across population compositions" in pre["null_hypothesis"]
    assert "excludes zero in BOTH arms with opposite signs" in pre["decision_rule"]


# ---------------------------------------------------------------------------
# Reversal / stable / indeterminate decisions
# ---------------------------------------------------------------------------


def test_clean_reversal_is_detected() -> None:
    """Opposite determined orderings across arms -> reject H0, one reversal."""
    result = pre_specified_rank_reversal_test(
        _reversal_fixture(), metric_key=_METRIC, planners=["A", "B"], num_bootstrap=200, seed=42
    )
    assert result["status"] == "ready"
    assert result["schema_version"] == RANK_REVERSAL_TEST_SCHEMA
    assert result["decision"] == "reject_null_rank_stability"
    assert result["reversal_count"] == 1

    pair = result["pairwise"][0]
    assert pair["planners"] == ["A", "B"]
    assert pair["verdict"] == "reversal_detected"
    assert pair["reversal"] is True
    # Heterogeneous: A ahead (CI strictly positive); homogeneous: B ahead (CI strictly negative).
    assert pair["heterogeneous"]["ci_lower"] > 0.0
    assert pair["mean_matched_homogeneous"]["ci_upper"] < 0.0
    assert pair["heterogeneous"]["A_ahead"] is True
    assert pair["mean_matched_homogeneous"]["A_ahead"] is False

    rev = result["reversals"][0]
    assert rev["planners"] == ["A", "B"]
    assert rev["heterogeneous_leader"] == "A"
    assert rev["mean_matched_homogeneous_leader"] == "B"


def test_stable_determined_pair_is_not_a_reversal() -> None:
    """Same determined ordering in both arms -> fail to reject H0."""
    result = pre_specified_rank_reversal_test(
        _stable_fixture(), metric_key=_METRIC, planners=["A", "B"], num_bootstrap=200, seed=42
    )
    assert result["decision"] == "fail_to_reject_null_rank_stability"
    assert result["reversal_count"] == 0
    assert result["pairwise"][0]["verdict"] == "stable_determined"
    assert result["pairwise"][0]["reversal"] is False


def test_indeterminate_pair_is_not_a_reversal() -> None:
    """A CI straddling zero yields ``indeterminate`` and does not reject H0."""

    # Near-tied planners with tiny noise so the bootstrap mean-difference CI straddles zero.
    import random

    rng = random.Random(0)
    records: list[dict[str, Any]] = []
    for seed in (1, 2, 3, 4, 5):
        for arm in (_HET, _HOM):
            records.append(_rec(arm, "A", seed, 5.0 + rng.uniform(-0.05, 0.05)))
            records.append(_rec(arm, "B", seed, 5.0 + rng.uniform(-0.05, 0.05)))
    result = pre_specified_rank_reversal_test(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=500, seed=42
    )
    assert result["reversal_count"] == 0
    assert result["pairwise"][0]["verdict"] == "indeterminate"
    # Indeterminate is not a confirmed reversal even if the two arms' point estimates differ.
    assert result["decision"] == "fail_to_reject_null_rank_stability"


def test_three_planner_reversal_isolated_to_one_pair() -> None:
    """A middle-rank swap (B/C) reverses only that pair; A stays on top in both arms."""
    records: list[dict[str, Any]] = []
    for seed in (1, 2, 3, 4, 5):
        records.append(_rec(_HET, "A", seed, 30.0 + seed))
        records.append(_rec(_HET, "B", seed, 20.0 + seed))
        records.append(_rec(_HET, "C", seed, 10.0 + seed))
        records.append(_rec(_HOM, "A", seed, 30.0 + seed))
        records.append(_rec(_HOM, "B", seed, 10.0 + seed))
        records.append(_rec(_HOM, "C", seed, 20.0 + seed))
    result = pre_specified_rank_reversal_test(
        records, metric_key=_METRIC, planners=["A", "B", "C"], num_bootstrap=200, seed=42
    )
    assert result["reversal_count"] == 1
    verdicts = {tuple(pair["planners"]): pair["verdict"] for pair in result["pairwise"]}
    assert verdicts[("A", "B")] == "stable_determined"
    assert verdicts[("A", "C")] == "stable_determined"
    assert verdicts[("B", "C")] == "reversal_detected"


def test_lower_is_safer_flips_sign_logic_but_preserves_reversal_count() -> None:
    """``higher_is_safer=False`` negates differences; a true swap is still a reversal."""
    records: list[dict[str, Any]] = []
    for seed in (1, 2, 3, 4, 5):
        records.append(_rec(_HET, "A", seed, 30.0 + seed))
        records.append(_rec(_HET, "B", seed, 20.0 + seed))
        records.append(_rec(_HET, "C", seed, 10.0 + seed))
        records.append(_rec(_HOM, "A", seed, 30.0 + seed))
        records.append(_rec(_HOM, "B", seed, 10.0 + seed))
        records.append(_rec(_HOM, "C", seed, 20.0 + seed))
    result = pre_specified_rank_reversal_test(
        records,
        metric_key=_METRIC,
        planners=["A", "B", "C"],
        higher_is_safer=False,
        num_bootstrap=200,
        seed=42,
    )
    # Under lower-is-better the numeric ranking inverts, but the B/C swap remains a reversal.
    assert result["reversal_count"] == 1
    assert result["pre_registration"]["higher_is_safer"] is False


# ---------------------------------------------------------------------------
# Blocked (fail-closed) shapes
# ---------------------------------------------------------------------------


def test_missing_arm_is_blocked_with_precise_blocker() -> None:
    """A wholly absent arm yields ``blocked`` naming the missing arm."""
    result = pre_specified_rank_reversal_test(
        _stable_fixture(),
        metric_key=_METRIC,
        planners=["A", "B"],
        arms=("nonexistent", _HOM),
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["no valid episode records for arm 'nonexistent'"]
    # The decision specification is still echoed so the blocked result is auditable.
    assert result["pre_registration"]["heterogeneous_arm"] == "nonexistent"


def test_missing_planner_within_arm_is_blocked() -> None:
    """An arm missing one declared planner is blocked naming the planner."""
    records: list[dict[str, Any]] = []
    for seed in (1, 2):
        records.append(_rec(_HET, "A", seed, 10.0 + seed))
        records.append(_rec(_HET, "B", seed, 2.0 + seed))
        records.append(_rec(_HOM, "A", seed, 10.0 + seed))
        # _HOM has no planner "B"
    result = pre_specified_rank_reversal_test(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=100, seed=42
    )
    assert result["status"] == "blocked"
    assert any("planner 'B'" in blocker for blocker in result["blockers"])


def test_insufficient_paired_seeds_is_blocked() -> None:
    """Fewer than two common paired seeds across planners blocks the test."""
    records = [
        _rec(_HET, "A", 1, 10.0),
        _rec(_HET, "B", 1, 2.0),
        _rec(_HOM, "A", 1, 2.0),
        _rec(_HOM, "B", 1, 10.0),
    ]
    result = pre_specified_rank_reversal_test(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=100, seed=42
    )
    assert result["status"] == "blocked"
    assert any("paired seed" in blocker and "need >=2" in blocker for blocker in result["blockers"])


# ---------------------------------------------------------------------------
# Record-parsing contract (parity with compute_bootstrap_rank_sensitivity)
# ---------------------------------------------------------------------------


def test_scenario_params_fallback_supplies_arm_planner_seed() -> None:
    """Records without top-level fields fall back to ``scenario_params``."""
    records = [
        {
            "scenario_params": {"population_arm": _HET, "planner": "A", "seed": 1},
            "metrics": {_METRIC: 10.0},
        },
        {
            "scenario_params": {"population_arm": _HET, "planner": "A", "seed": 2},
            "metrics": {_METRIC: 12.0},
        },
        {
            "scenario_params": {"population_arm": _HET, "planner": "B", "seed": 1},
            "metrics": {_METRIC: 2.0},
        },
        {
            "scenario_params": {"population_arm": _HET, "planner": "B", "seed": 2},
            "metrics": {_METRIC: 3.0},
        },
        {
            "scenario_params": {"population_arm": _HOM, "planner": "A", "seed": 1},
            "metrics": {_METRIC: 2.0},
        },
        {
            "scenario_params": {"population_arm": _HOM, "planner": "A", "seed": 2},
            "metrics": {_METRIC: 3.0},
        },
        {
            "scenario_params": {"population_arm": _HOM, "planner": "B", "seed": 1},
            "metrics": {_METRIC: 10.0},
        },
        {
            "scenario_params": {"population_arm": _HOM, "planner": "B", "seed": 2},
            "metrics": {_METRIC: 12.0},
        },
    ]
    result = pre_specified_rank_reversal_test(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=100, seed=42
    )
    assert result["status"] == "ready"
    assert result["reversal_count"] == 1


def test_non_finite_metrics_are_dropped() -> None:
    """NaN/Inf metrics are dropped, not propagated into the bootstrap arrays."""
    records = [
        _rec(_HET, "A", 1, 10.0),
        _rec(_HET, "A", 2, 12.0),
        _rec(_HET, "A", 3, float("inf")),  # dropped
        _rec(_HET, "B", 1, 2.0),
        _rec(_HET, "B", 2, 3.0),
        _rec(_HET, "B", 3, 4.0),
        _rec(_HOM, "A", 1, 2.0),
        _rec(_HOM, "A", 2, 3.0),
        _rec(_HOM, "A", 3, 4.0),
        _rec(_HOM, "B", 1, 10.0),
        _rec(_HOM, "B", 2, 12.0),
        _rec(_HOM, "B", 3, float("nan")),  # dropped
    ]
    result = pre_specified_rank_reversal_test(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=100, seed=42
    )
    # After dropping seed 3 from both arms, seeds {1,2} remain: het A>B, hom B>A -> reversal.
    assert result["status"] == "ready"
    assert result["common_seeds_count"] == 2
    assert result["reversal_count"] == 1


def test_float_and_whitespace_seeds_are_normalized() -> None:
    """Float seeds coerce to int and surrounding whitespace is stripped.

    After stripping, the arm names align with the caller-declared names, proving parity with
    the record-parsing contract of ``compute_bootstrap_rank_sensitivity``.
    """
    records = [
        {"population_arm": " het ", "planner": " A ", "seed": 1.0, "metrics": {_METRIC: 10.0}},
        {"population_arm": " het ", "planner": " A ", "seed": 2, "metrics": {_METRIC: 12.0}},
        {"population_arm": " het ", "planner": " B ", "seed": 1, "metrics": {_METRIC: 2.0}},
        {"population_arm": " het ", "planner": " B ", "seed": 2, "metrics": {_METRIC: 3.0}},
        {"population_arm": " hom ", "planner": "A", "seed": 1.0, "metrics": {_METRIC: 2.0}},
        {"population_arm": " hom ", "planner": "A", "seed": 2, "metrics": {_METRIC: 3.0}},
        {"population_arm": " hom ", "planner": "B", "seed": 1, "metrics": {_METRIC: 10.0}},
        {"population_arm": " hom ", "planner": "B", "seed": 2, "metrics": {_METRIC: 12.0}},
    ]
    result = pre_specified_rank_reversal_test(
        records,
        metric_key=_METRIC,
        planners=["A", "B"],
        arms=("het", "hom"),
        num_bootstrap=100,
        seed=42,
    )
    # Whitespace is stripped (" het " -> "het", " hom " -> "hom") so the arms pair up;
    # float seed 1.0 coerces to int 1 so it pairs with the int-seed record.
    assert result["status"] == "ready"
    assert result["common_seeds_count"] == 2
    assert result["reversal_count"] == 1


# ---------------------------------------------------------------------------
# Determinism and claim boundary
# ---------------------------------------------------------------------------


def test_same_seed_is_deterministic() -> None:
    """Same input + same seed -> identical ready result (locks the RNG contract).."""
    kwargs = {
        "records": _reversal_fixture(),
        "metric_key": _METRIC,
        "planners": ["A", "B"],
        "num_bootstrap": 200,
        "seed": 123,
    }
    a = pre_specified_rank_reversal_test(**kwargs)
    b = pre_specified_rank_reversal_test(**kwargs)
    assert a == b


def test_claim_boundary_is_explicit_and_non_claiming() -> None:
    """The ready result states no benchmark/rank/realism claim is established here."""
    result = pre_specified_rank_reversal_test(
        _reversal_fixture(), metric_key=_METRIC, planners=["A", "B"], num_bootstrap=100, seed=42
    )
    boundary = result["claim_boundary"]
    assert "Preregistered rank-reversal test primitive" in boundary
    assert "No benchmark campaign" in boundary
