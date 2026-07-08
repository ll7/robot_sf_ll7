"""Characterization baseline tests for ``robot_sf/benchmark/heterogeneous_rank_sensitivity.py``.

These tests pin the *current observable behavior* of
``compute_bootstrap_rank_sensitivity`` on small synthetic record sets. They are
table-driven and assert exact golden values, including edge cases (empty input,
single paired seed, NaN/Inf metrics at the finite-guard boundary, missing
metric, seed coercion, whitespace stripping) and the exact blocked / ready
result shapes.

Purpose (issue #4881, wave 2; Refs #4874, #4770): lock a behavioral baseline so
the post-submission refactor wave can prove behavior-preservation. If a test
reveals a genuine bug, do NOT fix it here — document it and file a separate fix
issue.

These tests are additive: they pin the error contract, the exact blocked-status
shapes and messages, the ``scenario_params`` fallback path, finite-guard
skipping, ranking direction, exact ``observed_means``, the pairwise-key
contract, the reversal-dict shape, arm sorting, and same-seed determinism. They
do not duplicate the ranking/probability/reversal happy-path coverage or the
#4618 falsy-field regression in ``test_heterogeneous_rank_sensitivity.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from robot_sf.benchmark.heterogeneous_rank_sensitivity import (
    compute_bootstrap_rank_sensitivity,
)

_SCHEMA = "heterogeneous_rank_sensitivity.v1"
_METRIC = "clearance_m"


def _rec(
    arm: str,
    planner: str,
    seed: Any,
    value: float,
    *,
    metric: str = _METRIC,
    **extra: Any,
) -> dict[str, Any]:
    """Build a minimal episode record; ``extra`` overrides/extends fields."""
    rec: dict[str, Any] = {
        "population_arm": arm,
        "planner": planner,
        "seed": seed,
        "metrics": {metric: value},
    }
    rec.update(extra)
    return rec


def _two_planner_two_seed() -> list[dict[str, Any]]:
    """Arm ``het``: A mean=11.0, B mean=6.0 (A strictly dominates B)."""
    return [
        _rec("het", "A", 1, 10.0),
        _rec("het", "A", 2, 12.0),
        _rec("het", "B", 1, 5.0),
        _rec("het", "B", 2, 7.0),
    ]


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------


def test_fewer_than_two_planners_raises_value_error_with_pinned_message() -> None:
    """The guard requires >=2 planners and raises a pinned message."""
    with pytest.raises(ValueError, match="At least 2 planners are required to compare ranks"):
        compute_bootstrap_rank_sensitivity([], metric_key=_METRIC, planners=["A"])


# ---------------------------------------------------------------------------
# Blocked-status shapes (exact)
# ---------------------------------------------------------------------------


def test_empty_records_returns_blocked_no_data_shape() -> None:
    """No records at all -> blocked with the exact 'no valid data' blocker."""
    result = compute_bootstrap_rank_sensitivity([], metric_key=_METRIC, planners=["A", "B"])
    assert set(result) == {"schema_version", "metric_key", "status", "blockers"}
    assert result["schema_version"] == _SCHEMA
    assert result["metric_key"] == _METRIC
    assert result["status"] == "blocked"
    assert result["blockers"] == ["No valid episode data matches configuration"]


def test_records_all_filtered_out_returns_blocked_no_data() -> None:
    """Records that all fail parsing (unknown planner) -> same blocked shape."""
    records = [
        _rec("het", "C", 1, 10.0),  # planner C not in planners list -> dropped
        _rec("het", "C", 2, 12.0),
    ]
    result = compute_bootstrap_rank_sensitivity(records, metric_key=_METRIC, planners=["A", "B"])
    assert result["status"] == "blocked"
    assert result["blockers"] == ["No valid episode data matches configuration"]


def test_insufficient_paired_seeds_blocks_with_exact_interpolated_message() -> None:
    """Only one common seed -> blocked with the exact count-interpolated message."""
    records = [
        _rec("het", "A", 1, 10.0),
        _rec("het", "B", 1, 5.0),
    ]
    result = compute_bootstrap_rank_sensitivity(records, metric_key=_METRIC, planners=["A", "B"])
    assert result["status"] == "blocked"
    assert result["blockers"] == [
        "Insufficient paired seeds across planners/arms (found 1, need >=2)"
    ]


# ---------------------------------------------------------------------------
# Record-parsing contract (finite guards, fallbacks, coercion)
# ---------------------------------------------------------------------------


def test_scenario_params_fallback_path_is_used_when_top_level_absent() -> None:
    """Top-level arm/planner/seed absent -> scenario_params supplies them."""
    records = [
        {
            "scenario_params": {"population_arm": "het", "planner": "A", "seed": 1},
            "metrics": {"clearance_m": 10.0},
        },
        {
            "scenario_params": {"population_arm": "het", "planner": "A", "seed": 2},
            "metrics": {"clearance_m": 12.0},
        },
        {
            "scenario_params": {"population_arm": "het", "planner": "B", "seed": 1},
            "metrics": {"clearance_m": 5.0},
        },
        {
            "scenario_params": {"population_arm": "het", "planner": "B", "seed": 2},
            "metrics": {"clearance_m": 7.0},
        },
    ]
    result = compute_bootstrap_rank_sensitivity(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=10, seed=42
    )
    assert result["status"] == "ready"
    assert list(result["arms"]) == ["het"]  # fallback arm name used verbatim
    assert result["arms"]["het"]["ranking"] == ["A", "B"]


def test_nan_metric_record_is_dropped_from_window() -> None:
    """A NaN metric fails the finite guard and is dropped; mean is over the rest."""
    records = [
        _rec("het", "A", 1, 10.0),
        _rec("het", "A", 2, 12.0),
        _rec("het", "A", 3, float("nan")),  # dropped by math.isfinite guard
        _rec("het", "B", 1, 5.0),
        _rec("het", "B", 2, 7.0),
        _rec("het", "B", 3, 9.0),
    ]
    result = compute_bootstrap_rank_sensitivity(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=10, seed=42
    )
    assert result["status"] == "ready"
    # Seed 3 disappears from A -> common seeds {1, 2}; A mean = (10+12)/2 = 11.0.
    assert result["common_seeds_count"] == 2
    assert result["arms"]["het"]["observed_means"]["A"] == pytest.approx(11.0)


def test_inf_metric_record_is_dropped_from_window() -> None:
    """An Inf metric fails the finite guard and is dropped (not propagated)."""
    records = [
        _rec("het", "A", 1, 10.0),
        _rec("het", "A", 2, 12.0),
        _rec("het", "A", 3, float("inf")),  # dropped
        _rec("het", "B", 1, 5.0),
        _rec("het", "B", 2, 7.0),
        _rec("het", "B", 3, 9.0),
    ]
    result = compute_bootstrap_rank_sensitivity(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=10, seed=42
    )
    assert result["status"] == "ready"
    assert result["arms"]["het"]["observed_means"]["A"] == pytest.approx(11.0)


def test_missing_metric_key_skips_record() -> None:
    """A record whose metrics lack the key (or metrics is not a mapping) is dropped."""
    records = [
        _rec("het", "A", 1, 10.0),
        _rec("het", "A", 2, 12.0),
        {"population_arm": "het", "planner": "A", "seed": 3, "metrics": {"other": 1.0}},
        _rec("het", "B", 1, 5.0),
        _rec("het", "B", 2, 7.0),
        _rec("het", "B", 3, 9.0),
    ]
    result = compute_bootstrap_rank_sensitivity(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=10, seed=42
    )
    assert result["status"] == "ready"
    assert result["common_seeds_count"] == 2  # seed 3 dropped from A


def test_float_seed_is_coerced_to_int_for_pairing() -> None:
    """``int(s_val)`` coerces a float seed so it pairs with an int seed."""
    records = [
        _rec("het", "A", 1, 10.0),  # int seed 1
        _rec("het", "A", 2, 12.0),
        _rec("het", "B", 1.0, 5.0),  # float seed 1.0 -> coerced to 1
        _rec("het", "B", 2, 7.0),
    ]
    result = compute_bootstrap_rank_sensitivity(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=10, seed=42
    )
    # Without coercion, B would have keys {1.0, 2} and common with A's {1, 2} would
    # be just {2} -> blocked. Ready-with-2 proves the float->int coercion.
    assert result["status"] == "ready"
    assert result["common_seeds_count"] == 2


def test_arm_and_planner_whitespace_is_stripped() -> None:
    """``str(x).strip()`` normalizes surrounding whitespace on arm and planner."""
    records = [
        _rec(" het ", " A ", 1, 10.0),
        _rec(" het ", " A ", 2, 12.0),
        _rec(" het ", " B ", 1, 5.0),
        _rec(" het ", " B ", 2, 7.0),
    ]
    result = compute_bootstrap_rank_sensitivity(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=10, seed=42
    )
    assert result["status"] == "ready"
    assert list(result["arms"]) == ["het"]


def test_unknown_planner_records_are_ignored() -> None:
    """Records for a planner not in the list are dropped; result is unaffected."""
    base = _two_planner_two_seed()
    extra = [_rec("het", "C", 1, 99.0), _rec("het", "C", 2, 99.0)]
    result = compute_bootstrap_rank_sensitivity(
        base + extra, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=10, seed=42
    )
    assert result["status"] == "ready"
    assert result["planners"] == ["A", "B"]  # C never admitted
    assert set(result["arms"]["het"]["pairwise_probabilities"]) == {
        "A_beats_B",
        "B_beats_A",
    }


# ---------------------------------------------------------------------------
# Ranking direction, observed means, pairwise-key contract
# ---------------------------------------------------------------------------


def test_observed_means_are_exact_python_floats() -> None:
    """Observed means are the plain per-planner mean over common seeds."""
    result = compute_bootstrap_rank_sensitivity(
        _two_planner_two_seed(),
        metric_key=_METRIC,
        planners=["A", "B"],
        num_bootstrap=10,
        seed=42,
    )
    means = result["arms"]["het"]["observed_means"]
    assert means == {"A": pytest.approx(11.0), "B": pytest.approx(6.0)}
    assert isinstance(means["A"], float)  # coerced via float(...)


@pytest.mark.parametrize("higher_is_safer,expected", [(True, ["A", "B"]), (False, ["B", "A"])])
def test_ranking_direction_respects_higher_is_safer(
    higher_is_safer: bool, expected: list[str]
) -> None:
    """``higher_is_safer`` flips the observed-mean sort order."""
    result = compute_bootstrap_rank_sensitivity(
        _two_planner_two_seed(),
        metric_key=_METRIC,
        planners=["A", "B"],
        higher_is_safer=higher_is_safer,
        num_bootstrap=10,
        seed=42,
    )
    assert result["arms"]["het"]["ranking"] == expected
    assert result["higher_is_safer"] is higher_is_safer


def test_dominant_planner_has_unit_and_zero_pairwise_probabilities() -> None:
    """A strictly-dominant planner yields P=1.0 forward and P=0.0 reverse."""
    result = compute_bootstrap_rank_sensitivity(
        _two_planner_two_seed(),
        metric_key=_METRIC,
        planners=["A", "B"],
        num_bootstrap=50,
        seed=7,
    )
    pair = result["arms"]["het"]["pairwise_probabilities"]
    assert pair["A_beats_B"] == 1.0  # A > B at every seed -> always
    assert pair["B_beats_A"] == 0.0


def test_three_planners_yield_six_directed_pairwise_keys() -> None:
    """Pairwise keys cover all ordered i!=j pairs; self-pairs are excluded."""
    records = [
        _rec("het", "A", 1, 30.0),
        _rec("het", "A", 2, 32.0),
        _rec("het", "B", 1, 20.0),
        _rec("het", "B", 2, 22.0),
        _rec("het", "C", 1, 10.0),
        _rec("het", "C", 2, 12.0),
    ]
    result = compute_bootstrap_rank_sensitivity(
        records, metric_key=_METRIC, planners=["A", "B", "C"], num_bootstrap=10, seed=42
    )
    keys = set(result["arms"]["het"]["pairwise_probabilities"])
    assert keys == {"A_beats_B", "A_beats_C", "B_beats_A", "B_beats_C", "C_beats_A", "C_beats_B"}
    assert len(keys) == 3 * 2  # n * (n - 1)


def test_planners_order_is_preserved_and_echoed() -> None:
    """``result['planners']`` echoes the input order (not re-sorted)."""
    records = [
        _rec("het", "B", 1, 10.0),
        _rec("het", "B", 2, 12.0),
        _rec("het", "A", 1, 5.0),
        _rec("het", "A", 2, 7.0),
    ]
    result = compute_bootstrap_rank_sensitivity(
        records, metric_key=_METRIC, planners=["B", "A"], num_bootstrap=10, seed=42
    )
    assert result["planners"] == ["B", "A"]
    # observed_means keys preserve the declared planner order.
    assert list(result["arms"]["het"]["observed_means"]) == ["B", "A"]


# ---------------------------------------------------------------------------
# Reversal detection (exact shape)
# ---------------------------------------------------------------------------


def _reversal_dataset() -> list[dict[str, Any]]:
    """het ranks A>B; mean_matched_homogeneous ranks B>A -> one reversal."""
    return [
        # heterogeneous: A mean 11, B mean 6 -> [A, B]
        _rec("heterogeneous", "A", 1, 10.0),
        _rec("heterogeneous", "A", 2, 12.0),
        _rec("heterogeneous", "B", 1, 5.0),
        _rec("heterogeneous", "B", 2, 7.0),
        # mean_matched_homogeneous: A mean 6, B mean 11 -> [B, A]
        _rec("mean_matched_homogeneous", "A", 1, 5.0),
        _rec("mean_matched_homogeneous", "A", 2, 7.0),
        _rec("mean_matched_homogeneous", "B", 1, 10.0),
        _rec("mean_matched_homogeneous", "B", 2, 12.0),
    ]


def test_rank_reversal_dict_shape_is_pinned() -> None:
    """When the two special arms disagree, the exact reversal dict is emitted."""
    result = compute_bootstrap_rank_sensitivity(
        _reversal_dataset(), metric_key=_METRIC, planners=["A", "B"], num_bootstrap=10, seed=42
    )
    assert len(result["reversals"]) == 1
    rev = result["reversals"][0]
    assert set(rev) == {
        "type",
        "heterogeneous_ranking",
        "mean_matched_homogeneous_ranking",
        "description",
    }
    assert rev["type"] == "rank_order_disagreement"
    assert rev["heterogeneous_ranking"] == ["A", "B"]
    assert rev["mean_matched_homogeneous_ranking"] == ["B", "A"]
    assert rev["description"] == (
        "Heterogeneous ranking ['A', 'B'] differs from homogeneous ['B', 'A']"
    )


def test_no_reversal_when_special_arms_absent() -> None:
    """Arms not named heterogeneous / mean_matched_homogeneous never trigger reversals."""
    # alpha ranks A>B, beta ranks B>A, but neither special arm name is present.
    records = [
        _rec("alpha", "A", 1, 10.0),
        _rec("alpha", "A", 2, 12.0),
        _rec("alpha", "B", 1, 5.0),
        _rec("alpha", "B", 2, 7.0),
        _rec("beta", "A", 1, 5.0),
        _rec("beta", "A", 2, 7.0),
        _rec("beta", "B", 1, 10.0),
        _rec("beta", "B", 2, 12.0),
    ]
    result = compute_bootstrap_rank_sensitivity(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=10, seed=42
    )
    assert result["reversals"] == []


def test_no_reversal_when_special_arms_agree() -> None:
    """Both special arms ranking identically -> no reversal."""
    records = [
        _rec("heterogeneous", "A", 1, 10.0),
        _rec("heterogeneous", "A", 2, 12.0),
        _rec("heterogeneous", "B", 1, 5.0),
        _rec("heterogeneous", "B", 2, 7.0),
        _rec("mean_matched_homogeneous", "A", 1, 10.0),
        _rec("mean_matched_homogeneous", "A", 2, 12.0),
        _rec("mean_matched_homogeneous", "B", 1, 5.0),
        _rec("mean_matched_homogeneous", "B", 2, 7.0),
    ]
    result = compute_bootstrap_rank_sensitivity(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=10, seed=42
    )
    assert result["reversals"] == []


def test_arms_emitted_in_sorted_key_order() -> None:
    """Multiple arms are emitted in sorted (alphabetical) key order."""
    records = [
        _rec("zeta", "A", 1, 10.0),
        _rec("zeta", "A", 2, 12.0),
        _rec("zeta", "B", 1, 5.0),
        _rec("zeta", "B", 2, 7.0),
        _rec("alpha", "A", 1, 10.0),
        _rec("alpha", "A", 2, 12.0),
        _rec("alpha", "B", 1, 5.0),
        _rec("alpha", "B", 2, 7.0),
    ]
    result = compute_bootstrap_rank_sensitivity(
        records, metric_key=_METRIC, planners=["A", "B"], num_bootstrap=10, seed=42
    )
    assert list(result["arms"]) == ["alpha", "zeta"]


# ---------------------------------------------------------------------------
# Ready-status shape + determinism
# ---------------------------------------------------------------------------


def test_ready_status_top_level_field_set_is_pinned() -> None:
    """The ready result exposes exactly the documented top-level field set."""
    result = compute_bootstrap_rank_sensitivity(
        _two_planner_two_seed(),
        metric_key=_METRIC,
        planners=["A", "B"],
        num_bootstrap=10,
        seed=42,
    )
    assert set(result) == {
        "schema_version",
        "status",
        "metric_key",
        "higher_is_safer",
        "common_seeds_count",
        "planners",
        "arms",
        "reversals",
    }
    assert result["schema_version"] == _SCHEMA
    assert result["status"] == "ready"
    assert result["common_seeds_count"] == 2


def test_same_seed_is_bit_for_bit_deterministic() -> None:
    """Same input + same seed -> identical pairwise probabilities (locks the RNG contract)."""
    kwargs = {
        "records": _two_planner_two_seed(),
        "metric_key": _METRIC,
        "planners": ["A", "B"],
        "num_bootstrap": 50,
        "seed": 123,
    }
    a = compute_bootstrap_rank_sensitivity(**kwargs)
    b = compute_bootstrap_rank_sensitivity(**kwargs)
    assert a["arms"]["het"]["pairwise_probabilities"] == b["arms"]["het"]["pairwise_probabilities"]
    assert a["arms"]["het"]["observed_means"] == b["arms"]["het"]["observed_means"]
