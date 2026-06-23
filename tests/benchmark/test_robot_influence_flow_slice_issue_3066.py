"""Unit tests for the issue #3066 robot-influence flow slice driver.

These tests exercise the pure classification/aggregation/delta-vs-variance logic
on small *synthetic* episode rows. They deliberately do NOT launch a live
campaign: the simulator is exercised only by the script's runtime path, while the
analysis logic that decides the overall classification is what needs locking down.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "run_robot_influence_flow_slice_issue_3066.py"
)
_spec = importlib.util.spec_from_file_location("rif3066", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
rif = importlib.util.module_from_spec(_spec)
# Register before exec so dataclass annotation resolution can find the module.
sys.modules["rif3066"] = rif
_spec.loader.exec_module(rif)


def _record(
    *,
    algo: str,
    scenario: str,
    seed: int,
    status: str = "failure",
    accel_delta: float | None = 0.3,
    turn_delta: float | None = 0.5,
    near_samples: float = 5.0,
    extra_meta: dict | None = None,
) -> dict:
    """Build a synthetic runner episode record.

    Returns:
        A dict shaped like a runner JSONL episode record.
    """
    metrics = {
        "ped_impact_accel_delta_mean": accel_delta,
        "ped_impact_turn_rate_delta_mean": turn_delta,
        "ped_impact_near_samples": near_samples,
        "ped_impact_far_samples": 100.0,
        "ped_impact_ped_count": 2.0,
        "ped_impact_accel_delta_valid": 1.0,
        "ped_impact_turn_rate_delta_valid": 1.0,
        "avg_speed": 1.0,
        "robot_ped_within_5m_frac": 0.3,
        "social_proxemic_intrusion_frac": 0.1,
        "min_distance": 1.5,
        "mean_clearance": 5.0,
        "near_misses": 1.0,
        "success": status == "success",
        "total_collision_count": 0.0,
        "ped_collision_count": 0.0,
    }
    rec = {
        "algo": algo,
        "scenario_id": scenario,
        "seed": seed,
        "status": status,
        "metrics": metrics,
    }
    if extra_meta is not None:
        rec["algorithm_metadata"] = extra_meta
    return rec


# --------------------------------------------------------------------------- #
# Vocabulary stability.
# --------------------------------------------------------------------------- #


def test_classification_vocabulary_stable() -> None:
    """The classification vocabulary literals must remain stable."""
    assert rif.CLASSIFICATIONS == ("benchmark", "diagnostic", "blocked", "non_claim")
    assert rif.CLAIM_BOUNDARY == "diagnostic_only"
    assert rif.EVIDENCE_TIER == "smoke"
    assert rif.PAPER_GRADE is False


# --------------------------------------------------------------------------- #
# Row classification + fail-closed.
# --------------------------------------------------------------------------- #


def test_nav_failure_is_still_usable() -> None:
    """A nav timeout (status='failure') is a usable physics row, not fail-closed."""
    row = rif.classify_row(_record(algo="social_force", scenario="corridor", seed=111))
    assert row.usable is True
    assert row.degraded_reason is None
    assert row.policy == "social_force"


@pytest.mark.parametrize("bad_status", ["degraded", "fallback", "unavailable", "error", "failed"])
def test_degraded_rows_are_failclosed(bad_status: str) -> None:
    """Degraded/fallback/unavailable/error rows are never usable evidence."""
    row = rif.classify_row(_record(algo="orca", scenario="corridor", seed=111, status=bad_status))
    assert row.usable is False
    assert row.degraded_reason is not None


def test_algorithm_fallback_flag_is_failclosed() -> None:
    """An algorithm-level fallback flag forces a row to fail-closed."""
    row = rif.classify_row(
        _record(
            algo="orca",
            scenario="corridor",
            seed=111,
            status="success",
            extra_meta={"fallback": True},
        )
    )
    assert row.usable is False
    assert "fallback" in (row.degraded_reason or "")


def test_missing_metric_becomes_nan() -> None:
    """A missing influence metric coerces to NaN rather than raising."""
    row = rif.classify_row(_record(algo="orca", scenario="c", seed=1, accel_delta=None))
    assert math.isnan(row.influence["ped_impact_accel_delta_mean"])


# --------------------------------------------------------------------------- #
# Aggregation: degraded rows excluded from means.
# --------------------------------------------------------------------------- #


def test_aggregate_excludes_degraded_from_means() -> None:
    """Degraded rows are counted but excluded from metric means (fail-closed)."""
    rows = [
        rif.classify_row(_record(algo="sf", scenario="c", seed=111, accel_delta=0.2)),
        rif.classify_row(_record(algo="sf", scenario="c", seed=112, accel_delta=0.4)),
        rif.classify_row(
            _record(algo="sf", scenario="c", seed=113, status="degraded", accel_delta=99.0)
        ),
    ]
    agg = rif.aggregate_cell(rows)
    assert agg.n_rows == 3
    assert agg.n_usable == 2
    assert agg.n_degraded == 1
    # Mean must ignore the degraded 99.0 outlier -> (0.2 + 0.4)/2 = 0.3.
    assert agg.influence_mean["ped_impact_accel_delta_mean"] == pytest.approx(0.3)


def test_aggregate_requires_rows() -> None:
    """Aggregating an empty cell raises a clear error."""
    with pytest.raises(ValueError, match="at least one row"):
        rif.aggregate_cell([])


# --------------------------------------------------------------------------- #
# Determinism.
# --------------------------------------------------------------------------- #


def test_aggregation_is_deterministic() -> None:
    """Aggregation over the same rows yields identical means twice."""
    rows = [
        rif.classify_row(_record(algo="sf", scenario="c", seed=s, accel_delta=0.1 * i))
        for i, s in enumerate((111, 112, 113), start=1)
    ]
    a = rif.aggregate_cell(rows)
    b = rif.aggregate_cell(rows)
    assert a.influence_mean == b.influence_mean
    assert a.influence_std == b.influence_std


# --------------------------------------------------------------------------- #
# Flow-delta-vs-variance logic.
# --------------------------------------------------------------------------- #


def _build_aggregates(spec: dict[tuple[str, str], list[float]]):
    """Build aggregates from {(policy, scenario): [accel_deltas...]} synthetic data.

    Returns:
        A mapping of (policy, scenario) -> aggregate.
    """
    aggregates = {}
    for (policy, scenario), deltas in spec.items():
        rows = [
            rif.classify_row(_record(algo=policy, scenario=scenario, seed=111 + i, accel_delta=v))
            for i, v in enumerate(deltas)
        ]
        aggregates[(policy, scenario)] = rif.aggregate_cell(rows)
    return aggregates


def test_delta_exceeds_seed_variance() -> None:
    """A large, tight separation between policies exceeds pooled seed variance."""
    aggregates = _build_aggregates(
        {
            ("social_force", "corridor"): [0.10, 0.11, 0.09],  # mean ~0.10, tiny std
            ("orca", "corridor"): [0.90, 0.91, 0.89],  # mean ~0.90, tiny std
        }
    )
    deltas = rif.compute_flow_deltas(aggregates, "social_force", "orca")
    accel = next(d for d in deltas if d.metric == "ped_impact_accel_delta_mean")
    assert accel.powered is True
    assert accel.exceeds_variance is True
    assert accel.delta == pytest.approx(0.80, abs=0.02)


def test_delta_within_seed_variance_is_null() -> None:
    """Overlapping noisy policies produce a within-variance (null) delta."""
    aggregates = _build_aggregates(
        {
            ("social_force", "corridor"): [0.10, 0.90, 0.50],  # huge std
            ("orca", "corridor"): [0.15, 0.85, 0.55],  # huge std, similar mean
        }
    )
    deltas = rif.compute_flow_deltas(aggregates, "social_force", "orca")
    accel = next(d for d in deltas if d.metric == "ped_impact_accel_delta_mean")
    assert accel.powered is True
    assert accel.exceeds_variance is False


def test_underpowered_cell_is_not_powered() -> None:
    """A single usable row per cell is underpowered and never asserts influence."""
    aggregates = _build_aggregates(
        {
            ("social_force", "corridor"): [0.10],
            ("orca", "corridor"): [0.90],
        }
    )
    deltas = rif.compute_flow_deltas(aggregates, "social_force", "orca")
    accel = next(d for d in deltas if d.metric == "ped_impact_accel_delta_mean")
    assert accel.powered is False
    assert accel.exceeds_variance is False


# --------------------------------------------------------------------------- #
# Overall classification (fail-closed -> blocked; null -> diagnostic).
# --------------------------------------------------------------------------- #


def test_blocked_when_policy_has_zero_usable_rows() -> None:
    """If a requested policy has no usable rows, the campaign is blocked."""
    rows = [
        rif.classify_row(_record(algo="social_force", scenario="c", seed=111)),
        rif.classify_row(_record(algo="orca", scenario="c", seed=111, status="degraded")),
    ]
    cells: dict[tuple[str, str], list] = {}
    for r in rows:
        cells.setdefault((r.policy, r.scenario), []).append(r)
    aggregates = {k: rif.aggregate_cell(v) for k, v in cells.items()}
    deltas = rif.compute_flow_deltas(aggregates, "social_force", "orca")
    classification, rationale = rif.classify_overall(aggregates, deltas, ("social_force", "orca"))
    assert classification == "blocked"
    assert "orca" in rationale


def test_diagnostic_on_null_result() -> None:
    """A within-variance comparison classifies as diagnostic (honest null)."""
    aggregates = _build_aggregates(
        {
            ("social_force", "corridor"): [0.10, 0.90, 0.50],
            ("orca", "corridor"): [0.15, 0.85, 0.55],
        }
    )
    deltas = rif.compute_flow_deltas(aggregates, "social_force", "orca")
    classification, rationale = rif.classify_overall(aggregates, deltas, ("social_force", "orca"))
    assert classification == "diagnostic"
    assert "null" in rationale or "within" in rationale


def test_diagnostic_even_when_signal_present() -> None:
    """A powered exceed-variance delta still stays diagnostic at v0 smoke tier."""
    aggregates = _build_aggregates(
        {
            ("social_force", "corridor"): [0.10, 0.11, 0.09],
            ("orca", "corridor"): [0.90, 0.91, 0.89],
        }
    )
    deltas = rif.compute_flow_deltas(aggregates, "social_force", "orca")
    classification, rationale = rif.classify_overall(aggregates, deltas, ("social_force", "orca"))
    assert classification == "diagnostic"
    assert "exceed" in rationale


def test_classification_in_vocabulary() -> None:
    """Whatever classify_overall returns must be in the stable vocabulary."""
    aggregates = _build_aggregates(
        {
            ("social_force", "corridor"): [0.10, 0.20, 0.30],
            ("orca", "corridor"): [0.40, 0.50, 0.60],
        }
    )
    deltas = rif.compute_flow_deltas(aggregates, "social_force", "orca")
    classification, _ = rif.classify_overall(aggregates, deltas, ("social_force", "orca"))
    assert classification in rif.CLASSIFICATIONS


def test_build_report_shape_and_separation() -> None:
    """build_report emits separated influence/nav blocks and a valid classification."""
    records = []
    for seed in (111, 112, 113):
        records.append(
            _record(algo="social_force", scenario="corridor", seed=seed, accel_delta=0.1)
        )
        records.append(_record(algo="orca", scenario="corridor", seed=seed, accel_delta=0.9))
    report = rif.build_report(
        records=records,
        policies=("social_force", "orca"),
        scenarios=("corridor",),
        seeds=(111, 112, 113),
        git_hash="deadbeef",
        horizon=240,
    )
    assert report["classification"] in rif.CLASSIFICATIONS
    assert report["claim_boundary"] == "diagnostic_only"
    assert report["paper_grade"] is False
    # Influence and nav are reported in separate structures.
    assert "flow_deltas" in report
    assert all("nav_mean" in agg for agg in report["aggregates"])
    # Markdown renders without error.
    md = rif.render_markdown(report)
    assert "Robot-influence flow deltas" in md
    assert "Nav performance" in md
