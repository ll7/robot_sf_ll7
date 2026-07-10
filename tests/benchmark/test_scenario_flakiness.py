"""Tests for the scenario flakiness audit (issue #4978).

Covers exact-repeat determinism detection, per-cell outcome-stability scoring,
knife-edge flagging, fail-closed behavior, and planner-cell separation.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.scenario_flakiness import (
    SCHEMA_VERSION,
    compute_flakiness_audit,
)


def _rec(scenario_id: str, planner: str, seed: int, success: int) -> dict:
    """Build a minimal episode record with a binary ``success`` outcome."""
    return {
        "episode_id": f"{scenario_id}-{planner}-{seed}",
        "scenario_id": scenario_id,
        "algo": planner,
        "seed": seed,
        "metrics": {"success": success},
    }


def _cell(report: dict, cell_key: str) -> dict:
    """Return the cell dict for ``cell_key`` from an audit report."""
    return next(c for c in report["cells"] if c["cell_key"] == cell_key)


def test_stable_cell_scores_full_stability():
    """A cell whose seeds all agree is fully stable and not knife-edge."""
    records = [_rec("s1", "ppo", seed, 1) for seed in range(4)]
    report = compute_flakiness_audit(records, group_by="algo")
    cell = _cell(report, "s1::ppo")
    assert cell["stability_score"] == pytest.approx(1.0)
    assert cell["success_rate"] == pytest.approx(1.0)
    assert cell["knife_edge"] is False
    assert cell["assessable"] is True
    assert report["summary"]["n_knife_edge_cells"] == 0


def test_knife_edge_cell_is_flagged():
    """A 50/50 success/failure split across seeds is a knife-edge cell."""
    records = [
        _rec("s1", "ppo", 0, 1),
        _rec("s1", "ppo", 1, 1),
        _rec("s1", "ppo", 2, 0),
        _rec("s1", "ppo", 3, 0),
    ]
    report = compute_flakiness_audit(records, group_by="algo", stability_threshold=0.8)
    cell = _cell(report, "s1::ppo")
    assert cell["success_rate"] == pytest.approx(0.5)
    assert cell["stability_score"] == pytest.approx(0.5)
    assert cell["knife_edge"] is True
    assert report["summary"]["n_knife_edge_cells"] == 1
    assert report["summary"]["knife_edge_fraction"] == pytest.approx(1.0)


def test_threshold_boundary_excludes_equal_stability():
    """Stability exactly at the threshold is not flagged (strict less-than)."""
    # 4 seeds, 3 success / 1 fail -> dominant fraction 0.75.
    records = [
        _rec("s1", "ppo", 0, 1),
        _rec("s1", "ppo", 1, 1),
        _rec("s1", "ppo", 2, 1),
        _rec("s1", "ppo", 3, 0),
    ]
    report = compute_flakiness_audit(records, group_by="algo", stability_threshold=0.75)
    cell = _cell(report, "s1::ppo")
    assert cell["stability_score"] == pytest.approx(0.75)
    assert cell["knife_edge"] is False


def test_exact_repeat_determinism_pass():
    """Repeated runs of a seed that agree count as deterministic repeat groups."""
    records = [
        _rec("s1", "ppo", 0, 1),
        _rec("s1", "ppo", 0, 1),  # exact repeat, same outcome
        _rec("s1", "ppo", 1, 1),
        _rec("s1", "ppo", 1, 1),
    ]
    report = compute_flakiness_audit(records, group_by="algo")
    er = report["exact_repeat"]
    assert er["checked_repeat_groups"] == 2
    assert er["nondeterministic_repeat_groups"] == 0
    assert er["is_deterministic"] is True
    assert er["examples"] == []


def test_exact_repeat_nondeterminism_detected():
    """Repeated runs of a seed that disagree are flagged as non-deterministic."""
    records = [
        _rec("s1", "ppo", 0, 1),
        _rec("s1", "ppo", 0, 0),  # same seed, flipped outcome -> reproducibility bug
        _rec("s1", "ppo", 1, 1),
        _rec("s1", "ppo", 2, 1),
    ]
    report = compute_flakiness_audit(records, group_by="algo")
    er = report["exact_repeat"]
    assert er["checked_repeat_groups"] == 1
    assert er["nondeterministic_repeat_groups"] == 1
    assert er["is_deterministic"] is False
    assert len(er["examples"]) == 1
    example = er["examples"][0]
    assert example["scenario_id"] == "s1"
    assert example["seed"] == "0"
    assert sorted(example["outcomes"]) == [0, 1]
    # The nondeterministic seed is recorded on the cell too.
    cell = _cell(report, "s1::ppo")
    assert cell["nondeterministic_seeds"] == 1
    assert cell["has_within_seed_repeats"] is True


def test_determinism_unknown_without_repeat_data():
    """With no exact-repeat data, determinism is unknown (fail closed, not True)."""
    records = [_rec("s1", "ppo", seed, 1) for seed in range(3)]
    report = compute_flakiness_audit(records, group_by="algo")
    er = report["exact_repeat"]
    assert er["checked_repeat_groups"] == 0
    assert er["is_deterministic"] is None


def test_single_seed_cell_not_assessable():
    """A cell with fewer than min_seeds is reported but not scored or flagged."""
    records = [_rec("s1", "ppo", 0, 0)]
    report = compute_flakiness_audit(records, group_by="algo", min_seeds=2)
    cell = _cell(report, "s1::ppo")
    assert cell["assessable"] is False
    assert cell["stability_score"] is None
    assert cell["knife_edge"] is False
    assert report["summary"]["n_assessable_cells"] == 0
    assert report["summary"]["knife_edge_fraction"] is None


def test_planner_cells_are_separated():
    """Records for different planners on the same scenario form distinct cells."""
    records = [
        _rec("s1", "ppo", 0, 1),
        _rec("s1", "ppo", 1, 1),
        _rec("s1", "orca", 0, 0),
        _rec("s1", "orca", 1, 0),
    ]
    report = compute_flakiness_audit(records, group_by="algo")
    keys = {c["cell_key"] for c in report["cells"]}
    assert keys == {"s1::ppo", "s1::orca"}
    assert _cell(report, "s1::ppo")["success_rate"] == pytest.approx(1.0)
    assert _cell(report, "s1::orca")["success_rate"] == pytest.approx(0.0)


def test_missing_outcome_records_counted_not_scored():
    """Records lacking the outcome metric are counted, not silently scored."""
    records = [
        _rec("s1", "ppo", 0, 1),
        _rec("s1", "ppo", 1, 1),
        {"episode_id": "x", "scenario_id": "s1", "algo": "ppo", "seed": 2, "metrics": {}},
    ]
    report = compute_flakiness_audit(records, group_by="algo")
    assert report["summary"]["n_records_missing_outcome"] == 1
    # The scored cell only reflects the two records that carried an outcome.
    assert _cell(report, "s1::ppo")["n_seeds"] == 2


def test_schema_version_and_metadata_present():
    """The report advertises its schema version and audit parameters."""
    report = compute_flakiness_audit([_rec("s1", "ppo", 0, 1)], group_by="algo")
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["outcome_metric"] == "success"
    assert report["stability_threshold"] == pytest.approx(0.8)


def test_empty_records_fail_closed():
    """An audit of no records raises rather than asserting stability."""
    with pytest.raises(ValueError, match="at least one episode record"):
        compute_flakiness_audit([])


def test_invalid_threshold_rejected():
    """A threshold outside (0, 1] is rejected."""
    with pytest.raises(ValueError, match="stability_threshold"):
        compute_flakiness_audit([_rec("s1", "ppo", 0, 1)], stability_threshold=1.5)


@pytest.mark.parametrize("min_seeds", [0, -1])
def test_nonpositive_min_seeds_rejected(min_seeds: int):
    """A nonpositive evidence minimum cannot silently assess every cell."""
    with pytest.raises(ValueError, match="min_seeds"):
        compute_flakiness_audit([_rec("s1", "ppo", 0, 1)], min_seeds=min_seeds)


def test_all_unusable_records_fail_closed():
    """Records without a usable scenario/outcome cannot yield an empty success report."""
    records = [
        {"scenario_id": "s1", "algo": "ppo", "seed": 0, "metrics": {}},
        {"algo": "ppo", "seed": 1, "metrics": {"success": 1}},
    ]
    with pytest.raises(ValueError, match="no usable evidence"):
        compute_flakiness_audit(records, group_by="algo")
