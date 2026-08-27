"""Tests for effective algorithm-branch enumeration and witness coverage (issue #7937)."""

from __future__ import annotations

from robot_sf.benchmark.effective_algorithm_branches import (
    WITNESS_KINDS,
    check_witness_coverage,
    enumerate_effective_branches,
)

HYBRID_CANDIDATE = {
    "id": "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "algo": "hybrid_rule_local_planner",
    "scenario_algo_overrides": {
        "francis2023_leave_group": {
            "algo": "orca",
            "base_config_path": "configs/algos/issue707_orca_tuned.yaml",
        },
        "classic_realworld_double_bottleneck_high": {
            "algo": "orca",
            "base_config_path": "configs/algos/issue707_orca_tuned.yaml",
        },
    },
}


def test_enumerate_effective_branches_lists_overrides() -> None:
    branches = enumerate_effective_branches(HYBRID_CANDIDATE)
    assert branches == [
        {
            "arm": "scenario_adaptive_hybrid_orca_v2_collision_guard",
            "scenario": "classic_realworld_double_bottleneck_high",
            "algorithm": "orca",
        },
        {
            "arm": "scenario_adaptive_hybrid_orca_v2_collision_guard",
            "scenario": "francis2023_leave_group",
            "algorithm": "orca",
        },
    ]


def test_enumerate_respects_allowed_scenario_ids() -> None:
    branches = enumerate_effective_branches(
        HYBRID_CANDIDATE,
        allowed_scenario_ids={"francis2023_leave_group"},
    )
    assert len(branches) == 1
    assert branches[0]["scenario"] == "francis2023_leave_group"


def test_enumerate_ignores_malformed_overrides() -> None:
    candidate = {
        "id": "arm",
        "algo": "hybrid_rule_local_planner",
        "scenario_algo_overrides": {
            "good": {"algo": "orca"},
            "bad": "not-a-mapping",
        },
    }
    branches = enumerate_effective_branches(candidate)
    assert len(branches) == 1
    assert branches[0]["scenario"] == "good"


def test_enumerate_no_overrides_returns_empty() -> None:
    assert enumerate_effective_branches({"id": "plain", "algo": "sf"}) == []


def test_witness_coverage_passes_with_exact_branch_key() -> None:
    branches = enumerate_effective_branches(HYBRID_CANDIDATE)
    witnesses = [
        {
            "kind": "scenario_cell",
            "branch_key": (
                "scenario_adaptive_hybrid_orca_v2_collision_guard|francis2023_leave_group|orca"
            ),
        },
        {
            "kind": "scenario_cell",
            "branch_key": (
                "scenario_adaptive_hybrid_orca_v2_collision_guard|"
                "classic_realworld_double_bottleneck_high|orca"
            ),
        },
    ]
    assert check_witness_coverage(branches, witnesses) == []


def test_witness_coverage_passes_with_field_match() -> None:
    branches = enumerate_effective_branches(HYBRID_CANDIDATE)
    witnesses = [
        {
            "kind": "episode_row",
            "arm": "scenario_adaptive_hybrid_orca_v2_collision_guard",
            "scenario": "francis2023_leave_group",
            "algorithm": "orca",
        },
        {
            "kind": "episode_row",
            "arm": "scenario_adaptive_hybrid_orca_v2_collision_guard",
            "scenario": "classic_realworld_double_bottleneck_high",
            "algorithm": "orca",
        },
    ]
    assert check_witness_coverage(branches, witnesses) == []


def test_witness_coverage_missing_branch_fails() -> None:
    branches = enumerate_effective_branches(HYBRID_CANDIDATE)
    witnesses = [
        {
            "kind": "scenario_cell",
            "branch_key": (
                "scenario_adaptive_hybrid_orca_v2_collision_guard|francis2023_leave_group|orca"
            ),
        }
    ]
    problems = check_witness_coverage(branches, witnesses)
    assert len(problems) == 1
    assert "classic_realworld_double_bottleneck_high" in problems[0]
    assert "missing diagnostic witness" in problems[0]


def test_witness_coverage_wrong_arm_attribution_fails() -> None:
    branches = enumerate_effective_branches(HYBRID_CANDIDATE)
    witnesses = [
        {
            "kind": "scenario_cell",
            "branch_key": "wrong_arm|francis2023_leave_group|orca",
        },
        {
            "kind": "scenario_cell",
            "branch_key": (
                "scenario_adaptive_hybrid_orca_v2_collision_guard|"
                "classic_realworld_double_bottleneck_high|orca"
            ),
        },
    ]
    problems = check_witness_coverage(branches, witnesses)
    assert any("francis2023_leave_group" in problem for problem in problems)


def test_witness_coverage_unknown_kind_ignored() -> None:
    branches = enumerate_effective_branches(HYBRID_CANDIDATE)
    witnesses = [
        {
            "kind": "not_a_kind",
            "branch_key": (
                "scenario_adaptive_hybrid_orca_v2_collision_guard|francis2023_leave_group|orca"
            ),
        }
    ]
    assert check_witness_coverage(branches, witnesses)  # unknown kind gives no coverage


def test_witness_kinds_vocabulary() -> None:
    assert WITNESS_KINDS == {"scenario_cell", "episode_row", "diagnostic_row"}
