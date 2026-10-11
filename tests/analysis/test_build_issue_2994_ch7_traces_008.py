"""Tests for the issue #2994 Chapter 7 trace selection helper."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "build_issue_2994_ch7_traces_008.py"
)
SPEC = importlib.util.spec_from_file_location("build_issue_2994_ch7_traces_008", SCRIPT_PATH)
assert SPEC is not None
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _row(seed: int, label: str) -> dict:
    outcome = {
        "collision_event": label == "collision",
        "timeout_event": label == "timeout",
        "route_complete": label == "success",
    }
    return {
        "seed": seed,
        "metrics": {"success": label == "success"},
        "outcome": outcome,
        "termination_reason": "timeout" if label == "timeout" else label,
    }


def test_select_shared_subset_matches_two_planner_marginals_exactly() -> None:
    rows_by_planner = {
        "orca": [
            _row(seed, "collision" if 1031 <= seed <= 1040 else "success")
            for seed in range(1001, 1051)
        ],
        "social_force": [
            _row(seed, "success" if seed <= 1030 else "timeout") for seed in range(1001, 1051)
        ],
    }
    target = {
        "orca": Counter({"success": 20, "collision": 10}),
        "social_force": Counter({"success": 15, "timeout": 15}),
    }

    seeds, status = builder.select_shared_subset(rows_by_planner, target)

    # The first 30 seeds are all successes; an exact match must skip some of them.
    assert seeds == [*range(1001, 1016), *range(1031, 1046)]
    assert status == "exact"
    assert len(seeds) == 30
    assert (
        builder.outcome_counts([rows_by_planner["orca"][seed - 1001] for seed in seeds])
        == target["orca"]
    )
    assert (
        builder.outcome_counts([rows_by_planner["social_force"][seed - 1001] for seed in seeds])
        == target["social_force"]
    )


def test_select_shared_subset_minimizes_marginal_error_when_exact_is_impossible() -> None:
    labels = {
        1001: ("collision", "timeout"),
        1002: ("collision", "timeout"),
        1003: ("collision", "timeout"),
        1004: ("success", "success"),
        1005: ("success", "success"),
    }
    rows = {
        planner: [_row(seed, pair[index]) for seed, pair in labels.items()]
        for index, planner in enumerate(("orca", "social_force"))
    }
    target = {
        "orca": Counter({"success": 2, "collision": 1}),
        "social_force": Counter({"success": 3}),
    }

    seeds, status = builder.select_shared_subset(rows, target, subset_size=3)

    assert status == "nearest"
    assert seeds == [1001, 1004, 1005]

    def distance(subset: tuple[int, ...] | list[int]) -> int:
        total = 0
        for index, planner in enumerate(rows):
            counts = Counter(labels[seed][index] for seed in subset)
            total += sum(
                abs(counts[label] - target[planner][label])
                for label in counts.keys() | target[planner].keys()
            )
        return total

    assert distance(seeds) == 2
    assert distance(seeds) == min(distance(subset) for subset in combinations(labels, 3))
    assert distance([1001, 1002, 1003]) == 10


@pytest.mark.parametrize("seed", [1000, 1201], ids=["below-range", "above-range"])
@pytest.mark.parametrize("exact", [True, False], ids=["exact", "nearest"])
def test_select_shared_subset_rejects_non_development_seeds(seed: int, exact: bool) -> None:
    rows = {"orca": [_row(seed, "success")]}
    target = {"orca": Counter({"success" if exact else "collision": 1})}

    with pytest.raises(ValueError, match=r"development range 1001\.\.1200"):
        builder.select_shared_subset(rows, target, subset_size=1)


def test_select_shared_subset_accepts_development_range_boundaries() -> None:
    rows = {"orca": [_row(seed, "success") for seed in (1001, 1200)]}
    assert builder.select_shared_subset(rows, {"orca": Counter(success=2)}, subset_size=2) == (
        [1001, 1200],
        "exact",
    )


def test_select_shared_subset_rejects_too_few_shared_seeds() -> None:
    rows = {"orca": [_row(1001, "success")], "social_force": [_row(1002, "success")]}
    target = {planner: Counter(success=1) for planner in rows}
    with pytest.raises(ValueError, match="not enough shared development seeds"):
        builder.select_shared_subset(rows, target, subset_size=1)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"metrics": {"success": True}}, "success"),
        ({"outcome": {"collision_event": True}}, "collision"),
        ({"metrics": {"total_collision_count": 1}}, "collision"),
        ({"outcome": {"timeout_event": True}}, "timeout"),
        ({"termination_reason": "timeout"}, "timeout"),
        ({"outcome": {"route_complete": True}}, "route_complete_non_success"),
        ({}, "other"),
        ({"metrics": None, "outcome": None}, "other"),
        (
            {"outcome": {"collision_event": True, "timeout_event": True, "route_complete": True}},
            "collision",
        ),
        (
            {"metrics": {"total_collision_count": 1}, "termination_reason": "timeout"},
            "collision",
        ),
        (
            {"outcome": {"timeout_event": True, "route_complete": True}},
            "timeout",
        ),
        (
            {"termination_reason": "timeout", "outcome": {"route_complete": True}},
            "timeout",
        ),
        (
            {
                "metrics": {"success": True, "total_collision_count": 1},
                "outcome": {
                    "collision_event": True,
                    "timeout_event": True,
                    "route_complete": True,
                },
                "termination_reason": "timeout",
            },
            "success",
        ),
    ],
    ids=[
        "success",
        "collision-event",
        "collision-count",
        "timeout-event",
        "timeout-reason",
        "route-complete",
        "empty",
        "null",
        "collision-before-timeout-and-route",
        "collision-count-before-timeout",
        "timeout-before-route",
        "timeout-reason-before-route",
        "success-before-all-failures",
    ],
)
def test_outcome_label_paths_and_precedence(row: dict, expected: str) -> None:
    assert builder.outcome_label(row) == expected
