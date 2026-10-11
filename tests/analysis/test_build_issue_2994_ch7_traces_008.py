"""Tests for the issue #2994 Chapter 7 trace selection helper."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

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
        "orca": [_row(seed, "success" if seed <= 20 else "collision") for seed in range(1, 41)],
        "social_force": [
            _row(seed, "success" if seed <= 15 else "timeout" if seed <= 30 else "collision")
            for seed in range(1, 41)
        ],
    }
    target = {
        "orca": Counter({"success": 20, "collision": 10}),
        "social_force": Counter({"success": 15, "timeout": 15}),
    }

    seeds, status = builder.select_shared_subset(rows_by_planner, target)

    assert status == "exact"
    assert len(seeds) == 30
    assert (
        builder.outcome_counts([rows_by_planner["orca"][seed - 1] for seed in seeds])
        == target["orca"]
    )
    assert (
        builder.outcome_counts([rows_by_planner["social_force"][seed - 1] for seed in seeds])
        == target["social_force"]
    )


def test_outcome_label_prefers_collision_and_timeout_for_failures() -> None:
    assert builder.outcome_label(_row(1, "success")) == "success"
    assert builder.outcome_label(_row(2, "collision")) == "collision"
    assert builder.outcome_label(_row(3, "timeout")) == "timeout"
