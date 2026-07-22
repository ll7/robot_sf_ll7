"""Tests for fixed-episode per-context determinism smoke check (issue #6126)."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.step_trace_comparator import (
    canonical_step_trace_digest,
    compare_step_traces,
    find_first_trace_difference,
)
from scripts.validation.run_per_context_determinism_smoke import (
    run_determinism_smoke,
    run_negative_test_smoke,
)


def test_per_context_determinism_smoke() -> None:
    """Run two in-process episodes and verify that canonical step traces match."""
    res = run_determinism_smoke(horizon=20)
    assert res["status"] == "pass"
    assert res["step_count"] == 20
    assert isinstance(res["trace_sha256"], str)
    assert len(res["trace_sha256"]) == 64


def test_per_context_determinism_smoke_negative() -> None:
    """Verify that a trace divergence produces an actionable first-difference report."""
    res = run_negative_test_smoke(horizon=20)
    assert res["status"] == "pass"
    assert res["negative_test"] is True
    assert "steps[2].robot.position[0]" in res["diff_report"]


def test_step_trace_comparator_unit() -> None:
    """Unit test canonical step-trace comparator edge cases and difference reports."""
    t1 = {
        "steps": [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 1.0], "heading": 0.0},
            },
            {
                "step": 1,
                "time_s": 0.2,
                "robot": {"position": [0.1, 1.0], "heading": 0.1},
            },
        ]
    }
    t2 = {
        "steps": [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 1.0], "heading": 0.0},
            },
            {
                "step": 1,
                "time_s": 0.2,
                "robot": {"position": [0.1, 1.0], "heading": 0.1},
            },
        ]
    }

    # Identical traces
    equal, diff = compare_step_traces(t1, t2)
    assert equal is True
    assert diff is None

    # Digest stability
    d1 = canonical_step_trace_digest(t1)
    d2 = canonical_step_trace_digest(t2)
    assert d1 == d2
    assert len(d1) == 64

    # Float mismatch
    t3 = {
        "steps": [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 1.0], "heading": 0.0},
            },
            {
                "step": 1,
                "time_s": 0.2,
                "robot": {"position": [0.15, 1.0], "heading": 0.1},
            },
        ]
    }
    equal_diff, diff_msg = compare_step_traces(t1, t3)
    assert equal_diff is False
    assert diff_msg is not None
    assert "Value mismatch at 'steps[1].robot.position[0]'" in diff_msg

    # Key mismatch
    t4 = {
        "steps": [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 1.0]},
            }
        ]
    }
    t5 = {
        "steps": [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 1.0], "extra_key": True},
            }
        ]
    }
    equal_key, diff_key = compare_step_traces(t4, t5)
    assert equal_key is False
    assert diff_key is not None
    assert "Key mismatch at 'steps[0].robot'" in diff_key

    # Length mismatch
    t6 = {"steps": t1["steps"][:1]}
    equal_len, diff_len = compare_step_traces(t1, t6)
    assert equal_len is False
    assert diff_len is not None
    assert "Length mismatch at 'steps'" in diff_len

    # Type mismatch
    diff_type = find_first_trace_difference("string_val", 123, path="test")
    assert diff_type is not None
    assert "Type mismatch at 'test'" in diff_type

    # Invalid input format
    with pytest.raises(ValueError, match="Input trace dictionary must contain a 'steps' list"):
        compare_step_traces({"invalid": 123}, t1)
