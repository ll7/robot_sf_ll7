"""Tests for fixed-episode per-context determinism smoke check (issue #6126)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.step_trace_comparator import (
    canonical_step_trace_digest,
    compare_step_traces,
    find_first_trace_difference,
)
from scripts.validation import run_per_context_determinism_smoke as smoke_module
from scripts.validation.run_per_context_determinism_smoke import (
    DEFAULT_SCENARIO_ID,
    run_determinism_smoke,
    run_negative_test_smoke,
    run_single_episode_trace,
)


def test_per_context_determinism_smoke() -> None:
    """Run two in-process episodes and verify that canonical step traces match."""
    res = run_determinism_smoke(horizon=20)
    assert res["status"] == "pass"
    assert res["scenario_id"] == DEFAULT_SCENARIO_ID
    assert res["planner"] == "goal"
    assert res["seed"] == 42
    assert res["horizon"] == 20
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

    # Matching non-finite values are invalid trace evidence, not deterministic success.
    non_finite_trace = {"steps": [{"value": float("nan")}]}
    equal_non_finite, diff_non_finite = compare_step_traces(non_finite_trace, non_finite_trace)
    assert equal_non_finite is False
    assert diff_non_finite is not None
    assert "Non-finite value at 'steps[0].value'" in diff_non_finite

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


def _episode_row(
    *, scenario_id: str = DEFAULT_SCENARIO_ID, seed: int = 42, horizon: int = 20
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "horizon": horizon,
        "algorithm_metadata": {
            "algorithm": "goal",
            "simulation_step_trace": {"steps": [{"step": 0}]},
        },
    }


def _patch_scenario_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        smoke_module,
        "load_scenario_matrix",
        lambda _path: [
            {"name": DEFAULT_SCENARIO_ID, "map_file": "map.svg", "seeds": [101, 102, 103]},
            {"name": "other_scenario", "map_file": "map.svg", "seeds": [101]},
        ],
    )


def test_single_episode_trace_schedules_exact_requested_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The smoke narrows a manifest to one scenario and one requested seed."""
    _patch_scenario_loader(monkeypatch)
    captured: dict[str, Any] = {}

    def _fake_run_map_batch(
        scenarios: list[dict[str, Any]], out_path: str, **kwargs: Any
    ) -> dict[str, Any]:
        captured["scenarios"] = scenarios
        captured["kwargs"] = kwargs
        Path(out_path).write_text(json.dumps(_episode_row()) + "\n", encoding="utf-8")
        return {"written": 1}

    monkeypatch.setattr(smoke_module, "run_map_batch", _fake_run_map_batch)

    row, _trace = run_single_episode_trace(horizon=20)

    assert row["scenario_id"] == DEFAULT_SCENARIO_ID
    assert captured["scenarios"] == [
        {"name": DEFAULT_SCENARIO_ID, "map_file": "map.svg", "seeds": [42]}
    ]
    assert captured["kwargs"]["scenario_path"] == smoke_module.DEFAULT_SCENARIO_PATH
    assert captured["kwargs"]["horizon"] == 20
    assert captured["kwargs"]["algo"] == "goal"
    assert captured["kwargs"]["workers"] == 1
    assert captured["kwargs"]["resume"] is False


@pytest.mark.parametrize("matching_count", [0, 2])
def test_single_episode_trace_rejects_ambiguous_scenario_selection(
    monkeypatch: pytest.MonkeyPatch,
    matching_count: int,
) -> None:
    """Fail closed unless the manifest identifies exactly one requested scenario."""
    monkeypatch.setattr(
        smoke_module,
        "load_scenario_matrix",
        lambda _path: [
            {"name": DEFAULT_SCENARIO_ID, "map_file": "map.svg", "seeds": [101]}
            for _ in range(matching_count)
        ],
    )

    with pytest.raises(RuntimeError, match=rf"found {matching_count}\."):
        run_single_episode_trace(horizon=20)


def test_negative_cli_reports_clean_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Negative-mode validation failures return one clean CI error without a traceback."""

    def _fail_negative_smoke(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic comparator failure")

    monkeypatch.setattr(smoke_module, "run_negative_test_smoke", _fail_negative_smoke)
    monkeypatch.setattr(sys, "argv", ["run_per_context_determinism_smoke.py", "--negative-test"])

    assert smoke_module.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Negative test FAILED: synthetic comparator failure\n"


def test_single_episode_trace_rejects_multirow_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the old path emitted nine rows and silently consumed the first."""
    _patch_scenario_loader(monkeypatch)

    def _fake_run_map_batch(
        _scenarios: list[dict[str, Any]], out_path: str, **_kwargs: Any
    ) -> dict[str, Any]:
        rows = [_episode_row(seed=101) for _ in range(9)]
        Path(out_path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return {"written": 9}

    monkeypatch.setattr(smoke_module, "run_map_batch", _fake_run_map_batch)

    with pytest.raises(RuntimeError, match="must execute exactly one episode"):
        run_single_episode_trace(horizon=20)


def test_single_episode_trace_rejects_false_seed_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: requested seed 42 must not conceal an actual seed-101 episode."""
    _patch_scenario_loader(monkeypatch)

    def _fake_run_map_batch(
        _scenarios: list[dict[str, Any]], out_path: str, **_kwargs: Any
    ) -> dict[str, Any]:
        Path(out_path).write_text(json.dumps(_episode_row(seed=101)) + "\n", encoding="utf-8")
        return {"written": 1}

    monkeypatch.setattr(smoke_module, "run_map_batch", _fake_run_map_batch)

    with pytest.raises(RuntimeError, match="Executed episode identity does not match"):
        run_single_episode_trace(horizon=20)
