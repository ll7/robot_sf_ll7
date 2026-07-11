"""Unit contracts for the issue #5319 DWA route-rescue trace exporter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "benchmark" / "trace_dwa_route_rescue_issue_5319.py"


def _load_trace_module():
    spec = importlib.util.spec_from_file_location("_dwa_trace_issue_5319", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_route_progress_summary_skips_and_records_non_finite_trace_values() -> None:
    """NaN/Infinity diagnostic cells never propagate into derived evidence."""
    trace = _load_trace_module()

    summary = trace._route_progress_summary(
        [
            {"distance_to_goal_m": "3.0", "route_progress_from_start_m": "0.0"},
            {"distance_to_goal_m": "nan", "route_progress_from_start_m": "inf"},
            {"distance_to_goal_m": "2.0", "route_progress_from_start_m": "1.0"},
        ]
    )

    assert summary["initial_distance_to_goal_m"] == 3.0
    assert summary["final_distance_to_goal_m"] == 2.0
    assert summary["skipped_non_finite_rows"] == 1
    assert summary["skipped_non_finite_cells"] == 2


def test_load_scenario_resolves_relative_paths_from_matrix_parent(monkeypatch) -> None:
    """The matrix parent, rather than its filename, is the loader base directory."""
    trace = _load_trace_module()
    matrix = ROOT / "configs" / "scenarios" / "classic_interactions.yaml"
    captured: dict[str, Path] = {}

    def fake_load(path: Path, *, base_dir: Path):
        captured["path"] = path
        captured["base_dir"] = base_dir
        return [{"name": "target"}]

    monkeypatch.setattr(trace, "load_scenarios", fake_load)

    assert trace._load_scenario("target", 7, matrix) == {"name": "target", "seeds": [7]}
    assert captured == {"path": matrix, "base_dir": matrix.parent}


def test_repo_relative_path_avoids_worktree_specific_evidence_paths() -> None:
    """Durable evidence records repository-relative paths, not the active worktree."""
    trace = _load_trace_module()
    assert trace._repo_relative_path(ROOT / "configs/algos/dwa_route_rescue.yaml") == (
        "configs/algos/dwa_route_rescue.yaml"
    )


def test_route_rescue_activation_steps_extracts_active_steps() -> None:
    """Only steps with route_rescue_active=True are returned."""
    trace = _load_trace_module()

    steps = [
        {"step": 0, "route_rescue_active": False},
        {"step": 1, "route_rescue_active": True},
        {"step": 2, "route_rescue_active": True},
        {"step": 3, "route_rescue_active": False},
    ]
    assert trace._route_rescue_activation_steps(steps) == [1, 2]


def test_feasibility_slowdown_steps_extracts_active_steps() -> None:
    """Only steps with feasibility_slowdown_active=True are returned."""
    trace = _load_trace_module()

    steps = [
        {"step": 0, "feasibility_slowdown_active": False},
        {"step": 1, "feasibility_slowdown_active": True},
        {"step": 2, "feasibility_slowdown_active": False},
    ]
    assert trace._feasibility_slowdown_steps(steps) == [1]


def test_flatten_step_includes_rescue_fields() -> None:
    """Flattened step rows include the rescue-specific diagnostic fields."""
    trace = _load_trace_module()

    step = {
        "step": 5,
        "selected_command": [0.5, 0.1],
        "selected_score": 2.5,
        "constraint_reason": "best_feasible",
        "candidate_total": 63,
        "candidate_feasible": 60,
        "candidate_infeasible": 3,
        "dynamic_window": {"v_min": 0.0, "v_max": 1.0, "w_min": -0.3, "w_max": 0.3},
        "target_goal": {"kind": "next", "x": 5.0, "y": 3.0},
        "distance_to_goal_m": 2.5,
        "route_rescue_active": True,
        "route_rescue_type": "route_rescue",
        "feasibility_slowdown_active": False,
    }
    row = trace._flatten_step(step, episode_id="test", scenario_id="test_scenario", seed=42)
    assert row["route_rescue_active"] is True
    assert row["route_rescue_type"] == "route_rescue"
    assert row["feasibility_slowdown_active"] is False


def test_summarize_episode_includes_rescue_counts() -> None:
    """Episode summary includes rescue and slowdown step counts."""
    trace = _load_trace_module()

    steps = [
        {
            "step": i,
            "selected_command": [0.5, 0.0],
            "selected_score": 2.5,
            "constraint_reason": "best_feasible",
            "candidate_total": 63,
            "candidate_feasible": 63,
            "candidate_infeasible": 0,
            "dynamic_window": {"v_min": 0.0, "v_max": 1.0, "w_min": -0.3, "w_max": 0.3},
            "target_goal": {"kind": "next", "x": 5.0, "y": 3.0},
            "distance_to_goal_m": 3.0 - i * 0.1,
            "route_rescue_active": i >= 5,
            "route_rescue_type": "route_rescue" if i >= 5 else None,
            "feasibility_slowdown_active": False,
        }
        for i in range(10)
    ]
    record = {
        "scenario_id": "test",
        "seed": 42,
        "termination_reason": "max_steps",
        "steps": 10,
        "outcome": {"route_complete": False, "collision_event": False, "timeout_event": True},
    }
    summary = trace._summarize_episode(episode_id="test_ep", record=record, steps=steps)
    assert summary["route_rescue_active_step_count"] == 5
    assert summary["route_rescue_first_active_step"] == 5
    assert summary["feasibility_slowdown_active_step_count"] == 0
    assert summary["feasibility_slowdown_first_active_step"] is None
