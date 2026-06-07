"""Tests for static recenter activation trace summaries."""

from __future__ import annotations

from scripts.validation.run_static_recenter_activation_trace import (
    _activation_summary,
    _episode_summary,
    _trace_run_summary,
)


def _record(
    *,
    scenario_id: str = "narrow_passage",
    seed: int = 111,
    success: bool = False,
    row_status: str = "completed",
    execution_mode: str = "adapter",
    activation_values: list[float] | None = None,
) -> dict:
    """Return a minimal map-runner record with static-deadlock suite fields."""
    activation_values = activation_values if activation_values is not None else [0.0, 0.0]
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "status": "success" if success else "failure",
        "termination_reason": "success" if success else "max_steps",
        "steps": len(activation_values),
        "metrics": {"success": success, "near_misses": 0, "collisions": 0},
        "row_status": row_status,
        "low_progress_window": {"active": not success},
        "recenter_activation_count": sum(1 for value in activation_values if value > 0.0),
        "distance_to_goal_delta": {"delta_m": 0.0 if not success else 1.0},
        "local_minimum_indicator": {"is_local_minimum": not success},
        "algorithm_metadata": {
            "planner_kinematics": {"execution_mode": execution_mode},
            "planner_decision_trace": {
                "steps": [
                    {
                        "step": step,
                        "static_recenter": value,
                        "selected_source": "static_recenter" if value > 0.0 else "progress",
                        "route_progress_from_start_m": float(step),
                        "robot_x_m": float(step),
                        "robot_y_m": 0.0,
                    }
                    for step, value in enumerate(activation_values)
                ]
            },
        },
    }


def test_episode_summary_reports_static_deadlock_trace_fields() -> None:
    """Episode summaries expose the fields required by the static-deadlock suite."""
    summary = _episode_summary(_record(activation_values=[0.0, 1.0, 0.5]))

    assert summary["row_status"] == "completed"
    assert summary["execution_mode"] == "adapter"
    assert summary["recenter_activation_count"] == 2
    assert summary["missing_required_trace_fields"] == []
    assert summary["all_required_trace_fields_present"] is True


def test_activation_summary_classifies_active_terminal_change() -> None:
    """Active intervention rows with a changed terminal outcome are not called irrelevant."""
    row = _activation_summary(
        baseline_record=_record(success=False, activation_values=[0.0, 0.0, 0.0]),
        mechanism_record=_record(success=True, activation_values=[0.0, 1.0, 1.0]),
    )

    assert row["paired_row_status"] == "completed"
    assert row["activation_count"] == 2
    assert row["classification"] == "mechanism_active_terminal_changed"
    assert row["terminal_outcome_changed"] is True


def test_activation_summary_separates_trace_change_from_terminal_change() -> None:
    """Trace deltas alone do not become terminal-outcome changes."""
    row = _activation_summary(
        baseline_record=_record(success=False, activation_values=[0.0, 0.0, 0.0]),
        mechanism_record=_record(success=False, activation_values=[0.0, 1.0, 1.0]),
    )

    assert row["classification"] == "mechanism_active_trace_changed"
    assert row["trace_changed"] is True
    assert row["terminal_outcome_changed"] is False


def test_trace_run_summary_counts_missing_fields_and_pair_status() -> None:
    """Run summaries keep missing-field rows visible instead of overclaiming."""
    complete_row = _activation_summary(
        baseline_record=_record(),
        mechanism_record=_record(activation_values=[0.0, 1.0]),
    )
    missing_record = _record()
    missing_record.pop("row_status")
    missing_row = _activation_summary(
        baseline_record=missing_record,
        mechanism_record=_record(),
    )

    summary = _trace_run_summary([complete_row, missing_row])

    assert summary["rows"] == 2
    assert summary["paired_row_status_counts"] == {"completed": 1, "excluded": 1}
    assert summary["all_required_trace_fields_present"] is False
    assert summary["missing_required_trace_fields"] == {
        "narrow_passage:111": {"baseline": ["row_status"], "mechanism": []}
    }
