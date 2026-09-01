"""Canonical benchmark trace attachment for opt-in simulator truth (#8065)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_sf.benchmark.map_runner.map_runner_episode import (
    _step_build_simulation_trace,
    _StepLoopState,
)


def test_map_runner_keeps_oracle_trace_as_evaluator_only_step_sibling() -> None:
    """Existing simulation-step traces carry oracle data without changing planner fields."""
    state = _StepLoopState(obs={})
    state.previous_trace_robot_pos = np.zeros(2, dtype=float)
    state.previous_trace_ped_pos = np.empty((0, 2), dtype=float)
    oracle_payload = {
        "schema_version": "oracle_transition_trace.v1",
        "transitions": [],
    }
    slc = SimpleNamespace(
        record_simulation_step_trace=True,
        record_forces=False,
        config=SimpleNamespace(sim_config=SimpleNamespace(time_per_step_in_secs=0.1)),
        single_pedestrian_intent_metadata=None,
        single_pedestrian_vru_metadata=None,
    )
    sim = SimpleNamespace(
        robot_pos=np.array([1.0, 2.0], dtype=float),
        peds=np.empty((0, 2), dtype=float),
        heading=0.25,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"oracle_transition_trace": oracle_payload},
        selected_action_payload={"linear_velocity": 0.0},
        applied_environment_action_payload={"linear_velocity": 0.0},
        actuation_step=None,
        step_visible=None,
        step_confidence=None,
        step_visibility_status="not_available",
        step_visibility_reason=None,
    )

    _step_build_simulation_trace(state, slc, step_idx=0, sim=sim)

    entry = state.simulation_step_trace[0]
    assert entry["oracle_transition_trace"] == oracle_payload
    assert entry["planner"]["event"] == "step"
    assert "oracle_transition_trace" not in entry["planner"]
