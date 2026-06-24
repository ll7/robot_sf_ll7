"""Tests for the benchmark belief-mode policy hook + map_runner wiring (#3556)."""

from __future__ import annotations

from typing import Any

import pytest

from robot_sf.benchmark.scenario_belief_policy_hook import (
    BELIEF_MODES,
    BeliefModeStreamGapAdapter,
    augment_observation_with_belief,
)
from robot_sf.planner.stream_gap import StreamGapPlannerAdapter, StreamGapPlannerConfig


def _obs() -> dict[str, Any]:
    """A benchmark-style SOCNAV observation: one agent ahead (in FOV), one behind (out of FOV)."""
    return {
        "robot": {"position": [5.0, 5.0], "heading": 0.0},
        "goal": {"current": [12.0, 5.0], "next": [12.0, 5.0]},
        "pedestrians": {
            "positions": [[7.0, 5.4], [3.0, 5.0]],
            "velocities": [[0.0, -0.4], [0.0, 0.0]],
            "count": [2],
        },
    }


def _existence(aug: dict[str, Any]) -> list[float]:
    rows = aug["pedestrians"].get("uncertainty") or []
    return [round(float(r.get("existence_probability", 1.0)), 2) for r in rows]


def test_oracle_keeps_all_agents_certain():
    """Oracle mode produces one certain uncertainty row per observed pedestrian."""
    aug = augment_observation_with_belief(_obs(), mode="oracle", fov_degrees=120.0)
    ex = _existence(aug)
    assert len(ex) == 2
    assert all(e > 0.5 for e in ex)


def test_out_of_view_agent_is_degraded_but_rows_stay_aligned():
    """Uncertain modes degrade the out-of-FOV agent yet keep rows 1:1 with the observation."""
    aug = augment_observation_with_belief(_obs(), mode="uncertain_dropped", fov_degrees=120.0)
    ex = _existence(aug)
    assert len(ex) == 2  # alignment preserved (no agent removed)
    assert min(ex) < 0.5 < max(ex)  # the behind agent degraded, the ahead agent kept


def test_dropped_mode_drops_uncertain_agent_changing_the_plan():
    """The gate drops the degraded agent only in dropped mode, changing the planner command."""
    commands: dict[str, tuple[float, float]] = {}
    dropped_counts: dict[str, int] = {}
    for mode in ("oracle", "uncertain_retained", "uncertain_dropped"):
        inner = StreamGapPlannerAdapter(
            StreamGapPlannerConfig(uncertainty_gating_enabled=BELIEF_MODES[mode]["gate"])
        )
        commands[mode] = BeliefModeStreamGapAdapter(inner, mode=mode, fov_degrees=120.0).plan(
            _obs()
        )
        dropped_counts[mode] = int(inner.last_uncertainty_gate.get("dropped_count", 0))

    # Retaining (gate off) matches oracle; only dropping removes the uncertain agent.
    assert dropped_counts["oracle"] == 0
    assert dropped_counts["uncertain_retained"] == 0
    assert dropped_counts["uncertain_dropped"] == 1
    assert commands["uncertain_retained"] == commands["oracle"]
    assert commands["uncertain_dropped"] != commands["uncertain_retained"]


def test_augment_fails_closed_on_empty_or_unknown():
    """Unknown mode or empty pedestrian set returns the observation unchanged (no sidecar)."""
    assert augment_observation_with_belief(_obs(), mode="not_a_mode") == _obs()
    empty = {
        "robot": {"position": [0, 0], "heading": 0.0},
        "pedestrians": {"positions": [], "count": [0]},
    }
    assert "uncertainty" not in augment_observation_with_belief(
        empty, mode="uncertain_dropped"
    ).get("pedestrians", {})


def test_adapter_delegates_non_plan_attributes():
    """The wrapper delegates reset/diagnostics-style attributes to the inner adapter."""
    inner = StreamGapPlannerAdapter(StreamGapPlannerConfig())
    wrapped = BeliefModeStreamGapAdapter(inner, mode="oracle")
    assert wrapped.config is inner.config  # delegated attribute access
    assert wrapped.last_uncertainty_gate is inner.last_uncertainty_gate


def test_map_runner_wires_belief_mode_adapter():
    """`stream_gap` with a `belief_mode` builds a BeliefModeStreamGapAdapter and sets the gate."""
    pytest.importorskip("torch", reason="map_runner imports torch-backed planners")
    from robot_sf.benchmark.map_runner import _build_policy

    policy, _meta = _build_policy(
        "stream_gap",
        {"belief_mode": "uncertain_dropped", "belief_fov_degrees": 120.0},
    )
    adapter = policy._planner_adapter
    assert isinstance(adapter, BeliefModeStreamGapAdapter)
    assert adapter._inner.config.uncertainty_gating_enabled is True

    # Plain stream_gap (no belief_mode) is unwrapped and gate-off by default.
    plain_policy, _ = _build_policy("stream_gap", {})
    assert isinstance(plain_policy._planner_adapter, StreamGapPlannerAdapter)
    assert not isinstance(plain_policy._planner_adapter, BeliefModeStreamGapAdapter)
