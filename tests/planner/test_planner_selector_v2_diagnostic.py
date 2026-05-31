"""Tests for the diagnostic planner-selector v2 contract."""

from __future__ import annotations

import json

from robot_sf.planner.planner_selector_v2_diagnostic import (
    PlannerSelectorV2DiagnosticAdapter,
    PlannerSelectorV2DiagnosticConfig,
)


class _DummyAdapter:
    """Tiny planner adapter test double."""

    def __init__(self, command: tuple[float, float]) -> None:
        self.command = command
        self.calls = 0
        self.bound_env = None
        self.reset_calls = 0

    def bind_env(self, env: object) -> None:
        """Record bind propagation."""
        self.bound_env = env

    def reset(self) -> None:
        """Record reset propagation."""
        self.calls = 0
        self.reset_calls += 1

    def plan(self, observation: dict[str, object]) -> tuple[float, float]:
        """Return the configured command."""
        _ = observation
        self.calls += 1
        return self.command


def _obs(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    goal: tuple[float, float] = (3.0, 0.0),
    peds: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Build a compact SocNav-style observation."""
    return {
        "robot": {"position": list(robot), "heading": [0.0]},
        "goal": {"current": list(goal)},
        "pedestrians": {"positions": [list(pos) for pos in (peds or [])]},
    }


def _selector() -> PlannerSelectorV2DiagnosticAdapter:
    """Build a selector with deterministic dummy heads."""
    return PlannerSelectorV2DiagnosticAdapter(
        config=PlannerSelectorV2DiagnosticConfig(
            scenario_id="classic_realworld_double_bottleneck_high",
            scenario_family="classic",
            seed=116,
            topology_scenarios=("classic_realworld_double_bottleneck_high",),
            seed_sensitive_scenarios=("classic_open_low_progress",),
            hard_seed_values=(116,),
            dense_ped_count=3,
            comfort_distance_m=1.1,
        ),
        candidate_adapters={
            "baseline": _DummyAdapter((0.1, 0.0)),
            "topology_route": _DummyAdapter((0.2, 0.0)),
            "proxemic_conservative": _DummyAdapter((0.3, 0.0)),
            "fast_progress_static_escape": _DummyAdapter((0.4, 0.0)),
        },
    )


def test_topology_rule_wins_before_seed_and_pedestrian_rules() -> None:
    """Predeclared topology signatures should select the route-lookahead head."""
    selector = _selector()

    command = selector.plan(_obs(peds=[(0.5, 0.0), (0.6, 0.2), (0.7, -0.2)]))

    assert command == (0.2, 0.0)
    diagnostics = selector.diagnostics()
    assert diagnostics["selected_candidate"] == "topology_route"
    assert diagnostics["last_decision"]["selected_head"] == "topology_route"
    assert diagnostics["last_decision"]["trigger_reason"] == "predeclared_topology_signature"
    assert diagnostics["last_decision"]["rule_inputs"]["scenario_id"] == (
        "classic_realworld_double_bottleneck_high"
    )
    assert diagnostics["no_leakage"]["current_episode_outcome_fields_used"] == []
    assert diagnostics["no_leakage"]["future_observation_fields_used"] == []
    json.dumps(diagnostics, allow_nan=False)


def test_dense_social_rule_selects_proxemic_head_without_topology_signature() -> None:
    """Dense near-field pedestrians should select the conservative proxemic profile."""
    selector = PlannerSelectorV2DiagnosticAdapter(
        config=PlannerSelectorV2DiagnosticConfig(
            scenario_id="francis2023_crossing",
            scenario_family="francis2023",
            dense_ped_count=2,
            comfort_distance_m=1.1,
        ),
        candidate_adapters={
            "baseline": _DummyAdapter((0.1, 0.0)),
            "topology_route": _DummyAdapter((0.2, 0.0)),
            "proxemic_conservative": _DummyAdapter((0.3, 0.0)),
            "fast_progress_static_escape": _DummyAdapter((0.4, 0.0)),
        },
    )

    command = selector.plan(_obs(peds=[(0.7, 0.0), (0.9, 0.2)]))

    assert command == (0.3, 0.0)
    assert selector.last_decision()["trigger_reason"] == "dense_social_or_comfort_risk"


def test_seed_sensitive_open_space_rule_selects_fast_progress_head() -> None:
    """Predeclared low-progress risk should select the static-escape progress head."""
    selector = PlannerSelectorV2DiagnosticAdapter(
        config=PlannerSelectorV2DiagnosticConfig(
            scenario_id="classic_open_low_progress",
            scenario_family="classic",
            seed=116,
            seed_sensitive_scenarios=("classic_open_low_progress",),
            hard_seed_values=(116,),
        ),
        candidate_adapters={
            "baseline": _DummyAdapter((0.1, 0.0)),
            "topology_route": _DummyAdapter((0.2, 0.0)),
            "proxemic_conservative": _DummyAdapter((0.3, 0.0)),
            "fast_progress_static_escape": _DummyAdapter((0.4, 0.0)),
        },
    )

    command = selector.plan(_obs())

    assert command == (0.4, 0.0)
    decision = selector.last_decision()
    assert decision["selected_head"] == "fast_progress_static_escape"
    assert decision["trigger_reason"] == "predeclared_seed_sensitive_low_progress_risk"


def test_default_rule_selects_baseline_and_reset_clears_diagnostics() -> None:
    """Unknown scenes should use the safe baseline and reset should clear state."""
    selector = PlannerSelectorV2DiagnosticAdapter(
        config=PlannerSelectorV2DiagnosticConfig(
            scenario_id="planner_sanity_simple",
            scenario_family="nominal",
        ),
        candidate_adapters={
            "baseline": _DummyAdapter((0.1, 0.0)),
            "topology_route": _DummyAdapter((0.2, 0.0)),
            "proxemic_conservative": _DummyAdapter((0.3, 0.0)),
            "fast_progress_static_escape": _DummyAdapter((0.4, 0.0)),
        },
    )
    env = object()
    selector.bind_env(env)

    assert selector.plan(_obs()) == (0.1, 0.0)
    assert selector.last_decision()["trigger_reason"] == "default_safe_baseline"
    assert selector.diagnostics()["selected_candidate_counts"] == {"baseline": 1}

    selector.reset()

    assert selector.diagnostics()["last_decision"] is None
    assert selector.diagnostics()["selected_candidate_counts"] == {}
    for adapter in selector.candidate_adapters.values():
        assert adapter.bound_env is env
        assert adapter.reset_calls == 1
