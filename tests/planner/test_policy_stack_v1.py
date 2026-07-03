"""Tests for the minimal policy_stack_v1 runtime contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.planner.policy_stack_v1 import (
    PolicyStackV1Adapter,
    PolicyStackV1Config,
    build_policy_stack_v1_build_config,
)


def _obs(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    goal: tuple[float, float] = (1.0, 0.0),
    peds: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Build a small SocNav-style observation payload for policy-stack tests."""
    return {
        "robot": {"position": list(robot), "heading": [0.0]},
        "goal": {"current": list(goal)},
        "pedestrians": {"positions": [list(p) for p in (peds or [])]},
    }


def _load_atomic_topology_smoke_scenario(name: str) -> dict[str, object]:
    """Load one atomic topology scenario with absolute map paths for inline map-runner use."""
    from robot_sf.training.scenario_loader import load_scenarios

    scenario_path = Path("configs/scenarios/sets/atomic_navigation_minimal_full_v1.yaml")
    scenario_root = scenario_path.parent.resolve()
    for scenario in load_scenarios(scenario_path, base_dir=scenario_path):
        if scenario.get("name") != name:
            continue
        selected = dict(scenario)
        # Explicit reset seeding makes this smoke use the seed literally; keep a
        # deterministic success seed so the test continues to exercise policy-stack execution.
        selected["seeds"] = [4]
        map_file = selected.get("map_file")
        if isinstance(map_file, str) and map_file.strip():
            map_path = Path(map_file)
            if not map_path.is_absolute():
                selected["map_file"] = str((scenario_root / map_path).resolve())
        return selected
    raise AssertionError(f"Scenario not found: {name}")


class _DummyRiskDWA:
    """Small adapter test double using the public ``plan``/``reset`` contract."""

    def __init__(
        self,
        command: tuple[float, float] = (0.2, 0.0),
        *,
        error: Exception | None = None,
    ) -> None:
        self.command = command
        self.error = error
        self.calls = 0
        self.reset_calls = 0

    def plan(self, observation: dict[str, object]) -> tuple[float, float]:
        """Return a configured command or raise the configured error."""
        _ = observation
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.command

    def reset(self) -> None:
        """Track reset propagation."""
        self.reset_calls += 1


def test_policy_stack_selects_goal_and_records_two_proposal_modes() -> None:
    """The minimal stack should score goal/native and risk_dwa/adapter proposals."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(proposal_sources=("goal", "risk_dwa")),
        risk_dwa=_DummyRiskDWA(command=(0.0, 1.0)),
    )

    command = stack.plan(_obs())
    diagnostics = stack.diagnostics()
    last = diagnostics["last_step"]

    assert command[0] > 0.0
    assert last["selected_proposal_key"] == "goal"
    assert last["selected_mode"] == "native"
    assert last["candidate_count"] == 2
    assert last["proposal_status_counts"]["native"] == 1
    assert last["proposal_status_counts"]["adapter"] == 1
    assert set(last["risk_score_components"]) == {"goal", "risk_dwa"}
    assert last["candidate_ranking"][0]["proposal_key"] == "goal"
    json.dumps(diagnostics, allow_nan=False)


def test_policy_stack_records_failed_and_not_available_without_fallback_success() -> None:
    """Failed and missing proposals should remain diagnostic caveats, not successes."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(
            proposal_sources=("goal", "risk_dwa", "missing_optional"),
            optional_sources=("missing_optional",),
        ),
        risk_dwa=_DummyRiskDWA(error=RuntimeError("risk solver failed")),
    )

    command = stack.plan(_obs())
    last = stack.diagnostics()["last_step"]

    assert command[0] > 0.0
    assert last["selected_proposal_key"] == "goal"
    assert last["proposal_status_counts"]["failed"] == 1
    assert last["proposal_status_counts"]["not_available"] == 1
    assert last["failed_count"] == 1
    assert last["unavailable_count"] == 1
    assert "risk solver failed" in last["rejection_reasons"]["risk_dwa"]
    assert "not available" in last["rejection_reasons"]["missing_optional"]


def test_policy_stack_arbitration_trace_packet_contract() -> None:
    """Trace packets should define the future arbiter contract without enabling training."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(
            proposal_sources=("goal", "missing_optional"),
            optional_sources=("missing_optional",),
        ),
        risk_dwa=_DummyRiskDWA(),
    )

    stack.plan(_obs())
    packet = stack.arbitration_trace_packet()

    assert packet["schema_version"] == "policy_stack_v1.arbitration_trace_packet.v1"
    assert packet["training_enabled"] is False
    assert packet["proposal_sources"] == ["goal", "missing_optional"]
    assert packet["command_contract"]["action_space"] == "unicycle_vw"
    assert packet["switching_contract"]["min_dwell_steps"] == 1
    assert (
        "forecast_risk_channel"
        not in packet["observation_contract"]["inference_available_features"]
    )
    assert "future_trajectory" in packet["observation_contract"]["leakage_exclusions"]
    assert "not_available" in packet["status_policy"]["not_available_statuses"]
    assert packet["status_policy"]["non_executable_statuses"] == [
        "failed",
        "not_available",
        "rejected",
    ]
    assert packet["trace"]["last_step"]["proposal_status_counts"]["not_available"] == 1
    assert packet["trace"]["last_step"]["candidate_ranking"][0]["proposal_key"] == "goal"
    assert packet["trace"]["last_step"]["executed_command"] == [1.0, 0.0]
    assert stack.last_decision() == packet["trace"]["last_step"]
    json.dumps(packet, allow_nan=False)


def test_policy_stack_rejects_nonfinite_adapter_command_before_scoring() -> None:
    """Non-finite adapter commands should be rejected instead of entering risk scoring."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(proposal_sources=("goal", "risk_dwa")),
        risk_dwa=_DummyRiskDWA(command=(float("nan"), 0.0)),
    )

    command = stack.plan(_obs())
    last = stack.diagnostics()["last_step"]

    assert command[0] > 0.0
    assert last["selected_proposal_key"] == "goal"
    assert last["proposal_statuses"]["risk_dwa"] == "rejected"
    assert last["proposal_status_counts"]["rejected"] == 1
    assert last["rejected_count"] == 1
    assert "non-finite command" in last["rejection_reasons"]["risk_dwa"]
    assert set(last["risk_score_components"]) == {"goal"}
    json.dumps(stack.diagnostics(), allow_nan=False)


def test_policy_stack_rejects_adapter_command_outside_configured_bounds() -> None:
    """Out-of-bounds adapter commands should fail closed at the proposal boundary."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(
            proposal_sources=("goal", "risk_dwa"),
            max_linear_speed=0.8,
            max_angular_speed=0.6,
        ),
        risk_dwa=_DummyRiskDWA(command=(1.5, 0.8)),
    )

    command = stack.plan(_obs())
    last = stack.diagnostics()["last_step"]

    assert command[0] > 0.0
    assert last["proposal_statuses"]["risk_dwa"] == "rejected"
    assert last["proposal_status_counts"]["rejected"] == 1
    assert last["rejected_count"] == 1
    assert "outside configured command bounds" in last["rejection_reasons"]["risk_dwa"]
    assert set(last["risk_score_components"]) == {"goal"}


def test_policy_stack_fail_closes_when_mandatory_source_is_unavailable() -> None:
    """Mandatory proposal source gaps should raise instead of becoming fallback success."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(
            proposal_sources=("missing_mandatory",),
            mandatory_sources=("missing_mandatory",),
        ),
        risk_dwa=_DummyRiskDWA(),
    )

    with pytest.raises(RuntimeError, match="mandatory proposal source 'missing_mandatory'"):
        stack.plan(_obs())


def test_policy_stack_hard_shield_stops_unsafe_moving_command_and_resets() -> None:
    """The hard shield should report intervention when a moving command violates clearance."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(
            proposal_sources=("risk_dwa",),
            hard_stop_clearance=0.4,
        ),
        risk_dwa=_DummyRiskDWA(command=(0.7, 0.0)),
    )

    assert stack.plan(_obs(peds=[(0.1, 0.0)])) == (0.0, 0.0)
    diagnostics = stack.diagnostics()

    assert diagnostics["steps"] == 1
    assert diagnostics["shield_intervention_count"] == 1
    assert diagnostics["last_step"]["shield_intervened"] is True
    assert diagnostics["last_step"]["selected_proposal_key"] == "shield_stop"
    assert diagnostics["last_step"]["selected_command"] == [0.0, 0.0]
    assert diagnostics["last_step"]["executed_command"] == [0.0, 0.0]
    assert diagnostics["last_step"]["proposal_commands"]["shield_stop"] == [0.0, 0.0]

    packet = stack.arbitration_trace_packet()
    assert packet["trace"]["last_step"]["selected_proposal_key"] == "shield_stop"
    assert packet["trace"]["last_step"]["executed_command"] == [0.0, 0.0]

    stack.reset()
    assert stack.diagnostics()["steps"] == 0
    assert stack.diagnostics()["last_step"] is None
    assert stack.risk_dwa.reset_calls == 1


def test_policy_stack_build_config_preserves_nested_risk_dwa_fields() -> None:
    """Config builder should pass nested risk_dwa settings to the existing builder."""
    build = build_policy_stack_v1_build_config(
        {
            "proposal_sources": ["goal", "risk_dwa"],
            "hard_stop_clearance": 0.3,
            "risk_dwa": {"goal_progress_weight": 8.5},
        }
    )

    assert build.policy_stack.proposal_sources == ("goal", "risk_dwa")
    assert build.policy_stack.hard_stop_clearance == 0.3
    assert build.risk_dwa.goal_progress_weight == 8.5


def test_policy_stack_map_runner_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map-runner should expose policy_stack_v1 as an experimental adapter planner."""
    from robot_sf.benchmark.map_runner import _build_policy

    class _DummyStack:
        """Minimal policy-stack adapter used to verify map-runner registration."""

        def __init__(self, *, config, risk_dwa) -> None:
            self.config = config
            self.risk_dwa = risk_dwa

        def plan(self, observation: dict[str, object]) -> tuple[float, float]:
            """Return a deterministic velocity command for registration checks."""
            _ = observation
            return (0.25, 0.0)

        def diagnostics(self) -> dict[str, object]:
            """Return the minimal diagnostics contract expected by map-runner."""
            return {
                "last_step": {
                    "selected_proposal_key": "goal",
                    "selected_mode": "native",
                }
            }

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner_policies.rule_and_grid.PolicyStackV1Adapter", _DummyStack
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner_policies.rule_and_grid.build_policy_stack_v1_build_config",
        lambda cfg: SimpleNamespace(policy_stack=cfg, risk_dwa=SimpleNamespace()),
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner_policies.rule_and_grid.RiskDWAPlannerAdapter",
        lambda config: SimpleNamespace(config=config),
    )

    policy, meta = _build_policy(
        "policy_stack_v1",
        {"allow_testing_algorithms": True, "proposal_sources": ["goal", "risk_dwa"]},
    )

    assert policy(_obs()) == (0.25, 0.0)
    assert policy._planner_stats()["last_step"]["selected_proposal_key"] == "goal"
    assert meta["status"] == "ok"
    assert meta["baseline_category"] == "classical"
    assert meta["policy_semantics"] == "policy_stack_v1_portfolio"
    assert meta["planner_kinematics"]["execution_mode"] == "adapter"


def test_policy_stack_runs_atomic_topology_smoke_through_map_runner(tmp_path: Path) -> None:
    """The stack should execute a topology-heavy atomic scenario through the benchmark runner."""
    from robot_sf.benchmark.map_runner import run_map_batch

    out_path = tmp_path / "episodes.jsonl"
    summary = run_map_batch(
        [_load_atomic_topology_smoke_scenario("corridor_following")],
        out_path,
        schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
        algo="policy_stack_v1",
        algo_config_path="configs/algos/policy_stack_v1.yaml",
        horizon=180,
        dt=0.1,
        record_forces=False,
        workers=1,
        resume=False,
        benchmark_profile="experimental",
    )
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    assert summary["total_jobs"] == 1
    assert summary["successful_jobs"] == 1
    assert summary["failed_jobs"] == 0
    assert summary["benchmark_availability"]["benchmark_success"] is True
    assert len(rows) == 1

    record = rows[0]
    assert record["scenario_id"] == "corridor_following"
    assert record["status"] == "success"
    assert record["termination_reason"] == "success"
    assert record["steps"] > 0

    metadata = record["algorithm_metadata"]
    assert metadata["policy_semantics"] == "policy_stack_v1_portfolio"
    assert metadata["planner_kinematics"]["execution_mode"] == "adapter"
    assert metadata["kinematics_feasibility"]["commands_evaluated"] == record["steps"]

    runtime = metadata["planner_runtime"]
    assert runtime["steps"] == record["steps"]
    assert runtime["proposal_status_counts"]["native"] > 0
    assert runtime["proposal_status_counts"]["adapter"] > 0
    assert runtime["last_step"]["selected_proposal_key"] in {"goal", "risk_dwa", "shield_stop"}


def test_forecast_risk_default_off_ignores_channel() -> None:
    """Default weight=0.0 should not inject forecast risk keys into scores."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(proposal_sources=("goal",)),
        risk_dwa=_DummyRiskDWA(),
    )
    obs = _obs()
    obs["forecast_risk_channel"] = {"status": "available", "risk": 1.0}
    stack.plan(obs)
    last = stack.diagnostics()["last_step"]
    for components in last["risk_score_components"].values():
        assert "forecast_risk_raw" not in components
        assert "forecast_risk_penalty" not in components


def test_forecast_risk_high_can_shift_selection_to_risk_dwa() -> None:
    """High forecast risk with weight>0 should penalize goal enough to select risk_dwa."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(
            proposal_sources=("goal", "risk_dwa"),
            forecast_risk_weight=5.0,
        ),
        risk_dwa=_DummyRiskDWA(command=(0.2, 0.0)),
    )
    obs = _obs()
    obs["forecast_risk_channel"] = {"status": "available", "risk": 1.0}
    stack.plan(obs)
    last = stack.diagnostics()["last_step"]
    assert last["selected_proposal_key"] == "risk_dwa"
    goal_components = last["risk_score_components"]["goal"]
    assert goal_components["forecast_risk_available"] == 1.0
    assert goal_components["forecast_risk_raw"] == 1.0
    assert goal_components["forecast_risk_penalty"] > 0.0
    assert last["candidate_ranking"][0]["proposal_key"] == "risk_dwa"


def test_forecast_risk_false_positive_suppresses_penalty() -> None:
    """false_positive_risk=1 should cancel the forecast penalty; goal stays preferred."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(
            proposal_sources=("goal",),
            forecast_risk_weight=10.0,
        ),
        risk_dwa=_DummyRiskDWA(),
    )
    obs = _obs()
    obs["forecast_risk_channel"] = {
        "status": "available",
        "risk": 1.0,
        "false_positive_risk": 1.0,
    }
    stack.plan(obs)
    last = stack.diagnostics()["last_step"]
    assert last["selected_proposal_key"] == "goal"
    goal_components = last["risk_score_components"]["goal"]
    assert goal_components["forecast_risk_penalty"] == 0.0
    assert goal_components["forecast_risk_false_positive"] == 1.0
    assert goal_components["forecast_risk_effective"] == 0.0


def test_forecast_risk_unavailable_status_is_trace_visible_without_penalty() -> None:
    """Unavailable forecast-risk payloads should be visible but non-penalizing."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(
            proposal_sources=("goal",),
            forecast_risk_weight=10.0,
        ),
        risk_dwa=_DummyRiskDWA(),
    )
    obs = _obs()
    obs["forecast_risk_channel"] = {"status": "unavailable", "risk": 1.0}
    stack.plan(obs)
    goal_components = stack.diagnostics()["last_step"]["risk_score_components"]["goal"]
    assert goal_components["forecast_risk_available"] == 0.0
    assert goal_components["forecast_risk_raw"] == 0.0
    assert goal_components["forecast_risk_penalty"] == 0.0


@pytest.mark.parametrize("payload", [{"risk": 1.0}, {"status": "", "risk": 1.0}])
def test_forecast_risk_requires_explicit_active_status(payload: dict[str, float | str]) -> None:
    """Missing or blank forecast-risk status should not change scoring."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(
            proposal_sources=("goal",),
            forecast_risk_weight=10.0,
        ),
        risk_dwa=_DummyRiskDWA(),
    )
    obs = _obs()
    obs["forecast_risk_channel"] = payload
    stack.plan(obs)
    goal_components = stack.diagnostics()["last_step"]["risk_score_components"]["goal"]
    assert goal_components["forecast_risk_available"] == 0.0
    assert goal_components["forecast_risk_raw"] == 0.0
    assert goal_components["forecast_risk_penalty"] == 0.0


def test_forecast_risk_trace_packet_declares_enabled_observation_key() -> None:
    """Forecast-risk-enabled traces should declare the configured diagnostic input."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(
            proposal_sources=("goal",),
            forecast_risk_weight=2.0,
            forecast_risk_observation_key="my_forecast_risk",
        ),
        risk_dwa=_DummyRiskDWA(),
    )
    stack.plan(_obs())
    observation_contract = stack.arbitration_trace_packet()["observation_contract"]
    assert "my_forecast_risk" in observation_contract["inference_available_features"]
    assert observation_contract["forecast_risk_observation_key"] == "my_forecast_risk"
    assert "" not in observation_contract["forecast_risk_active_statuses"]
    assert "diagnostic" in observation_contract["forecast_risk_active_statuses"]


def test_forecast_risk_build_config_parses_fields() -> None:
    """Build config should parse forecast_risk_weight and forecast_risk_observation_key."""
    build = build_policy_stack_v1_build_config(
        {
            "proposal_sources": ["goal", "risk_dwa"],
            "forecast_risk_weight": 3.5,
            "forecast_risk_observation_key": "my_custom_channel",
        }
    )
    assert build.policy_stack.forecast_risk_weight == 3.5
    assert build.policy_stack.forecast_risk_observation_key == "my_custom_channel"


def test_forecast_risk_build_config_defaults() -> None:
    """Build config should default forecast fields when absent."""
    build = build_policy_stack_v1_build_config({"proposal_sources": ["goal"]})
    assert build.policy_stack.forecast_risk_weight == 0.0
    assert build.policy_stack.forecast_risk_observation_key == "forecast_risk_channel"


def test_forecast_risk_build_config_treats_null_fields_as_defaults() -> None:
    """Explicit null forecast-risk config values should use safe defaults."""
    build = build_policy_stack_v1_build_config(
        {
            "proposal_sources": ["goal"],
            "forecast_risk_weight": None,
            "forecast_risk_observation_key": None,
        }
    )
    assert build.policy_stack.forecast_risk_weight == 0.0
    assert build.policy_stack.forecast_risk_observation_key == "forecast_risk_channel"


def test_forecast_risk_diagnostics_json_safe() -> None:
    """Diagnostics and trace packet should remain JSON-serializable with forecast risk."""
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(
            proposal_sources=("goal",),
            forecast_risk_weight=2.0,
        ),
        risk_dwa=_DummyRiskDWA(),
    )
    obs = _obs()
    obs["forecast_risk_channel"] = {
        "status": "diagnostic",
        "risk": 0.5,
        "occupancy_risk": 0.3,
        "collision_relevance": 0.2,
        "false_positive_risk": 0.1,
        "unnecessary_stop_risk": 0.05,
    }
    stack.plan(obs)
    diagnostics = stack.diagnostics()
    goal_components = diagnostics["last_step"]["risk_score_components"]["goal"]
    assert goal_components["forecast_risk_raw"] == 0.5
    assert goal_components["forecast_risk_false_positive"] == 0.1
    assert goal_components["forecast_risk_effective"] == pytest.approx(0.45)
    packet = stack.arbitration_trace_packet()
    json.dumps(diagnostics, allow_nan=False)
    json.dumps(packet, allow_nan=False)
