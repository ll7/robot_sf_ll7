"""Tests for the minimal policy_stack_v1 runtime contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.planner.policy_stack_v1 import (
    PolicyStackV1Adapter,
    PolicyStackV1Config,
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
        selected["seeds"] = [59621]
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

    stack.reset()
    assert stack.diagnostics()["steps"] == 0
    assert stack.diagnostics()["last_step"] is None
    assert stack.risk_dwa.reset_calls == 1


def test_policy_stack_build_config_preserves_nested_risk_dwa_fields() -> None:
    """Config builder should pass nested risk_dwa settings to the existing builder."""
    from robot_sf.planner.policy_stack_v1 import build_policy_stack_v1_build_config

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
        def __init__(self, *, config, risk_dwa) -> None:
            self.config = config
            self.risk_dwa = risk_dwa

        def plan(self, observation: dict[str, object]) -> tuple[float, float]:
            _ = observation
            return (0.25, 0.0)

        def diagnostics(self) -> dict[str, object]:
            return {
                "last_step": {
                    "selected_proposal_key": "goal",
                    "selected_mode": "native",
                }
            }

    monkeypatch.setattr("robot_sf.benchmark.map_runner.PolicyStackV1Adapter", _DummyStack)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.build_policy_stack_v1_build_config",
        lambda cfg: SimpleNamespace(policy_stack=cfg, risk_dwa=SimpleNamespace()),
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.RiskDWAPlannerAdapter",
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
