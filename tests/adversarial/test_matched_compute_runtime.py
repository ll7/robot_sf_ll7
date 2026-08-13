"""Diagnostic-only runtime accounting tests for issue #4360/#6921."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from robot_sf.adversarial import search as adversarial_search
from robot_sf.adversarial.certification import passed_status
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    Pose2D,
    RangeConfig,
    SearchConfig,
    SearchRunResult,
    SearchSpaceConfig,
)
from robot_sf.adversarial.matched_compute import (
    MATCHED_COMPUTE_RUNTIME_SCHEMA,
    MATCHED_COMPUTE_RUNTIME_SCHEMA_VERSION,
    MatchedComputeRuntimeTrace,
    ReactiveRuntimeSnapshot,
    probe_open_loop_runtime,
    probe_reactive_runtime,
)
from robot_sf.ped_npc.residual_adversary import ResidualAdversaryConfig
from robot_sf.ped_npc.residual_search import ResidualSearchConfig


def _reactive_trace() -> MatchedComputeRuntimeTrace:
    """Return a one-step native reactive accounting trace."""
    return probe_reactive_runtime(
        ResidualSearchConfig(grid_points_per_dim=2, max_candidates=4, seed=7),
        ResidualAdversaryConfig(
            is_active=True,
            target_ped_idx=0,
            max_residual_accel_mps2=1.5,
            max_jerk_mps3=20.0,
        ),
        snapshot=ReactiveRuntimeSnapshot(
            dt_s=0.1,
            positions=[[0.0, 0.0]],
            velocities=[[0.4, 0.0]],
            max_speeds=[2.0],
            robot_pose=((3.0, 0.0), 0.0),
            scenario_seed=123,
        ),
    )


def _candidate(seed: int = 7) -> CandidateSpec:
    """Build a valid synthetic open-loop candidate."""
    return CandidateSpec(
        start=Pose2D(0.0, 0.0),
        goal=Pose2D(2.0, 0.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=seed,
    )


def _search_config(tmp_path: Path, *, budget: int = 2) -> SearchConfig:
    """Build a minimal programmatic search config for injected runners."""
    template = tmp_path / "template.yaml"
    template.write_text(
        "scenarios:\n"
        "  - name: matched-compute-test\n"
        "    map_id: classic_cross_trap\n"
        "    simulation_config: {}\n"
        "    robot_config: {}\n",
        encoding="utf-8",
    )
    return SearchConfig(
        policy="social_force",
        scenario_template=template,
        search_space_path=tmp_path / "space.yaml",
        search_space=SearchSpaceConfig(
            start_x=RangeConfig(0.0, 1.0),
            start_y=RangeConfig(0.0, 1.0),
            goal_x=RangeConfig(2.0, 3.0),
            goal_y=RangeConfig(0.0, 1.0),
            scenario_seed=RangeConfig(7.0, 7.0),
        ),
        objective="minimize_episode_min_robot_distance",
        output_dir=tmp_path / "out",
        budget=budget,
        seed=11,
    )


def _search_result(
    tmp_path: Path, *, steps: int | list[dict[str, int]] | None = 5
) -> SearchRunResult:
    """Build an injected search result with an optional best episode record."""
    episode_path = tmp_path / "episode_records.jsonl"
    if steps is not None:
        episode_path.write_text(json.dumps({"episode_id": "e0", "steps": steps}) + "\n")
    evaluation = CandidateEvaluation(
        candidate=_candidate(),
        certification_status=passed_status("matched compute test"),
        objective_value=1.0,
        failure_attribution=None,
        episode_record_path=episode_path if steps is not None else None,
        trajectory_csv_path=None,
        scenario_yaml_path=tmp_path / "scenario.yaml",
        bundle_path=tmp_path,
    )
    return SearchRunResult(
        manifest_path=tmp_path / "manifest.json",
        best_candidate=evaluation,
        best_bundle_path=tmp_path,
        num_candidates=2,
        num_valid_candidates=2,
        num_invalid_candidates=0,
        num_failed_evaluations=0,
    )


def _search_manifest(tmp_path: Path, steps_by_candidate: list[int]) -> Path:
    """Write a minimal adversarial-search manifest with per-candidate records."""
    candidates = []
    for index, steps in enumerate(steps_by_candidate):
        candidate_dir = tmp_path / f"candidate_{index:04d}"
        candidate_dir.mkdir(exist_ok=True)
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(
            json.dumps({"episode_id": f"e{index}", "steps": steps}) + "\n",
            encoding="utf-8",
        )
        candidates.append({"episode_record_path": episode_path.as_posix()})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
    return manifest_path


def _synthetic_evaluation(candidate_dir: Path) -> CandidateEvaluation:
    """Return a candidate evaluation without launching benchmark execution."""
    return CandidateEvaluation(
        candidate=_candidate(),
        certification_status=passed_status("matched compute test"),
        objective_value=1.0,
        failure_attribution=None,
        episode_record_path=candidate_dir / "episode_records.jsonl",
        trajectory_csv_path=None,
        scenario_yaml_path=candidate_dir / "scenario.yaml",
        bundle_path=candidate_dir,
    )


def test_reactive_adapter_uses_native_policy_and_controller() -> None:
    """Reactive accounting must come from the finite-grid residual runtime."""
    trace = _reactive_trace()

    assert trace.arm == "reactive"
    assert trace.schema_version == "matched_compute_trace.v1"
    assert trace.scenario_seed == 123
    assert trace.search_seed == 7
    assert trace.execution_mode == "native"
    assert trace.runtime_status == "native"
    assert trace.status == "native"
    assert "FiniteGridSearchPolicy" in trace.native_path
    assert "BoundedResidualAdversary" in trace.native_path
    assert trace.candidate_budget == 4
    assert trace.candidate_evaluations == 4
    assert trace.accepted >= 0
    assert trace.rejected >= 0
    assert trace.invalid >= 0
    assert trace.simulator_steps == 1
    assert trace.simulator_physics_steps == 1
    assert trace.macro_actions == 1
    assert trace.fallback is False
    assert trace.degraded is False
    assert trace.metadata["search_diagnostic_record"]["total_evaluated"] == 4


def test_shared_trace_schema_is_stable_json() -> None:
    """Both arms must serialize through the same schema fields."""
    trace = _reactive_trace()
    payload = trace.to_dict()
    parsed = json.loads(trace.to_json())

    assert MATCHED_COMPUTE_RUNTIME_SCHEMA["schema_version"] == (
        MATCHED_COMPUTE_RUNTIME_SCHEMA_VERSION
    )
    assert set(MATCHED_COMPUTE_RUNTIME_SCHEMA["required"]).issubset(payload)
    assert parsed == payload
    assert payload["schema_version"] == "matched_compute_trace.v1"
    assert payload["runtime_status"] == payload["status"]
    assert payload["simulator_steps"] == payload["simulator_physics_steps"]


@pytest.mark.parametrize("field", ["fallback", "degraded"])
def test_native_trace_rejects_non_native_flags(field: str) -> None:
    """Native preflight traces cannot hide fallback or degraded execution."""
    trace_kwargs = {
        "arm": "reactive",
        "scenario_seed": 123,
        "search_seed": 42,
        "execution_mode": "native",
        "simulator_physics_steps": 1,
        "macro_actions": 1,
        "candidate_evaluations": 1,
        "accepted": 1,
        "rejected": 0,
        "invalid": 0,
        "status": "native",
        "adapter": "test",
        "native_path": "test.native_path",
        "candidate_budget": 1,
        field: True,
    }
    with pytest.raises(ValueError, match=f"cannot be {field}"):
        MatchedComputeRuntimeTrace(**cast("Any", trace_kwargs))


def test_open_loop_adapter_uses_canonical_search_and_production_seams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Open-loop accounting must resolve the production seam without running a real batch."""
    calls: list[str] = []

    def factory() -> Any:
        calls.append("production_candidate_evaluator")
        return production_evaluator

    def production_evaluator(config: SearchConfig, candidate: CandidateSpec, index: int) -> Any:
        calls.append(f"production_candidate_evaluator_call_{index}")
        assert config.budget == 2
        assert candidate == _candidate()
        return _synthetic_evaluation(tmp_path / f"candidate_{index:04d}")

    def runner(config: SearchConfig, *, evaluator: Any) -> SearchRunResult:
        calls.append("run_adversarial_search")
        assert config.budget == 2
        candidate_dir = tmp_path / "candidate_0000"
        candidate_dir.mkdir()
        evaluator(config, _candidate(), candidate_dir / "scenario.yaml", candidate_dir)
        result = _search_result(tmp_path, steps=5)
        return SearchRunResult(
            manifest_path=_search_manifest(tmp_path, [5, 7]),
            best_candidate=result.best_candidate,
            best_bundle_path=result.best_bundle_path,
            num_candidates=result.num_candidates,
            num_valid_candidates=result.num_valid_candidates,
            num_invalid_candidates=result.num_invalid_candidates,
            num_failed_evaluations=result.num_failed_evaluations,
        )

    monkeypatch.setattr("robot_sf.adversarial.search.production_candidate_evaluator", factory)

    trace = probe_open_loop_runtime(_search_config(tmp_path), macro_actions=10, runner=runner)

    assert calls == [
        "production_candidate_evaluator",
        "run_adversarial_search",
        "production_candidate_evaluator_call_0",
    ]
    assert trace.arm == "open_loop"
    assert trace.schema_version == "matched_compute_trace.v1"
    assert trace.scenario_seed == 7
    assert trace.search_seed == 11
    assert trace.execution_mode == "native"
    assert trace.runtime_status == "native"
    assert trace.status == "native"
    assert trace.native_path == "robot_sf.adversarial.search.run_adversarial_search"
    assert trace.metadata["production_candidate_evaluator"] == (
        "robot_sf.adversarial.search.production_candidate_evaluator"
    )
    assert trace.candidate_budget == 2
    assert trace.candidate_evaluations == 2
    assert trace.accepted == 2
    assert trace.rejected == 0
    assert trace.invalid == 0
    assert trace.macro_actions == 10
    assert trace.simulator_steps == 12
    assert trace.simulator_physics_steps == 12
    assert trace.fallback is False
    assert trace.degraded is False


def test_open_loop_adapter_runs_actual_search_seam_without_batch_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canary must exercise the real search runner with a synthetic evaluator."""

    def production_evaluator(
        config: SearchConfig, candidate: CandidateSpec, index: int
    ) -> CandidateEvaluation:
        candidate_dir = config.output_dir / f"candidate_{index:04d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(
            json.dumps({"metrics": {"min_distance": 0.5}, "steps": 5}) + "\n",
            encoding="utf-8",
        )
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status("synthetic canary evaluator"),
            objective_value=None,
            failure_attribution=None,
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=candidate_dir / "scenario.yaml",
            bundle_path=candidate_dir,
        )

    monkeypatch.setattr(
        adversarial_search,
        "_default_certifier",
        lambda *_args, **_kwargs: passed_status("synthetic canary certification"),
    )
    trace = probe_open_loop_runtime(
        _search_config(tmp_path),
        macro_actions=10,
        production_evaluator_factory=lambda: production_evaluator,
    )

    assert trace.native_path == "robot_sf.adversarial.search.run_adversarial_search"
    assert trace.execution_mode == "native"
    assert trace.status == "native"
    assert trace.candidate_evaluations == 2
    assert trace.candidate_budget == 2
    assert trace.simulator_physics_steps == 10
    assert trace.fallback is False
    assert trace.degraded is False


def test_open_loop_adapter_accepts_trace_step_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Episode-record list accounting should count simulator steps deterministically."""
    monkeypatch.setattr(
        "robot_sf.adversarial.search.production_candidate_evaluator",
        lambda: lambda *_args, **_kwargs: None,
    )
    trace = probe_open_loop_runtime(
        _search_config(tmp_path),
        macro_actions=10,
        runner=lambda _config, **_kwargs: _search_result(tmp_path, steps=[{"i": 0}, {"i": 1}]),
    )

    assert trace.simulator_steps == 2


def test_open_loop_adapter_marks_missing_episode_record_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No returned episode record is explicit unavailability, not success evidence."""
    monkeypatch.setattr(
        "robot_sf.adversarial.search.production_candidate_evaluator",
        lambda: lambda *_args, **_kwargs: None,
    )
    trace = probe_open_loop_runtime(
        _search_config(tmp_path),
        macro_actions=10,
        runner=lambda _config, **_kwargs: _search_result(tmp_path, steps=None),
    )

    assert trace.runtime_status == "unavailable"
    assert trace.simulator_steps is None
    assert "episode record path" in str(trace.unavailability_reason)


def test_open_loop_adapter_fails_closed_on_malformed_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed episode step accounting must not become a runtime trace."""
    episode_path = tmp_path / "episode_records.jsonl"
    episode_path.write_text(json.dumps({"episode_id": "e0", "steps": -1}) + "\n")
    evaluation = CandidateEvaluation(
        candidate=_candidate(),
        certification_status=passed_status("matched compute test"),
        objective_value=1.0,
        failure_attribution=None,
        episode_record_path=episode_path,
        trajectory_csv_path=None,
        scenario_yaml_path=tmp_path / "scenario.yaml",
        bundle_path=tmp_path,
    )
    result = SearchRunResult(
        manifest_path=tmp_path / "manifest.json",
        best_candidate=evaluation,
        best_bundle_path=tmp_path,
        num_candidates=1,
        num_valid_candidates=1,
        num_invalid_candidates=0,
        num_failed_evaluations=0,
    )
    monkeypatch.setattr(
        "robot_sf.adversarial.search.production_candidate_evaluator",
        lambda: lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match="record.steps"):
        probe_open_loop_runtime(
            _search_config(tmp_path), macro_actions=10, runner=lambda _config, **_kwargs: result
        )


def test_open_loop_adapter_fails_closed_on_manifest_missing_candidate_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest candidates without episode records cannot be counted as preflight evidence."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"candidates": [{"episode_record_path": None}]}))
    result = _search_result(tmp_path, steps=3)
    result = SearchRunResult(
        manifest_path=manifest_path,
        best_candidate=result.best_candidate,
        best_bundle_path=result.best_bundle_path,
        num_candidates=result.num_candidates,
        num_valid_candidates=result.num_valid_candidates,
        num_invalid_candidates=result.num_invalid_candidates,
        num_failed_evaluations=result.num_failed_evaluations,
    )
    monkeypatch.setattr(
        "robot_sf.adversarial.search.production_candidate_evaluator",
        lambda: lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match="missing episode_record_path"):
        probe_open_loop_runtime(
            _search_config(tmp_path), macro_actions=10, runner=lambda _config, **_kwargs: result
        )


def test_reactive_adapter_fails_closed_on_missing_search_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reactive traces require SearchDiagnosticRecord candidate accounting."""

    class BadPolicy:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.last_record = object()

    class OneStepAdversary:
        step_index = 1
        macro_action_index = 1

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def step_residual(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
            return np.zeros((1, 2))

    monkeypatch.setattr("robot_sf.adversarial.matched_compute.FiniteGridSearchPolicy", BadPolicy)
    monkeypatch.setattr(
        "robot_sf.adversarial.matched_compute.BoundedResidualAdversary", OneStepAdversary
    )

    with pytest.raises(ValueError, match="SearchDiagnosticRecord accounting"):
        probe_reactive_runtime(
            ResidualSearchConfig(grid_points_per_dim=2, max_candidates=4),
            ResidualAdversaryConfig(is_active=True),
            snapshot=ReactiveRuntimeSnapshot(
                dt_s=0.1,
                positions=[[0.0, 0.0]],
                velocities=[[0.0, 0.0]],
                max_speeds=[1.0],
                robot_pose=((1.0, 0.0), 0.0),
                scenario_seed=123,
            ),
        )
