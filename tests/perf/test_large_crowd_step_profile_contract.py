"""Contract coverage for deterministic large-crowd step-profile telemetry."""

from argparse import Namespace
from pathlib import Path

from scripts.validation import performance_smoke_test


def test_large_crowd_step_profile_contract(monkeypatch) -> None:
    """Smoke profile output includes deterministic scenario + advisory metadata."""

    repo_root = Path(__file__).resolve().parents[2]
    scenario_path = repo_root / "configs/scenarios/single/dense_pedestrian_stress.yaml"
    scenario_metadata = performance_smoke_test.ScenarioProfileMetadata(
        scenario_id="dense_pedestrian_stress",
        scenario_name="dense_pedestrian_stress",
        scenario_path=str(scenario_path),
        density="high",
        density_advisory="diagnostic_stress_only",
    )

    monkeypatch.setattr(
        performance_smoke_test,
        "measure_environment_creation",
        lambda config=None: 1.25,
    )
    monkeypatch.setattr(
        performance_smoke_test,
        "measure_environment_performance",
        lambda num_resets=5, config=None: {
            "resets_per_sec": 4.0,
            "ms_per_reset": 250.0,
            "total_resets": float(num_resets),
            "total_time": 0.5,
        },
    )
    monkeypatch.setattr(
        performance_smoke_test,
        "measure_step_loop_performance",
        lambda step_samples=10, config=None, warmup_steps=0: performance_smoke_test.StepLoopMetrics(
            step_samples=step_samples,
            first_step_sec=0.2,
            step_loop_sec=1.0,
            steady_step_loop_sec=0.8,
            steps_per_sec=3.0,
            steady_steps_per_sec=2.6666666666666665,
            warmup_excluded=False,
            warmup_first_step_sec=None,
            warmup_step_loop_sec=None,
            warmup_steps_per_sec=None,
            measurement_mode="cold_only",
        ),
    )
    monkeypatch.setattr(
        performance_smoke_test,
        "_measure_profile_pedestrian_count",
        lambda config=None: 17,
    )
    monkeypatch.setattr(
        performance_smoke_test,
        "_load_scenario_config",
        lambda scenario, *, scenario_name=None: (
            performance_smoke_test.RobotSimulationConfig(),
            scenario_name or "dense_pedestrian_stress",
            scenario_metadata,
        ),
    )

    result = performance_smoke_test.run_performance_smoke_test(
        num_resets=2,
        step_samples=20,
        scenario=str(scenario_path),
        scenario_name="dense_pedestrian_stress",
        include_recommendations=False,
        creation_soft=3.0,
        creation_hard=8.0,
        reset_soft=0.5,
        reset_hard=0.2,
        enforce=False,
        on_ci=False,
    )

    payload = result.to_dict()
    step_profile = payload["step_profile"]

    assert payload["scenario"] == "dense_pedestrian_stress"
    assert step_profile is not None
    assert step_profile["scenario_id"] == "dense_pedestrian_stress"
    assert step_profile["scenario_name"] == "dense_pedestrian_stress"
    assert step_profile["scenario_path"] == str(scenario_path)
    assert step_profile["density"] == "high"
    assert step_profile["density_advisory"] == "diagnostic_stress_only"
    assert step_profile["step_samples"] == 20
    assert step_profile["first_step_sec"] == 0.2
    assert step_profile["step_loop_sec"] == 1.0
    assert step_profile["steady_step_loop_sec"] == 0.8
    assert step_profile["steps_per_sec"] == 3.0
    assert step_profile["steady_steps_per_sec"] == 2.6666666666666665
    assert step_profile["warmup_excluded"] is False
    assert step_profile["warmup_first_step_sec"] is None
    assert step_profile["warmup_step_loop_sec"] is None
    assert step_profile["warmup_steps_per_sec"] is None
    assert step_profile["measurement_mode"] == "cold_only"
    assert step_profile["pedestrian_count"] == 17
    assert step_profile["advisory"] is True
    assert step_profile["gating"] == "non-gating"


def test_large_crowd_profile_preset_selects_dense_diagnostic_fixture() -> None:
    """The CLI preset names the reproducible dense stress profiling command."""

    args = Namespace(
        large_crowd_profile=True,
        scenario=None,
        step_samples=None,
    )

    performance_smoke_test._apply_large_crowd_profile_preset(args)

    assert args.scenario == "configs/scenarios/single/dense_pedestrian_stress.yaml"
    assert args.step_samples == 20


def test_large_crowd_profile_preset_keeps_explicit_overrides() -> None:
    """Explicit scenario and sample choices win over the convenience preset."""

    args = Namespace(
        large_crowd_profile=True,
        scenario="configs/scenarios/archetypes/classic_group_crossing.yaml",
        step_samples=10,
    )

    performance_smoke_test._apply_large_crowd_profile_preset(args)

    assert args.scenario == "configs/scenarios/archetypes/classic_group_crossing.yaml"
    assert args.step_samples == 10


def test_large_crowd_step_profile_reuses_first_step_ped_count(monkeypatch) -> None:
    """Avoid extra simulation runs when step-loop already captured ped count."""

    def explode_if_called(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("pedestrian count fallback should not be used")

    monkeypatch.setattr(
        performance_smoke_test,
        "_measure_profile_pedestrian_count",
        explode_if_called,
    )

    loop_metrics = performance_smoke_test.StepLoopMetrics(
        step_samples=32,
        first_step_sec=0.3,
        step_loop_sec=3.0,
        steady_step_loop_sec=2.7,
        steps_per_sec=10.0,
        steady_steps_per_sec=12.0,
        warmup_excluded=False,
        warmup_first_step_sec=None,
        warmup_step_loop_sec=None,
        warmup_steps_per_sec=None,
        measurement_mode="cold_only",
        first_step_pedestrian_count=17,
    )

    profile = performance_smoke_test.measure_step_profile(
        step_samples=32,
        config=performance_smoke_test.RobotSimulationConfig(),
        step_loop=loop_metrics,
        scenario_metadata=None,
    )

    assert profile.pedestrian_count == 17
