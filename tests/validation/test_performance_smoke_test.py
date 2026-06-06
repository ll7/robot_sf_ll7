"""Tests for the lightweight performance smoke script contract."""

from __future__ import annotations

from scripts.validation import performance_smoke_test


def test_performance_smoke_result_includes_step_loop_attribution(monkeypatch) -> None:
    """Smoke JSON should expose advisory startup/steady step attribution."""

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
        lambda step_samples=10, config=None: performance_smoke_test.StepLoopMetrics(
            step_samples=step_samples,
            first_step_sec=0.2,
            step_loop_sec=1.0,
            steady_step_loop_sec=0.8,
            steps_per_sec=4.0,
            steady_steps_per_sec=3.75,
        ),
    )

    result = performance_smoke_test.run_performance_smoke_test(
        num_resets=2,
        step_samples=4,
        include_recommendations=False,
        creation_soft=3.0,
        creation_hard=8.0,
        reset_soft=0.5,
        reset_hard=0.2,
        enforce=False,
        on_ci=False,
    )

    payload = result.to_dict()

    assert result.statuses["overall"] == "PASS"
    assert payload["step_loop"] == {
        "step_samples": 4,
        "first_step_sec": 0.2,
        "step_loop_sec": 1.0,
        "steady_step_loop_sec": 0.8,
        "steps_per_sec": 4.0,
        "steady_steps_per_sec": 3.75,
    }


def test_telemetry_snapshot_includes_step_loop_attribution(tmp_path) -> None:
    """Telemetry JSONL should retain advisory step-loop attribution."""

    result = performance_smoke_test.SmokeTestResult(
        timestamp=performance_smoke_test.datetime.now(performance_smoke_test.UTC),
        creation_seconds=1.0,
        resets_per_sec=5.0,
        ms_per_reset=200.0,
        total_resets=2,
        total_time_sec=0.4,
        thresholds={},
        statuses={"overall": "PASS", "creation": "PASS", "reset": "PASS"},
        step_loop=performance_smoke_test.StepLoopMetrics(
            step_samples=4,
            first_step_sec=0.2,
            step_loop_sec=1.0,
            steady_step_loop_sec=0.8,
            steps_per_sec=4.0,
            steady_steps_per_sec=3.75,
        ),
    )

    output = tmp_path / "telemetry.jsonl"
    performance_smoke_test._write_telemetry_snapshot(output, result)

    payload = performance_smoke_test.json.loads(output.read_text())

    assert payload["first_step_sec"] == 0.2
    assert payload["step_loop_sec"] == 1.0
    assert payload["steady_step_loop_sec"] == 0.8
    assert payload["sim_steps_per_sec"] == 4.0
    assert payload["steady_sim_steps_per_sec"] == 3.75
