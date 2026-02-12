"""Unit tests for cold/warm performance regression utilities."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from robot_sf.benchmark import perf_cold_warm


def _sample(
    *,
    create: float,
    first: float,
    episode: float,
    sps: float,
) -> perf_cold_warm.PhaseMetrics:
    return perf_cold_warm.PhaseMetrics(
        env_create_sec=create,
        first_step_sec=first,
        episode_sec=episode,
        steps_per_sec=sps,
    )


def test_median_metrics_uses_per_metric_median() -> None:
    """Median helper should aggregate each metric independently."""
    samples = [
        _sample(create=2.0, first=0.20, episode=3.0, sps=20.0),
        _sample(create=1.0, first=0.40, episode=5.0, sps=10.0),
        _sample(create=3.0, first=0.10, episode=4.0, sps=30.0),
    ]
    median = perf_cold_warm.median_metrics(samples)
    assert median.env_create_sec == pytest.approx(2.0)
    assert median.first_step_sec == pytest.approx(0.20)
    assert median.episode_sec == pytest.approx(4.0)
    assert median.steps_per_sec == pytest.approx(20.0)


def test_median_metrics_rejects_empty_input() -> None:
    """Median helper should reject empty sample lists."""
    with pytest.raises(ValueError, match="samples must not be empty"):
        perf_cold_warm.median_metrics([])


def test_compare_snapshots_flags_startup_regression() -> None:
    """A startup-only slowdown should be diagnosed as startup overhead regression."""
    baseline = perf_cold_warm.SuiteSnapshot(
        cold=_sample(create=2.0, first=0.20, episode=3.0, sps=20.0),
        warm=_sample(create=1.0, first=0.10, episode=2.0, sps=25.0),
    )
    current = perf_cold_warm.SuiteSnapshot(
        cold=_sample(create=4.0, first=0.55, episode=3.1, sps=20.2),
        warm=_sample(create=1.1, first=0.30, episode=2.1, sps=24.5),
    )
    thresholds = perf_cold_warm.RegressionThresholds(
        max_slowdown_pct=0.50,
        max_throughput_drop_pct=0.50,
        min_seconds_delta=0.15,
        min_throughput_delta=0.5,
    )
    report = perf_cold_warm.compare_snapshots(current, baseline, thresholds)
    assert report.status == "fail"
    assert report.has_regression
    assert any("startup overhead" in line for line in report.diagnostics)
    assert any(f.is_regression and f.metric == "env_create_sec" for f in report.findings)


def test_compare_snapshots_flags_steady_state_regression() -> None:
    """Stepping slowdown should be diagnosed as steady-state throughput regression."""
    baseline = perf_cold_warm.SuiteSnapshot(
        cold=_sample(create=2.0, first=0.20, episode=3.0, sps=20.0),
        warm=_sample(create=1.0, first=0.10, episode=2.0, sps=24.0),
    )
    current = perf_cold_warm.SuiteSnapshot(
        cold=_sample(create=2.1, first=0.22, episode=5.5, sps=8.0),
        warm=_sample(create=1.1, first=0.11, episode=3.8, sps=10.0),
    )
    thresholds = perf_cold_warm.RegressionThresholds(
        max_slowdown_pct=0.35,
        max_throughput_drop_pct=0.45,
        min_seconds_delta=0.15,
        min_throughput_delta=0.5,
    )
    report = perf_cold_warm.compare_snapshots(current, baseline, thresholds)
    assert report.status == "fail"
    assert report.has_regression
    assert any("steady-state stepping" in line for line in report.diagnostics)
    assert any(f.is_regression and f.metric == "steps_per_sec" for f in report.findings)


def test_compare_snapshots_ignores_small_noise() -> None:
    """Small changes under absolute and relative floors should not trigger failures."""
    baseline = perf_cold_warm.SuiteSnapshot(
        cold=_sample(create=2.0, first=0.20, episode=3.0, sps=20.0),
        warm=_sample(create=1.0, first=0.10, episode=2.0, sps=25.0),
    )
    current = perf_cold_warm.SuiteSnapshot(
        cold=_sample(create=2.2, first=0.28, episode=3.1, sps=19.7),
        warm=_sample(create=1.1, first=0.16, episode=2.1, sps=24.6),
    )
    thresholds = perf_cold_warm.RegressionThresholds(
        max_slowdown_pct=0.15,
        max_throughput_drop_pct=0.05,
        min_seconds_delta=0.25,
        min_throughput_delta=1.0,
    )
    report = perf_cold_warm.compare_snapshots(current, baseline, thresholds)
    assert report.status == "pass"
    assert not report.has_regression
    assert report.diagnostics == ("No meaningful regressions detected.",)


def test_load_snapshot_accepts_nested_median_shape(tmp_path: Path) -> None:
    """Baseline loader should support nested `{phase: {median: ...}}` payloads."""
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "cold": {"median": _sample(create=2.0, first=0.2, episode=3.0, sps=15.0).to_dict()},
                "warm": {"median": _sample(create=1.0, first=0.1, episode=2.0, sps=20.0).to_dict()},
            }
        ),
        encoding="utf-8",
    )
    loaded = perf_cold_warm.load_snapshot(baseline_path)
    assert loaded is not None
    assert loaded.cold.episode_sec == pytest.approx(3.0)
    assert loaded.warm.steps_per_sec == pytest.approx(20.0)


def test_load_snapshot_handles_missing_and_invalid_files(tmp_path: Path) -> None:
    """Loader should return None for missing/invalid baseline payloads."""
    missing = perf_cold_warm.load_snapshot(tmp_path / "missing.json")
    assert missing is None

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{bad", encoding="utf-8")
    assert perf_cold_warm.load_snapshot(invalid_json) is None

    invalid_shape = tmp_path / "invalid_shape.json"
    invalid_shape.write_text(json.dumps({"cold": []}), encoding="utf-8")
    assert perf_cold_warm.load_snapshot(invalid_shape) is None


def test_render_markdown_report_includes_diagnostics() -> None:
    """Markdown report should include comparison status and diagnostic bullet list."""
    current = perf_cold_warm.SuiteSnapshot(
        cold=_sample(create=2.0, first=0.2, episode=3.0, sps=15.0),
        warm=_sample(create=1.0, first=0.1, episode=2.0, sps=20.0),
    )
    baseline = perf_cold_warm.SuiteSnapshot(
        cold=_sample(create=1.0, first=0.1, episode=2.0, sps=20.0),
        warm=_sample(create=0.8, first=0.1, episode=1.8, sps=22.0),
    )
    report = perf_cold_warm.compare_snapshots(
        current,
        baseline,
        perf_cold_warm.RegressionThresholds(
            max_slowdown_pct=0.25,
            max_throughput_drop_pct=0.20,
            min_seconds_delta=0.10,
            min_throughput_delta=0.5,
        ),
    )
    rendered = perf_cold_warm.render_markdown_report(
        scenario_label="classic_crossing_low",
        episode_steps=64,
        cold_runs=1,
        warm_runs=2,
        current=current,
        baseline=baseline,
        report=report,
    )
    assert "Status: **FAIL**" in rendered
    assert "### Diagnostics" in rendered
    assert "classic_crossing_low" in rendered


def test_render_markdown_report_without_baseline() -> None:
    """Report should explicitly note when no baseline was available."""
    current = perf_cold_warm.SuiteSnapshot(
        cold=_sample(create=1.0, first=0.1, episode=1.0, sps=30.0),
        warm=_sample(create=0.5, first=0.05, episode=0.8, sps=40.0),
    )
    rendered = perf_cold_warm.render_markdown_report(
        scenario_label="classic_crossing_low",
        episode_steps=32,
        cold_runs=1,
        warm_runs=1,
        current=current,
        baseline=None,
        report=None,
    )
    assert "No baseline snapshot available." in rendered


def test_measure_once_validates_episode_steps() -> None:
    """Measurement should reject invalid episode step counts."""
    with pytest.raises(ValueError, match="episode_steps must be > 0"):
        perf_cold_warm.measure_once(config=object(), seed=1, episode_steps=0)  # type: ignore[arg-type]


def test_measure_once_with_fake_env(monkeypatch) -> None:
    """Measurement should compute metrics and close env even with resets during loop."""

    class _FakeEnv:
        def __init__(self) -> None:
            self.closed = False
            self.reset_calls = 0
            self.step_calls = 0

        def reset(self, seed: int) -> tuple[dict, dict]:
            self.reset_calls += 1
            return {}, {}

        def step(self, _action: tuple[float, float]) -> tuple[dict, float, bool, bool, dict]:
            self.step_calls += 1
            if self.step_calls == 1:
                return {}, 0.0, False, True, {}
            return {}, 0.0, False, False, {}

        def close(self) -> None:
            self.closed = True

    fake_env = _FakeEnv()
    monkeypatch.setattr(perf_cold_warm, "make_robot_env", lambda **_kwargs: fake_env)

    tick = {"value": 0.0}

    def _next_tick() -> float:
        tick["value"] += 0.1
        return tick["value"]

    monkeypatch.setattr(
        perf_cold_warm.time,
        "perf_counter",
        _next_tick,
    )

    metrics = perf_cold_warm.measure_once(config=object(), seed=7, episode_steps=3)  # type: ignore[arg-type]
    assert metrics.env_create_sec > 0.0
    assert metrics.first_step_sec > 0.0
    assert metrics.episode_sec > 0.0
    assert metrics.steps_per_sec > 0.0
    assert fake_env.reset_calls >= 2
    assert fake_env.closed


def test_run_suite_uses_warmup_then_warm_samples(monkeypatch) -> None:
    """Suite runner should perform one warmup run before warm measurements."""
    call_log: list[int] = []

    monkeypatch.setattr(
        perf_cold_warm,
        "_measure_cold_subprocess",
        lambda **kwargs: _sample(create=1.0 + kwargs["seed"], first=0.1, episode=1.0, sps=10.0),
    )

    def _fake_measure_once(*, config, seed, episode_steps):
        call_log.append(seed)
        return _sample(create=1.0, first=0.1, episode=1.0, sps=10.0)

    monkeypatch.setattr(perf_cold_warm, "measure_once", _fake_measure_once)
    cold, warm = perf_cold_warm.run_suite(
        config=object(),  # type: ignore[arg-type]
        script_path=Path("robot_sf/benchmark/perf_cold_warm.py"),
        scenario_config=Path("configs/scenarios/archetypes/classic_crossing.yaml"),
        scenario_name="classic_crossing_low",
        seed=10,
        episode_steps=8,
        cold_runs=2,
        warm_runs=2,
    )
    assert len(cold) == 2
    assert len(warm) == 2
    assert call_log == [1010, 1110, 1111]


def test_report_to_dict_and_diagnostics_mixed_regressions() -> None:
    """Report serialization should include findings and mixed-regression diagnostics."""
    findings = (
        perf_cold_warm.RegressionFinding(
            phase="cold",
            metric="env_create_sec",
            baseline=1.0,
            current=2.0,
            delta=1.0,
            delta_pct=100.0,
            is_regression=True,
            threshold_pct=50.0,
        ),
        perf_cold_warm.RegressionFinding(
            phase="warm",
            metric="steps_per_sec",
            baseline=10.0,
            current=2.0,
            delta=-8.0,
            delta_pct=-80.0,
            is_regression=True,
            threshold_pct=40.0,
        ),
    )
    diagnostics = perf_cold_warm._build_diagnostics(findings)
    assert any("spans startup and steady-state" in line for line in diagnostics)
    report = perf_cold_warm.RegressionReport(
        status="fail",
        findings=findings,
        diagnostics=diagnostics,
    )
    payload = perf_cold_warm._report_to_dict(report)
    assert payload["status"] == "fail"
    assert len(payload["findings"]) == 2
    assert perf_cold_warm._report_to_dict(None)["status"] == "no-baseline"


def test_load_scenario_config_and_not_found(monkeypatch) -> None:
    """Scenario loader should resolve named scenario and set sim-time override."""

    class _SimConfig:
        def __init__(self) -> None:
            self.time_per_step_in_secs = 0.2
            self.sim_time_in_secs = 0.0

    class _Config:
        def __init__(self) -> None:
            self.sim_config = _SimConfig()

    monkeypatch.setattr(
        perf_cold_warm,
        "load_scenarios",
        lambda _path: [{"name": "classic_crossing_low"}],
    )
    monkeypatch.setattr(
        perf_cold_warm,
        "build_robot_config_from_scenario",
        lambda _scenario, scenario_path: _Config(),
    )
    config, label = perf_cold_warm._load_scenario_config(
        Path("configs/scenarios/archetypes/classic_crossing.yaml"),
        "classic_crossing_low",
        20,
    )
    assert label == "classic_crossing_low"
    assert config.sim_config.sim_time_in_secs == pytest.approx(4.0)

    monkeypatch.setattr(perf_cold_warm, "load_scenarios", lambda _path: [])
    with pytest.raises(ValueError, match="not found"):
        perf_cold_warm._load_scenario_config(Path("x.yaml"), "missing", 10)


def test_measure_cold_subprocess_success_and_failures(monkeypatch) -> None:
    """Subprocess helper should parse JSON payloads and raise on failure modes."""

    class _Completed:
        def __init__(self, *, returncode: int, stdout: str, stderr: str) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(
        perf_cold_warm.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(
            returncode=0,
            stdout='line\n{"env_create_sec":1,"first_step_sec":0.2,"episode_sec":1.4,"steps_per_sec":9}\n',
            stderr="",
        ),
    )
    sample = perf_cold_warm._measure_cold_subprocess(
        script_path=Path("robot_sf/benchmark/perf_cold_warm.py"),
        scenario_config=Path("configs/scenarios/archetypes/classic_crossing.yaml"),
        scenario_name="classic_crossing_low",
        seed=1,
        episode_steps=10,
    )
    assert sample.steps_per_sec == pytest.approx(9.0)

    monkeypatch.setattr(
        perf_cold_warm.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(returncode=1, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="failed"):
        perf_cold_warm._measure_cold_subprocess(
            script_path=Path("robot_sf/benchmark/perf_cold_warm.py"),
            scenario_config=Path("configs/scenarios/archetypes/classic_crossing.yaml"),
            scenario_name="classic_crossing_low",
            seed=1,
            episode_steps=10,
        )

    monkeypatch.setattr(
        perf_cold_warm.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(returncode=0, stdout="no-json", stderr=""),
    )
    with pytest.raises(RuntimeError, match="did not return JSON"):
        perf_cold_warm._measure_cold_subprocess(
            script_path=Path("robot_sf/benchmark/perf_cold_warm.py"),
            scenario_config=Path("configs/scenarios/archetypes/classic_crossing.yaml"),
            scenario_name="classic_crossing_low",
            seed=1,
            episode_steps=10,
        )


def test_parse_args_custom_values() -> None:
    """Parser should accept overrides used by CI workflows."""
    parsed = perf_cold_warm.parse_args(
        [
            "--scenario-name",
            "classic_crossing_medium",
            "--episode-steps",
            "48",
            "--cold-runs",
            "2",
            "--warm-runs",
            "3",
            "--fail-on-regression",
        ]
    )
    assert parsed.scenario_name == "classic_crossing_medium"
    assert parsed.episode_steps == 48
    assert parsed.cold_runs == 2
    assert parsed.warm_runs == 3
    assert parsed.fail_on_regression is True


def test_main_internal_measure_path(monkeypatch, capsys) -> None:
    """Main should emit a JSON sample and exit 0 in internal measure mode."""
    monkeypatch.setattr(perf_cold_warm, "ensure_canonical_tree", lambda **kwargs: None)
    monkeypatch.setattr(
        perf_cold_warm,
        "parse_args",
        lambda _argv=None: Namespace(
            internal_measure_once=True,
            scenario_config=Path("configs/scenarios/archetypes/classic_crossing.yaml"),
            scenario_name="classic_crossing_low",
            episode_steps=8,
            seed=5,
        ),
    )
    monkeypatch.setattr(perf_cold_warm, "_load_scenario_config", lambda *args: (object(), "x"))
    monkeypatch.setattr(
        perf_cold_warm,
        "measure_once",
        lambda **kwargs: _sample(create=1.0, first=0.1, episode=1.2, sps=10.0),
    )
    exit_code = perf_cold_warm.main([])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert '"env_create_sec": 1.0' in out


def test_main_non_internal_paths(monkeypatch, tmp_path: Path) -> None:
    """Main should return expected exit codes for baseline and regression outcomes."""
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"

    def _common_args() -> Namespace:
        return Namespace(
            internal_measure_once=False,
            scenario_config=Path("configs/scenarios/archetypes/classic_crossing.yaml"),
            scenario_name="classic_crossing_low",
            episode_steps=16,
            seed=42,
            cold_runs=1,
            warm_runs=1,
            baseline=Path("configs/benchmarks/perf_baseline_classic_cold_warm_v1.json"),
            max_slowdown_pct=0.6,
            max_throughput_drop_pct=0.5,
            min_seconds_delta=0.15,
            min_throughput_delta=0.75,
            output_json=out_json,
            output_markdown=out_md,
            fail_on_regression=True,
            require_baseline=True,
        )

    monkeypatch.setattr(perf_cold_warm, "ensure_canonical_tree", lambda **kwargs: None)
    monkeypatch.setattr(
        perf_cold_warm, "_load_scenario_config", lambda *args: (object(), "scenario")
    )
    monkeypatch.setattr(
        perf_cold_warm,
        "run_suite",
        lambda **kwargs: (
            [_sample(create=1.0, first=0.2, episode=1.5, sps=20.0)],
            [_sample(create=0.8, first=0.1, episode=1.0, sps=25.0)],
        ),
    )

    # Missing baseline + require baseline => exit 2
    monkeypatch.setattr(perf_cold_warm, "parse_args", lambda _argv=None: _common_args())
    monkeypatch.setattr(perf_cold_warm, "load_snapshot", lambda _path: None)
    assert perf_cold_warm.main([]) == 2

    # Baseline present + regression + fail_on_regression => exit 1
    baseline = perf_cold_warm.SuiteSnapshot(
        cold=_sample(create=0.2, first=0.05, episode=0.6, sps=100.0),
        warm=_sample(create=0.2, first=0.05, episode=0.5, sps=100.0),
    )
    monkeypatch.setattr(perf_cold_warm, "load_snapshot", lambda _path: baseline)
    assert perf_cold_warm.main([]) == 1

    # Baseline present + no regression => exit 0
    monkeypatch.setattr(
        perf_cold_warm,
        "load_snapshot",
        lambda _path: perf_cold_warm.SuiteSnapshot(
            cold=_sample(create=1.0, first=0.2, episode=1.5, sps=20.0),
            warm=_sample(create=0.8, first=0.1, episode=1.0, sps=25.0),
        ),
    )
    assert perf_cold_warm.main([]) == 0
    assert out_json.exists()
    assert out_md.exists()
