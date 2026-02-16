"""Unit tests for trend-oriented performance benchmark helpers."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from robot_sf.benchmark import perf_trend


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_matrix_parses_valid_payload(tmp_path: Path) -> None:
    """Matrix loader should parse suite/scenario definitions."""
    matrix = tmp_path / "matrix.yaml"
    _write_text(
        matrix,
        """
schema_version: benchmark-perf-trend-matrix.v1
suite_name: suite-v1
scenarios:
  - scenario_config: configs/scenarios/archetypes/classic_crossing.yaml
    scenario_name: classic_crossing_low
    episode_steps: 96
    cold_runs: 1
    warm_runs: 2
    baseline: configs/benchmarks/perf_baseline_classic_cold_warm_v1.json
    require_baseline: true
""".strip()
        + "\n",
    )
    suite, scenarios = perf_trend.load_matrix(matrix)
    assert suite == "suite-v1"
    assert len(scenarios) == 1
    assert scenarios[0].scenario_name == "classic_crossing_low"
    assert scenarios[0].episode_steps == 96
    assert scenarios[0].require_baseline is True


def test_load_matrix_applies_defaults_with_entry_precedence(tmp_path: Path) -> None:
    """Matrix defaults should apply first, while scenario entries may override them."""
    matrix = tmp_path / "matrix_defaults.yaml"
    _write_text(
        matrix,
        """
schema_version: benchmark-perf-trend-matrix.v1
suite_name: suite-v1
scenarios:
  - scenario_config: configs/scenarios/archetypes/classic_crossing.yaml
    scenario_name: classic_crossing_low
    warm_runs: 5
""".strip()
        + "\n",
    )

    _suite, scenarios = perf_trend.load_matrix(matrix)
    assert len(scenarios) == 1
    assert scenarios[0].cold_runs == 1
    assert scenarios[0].warm_runs == 5
    assert scenarios[0].max_slowdown_pct == pytest.approx(0.60)


def test_load_matrix_rejects_invalid_schema(tmp_path: Path) -> None:
    """Loader should reject unsupported schema versions."""
    matrix = tmp_path / "matrix.yaml"
    _write_text(
        matrix,
        """
schema_version: old-schema
suite_name: suite-v1
scenarios: []
""".strip()
        + "\n",
    )
    with pytest.raises(ValueError, match="Unsupported matrix schema_version"):
        perf_trend.load_matrix(matrix)


def test_load_matrix_validation_errors(tmp_path: Path) -> None:
    """Loader should reject malformed matrix structures."""
    not_object = tmp_path / "not_object.yaml"
    _write_text(not_object, "- just-a-list\n")
    with pytest.raises(ValueError, match="expected object"):
        perf_trend.load_matrix(not_object)

    missing_suite = tmp_path / "missing_suite.yaml"
    _write_text(
        missing_suite,
        """
schema_version: benchmark-perf-trend-matrix.v1
scenarios: []
""".strip()
        + "\n",
    )
    with pytest.raises(ValueError, match="Missing suite_name"):
        perf_trend.load_matrix(missing_suite)

    bad_scenario = tmp_path / "bad_scenario.yaml"
    _write_text(
        bad_scenario,
        """
schema_version: benchmark-perf-trend-matrix.v1
suite_name: suite
scenarios:
  - {}
""".strip()
        + "\n",
    )
    with pytest.raises(ValueError, match="missing scenario_name"):
        perf_trend.load_matrix(bad_scenario)


def test_load_matrix_rejects_duplicate_scenario_slugs(tmp_path: Path) -> None:
    """Scenario names that normalize to the same artifact slug should be rejected."""
    matrix = tmp_path / "duplicate_slug.yaml"
    _write_text(
        matrix,
        """
schema_version: benchmark-perf-trend-matrix.v1
suite_name: suite-v1
scenarios:
  - scenario_config: configs/scenarios/archetypes/classic_crossing.yaml
    scenario_name: classic/crossing
  - scenario_config: configs/scenarios/archetypes/classic_crossing.yaml
    scenario_name: classic_crossing
""".strip()
        + "\n",
    )
    with pytest.raises(ValueError, match="duplicate output slug"):
        perf_trend.load_matrix(matrix)


def test_parse_args_accepts_overrides() -> None:
    """Argument parser should accept explicit matrix and threshold overrides."""
    parsed = perf_trend.parse_args(
        [
            "--matrix",
            "configs/benchmarks/perf_trend_matrix_classic_v1.yaml",
            "--seed",
            "123",
            "--history-glob",
            "output/benchmarks/perf/trend/history/*.json",
            "--history-limit",
            "5",
            "--max-history-slowdown-pct",
            "0.4",
            "--max-history-throughput-drop-pct",
            "0.2",
            "--min-history-seconds-delta",
            "0.12",
            "--min-history-throughput-delta",
            "0.8",
            "--fail-on-regression",
            "--fail-on-history-regression",
        ]
    )
    assert parsed.seed == 123
    assert parsed.history_limit == 5
    assert parsed.max_history_slowdown_pct == pytest.approx(0.4)
    assert parsed.fail_on_regression is True
    assert parsed.fail_on_history_regression is True


def test_compare_with_history_no_reports() -> None:
    """History comparison should report no-history when no snapshots exist."""
    result = perf_trend.compare_with_history(
        current_results=[],
        history_reports=[],
        thresholds=perf_trend.HistoryThresholds(),
    )
    assert result["status"] == "no-history"
    assert result["findings"] == []


def test_compare_with_history_flags_startup_regression() -> None:
    """Time-metric slowdown should produce startup-focused diagnostics."""
    current = [
        {
            "scenario_name": "classic_crossing_low",
            "cold_median": {
                "env_create_sec": 3.0,
                "first_step_sec": 0.40,
                "episode_sec": 1.2,
                "steps_per_sec": 15.0,
            },
            "warm_median": {
                "env_create_sec": 2.0,
                "first_step_sec": 0.30,
                "episode_sec": 1.0,
                "steps_per_sec": 16.0,
            },
        }
    ]
    history_reports = [
        {
            "schema_version": perf_trend.TREND_REPORT_SCHEMA_VERSION,
            "scenario_results": [
                {
                    "scenario_name": "classic_crossing_low",
                    "cold_median": {
                        "env_create_sec": 1.0,
                        "first_step_sec": 0.10,
                        "episode_sec": 1.2,
                        "steps_per_sec": 15.0,
                    },
                    "warm_median": {
                        "env_create_sec": 0.9,
                        "first_step_sec": 0.08,
                        "episode_sec": 1.0,
                        "steps_per_sec": 16.0,
                    },
                }
            ],
        }
    ]
    result = perf_trend.compare_with_history(
        current_results=current,
        history_reports=history_reports,
        thresholds=perf_trend.HistoryThresholds(
            max_slowdown_pct=0.35,
            max_throughput_drop_pct=0.30,
            min_seconds_delta=0.10,
            min_throughput_delta=0.75,
        ),
    )
    assert result["status"] == "fail"
    assert any("startup overhead" in d for d in result["diagnostics"])
    assert any(f["is_regression"] and f["metric"] == "env_create_sec" for f in result["findings"])


def test_compare_with_history_passes_when_within_thresholds() -> None:
    """Small metric drift should not be marked as historical regression."""
    current = [
        {
            "scenario_name": "classic_crossing_low",
            "cold_median": {
                "env_create_sec": 1.2,
                "first_step_sec": 0.12,
                "episode_sec": 1.3,
                "steps_per_sec": 14.8,
            },
            "warm_median": {
                "env_create_sec": 1.0,
                "first_step_sec": 0.10,
                "episode_sec": 1.1,
                "steps_per_sec": 15.2,
            },
        }
    ]
    history_reports = [
        {
            "schema_version": perf_trend.TREND_REPORT_SCHEMA_VERSION,
            "scenario_results": [
                {
                    "scenario_name": "classic_crossing_low",
                    "cold_median": {
                        "env_create_sec": 1.1,
                        "first_step_sec": 0.10,
                        "episode_sec": 1.25,
                        "steps_per_sec": 15.0,
                    },
                    "warm_median": {
                        "env_create_sec": 0.95,
                        "first_step_sec": 0.09,
                        "episode_sec": 1.05,
                        "steps_per_sec": 15.4,
                    },
                }
            ],
        }
    ]
    result = perf_trend.compare_with_history(
        current_results=current,
        history_reports=history_reports,
        thresholds=perf_trend.HistoryThresholds(
            max_slowdown_pct=0.35,
            max_throughput_drop_pct=0.30,
            min_seconds_delta=0.10,
            min_throughput_delta=0.75,
        ),
    )
    assert result["status"] == "pass"
    assert "No historical regressions detected." in result["diagnostics"][0]


def test_load_history_reports_filters_invalid_payloads(tmp_path: Path) -> None:
    """History loader should keep only schema-valid JSON objects."""
    valid = tmp_path / "valid.json"
    _write_text(valid, json.dumps({"schema_version": perf_trend.TREND_REPORT_SCHEMA_VERSION}))
    invalid_json = tmp_path / "invalid.json"
    _write_text(invalid_json, "{not json")
    wrong_schema = tmp_path / "wrong_schema.json"
    _write_text(wrong_schema, json.dumps({"schema_version": "other"}))

    loaded = perf_trend._load_history_reports(str(tmp_path / "*.json"), limit=10)
    assert len(loaded) == 1
    assert loaded[0]["schema_version"] == perf_trend.TREND_REPORT_SCHEMA_VERSION
    assert "_source_path" in loaded[0]


def test_run_scenario_builds_perf_cold_warm_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scenario runner should forward matrix settings to perf_cold_warm."""
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str]) -> int:
        captured["argv"] = list(argv)
        out_json = Path(argv[argv.index("--output-json") + 1])
        _write_text(
            out_json,
            json.dumps(
                {
                    "comparison": {"status": "pass"},
                    "cold": {"median": {"steps_per_sec": 10.0}},
                    "warm": {"median": {"steps_per_sec": 12.0}},
                }
            ),
        )
        return 0

    monkeypatch.setattr(perf_trend.perf_cold_warm, "main", _fake_main)
    spec = perf_trend.ScenarioSpec(
        scenario_config=Path("configs/scenarios/archetypes/classic_crossing.yaml"),
        scenario_name="classic_crossing_low",
        episode_steps=96,
        cold_runs=1,
        warm_runs=2,
        baseline=Path("configs/benchmarks/perf_baseline_classic_cold_warm_v1.json"),
        require_baseline=True,
        max_slowdown_pct=0.60,
        max_throughput_drop_pct=0.50,
        min_seconds_delta=0.15,
        min_throughput_delta=0.75,
        enforce_regression_gate=True,
    )
    result = perf_trend._run_scenario(
        spec=spec,
        seed=101,
        output_root=tmp_path,
        fail_on_regression=True,
    )
    argv = captured["argv"]
    assert "--scenario-name" in argv and "classic_crossing_low" in argv
    assert "--require-baseline" in argv
    assert "--fail-on-regression" in argv
    assert result["comparison_status"] == "pass"
    assert result["exit_code"] == 0


def test_run_scenario_uses_sanitized_slug_for_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scenario output filenames should use sanitized slugs."""
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str]) -> int:
        captured["argv"] = list(argv)
        out_json = Path(argv[argv.index("--output-json") + 1])
        _write_text(out_json, json.dumps({"comparison": {"status": "pass"}}))
        return 0

    monkeypatch.setattr(perf_trend.perf_cold_warm, "main", _fake_main)
    spec = perf_trend.ScenarioSpec(
        scenario_config=Path("configs/scenarios/archetypes/classic_crossing.yaml"),
        scenario_name="Classic/Crossing (Low)",
        episode_steps=96,
        cold_runs=1,
        warm_runs=2,
        baseline=Path("configs/benchmarks/perf_baseline_classic_cold_warm_v1.json"),
        require_baseline=True,
        max_slowdown_pct=0.60,
        max_throughput_drop_pct=0.50,
        min_seconds_delta=0.15,
        min_throughput_delta=0.75,
        enforce_regression_gate=True,
    )

    perf_trend._run_scenario(
        spec=spec,
        seed=101,
        output_root=tmp_path,
        fail_on_regression=False,
    )

    argv = captured["argv"]
    out_json = Path(argv[argv.index("--output-json") + 1])
    assert out_json.name == "classic_crossing_low.json"


def test_main_writes_report_and_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main should emit JSON/Markdown and exit 0 when no failures occur."""
    out_json = tmp_path / "trend.json"
    out_md = tmp_path / "trend.md"
    monkeypatch.setattr(perf_trend, "ensure_canonical_tree", lambda **kwargs: None)
    monkeypatch.setattr(
        perf_trend,
        "parse_args",
        lambda _argv=None: Namespace(
            matrix=Path("configs/benchmarks/perf_trend_matrix_classic_v1.yaml"),
            seed=101,
            output_json=out_json,
            output_markdown=out_md,
            history_glob="",
            history_limit=10,
            max_history_slowdown_pct=0.35,
            max_history_throughput_drop_pct=0.30,
            min_history_seconds_delta=0.10,
            min_history_throughput_delta=0.75,
            fail_on_regression=True,
            fail_on_history_regression=False,
        ),
    )
    monkeypatch.setattr(
        perf_trend,
        "load_matrix",
        lambda _path: (
            "suite-v1",
            [
                perf_trend.ScenarioSpec(
                    scenario_config=Path("x.yaml"),
                    scenario_name="scenario",
                    episode_steps=16,
                    cold_runs=1,
                    warm_runs=1,
                    baseline=Path("baseline.json"),
                    require_baseline=False,
                    max_slowdown_pct=0.6,
                    max_throughput_drop_pct=0.5,
                    min_seconds_delta=0.1,
                    min_throughput_delta=0.75,
                    enforce_regression_gate=True,
                )
            ],
        ),
    )
    monkeypatch.setattr(
        perf_trend,
        "_run_scenario",
        lambda **kwargs: {
            "scenario_name": "scenario",
            "exit_code": 0,
            "comparison_status": "pass",
            "cold_median": {"steps_per_sec": 10.0},
            "warm_median": {"steps_per_sec": 12.0},
        },
    )
    monkeypatch.setattr(perf_trend, "_load_history_reports", lambda *args: [])

    exit_code = perf_trend.main([])
    assert exit_code == 0
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == perf_trend.TREND_REPORT_SCHEMA_VERSION
    assert payload["history_comparison"]["status"] == "no-history"


def test_main_fails_on_history_regression_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """History regression mode should return non-zero when configured."""
    out_json = tmp_path / "trend.json"
    out_md = tmp_path / "trend.md"
    monkeypatch.setattr(perf_trend, "ensure_canonical_tree", lambda **kwargs: None)
    monkeypatch.setattr(
        perf_trend,
        "parse_args",
        lambda _argv=None: Namespace(
            matrix=Path("configs/benchmarks/perf_trend_matrix_classic_v1.yaml"),
            seed=101,
            output_json=out_json,
            output_markdown=out_md,
            history_glob="history/*.json",
            history_limit=10,
            max_history_slowdown_pct=0.35,
            max_history_throughput_drop_pct=0.30,
            min_history_seconds_delta=0.10,
            min_history_throughput_delta=0.75,
            fail_on_regression=False,
            fail_on_history_regression=True,
        ),
    )
    monkeypatch.setattr(perf_trend, "load_matrix", lambda _path: ("suite-v1", []))
    monkeypatch.setattr(
        perf_trend,
        "_load_history_reports",
        lambda *args: [{"schema_version": perf_trend.TREND_REPORT_SCHEMA_VERSION}],
    )
    monkeypatch.setattr(
        perf_trend,
        "compare_with_history",
        lambda **kwargs: {"status": "fail", "findings": [], "diagnostics": ["regression"]},
    )
    assert perf_trend.main([]) == 1


def test_main_fails_on_scenario_regression_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main should return non-zero when scenario regression gate is enabled."""
    out_json = tmp_path / "trend.json"
    out_md = tmp_path / "trend.md"
    monkeypatch.setattr(perf_trend, "ensure_canonical_tree", lambda **kwargs: None)
    monkeypatch.setattr(
        perf_trend,
        "parse_args",
        lambda _argv=None: Namespace(
            matrix=Path("configs/benchmarks/perf_trend_matrix_classic_v1.yaml"),
            seed=101,
            output_json=out_json,
            output_markdown=out_md,
            history_glob="",
            history_limit=10,
            max_history_slowdown_pct=0.35,
            max_history_throughput_drop_pct=0.30,
            min_history_seconds_delta=0.10,
            min_history_throughput_delta=0.75,
            fail_on_regression=True,
            fail_on_history_regression=False,
        ),
    )
    monkeypatch.setattr(
        perf_trend,
        "load_matrix",
        lambda _path: (
            "suite-v1",
            [
                perf_trend.ScenarioSpec(
                    scenario_config=Path("x.yaml"),
                    scenario_name="scenario",
                    episode_steps=16,
                    cold_runs=1,
                    warm_runs=1,
                    baseline=Path("baseline.json"),
                    require_baseline=False,
                    max_slowdown_pct=0.6,
                    max_throughput_drop_pct=0.5,
                    min_seconds_delta=0.1,
                    min_throughput_delta=0.75,
                    enforce_regression_gate=True,
                )
            ],
        ),
    )
    monkeypatch.setattr(
        perf_trend,
        "_run_scenario",
        lambda **kwargs: {
            "scenario_name": "scenario",
            "exit_code": 1,
            "comparison_status": "fail",
            "cold_median": {"steps_per_sec": 10.0},
            "warm_median": {"steps_per_sec": 12.0},
        },
    )
    monkeypatch.setattr(perf_trend, "_load_history_reports", lambda *args: [])

    assert perf_trend.main([]) == 1
