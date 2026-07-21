"""Focused contract tests for the optional benchmark reproducibility check."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_repro_check.py"
SPEC = importlib.util.spec_from_file_location("benchmark_repro_check", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_minimal_simple_policy_run_matches_current_aggregate_contract(tmp_path: Path):
    """A real minimal run passes schema validation and emits the required aggregate shape."""
    result = MODULE.run_benchmark_pipeline(tmp_path, seed=123)
    assert result["status"] == "passed"
    assert result["episodes_count"] == 2

    summary = json.loads(result["summary_file"].read_text(encoding="utf-8"))
    diagnostic = MODULE.validate_simple_policy_aggregate(summary)

    assert diagnostic["status"] == "passed"
    assert diagnostic["missing_metrics"] == []
    assert diagnostic["missing_statistics"] == []


def test_reports_missing_aggregate_metric_as_structured_failure():
    """A stale fixture fails closed with a machine-readable missing-metric diagnostic."""
    summary = {"simple_policy": {"path_efficiency": {"mean": 1.0, "median": 1.0, "p95": 1.0}}}

    diagnostic = MODULE.validate_simple_policy_aggregate(summary)

    assert diagnostic["status"] == "failed"
    assert diagnostic["schema"] == "benchmark_repro_check.aggregate_validation.v1"
    assert diagnostic["missing_metrics"] == ["collisions", "success"]
    assert diagnostic["missing_statistics"] == []


def test_same_seed_comparison_rejects_near_drift(tmp_path: Path):
    """A 0.04 aggregate drift is a deterministic reproducibility failure."""

    def write_summary(path: Path, collision_mean: float) -> None:
        statistics = {"mean": 0.0, "median": 0.0, "p95": 0.0}
        summary = {
            "simple_policy": {
                "collisions": {**statistics, "mean": collision_mean},
                "path_efficiency": dict(statistics),
                "success": dict(statistics),
            }
        }
        path.write_text(json.dumps(summary), encoding="utf-8")

    summary1 = tmp_path / "summary1.json"
    summary2 = tmp_path / "summary2.json"
    write_summary(summary1, 1.0)
    write_summary(summary2, 1.04)
    result1 = {"summary_file": summary1, "episodes_count": 2}
    result2 = {"summary_file": summary2, "episodes_count": 2}

    assert MODULE.compare_reproducibility(result1, result2) is False


def test_main_writes_structured_report_before_pipeline_failure_exit(tmp_path: Path, monkeypatch):
    """The optional lane leaves an uploadable report when a pipeline stage fails."""
    failure = {
        "status": "failed",
        "stage": "aggregate_validation",
        "error": {"missing_metrics": ["collisions"]},
    }
    monkeypatch.setattr(MODULE, "ensure_canonical_tree", lambda **_kwargs: None)
    monkeypatch.setattr(MODULE, "get_artifact_category_path", lambda _category: tmp_path)
    monkeypatch.setattr(MODULE, "ensure_output_dir", lambda _path: None)
    monkeypatch.setattr(MODULE, "run_benchmark_pipeline", lambda _path, seed: failure)

    assert MODULE.main() == 1

    report = json.loads((tmp_path / "reproducibility_check.json").read_text(encoding="utf-8"))
    assert report["schema"] == "benchmark_repro_check.report.v1"
    assert report["status"] == "failed"
    assert report["stage"] == "run1.aggregate_validation"
    assert report["error"] == {"missing_metrics": ["collisions"]}
    assert report["reproducible"] is False


def test_main_reports_expected_setup_error_with_narrow_handler(tmp_path: Path, monkeypatch):
    """Expected filesystem setup failures retain the structured-report contract."""

    def fail_setup(**_kwargs) -> None:
        raise OSError("artifact root unavailable")

    monkeypatch.setattr(MODULE, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "ensure_canonical_tree", fail_setup)

    assert MODULE.main() == 1

    report = json.loads(
        (tmp_path / "output" / "benchmarks" / "reproducibility_check.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["stage"] == "setup_or_execution"
    assert report["error"] == "OSError: artifact root unavailable"
