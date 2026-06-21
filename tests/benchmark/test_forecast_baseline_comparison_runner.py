"""Tests for the #2915 forecast-baseline comparison runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from robot_sf.benchmark.forecast_batch import validate_forecast_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/run_forecast_baseline_comparison.py"
CONFIG_PATH = REPO_ROOT / "configs/research/forecast_baseline_comparison_issue_2915.yaml"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_forecast_baseline_comparison", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tmp_config(tmp_path: Path) -> Path:
    config = _load_runner()._load_config(CONFIG_PATH)
    evidence_dir = tmp_path / "evidence"
    config["output"] = {
        "evidence_dir": str(evidence_dir),
        "forecast_jsonl_dir": str(evidence_dir / "forecast_batches"),
        "comparison_table": str(evidence_dir / "comparison_table.csv"),
        "comparison_report_json": str(evidence_dir / "comparison_report.json"),
        "comparison_report_md": str(evidence_dir / "comparison_report.md"),
        "summary_json": str(evidence_dir / "summary.json"),
    }
    config["scenario_families"] = [
        "corridor_interaction",
        "crossing_proxy",
        "dense_pedestrian_interaction",
    ]
    config["seeds"] = [111]
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_runner_writes_schema_valid_forecast_batches_on_identical_inputs(tmp_path: Path) -> None:
    """Each baseline emits ForecastBatch.v1 rows over the same evaluated trace keys."""
    runner = _load_runner()
    report = runner.run(_tmp_config(tmp_path), date="2026-06-20")

    assert report["summary"]["evidence_tier"] == "analysis_only"
    assert report["summary"]["row_status_counts"]["evaluated"] > 0

    keys_by_baseline: dict[str, list[tuple[str, str, int]]] = {}
    for baseline in report["summary"]["baselines"]:
        path = tmp_path / "evidence" / "forecast_batches" / f"{baseline}.forecast_batch.v1.jsonl"
        payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert payloads
        keys_by_baseline[baseline] = [
            (
                payload["metadata"]["scenario_family"],
                payload["metadata"]["trace_label"],
                payload["provenance"]["seed"],
            )
            for payload in payloads
        ]
        for payload in payloads:
            batch = validate_forecast_batch(payload)
            assert batch.provenance.predictor_id == baseline
            assert batch.provenance.fallback_status == "native"
            assert batch.provenance.degraded_status == "not_degraded"
            assert batch.metadata["evidence_tier"] == "analysis_only"

    assert len({tuple(keys) for keys in keys_by_baseline.values()}) == 1


def test_runner_fails_closed_for_unavailable_configured_families(tmp_path: Path) -> None:
    """Unavailable selected families are recorded as not available, not successful rows."""
    runner = _load_runner()
    report = runner.run(_tmp_config(tmp_path), date="2026-06-20")

    summary = report["summary"]
    assert "dense_pedestrian_interaction" in {
        item["family"] for item in summary["missing_selected_families"]
    }
    strongest = report["strongest_by_family"]
    assert strongest["crossing_proxy"]["best"] is None
    assert strongest["crossing_proxy"]["reason"] == "no evaluated baseline rows"
