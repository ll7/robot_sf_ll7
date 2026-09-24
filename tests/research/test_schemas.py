"""Schema validation tests for research reporting artifacts.

Validate lightweight structural schemas for hypothesis and metrics JSON outputs
produced by ``ReportOrchestrator.generate_report``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from robot_sf.research.exceptions import ValidationError
from robot_sf.research.orchestrator import ReportOrchestrator
from robot_sf.research.schema_loader import load_schema, validate_data


@pytest.fixture(name="report_dir")
def report_artifacts_dir(tmp_path: Path) -> Path:
    """Generate a report bundle with hypothesis, metrics, and metadata.

    Args:
        tmp_path: Temporary directory used as the report parent.

    Returns:
        Path to the generated report directory.
    """
    out_dir = tmp_path / "report"
    orchestrator = ReportOrchestrator(out_dir)
    seeds = [1, 2, 3]
    # Synthetic metric records (baseline vs pretrained)
    metric_records = []
    for s in seeds:
        metric_records.append(
            {"seed": s, "policy_type": "baseline", "reward": 50 + s, "timesteps": 1000 + s * 10}
        )
        metric_records.append(
            {"seed": s, "policy_type": "pretrained", "reward": 70 + s, "timesteps": 700 + s * 5}
        )
    orchestrator.generate_report(
        experiment_name="demo",
        metric_records=metric_records,
        run_id="r1",
        seeds=seeds,
        baseline_timesteps=[1000, 1010, 1020],
        pretrained_timesteps=[700, 705, 710],
        baseline_rewards=[[0, 1, 2], [0, 1.1, 2.1], [0, 1.2, 2.2]],
        pretrained_rewards=[[0, 1.5, 2.5], [0, 1.6, 2.6], [0, 1.7, 2.7]],
        telemetry={"cpu_percent": 10.0, "mem_mb": 50.0},
    )
    return out_dir


def _load(path: Path) -> dict:
    """Load a JSON artifact from disk.

    Args:
        path: Path to the JSON artifact.

    Returns:
        Parsed JSON payload.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_hypothesis_schema(report_dir: Path):
    """Validate the generated hypothesis artifact against its schema.

    Args:
        report_dir: Generated report directory containing ``data/hypothesis.json``.
    """
    hypothesis_path = report_dir / "data" / "hypothesis.json"
    assert hypothesis_path.exists(), "hypothesis.json missing"
    data = _load(hypothesis_path)
    schema = load_schema("hypothesis_result.schema.json")
    validate_data(data, schema)
    assert data["schema_version"].startswith("1.0.")


def test_metrics_schema(report_dir: Path):
    """Validate the generated aggregate metrics artifact against its schema.

    Args:
        report_dir: Generated report directory containing ``data/metrics.json``.
    """
    metrics_path = report_dir / "data" / "metrics.json"
    assert metrics_path.exists(), "metrics.json missing"
    data = _load(metrics_path)
    schema = load_schema("aggregated_metrics.schema.json")
    validate_data(data, schema)
    assert data["metrics"], "metrics list empty"


def test_metrics_schema_accepts_zero_convergence_summary() -> None:
    """The aggregate schema admits zero-valued convergence summaries."""
    schema = load_schema("aggregated_metrics.schema.json")
    validate_data(
        {
            "schema_version": "1.0.0",
            "metrics": [
                {
                    "metric_name": "timesteps_to_convergence",
                    "condition": "baseline",
                    "mean": 0,
                    "median": 0,
                    "p95": 0,
                    "std": 0,
                    "ci_low": None,
                    "ci_high": None,
                    "ci_confidence": 0.95,
                    "sample_size": 1,
                    "effect_size": None,
                }
            ],
        },
        schema,
    )


def test_metrics_schema_rejects_negative_convergence_summary() -> None:
    """The aggregate schema rejects negative convergence summaries."""
    schema = load_schema("aggregated_metrics.schema.json")
    with pytest.raises(ValidationError, match="less than the minimum of 0"):
        validate_data(
            {
                "schema_version": "1.0.0",
                "metrics": [
                    {
                        "metric_name": "timesteps_to_convergence",
                        "condition": "baseline",
                        "mean": -1,
                        "median": 0,
                        "p95": 0,
                        "std": 0,
                        "ci_low": None,
                        "ci_high": None,
                        "ci_confidence": 0.95,
                        "sample_size": 1,
                        "effect_size": None,
                    }
                ],
            },
            schema,
        )


def test_metadata_schema(report_dir: Path):
    """Validate the generated report metadata and artifact manifest.

    Args:
        report_dir: Generated report directory containing ``metadata.json``.
    """
    metadata_path = report_dir / "metadata.json"
    assert metadata_path.exists(), "metadata.json missing"
    data = _load(metadata_path)
    schema = load_schema("report_metadata.schema.v1.json")
    validate_data(data, schema)
    assert data["experiment_name"]
    assert data["artifacts"], "artifacts manifest missing"
