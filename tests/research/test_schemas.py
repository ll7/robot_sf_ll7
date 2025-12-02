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

from robot_sf.research.orchestrator import ReportOrchestrator
from robot_sf.research.schema_loader import load_schema, validate_data


@pytest.fixture(name="report_dir")
def report_artifacts_dir(tmp_path: Path) -> Path:
    """Report artifacts dir.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        Path: Auto-generated placeholder description.
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
    """Load.

    Args:
        path: Auto-generated placeholder description.

    Returns:
        dict: Auto-generated placeholder description.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_hypothesis_schema(report_dir: Path):
    """Test hypothesis schema.

    Args:
        report_dir: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    hypothesis_path = report_dir / "data" / "hypothesis.json"
    assert hypothesis_path.exists(), "hypothesis.json missing"
    data = _load(hypothesis_path)
    schema = load_schema("hypothesis_result.schema.json")
    validate_data(data, schema)
    assert data["schema_version"].startswith("1.0.")


def test_metrics_schema(report_dir: Path):
    """Test metrics schema.

    Args:
        report_dir: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    metrics_path = report_dir / "data" / "metrics.json"
    assert metrics_path.exists(), "metrics.json missing"
    data = _load(metrics_path)
    schema = load_schema("aggregated_metrics.schema.json")
    validate_data(data, schema)
    assert data["metrics"], "metrics list empty"


def test_metadata_schema(report_dir: Path):
    """Test metadata schema.

    Args:
        report_dir: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    metadata_path = report_dir / "metadata.json"
    assert metadata_path.exists(), "metadata.json missing"
    data = _load(metadata_path)
    schema = load_schema("report_metadata.schema.v1.json")
    validate_data(data, schema)
    assert data["experiment_name"]
    assert data["artifacts"], "artifacts manifest missing"
