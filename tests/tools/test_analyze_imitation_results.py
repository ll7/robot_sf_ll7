"""TODO docstring. Document this module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.benchmark.imitation_manifest import get_training_run_manifest_path
from scripts.tools.analyze_imitation_results import main

if TYPE_CHECKING:
    from pathlib import Path


def _write_manifest(run_id: str, root: Path, metrics: dict[str, float]) -> Path:
    """TODO docstring. Document this function.

    Args:
        run_id: TODO docstring.
        root: TODO docstring.
        metrics: TODO docstring.

    Returns:
        TODO docstring.
    """
    path = get_training_run_manifest_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_payload: dict[str, Any] = {}
    for name, val in metrics.items():
        metric_payload[name] = {"mean": val, "median": val, "p95": val}
        metric_payload[f"{name}_samples"] = [float(val), float(val)]
    payload = {
        "run_id": run_id,
        "run_type": "baseline_ppo",
        "input_artefacts": [],
        "seeds": [1],
        "metrics": metric_payload,
        "episode_log_path": "",
        "wall_clock_hours": 0.0,
        "status": "completed",
        "notes": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_analyze_imitation_results_generates_summary(tmp_path: Path, monkeypatch):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        monkeypatch: TODO docstring.
    """
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    baseline_metrics = {
        "timesteps_to_convergence": 1000.0,
        "success_rate": 0.6,
        "collision_rate": 0.2,
        "snqi": 0.5,
    }
    pretrained_metrics = {
        "timesteps_to_convergence": 400.0,
        "success_rate": 0.8,
        "collision_rate": 0.1,
        "snqi": 0.7,
    }
    _write_manifest("baseline_run", tmp_path, baseline_metrics)
    _write_manifest("pretrained_run", tmp_path, pretrained_metrics)

    out_dir = tmp_path / "analysis_out"
    exit_code = main(
        [
            "--group",
            "demo-group",
            "--baseline",
            "baseline_run",
            "--pretrained",
            "pretrained_run",
            "--output",
            str(out_dir),
        ]
    )
    assert exit_code == 0

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["aggregate_metrics"]["sample_efficiency_ratio"] == pytest.approx(2.5)
    figures_dir = out_dir / "figures"
    assert figures_dir.exists()
    expected_pngs = {
        "fig-sample-efficiency.png",
        "fig-success_rate-distribution.png",
        "fig-collision_rate-distribution.png",
        "fig-snqi-distribution.png",
        "fig-improvement-summary.png",
    }
    png_names = {path.name for path in figures_dir.glob("*.png")}
    assert png_names == expected_pngs
