"""Tests for imitation report generation (Markdown + LaTeX)."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


from robot_sf.research.imitation_report import (
    ImitationReportConfig,
    _ci_from_samples,
    _fmt_ci,
    generate_imitation_report,
)
from scripts.research.generate_imitation_report import _parse_hparams


def _minimal_record(name: str, timesteps: float, success: float, collision: float, snqi: float):
    """Build a minimal extractor record with multi-sample metrics for testing."""

    return {
        "config_name": name,
        "status": "success",
        "start_time": "2025-01-01T00:00:00Z",
        "end_time": "2025-01-01T00:01:00Z",
        "duration_seconds": 60.0,
        "hardware_profile": {
            "platform": "linux",
            "arch": "x86_64",
            "python_version": "3.11",
            "workers": 1,
        },
        "worker_mode": "single-thread",
        "training_steps": int(timesteps),
        "metrics": {
            "timesteps_to_convergence": timesteps,
            "timesteps_to_convergence_samples": [
                timesteps * 0.9,
                timesteps,
                timesteps * 1.1,
            ],
            "success_rate": success,
            "success_rate_samples": [success * 0.95, success, success * 1.05],
            "collision_rate": collision,
            "collision_rate_samples": [collision * 1.1, collision, collision * 0.9],
            "snqi": snqi,
            "snqi_samples": [snqi * 0.9, snqi, snqi * 1.05],
            "sample_efficiency_ratio": 1.0,
            "sample_efficiency_ratio_samples": [0.9, 1.0, 1.1],
        },
        "artifacts": {},
    }


def test_generate_imitation_report(tmp_path: Path):
    """Generate an imitation report and assert artifacts and stats are present."""

    summary = {
        "run_id": "demo",
        "created_at": "2025-01-01T00:00:00Z",
        "output_root": str(tmp_path / "analysis"),
        "hardware_overview": [],
        "extractor_results": [
            _minimal_record("baseline_run", 1000.0, 0.6, 0.2, 0.5),
            _minimal_record("pretrained_run", 600.0, 0.8, 0.1, 0.7),
        ],
        "aggregate_metrics": {
            "sample_efficiency_ratio": 1.5,
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    fig_dir = summary_path.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    (fig_dir / "timesteps_comparison.png").write_bytes(png_bytes)
    (fig_dir / "performance_metrics.png").write_bytes(png_bytes)

    cfg = ImitationReportConfig(
        experiment_name="imitation-demo",
        export_latex=True,
        ablation_label="bc_epochs=5",
        hyperparameters={"bc_epochs": "5", "dataset_size": "5000"},
    )
    out = generate_imitation_report(
        summary_path=summary_path,
        output_root=tmp_path,
        config=cfg,
    )

    assert out["report"].exists()
    report_text = out["report"].read_text(encoding="utf-8")
    assert "p-value: n/a" not in report_text
    assert "p-value:" in report_text
    assert "Ablation: bc_epochs=5" in report_text
    assert "bc_epochs: 5" in report_text
    assert "dataset_size: 5000" in report_text
    assert out["metadata"].exists()
    assert out["figures_dir"].exists()
    assert out["latex"] is not None and out["latex"].exists()


def test_ci_from_samples_requires_two_or_more():
    """_ci_from_samples returns n/a for insufficient samples."""

    assert _ci_from_samples([]) == "n/a"
    assert _ci_from_samples([1.0]) == "n/a"


def test_ci_from_samples_computes_interval():
    """_ci_from_samples computes a symmetric 95% CI for sample arrays."""

    samples = [1.0, 2.0, 3.0, 4.0]
    ci = _ci_from_samples(samples)
    assert ci is not None
    low, high = ci
    # Mean is 2.5; CI should contain the mean and be symmetric around it.
    assert low < 2.5 < high


def test_fmt_ci_formats_tuple_and_na():
    """_fmt_ci formats tuples and n/a consistently."""

    assert _fmt_ci((1.23456, 2.34567)) == "(1.2346, 2.3457)"
    assert _fmt_ci(None) == "n/a"
    assert _fmt_ci("n/a") == "n/a"


def test_parse_hparams_handles_valid_and_invalid(monkeypatch):
    """_parse_hparams returns parsed dict and ignores malformed pairs."""

    parsed = _parse_hparams(["epochs=5", " dataset = 1000 ", "invalidpair", "=novalue"])
    assert parsed["epochs"] == "5"
    assert parsed["dataset"] == "1000"
    assert "" not in parsed
