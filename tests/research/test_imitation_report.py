from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.research.imitation_report import (
    ImitationReportConfig,
    generate_imitation_report,
)


def _minimal_record(name: str, timesteps: float, success: float, collision: float, snqi: float):
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
            "success_rate": success,
            "collision_rate": collision,
            "snqi": snqi,
            "sample_efficiency_ratio": 1.0,
        },
        "artifacts": {},
    }


def test_generate_imitation_report(tmp_path: Path):
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
    (fig_dir / "timesteps_comparison.png").write_bytes(b"png")
    (fig_dir / "performance_metrics.png").write_bytes(b"png")

    cfg = ImitationReportConfig(experiment_name="imitation-demo", export_latex=True)
    out = generate_imitation_report(
        summary_path=summary_path,
        output_root=tmp_path,
        config=cfg,
    )

    assert out["report"].exists()
    assert out["metadata"].exists()
    assert out["figures_dir"].exists()
    assert out["latex"] is not None and out["latex"].exists()
