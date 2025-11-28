from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.research.extractor_report import ReportConfig, generate_extractor_report


def test_generate_extractor_report_smoke(tmp_path: Path):
    summary = {
        "run_id": "demo-run",
        "extractor_results": [
            {
                "config_name": "baseline",
                "status": "success",
                "metrics": {
                    "best_mean_reward": 1.2,
                    "convergence_timestep": 20.0,
                    "sample_efficiency_ratio": 1.0,
                },
            },
            {
                "config_name": "candidate",
                "status": "success",
                "metrics": {
                    "best_mean_reward": 1.4,
                    "convergence_timestep": 18.0,
                    "sample_efficiency_ratio": 1.3,
                },
            },
        ],
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    cfg = ReportConfig(
        experiment_name="imitation",
        hypothesis="Candidate improves reward",
        significance_level=0.05,
        export_latex=True,
        baseline_extractor="baseline",
    )

    out = generate_extractor_report(
        summary_path=summary_path,
        output_root=tmp_path,
        config=cfg,
    )

    assert out["report"].exists()
    assert (out["figures_dir"] / "final_performance.png").exists()
    assert (out["figures_dir"] / "sample_efficiency.png").exists()
    assert out["metadata"].exists()
    meta = json.loads(out["metadata"].read_text(encoding="utf-8"))
    assert "git" in meta and "hardware" in meta and "experiment" in meta
    assert out["latex"].exists()
    latex_text = out["latex"].read_text(encoding="utf-8")
    assert "baseline & success" in latex_text
    assert r"\section*{Statistical Comparison}" in latex_text
