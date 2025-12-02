"""Integration test T020: precision report structure.

Validate the statistical sufficiency report is produced with required keys:
final_pass (bool) and evaluations (list of per-scenario entries). Each entry
should contain metric_status list with metric entries.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


def test_precision_report_structure(config_factory):
    """Test precision report structure.

    Args:
        config_factory: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    cfg = config_factory(smoke=True)
    manifest = run_full_benchmark(cfg)
    report_path = Path(manifest.output_root) / "reports" / "statistical_sufficiency.json"
    assert report_path.exists(), "statistical_sufficiency.json missing"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(data.get("final_pass"), bool)
    evaluations = data.get("evaluations")
    assert isinstance(evaluations, list)
    if evaluations:
        first = evaluations[0]
        for key in ("scenario_id", "archetype", "density", "episodes", "metric_status", "all_pass"):
            assert key in first
        assert isinstance(first["metric_status"], list)
