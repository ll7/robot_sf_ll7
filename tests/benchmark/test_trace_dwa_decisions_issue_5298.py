"""Unit contracts for the issue #5298 DWA trace exporter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "benchmark" / "trace_dwa_decisions_issue_5298.py"


def _load_trace_module():
    spec = importlib.util.spec_from_file_location("_dwa_trace_issue_5298", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_route_progress_summary_skips_and_records_non_finite_trace_values() -> None:
    """NaN/Infinity diagnostic cells never propagate into derived evidence."""
    trace = _load_trace_module()

    summary = trace._route_progress_summary(
        [
            {"distance_to_goal_m": "3.0", "route_progress_from_start_m": "0.0"},
            {"distance_to_goal_m": "nan", "route_progress_from_start_m": "inf"},
            {"distance_to_goal_m": "2.0", "route_progress_from_start_m": "1.0"},
        ]
    )

    assert summary["initial_distance_to_goal_m"] == 3.0
    assert summary["final_distance_to_goal_m"] == 2.0
    assert summary["skipped_non_finite_rows"] == 1
    assert summary["skipped_non_finite_cells"] == 2
