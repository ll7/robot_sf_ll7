"""Planner package import regression tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_planner_package_import_does_not_preload_learned_risk_stack() -> None:
    """Core imports should not load optional learned-risk dependencies."""
    script = """
import json
import sys

import robot_sf.planner

print(json.dumps({
    "learned_risk_surface": "robot_sf.planner.learned_risk_surface" in sys.modules,
    "predictive_model": "robot_sf.planner.predictive_model" in sys.modules,
    "risk_dwa": "robot_sf.planner.risk_dwa" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == (
        '{"learned_risk_surface": false, "predictive_model": false, "risk_dwa": false}'
    )


def test_planner_package_preserves_legacy_global_planner_export() -> None:
    """Lazy exports should keep the public planner import surface intact."""
    script = """
from robot_sf.planner import GlobalPlanner, VisibilityPlanner

assert GlobalPlanner is VisibilityPlanner
print(GlobalPlanner.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "VisibilityPlanner"
