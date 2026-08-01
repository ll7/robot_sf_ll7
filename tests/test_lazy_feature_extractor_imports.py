"""Regression tests for optional ML dependency import boundaries."""

from __future__ import annotations

import subprocess
import sys


def test_feature_extractor_package_import_defers_ml_dependencies():
    """Importing the package and an extractor module must not load ML frameworks."""
    code = """\
import importlib
import sys

import robot_sf.feature_extractors
importlib.import_module("robot_sf.feature_extractors.attention_extractor")

assert "torch" not in sys.modules
assert "stable_baselines3" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, check=False, text=True
    )

    assert result.returncode == 0, result.stderr
