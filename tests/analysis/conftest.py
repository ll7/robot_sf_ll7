"""Shared fixtures for analysis tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_AMMV_PANEL_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "build_ammv_mechanism_panel_issue_2227.py"
)


def _load_ammv_panel_module():
    """Import the issue #2227 AMMV panel builder script by path."""
    spec = importlib.util.spec_from_file_location(
        "build_ammv_mechanism_panel_issue_2227", _AMMV_PANEL_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def panel_module():
    """Load the issue #2227 panel builder module once per test module."""
    return _load_ammv_panel_module()


@pytest.fixture(scope="module")
def panel_run(panel_module, tmp_path_factory):
    """Run the full issue #2227 panel build once and return (summary, output_dir)."""
    output_dir = tmp_path_factory.mktemp("ammv_panel")
    summary = panel_module.run(output_dir)
    return summary, output_dir
