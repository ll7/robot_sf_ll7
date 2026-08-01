"""Shared fixtures for analysis tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.support.script_loader import load_script_module

if TYPE_CHECKING:
    from types import ModuleType

_AMMV_PANEL_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "build_ammv_mechanism_panel_issue_2227.py"
)
_ISSUE_3078_DIAGNOSTIC_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "build_issue_3078_job_13521_diagnostic.py"
)


def _load_ammv_panel_module():
    """Import the issue #2227 AMMV panel builder script by path.

    Uses the shared :func:`tests.support.script_loader.load_script_module` helper
    (added for issue #5289) instead of the bare ``spec_from_file_location`` /
    ``module_from_spec`` / ``exec_module`` idiom. The shared helper registers the
    module in :data:`sys.modules` before ``exec_module`` so the script can use
    ``@dataclass(frozen=True)`` with bare ``InitVar`` / ``ClassVar`` annotations
    (the script already enables ``from __future__ import annotations``, which is
    the exact condition that surfaced the friction in issue #5289).
    """
    return load_script_module(
        _AMMV_PANEL_SCRIPT_PATH,
        name="build_ammv_mechanism_panel_issue_2227",
    )


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


@pytest.fixture(scope="session")
def diagnostic_builder() -> ModuleType:
    """Load the issue #3078 job-13521 diagnostic builder once per test session."""
    return load_script_module(
        _ISSUE_3078_DIAGNOSTIC_SCRIPT_PATH,
        name="build_issue_3078_job_13521_diagnostic",
    )
