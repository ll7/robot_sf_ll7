"""Tests for pytest configuration and strict marker fail-closed contracts (issue #8039)."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _load_pytest_ini_options() -> dict[str, object]:
    """Load pytest configuration from pyproject.toml."""
    raw = PYPROJECT_PATH.read_text(encoding="utf-8")
    data = tomllib.loads(raw)
    tool_section = data.get("tool", {})
    assert isinstance(tool_section, dict)
    pytest_section = tool_section.get("pytest", {})
    assert isinstance(pytest_section, dict)
    ini_options = pytest_section.get("ini_options", {})
    assert isinstance(ini_options, dict)
    return ini_options


def test_pytest_ini_options_declares_strict_flags() -> None:
    """Pytest addopts must include --strict-markers and --strict-config."""
    ini_options = _load_pytest_ini_options()
    addopts = ini_options.get("addopts", [])
    assert isinstance(addopts, list)
    assert "--strict-markers" in addopts
    assert "--strict-config" in addopts


def test_pytest_markers_are_unique_and_non_empty() -> None:
    """Declared markers in pyproject.toml must be non-empty and have unique marker names."""
    ini_options = _load_pytest_ini_options()
    markers = ini_options.get("markers", [])
    assert isinstance(markers, list)
    assert len(markers) > 0

    marker_names: list[str] = []
    for entry in markers:
        assert isinstance(entry, str)
        assert entry.strip() != ""
        name = entry.split(":", 1)[0].strip()
        assert name != ""
        marker_names.append(name)

    assert len(marker_names) == len(set(marker_names)), (
        f"Duplicate markers declared: {marker_names}"
    )


def test_strict_collection_fails_closed_on_unknown_marker(tmp_path: Path) -> None:
    """Pytest must fail collection when encountering an unregistered marker under strict mode."""
    test_file = tmp_path / "test_unknown_marker.py"
    test_file.write_text(
        "import pytest\n\n@pytest.mark.unregistered_custom_marker_probe\ndef test_dummy():\n    pass\n",
        encoding="utf-8",
    )

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(PYPROJECT_PATH),
            "--collect-only",
            str(test_file),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
    assert (
        "'unregistered_custom_marker_probe' not found in `markers` configuration option"
        in res.stderr
        or "'unregistered_custom_marker_probe' not found in `markers` configuration option"
        in res.stdout
    )
