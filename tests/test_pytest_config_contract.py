"""Contract tests for pytest configuration and strict marker declarations."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_pytest_addopts_contains_strict_flags() -> None:
    """pytest configuration enforces strict-markers and strict-config."""
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.is_file()

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    pytest_opts = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
    addopts = pytest_opts.get("addopts", [])

    assert "--strict-markers" in addopts, "addopts must include --strict-markers"
    assert "--strict-config" in addopts, "addopts must include --strict-config"


def test_pytest_markers_are_valid_and_unique() -> None:
    """Custom markers declared in pyproject.toml are non-empty, unique, and well-formed."""
    pyproject_path = REPO_ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    pytest_opts = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
    markers = pytest_opts.get("markers", [])

    assert len(markers) > 0, "At least one custom marker must be declared"

    marker_names: list[str] = []
    for marker_entry in markers:
        assert isinstance(marker_entry, str)
        assert marker_entry.strip(), "Marker entry cannot be empty or whitespace"
        assert ":" in marker_entry, (
            f"Marker entry must include description with colon: {marker_entry}"
        )

        name = marker_entry.split(":", 1)[0].strip()
        assert name.isidentifier(), f"Marker name must be a valid Python identifier: {name}"
        marker_names.append(name)

    # Check for duplicates
    assert len(marker_names) == len(set(marker_names)), (
        f"Duplicate markers detected: {marker_names}"
    )

    # Verify canonical expected markers are declared
    expected_markers = {"slow", "timeout", "base_sensitive"}
    for expected in expected_markers:
        assert expected in marker_names, (
            f"Expected marker {expected} is missing from pyproject.toml"
        )
