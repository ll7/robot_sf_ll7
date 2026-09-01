"""Contract tests for pytest configuration and strict fail-closed behavior."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _collect_with_config(config_path: Path, test_path: Path) -> subprocess.CompletedProcess[str]:
    """Collect one test file with the supplied pytest configuration."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(config_path), "--collect-only", str(test_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pytest_addopts_contains_strict_flags() -> None:
    """pytest configuration enforces strict-markers and strict-config."""
    assert PYPROJECT_PATH.is_file()

    with PYPROJECT_PATH.open("rb") as f:
        data = tomllib.load(f)

    pytest_opts = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
    addopts = pytest_opts.get("addopts", [])

    assert "--strict-markers" in addopts, "addopts must include --strict-markers"
    assert "--strict-config" in addopts, "addopts must include --strict-config"


def test_pytest_markers_are_valid_and_unique() -> None:
    """Custom markers declared in pyproject.toml are non-empty, unique, and well-formed."""
    with PYPROJECT_PATH.open("rb") as f:
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

    assert len(marker_names) == len(set(marker_names)), (
        f"Duplicate markers detected: {marker_names}"
    )

    expected_markers = {"slow", "timeout", "base_sensitive"}
    for expected in expected_markers:
        assert expected in marker_names, (
            f"Expected marker {expected} is missing from pyproject.toml"
        )


def test_strict_collection_fails_closed_on_unknown_marker(tmp_path: Path) -> None:
    """Default collection rejects a marker absent from the repository declaration list."""
    test_file = tmp_path / "test_unknown_marker.py"
    marker_name = "unregistered_custom_marker_probe"
    test_file.write_text(
        f"import pytest\n\n@pytest.mark.{marker_name}\ndef test_probe():\n    pass\n",
        encoding="utf-8",
    )

    result = _collect_with_config(PYPROJECT_PATH, test_file)

    assert result.returncode == 2
    assert f"'{marker_name}' not found in `markers` configuration option" in (
        result.stdout + result.stderr
    )


def test_strict_collection_fails_closed_on_unknown_config(tmp_path: Path) -> None:
    """Default strict-config rejects an unknown option in a copied repository config."""
    unknown_option = "unknown_config_option_probe"
    config_text = PYPROJECT_PATH.read_text(encoding="utf-8")
    section = "[tool.pytest.ini_options]"
    assert config_text.count(section) == 1
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        config_text.replace(section, f'{section}\n{unknown_option} = "enabled"', 1),
        encoding="utf-8",
    )
    test_file = tmp_path / "test_unknown_config.py"
    test_file.write_text("def test_probe():\n    pass\n", encoding="utf-8")

    result = _collect_with_config(config_path, test_file)

    assert result.returncode == 4
    assert f"Unknown config option: {unknown_option}" in result.stdout + result.stderr
