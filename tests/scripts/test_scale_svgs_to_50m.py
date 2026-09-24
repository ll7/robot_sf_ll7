"""Regression tests for the explicit-input SVG scaling utility (issue #8813)."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.scale_svgs_to_50m import main

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "scale_svgs_to_50m.py"
TRACKED_MAPS = (
    REPO_ROOT / "maps/svg_maps/static_humans.svg",
    REPO_ROOT / "maps/svg_maps/overtaking.svg",
    REPO_ROOT / "maps/svg_maps/crossing.svg",
    REPO_ROOT / "maps/svg_maps/door_passing.svg",
)

SVG_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">
  <rect x="40" y="80" width="20" height="10" />
  <path d="M 100 100 L 200 200" />
</svg>
"""


def _digest(path: Path) -> str:
    """Return the tracked-file digest used to prove a run left it untouched."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bare_invocation_fails_without_touching_tracked_maps() -> None:
    """A no-argument run must fail and leave every tracked scaled map byte-identical."""
    before = {path: _digest(path) for path in TRACKED_MAPS}

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 2
    assert "required" in result.stderr
    assert {path: _digest(path) for path in TRACKED_MAPS} == before


def test_dry_run_reports_without_writing(tmp_path: Path) -> None:
    """--dry-run reports the planned write and leaves the input untouched."""
    source = tmp_path / "scenario.svg"
    source.write_text(SVG_FIXTURE, encoding="utf-8")
    before = _digest(source)

    assert main([str(source), "--dry-run"]) == 0

    assert _digest(source) == before


def test_explicit_input_is_scaled_in_place(tmp_path: Path) -> None:
    """An explicitly named input still scales its viewBox and coordinates."""
    source = tmp_path / "scenario.svg"
    source.write_text(SVG_FIXTURE, encoding="utf-8")

    assert main([str(source)]) == 0

    scaled = source.read_text(encoding="utf-8")
    assert 'viewBox="0 0 40 40"' in scaled
    assert 'x="4"' in scaled
    assert "M 10 10 L 20 20" in scaled


def test_output_dir_writes_copies_and_leaves_sources(tmp_path: Path) -> None:
    """--output-dir writes a scaled copy and keeps the source unchanged."""
    source = tmp_path / "scenario.svg"
    source.write_text(SVG_FIXTURE, encoding="utf-8")
    before = _digest(source)
    output_dir = tmp_path / "scaled"

    assert main([str(source), "--output-dir", str(output_dir)]) == 0

    target = output_dir / "scenario.svg"
    assert target.is_file()
    assert _digest(source) == before
    assert 'viewBox="0 0 40 40"' in target.read_text(encoding="utf-8")


def test_missing_input_fails_closed(tmp_path: Path) -> None:
    """A named but absent input fails instead of being skipped silently."""
    with pytest.raises(SystemExit, match="input is not a file"):
        main([str(tmp_path / "missing.svg")])
