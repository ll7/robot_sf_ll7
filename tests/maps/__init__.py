"""Tests for map verification workflow.

This test module validates the map verification system across multiple scenarios:
- Map inventory loading and filtering
- Rule validation (file readability, XML validity, labeled layers)
- Environment instantiation decisions in the runner
- CLI integration
- Manifest output

See Also
--------
- robot_sf.maps.verification : Module under test
- specs/001-map-verification : Feature specification
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from robot_sf.maps.verification.context import (
    FactoryType,
    VerificationContext,
    VerificationRunSummary,
    VerificationStatus,
)
from robot_sf.maps.verification.manifest import write_manifest
from robot_sf.maps.verification.map_inventory import MapInventory, MapRecord
from robot_sf.maps.verification.rules import RuleSeverity, apply_all_rules
from robot_sf.maps.verification.runner import verify_single_map
from robot_sf.maps.verification.scope_resolver import ScopeResolver

if TYPE_CHECKING:
    from pathlib import Path


def _write_svg(path: Path, *, labeled: bool, valid: bool = True) -> Path:
    """Write a minimal SVG file used for verification tests.

    Args:
        path: Target file path.
        labeled: Whether to include an inkscape:label attribute on a group.
        valid: Whether to emit valid XML.
    """
    if not valid:
        path.write_text("<svg><g", encoding="utf-8")
        return path

    label_attr = 'inkscape:label="obstacles"' if labeled else ""
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="10" height="10">
  <g {label_attr}></g>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


# Fixtures
@pytest.fixture
def synthetic_valid_map(tmp_path: Path) -> Path:
    """Create a valid synthetic SVG map for testing."""
    return _write_svg(tmp_path / "valid.svg", labeled=True, valid=True)


@pytest.fixture
def synthetic_broken_map(tmp_path: Path) -> Path:
    """Create an invalid SVG map with XML issues."""
    return _write_svg(tmp_path / "broken.svg", labeled=True, valid=False)


@pytest.fixture
def synthetic_missing_metadata_map(tmp_path: Path) -> Path:
    """Create an SVG map missing required layer labels."""
    return _write_svg(tmp_path / "missing_labels.svg", labeled=False, valid=True)


class TestMapInventory:
    """Test map inventory loading and enumeration."""

    def test_load_all_maps(self, tmp_path: Path):
        """Test loading all maps from a temporary svg_maps directory."""
        _write_svg(tmp_path / "map_a.svg", labeled=True)
        _write_svg(tmp_path / "map_b.svg", labeled=True)
        inventory = MapInventory(maps_root=tmp_path)
        assert len(inventory.get_all_maps()) == 2

    def test_filter_ci_enabled_maps(self, tmp_path: Path):
        """Test filtering maps by ci_enabled flag."""
        _write_svg(tmp_path / "map_a.svg", labeled=True)
        _write_svg(tmp_path / "map_b.svg", labeled=True)
        inventory = MapInventory(maps_root=tmp_path)
        # Flip one map to simulate CI filtering
        first = inventory.get_map_by_id("map_a")
        assert first is not None
        first.ci_enabled = False
        assert len(inventory.get_ci_enabled_maps()) == 1

    def test_scope_resolver_all(self, tmp_path: Path):
        """Test scope resolver with 'all' scope."""
        _write_svg(tmp_path / "map_a.svg", labeled=True)
        _write_svg(tmp_path / "map_b.svg", labeled=True)
        inventory = MapInventory(maps_root=tmp_path)
        resolver = ScopeResolver(inventory)
        maps = resolver.resolve("all")
        assert len(maps) == 2

    def test_scope_resolver_glob(self, tmp_path: Path):
        """Test scope resolver with glob patterns."""
        _write_svg(tmp_path / "classic_one.svg", labeled=True)
        _write_svg(tmp_path / "classic_two.svg", labeled=True)
        _write_svg(tmp_path / "other.svg", labeled=True)
        inventory = MapInventory(maps_root=tmp_path)
        resolver = ScopeResolver(inventory)
        maps = resolver.resolve("classic_*.svg")
        assert {m.map_id for m in maps} == {"classic_one", "classic_two"}


class TestRuleValidation:
    """Test individual validation rules."""

    def test_valid_svg_passes(self, synthetic_valid_map: Path):
        """Valid SVG with labeled layers should not raise errors or warnings."""
        violations = apply_all_rules(synthetic_valid_map)
        assert all(v.severity == RuleSeverity.INFO for v in violations)

    def test_invalid_svg_fails(self, synthetic_broken_map: Path):
        """Invalid XML should trigger the R002 error."""
        violations = apply_all_rules(synthetic_broken_map)
        assert any(v.rule_id == "R002" and v.severity == RuleSeverity.ERROR for v in violations)

    def test_missing_layer_labels_warns(self, synthetic_missing_metadata_map: Path):
        """Missing inkscape labels should trigger the R004 warning."""
        violations = apply_all_rules(synthetic_missing_metadata_map)
        assert any(v.rule_id == "R004" and v.severity == RuleSeverity.WARNING for v in violations)


class TestEnvironmentInstantiation:
    """Test environment instantiation decisions during verification."""

    def test_robot_env_instantiation(self, synthetic_valid_map: Path, tmp_path: Path):
        """Robot maps should select the robot factory."""
        record = MapRecord(map_id="valid", file_path=synthetic_valid_map)
        context = VerificationContext(mode="local", artifact_root=tmp_path)
        result = verify_single_map(record, context)
        assert result.status == VerificationStatus.PASS
        assert result.factory_used == FactoryType.ROBOT

    def test_pedestrian_env_instantiation(self, synthetic_valid_map: Path, tmp_path: Path):
        """Pedestrian-only maps should select the pedestrian factory."""
        record = MapRecord(
            map_id="ped_only_map",
            file_path=synthetic_valid_map,
            tags={"pedestrian_only"},
        )
        context = VerificationContext(mode="local", artifact_root=tmp_path)
        result = verify_single_map(record, context)
        assert result.status == VerificationStatus.PASS
        assert result.factory_used == FactoryType.PEDESTRIAN

    def test_timing_budget_warning(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Slow maps should be reported in the summary slow_maps list."""
        _write_svg(tmp_path / "slow_map.svg", labeled=True)

        from robot_sf.maps.verification import runner

        class TinyTimeoutContext(VerificationContext):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.soft_timeout_s = 0.0

        monkeypatch.setattr(
            runner,
            "MapInventory",
            lambda: MapInventory(maps_root=tmp_path),
        )
        monkeypatch.setattr(runner, "VerificationContext", TinyTimeoutContext)

        summary = runner.verify_maps(scope="all", mode="local")
        assert summary.slow_maps == ["slow_map"]


class TestCLIIntegration:
    """Test CLI entry point behavior."""

    def test_cli_help(self):
        """Test CLI help output."""
        result = subprocess.run(
            [sys.executable, "scripts/validation/verify_maps.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--scope" in result.stdout

    def test_cli_local_mode(self, tmp_path: Path):
        """Test CLI in local mode."""
        output_path = tmp_path / "manifest.json"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/validation/verify_maps.py",
                "--scope",
                "classic_doorway.svg",
                "--mode",
                "local",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert output_path.exists()

    def test_cli_ci_mode_exit_codes(self, tmp_path: Path):
        """Test CLI exit codes in CI mode."""
        output_path = tmp_path / "manifest.json"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/validation/verify_maps.py",
                "--scope",
                "classic_doorway.svg",
                "--mode",
                "ci",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert output_path.exists()


class TestManifestOutput:
    """Test structured manifest generation."""

    def test_manifest_schema_compliance(self, synthetic_valid_map: Path, tmp_path: Path):
        """Test that generated manifest matches expected schema keys."""
        record = MapRecord(map_id="valid", file_path=synthetic_valid_map)
        context = VerificationContext(mode="local", artifact_root=tmp_path)
        result = verify_single_map(record, context)
        summary = VerificationRunSummary(
            run_id="test-run",
            git_sha=None,
            total_maps=1,
            passed=1,
            failed=0,
            warned=0,
            slow_maps=[],
            artifact_path=tmp_path / "manifest.json",
            started_at=result.timestamp,
            finished_at=result.timestamp,
            results=[result],
        )
        write_manifest(summary, summary.artifact_path)

        data = json.loads(summary.artifact_path.read_text(encoding="utf-8"))
        assert "run_id" in data
        assert "summary" in data
        assert "results" in data
        assert data["summary"]["total_maps"] == 1

    def test_manifest_filtering(self, synthetic_valid_map: Path, tmp_path: Path):
        """Test filtering manifest results by status."""
        record = MapRecord(map_id="valid", file_path=synthetic_valid_map)
        context = VerificationContext(mode="local", artifact_root=tmp_path)
        result = verify_single_map(record, context)
        summary = VerificationRunSummary(
            run_id="test-run",
            git_sha=None,
            total_maps=1,
            passed=1,
            failed=0,
            warned=0,
            slow_maps=[],
            artifact_path=tmp_path / "manifest.json",
            started_at=result.timestamp,
            finished_at=result.timestamp,
            results=[result],
        )
        write_manifest(summary, summary.artifact_path)

        data = json.loads(summary.artifact_path.read_text(encoding="utf-8"))
        passed = [r for r in data["results"] if r["status"] == "pass"]
        assert len(passed) == summary.passed


# Placeholder test to ensure module imports
def test_module_imports():
    """Test that verification module can be imported."""
    try:
        from robot_sf.maps import verification

        assert verification is not None
    except ImportError as e:
        pytest.fail(f"Failed to import verification module: {e}")
