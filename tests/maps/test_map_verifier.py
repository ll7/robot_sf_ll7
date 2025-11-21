"""Tests for map verification workflow.

This test module validates the map verification system across multiple scenarios:
- Map inventory loading and filtering
- Rule validation (geometry, metadata, spawn points)
- Environment instantiation
- CLI integration
- Manifest output

Test Fixtures
-------------
- synthetic_valid_map : Valid SVG map fixture
- synthetic_broken_map : Invalid SVG map fixture (geometry issues)
- synthetic_missing_metadata_map : Map missing required metadata

See Also
--------
- robot_sf.maps.verification : Module under test
- specs/001-map-verification : Feature specification
"""

from pathlib import Path

import pytest


def _run_verify(scope: str = "ci", mode: str = "ci", output: Path | None = None):
    """Helper to invoke verification runner directly for tests."""
    from robot_sf.maps.verification.runner import verify_maps

    return verify_maps(scope=scope, mode=mode, output_path=output)


# Placeholder test to ensure module imports
def test_module_imports():
    """Test that verification module can be imported."""
    try:
        from robot_sf.maps import verification

        assert verification is not None
    except ImportError as e:
        pytest.fail(f"Failed to import verification module: {e}")


def test_ci_mode_pass():
    """CI mode should pass (failed == 0) on healthy repository maps."""
    summary = _run_verify(scope="ci", mode="ci")
    assert summary.failed == 0, "Expected no failed maps in CI scope"


def test_ci_mode_failure_simulation(monkeypatch):
    """Simulate a failure by monkeypatching the runner's apply_all_rules symbol."""
    import robot_sf.maps.verification.runner as runner_module
    from robot_sf.maps.verification.rules import RuleSeverity, RuleViolation

    def fake_apply_all_rules(_map_path: Path):  # force one error per map
        return [
            RuleViolation(
                rule_id="TEST_ERR",
                severity=RuleSeverity.ERROR,
                message="Simulated failure",
                remediation="None",
            )
        ]

    # Patch the reference used by verify_single_map
    monkeypatch.setattr(runner_module, "apply_all_rules", fake_apply_all_rules)
    summary = _run_verify(scope="ci", mode="ci")
    assert summary.failed > 0, "Expected simulated failures to be recorded"
    assert any("TEST_ERR" in r.rule_ids for r in summary.results if r.status.value == "fail"), (
        "Failing results should include the TEST_ERR rule ID"
    )
