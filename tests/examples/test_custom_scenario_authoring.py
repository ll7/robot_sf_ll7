"""Focused tests for the custom scenario-authoring tutorial (issue #8738)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from robot_sf.benchmark.identity.hash_utils import stable_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "examples/advanced/35_custom_scenario_authoring.py"


def _load_tutorial_module():
    """Load the tutorial example by file path (numeric filenames are not importable)."""
    spec = importlib.util.spec_from_file_location("tutorial_authoring", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tutorial = _load_tutorial_module()
build_scenario = _tutorial.build_scenario
demonstrate_negative_fixtures = _tutorial.demonstrate_negative_fixtures
reload_and_compare = _tutorial.reload_and_compare
serialize_scenario = _tutorial.serialize_scenario
validate_scenario_file = _tutorial.validate_scenario_file


def test_build_produces_deterministic_identity() -> None:
    """Same inputs must yield the same canonical digest (deterministic ordering)."""
    left = build_scenario("../../maps/svg_maps/classic_head_on_corridor.svg")
    right = build_scenario("../../maps/svg_maps/classic_head_on_corridor.svg")
    assert stable_hash(left) == stable_hash(right)
    assert left["seeds"] == sorted(left["seeds"])


def test_serialize_validate_reload_roundtrip(tmp_path: Path) -> None:
    """Authored scenario serializes, validates clean, and reloads equivalently."""
    scenario = build_scenario("../../maps/svg_maps/classic_head_on_corridor.svg")
    out = serialize_scenario(scenario, tmp_path / "authored.yaml")
    assert out.exists()
    validate_scenario_file(out)
    reload_and_compare(out, stable_hash(scenario))
    assert yaml.safe_load(out.read_text())["scenarios"][0]["name"] == "tutorial_authored_corridor"


def test_negative_fixtures_fail_with_specific_errors() -> None:
    """Invalid geometry/identity inputs fail (or warn) with specific outcomes."""
    outcomes = demonstrate_negative_fixtures()
    assert outcomes["duplicate_identity"].startswith("ValueError: Duplicate single pedestrian IDs")
    assert "radius must be positive" in outcomes["invalid_radius"]
    assert "must be a 2-tuple" in outcomes["malformed_start"]
    assert outcomes["goal_equals_start"].startswith("accepted with warning")
    assert outcomes["path_escape"].startswith("accepted; flagged")


def test_manifest_entry_is_registered() -> None:
    """The tutorial example must be registered and CI-enabled in the manifest."""
    manifest = yaml.safe_load(Path("examples/examples_manifest.yaml").read_text())
    entries = [
        e for e in manifest["examples"] if e["path"] == "advanced/35_custom_scenario_authoring.py"
    ]
    assert len(entries) == 1
    assert entries[0]["ci_enabled"] is True
