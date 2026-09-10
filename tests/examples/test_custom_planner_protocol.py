"""Focused tests for the custom planner-protocol tutorial (issue #8737)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from robot_sf.planner.protocol import LocalPlannerProtocol

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "examples/advanced/37_custom_planner_protocol.py"


def _load_tutorial_module():
    """Load the tutorial example by file path (numeric filenames are not importable)."""
    spec = importlib.util.spec_from_file_location("tutorial_planner", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tutorial_planner"] = module
    spec.loader.exec_module(module)
    return module


_tutorial = _load_tutorial_module()


def test_planner_satisfies_protocol() -> None:
    """The tutorial class structurally satisfies LocalPlannerProtocol."""
    assert isinstance(_tutorial.TutorialPlanner(), LocalPlannerProtocol)


def test_same_inputs_produce_same_actions_and_diagnostics() -> None:
    """Same seed and inputs deterministically reproduce actions and diagnostics."""
    first = _tutorial.TutorialPlanner()
    first.reset(seed=11)
    second = _tutorial.TutorialPlanner()
    second.reset(seed=11)
    observation: dict[str, object] = {}
    assert [first.plan(observation) for _ in range(3)] == [
        second.plan(observation) for _ in range(3)
    ]
    assert first.diagnostics() == second.diagnostics()


def test_invalid_commands_fail_with_validation_error() -> None:
    """Wrong shape, non-finite, and out-of-bounds commands raise explicitly."""
    with pytest.raises(ValueError, match="must be a"):
        _tutorial.validate_command((0.3,))
    with pytest.raises(ValueError, match="finite"):
        _tutorial.validate_command((float("nan"), 0.0))
    with pytest.raises(ValueError, match="outside bounds"):
        _tutorial.validate_command((5.0, 0.0))
    with pytest.raises(ValueError, match="outside bounds"):
        _tutorial.validate_command((0.3, 9.0))


def test_close_is_idempotent() -> None:
    """Closing twice must not raise."""
    planner = _tutorial.TutorialPlanner()
    planner.close()
    planner.close()


def test_manifest_entry_is_registered() -> None:
    """The tutorial example must be registered and CI-enabled in the manifest."""
    import yaml

    manifest = yaml.safe_load(Path("examples/examples_manifest.yaml").read_text())
    entries = [
        entry
        for entry in manifest["examples"]
        if entry["path"] == "advanced/37_custom_planner_protocol.py"
    ]
    assert len(entries) == 1
    assert entries[0]["ci_enabled"] is True
