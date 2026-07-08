"""Tests for issue #4850 multiplier sweep helper scripts."""

import importlib.util
from pathlib import Path
from types import ModuleType

from robot_sf.benchmark.map_runner import build_map_policy

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(module_name: str, script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_format_probability_handles_missing_values() -> None:
    """Missing rank-probability fields should print as N/A instead of crashing."""
    module = _load_script_module(
        "generate_multiplier_sensitivity_report_4850",
        REPO_ROOT / "scripts/benchmark/generate_multiplier_sensitivity_report_4850.py",
    )

    assert module._format_probability(0.81234) == "0.812"
    assert module._format_probability(None) == "N/A"
    assert module._format_probability("N/A") == "N/A"


def test_multiplier_sweep_uses_public_policy_builder() -> None:
    """The multiplier sweep should import the public policy-builder wrapper."""
    module = _load_script_module(
        "run_multiplier_sweep_4850",
        REPO_ROOT / "scripts/benchmark/run_multiplier_sweep_4850.py",
    )

    assert module.build_map_policy is build_map_policy
