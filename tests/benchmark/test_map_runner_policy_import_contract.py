"""Import-contract checks for the map-runner policy namespace migration."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    ("module_name", "symbol_name"),
    (
        ("map_runner_policy_actions", "ppo_action_to_unicycle"),
        ("map_runner_policy_common", "build_adapter_policy"),
        ("map_runner_policy_metadata", "finalize_feasibility_metadata"),
        ("map_runner_policy_resolution", None),
        ("map_runner_actions", "policy_command_to_env_action"),
        ("map_runner_profile_metadata", "load_synthetic_actuation_profile"),
    ),
)
def test_legacy_policy_module_is_identity_alias(
    module_name: str,
    symbol_name: str | None,
) -> None:
    """Verify flat imports preserve module identity and representative public symbols.

    This matters because external callers and pickled references still use the flat paths; a
    non-identity alias would duplicate module state and break metadata wiring.
    """
    legacy_name = f"robot_sf.benchmark.{module_name}"
    canonical_name = f"robot_sf.benchmark.map_runner_policies.{module_name}"

    legacy = importlib.import_module(legacy_name)
    canonical = importlib.import_module(canonical_name)

    assert legacy is canonical
    assert sys.modules[legacy_name] is canonical
    assert canonical.__name__ == canonical_name
    if symbol_name is not None:
        assert getattr(legacy, symbol_name) is getattr(canonical, symbol_name)
