"""Policy-builder registry helpers for benchmark map-runner planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, build_risk_dwa_config

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class AdapterPolicySpec:
    """Planner adapter payload for map-runner policy wrapping."""

    algo_key: str
    algo_config: dict[str, Any]
    adapter: Any
    adapter_name: str
    limitations: str | None = None


def _build_risk_dwa_policy_spec(algo_config: dict[str, Any]) -> AdapterPolicySpec:
    """Build the Risk-DWA adapter spec without applying map-runner metadata.

    Returns:
        AdapterPolicySpec: Adapter construction payload for the map runner.
    """

    return AdapterPolicySpec(
        algo_key="risk_dwa",
        algo_config=algo_config,
        adapter=RiskDWAPlannerAdapter(config=build_risk_dwa_config(algo_config)),
        adapter_name="RiskDWAPlannerAdapter",
    )


_ADAPTER_POLICY_BUILDERS: dict[str, Callable[[dict[str, Any]], AdapterPolicySpec]] = {
    "risk_dwa": _build_risk_dwa_policy_spec,
}


def build_registered_adapter_policy_spec(
    algo_key: str,
    algo_config: dict[str, Any],
) -> AdapterPolicySpec | None:
    """Return a registered adapter policy spec for ``algo_key`` when migrated.

    Returns:
        AdapterPolicySpec | None: Registered adapter payload, or ``None`` for unmigrated keys.
    """

    builder = _ADAPTER_POLICY_BUILDERS.get(algo_key)
    if builder is None:
        return None
    return builder(algo_config)
