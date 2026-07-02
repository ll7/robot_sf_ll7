"""Policy-builder registry helpers for benchmark map-runner planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robot_sf.planner.prediction_mpc import (
    PredictionMPCPlannerAdapter,
    build_prediction_mpc_config,
)
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, build_risk_dwa_config
from robot_sf.planner.teb_commitment import (
    TEBCommitmentPlannerAdapter,
    build_teb_commitment_config,
)

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


def _build_teb_policy_spec(algo_config: dict[str, Any]) -> AdapterPolicySpec:
    """Build the TEB commitment adapter spec without applying map-runner metadata.

    Returns:
        AdapterPolicySpec: Adapter construction payload for the map runner.
    """

    return AdapterPolicySpec(
        algo_key="teb",
        algo_config=algo_config,
        adapter=TEBCommitmentPlannerAdapter(config=build_teb_commitment_config(algo_config)),
        adapter_name="TEBCommitmentPlannerAdapter",
    )


def _build_prediction_mpc_policy_spec(algo_config: dict[str, Any]) -> AdapterPolicySpec:
    """Build experimental prediction-aware MPC from algorithm config.

    Returns:
        AdapterPolicySpec: Map-runner adapter construction payload.
    """
    return AdapterPolicySpec(
        algo_key="prediction_mpc",
        algo_config=algo_config,
        adapter=PredictionMPCPlannerAdapter(config=build_prediction_mpc_config(algo_config)),
        adapter_name="PredictionMPCPlannerAdapter",
        limitations="experimental_prediction_aware_mpc_local_planner",
    )


_ADAPTER_POLICY_BUILDERS: dict[str, Callable[[dict[str, Any]], AdapterPolicySpec]] = {
    "cv_prediction_mpc": _build_prediction_mpc_policy_spec,
    "prediction_aware_mpc": _build_prediction_mpc_policy_spec,
    "prediction_mpc": _build_prediction_mpc_policy_spec,
    "risk_dwa": _build_risk_dwa_policy_spec,
    "teb": _build_teb_policy_spec,
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
