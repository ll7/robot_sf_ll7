"""Policy-builder registry helpers for benchmark map-runner planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robot_sf.planner.chance_constrained_mpc import build_chance_constrained_mpc_adapter
from robot_sf.planner.dwa import DWAPlannerAdapter, build_dwa_config
from robot_sf.planner.learned_prediction_mpc import (
    LEARNED_PREDICTION_MPC_ALIASES,
    build_learned_prediction_mpc_adapter,
)
from robot_sf.planner.prediction_mpc import (
    PredictionMPCPlannerAdapter,
    build_prediction_mpc_config,
)
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, build_risk_dwa_config
from robot_sf.planner.sipp_lattice import build_sipp_lattice_search_adapter
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


def _build_dwa_policy_spec(algo_config: dict[str, Any]) -> AdapterPolicySpec:
    """Build the classical DWA adapter spec without map-runner metadata.

    Returns:
        Adapter construction payload for the map runner.
    """
    return AdapterPolicySpec(
        algo_key="dwa",
        algo_config=algo_config,
        adapter=DWAPlannerAdapter(config=build_dwa_config(algo_config)),
        adapter_name="DWAPlannerAdapter",
        limitations="classical_dwa_experimental_testing_only",
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


def _build_chance_constrained_mpc_policy_spec(algo_config: dict[str, Any]) -> AdapterPolicySpec:
    """Build the experimental K-mode/GMM chance-constrained MPC arm.

    The builder intentionally fails closed until issue #2844 supplies a concrete
    multimodal predictor provider; no constant-velocity substitution is valid.

    Returns:
        The chance-constrained adapter specification when its provider is wired.
    """

    return AdapterPolicySpec(
        algo_key="chance_constrained_mpc",
        algo_config=algo_config,
        adapter=build_chance_constrained_mpc_adapter(algo_config),
        adapter_name="ChanceConstrainedMPCPlannerAdapter",
        limitations="blocked_until_issue_2844_k_mode_gmm_predictor_provider",
    )


def _build_learned_prediction_mpc_policy_spec(algo_config: dict[str, Any]) -> AdapterPolicySpec:
    """Build learned short-horizon prediction MPC algorithm config.

    Returns:
        AdapterPolicySpec: Map-runner adapter construction payload.
    """

    return AdapterPolicySpec(
        algo_key="learned_prediction_mpc",
        algo_config=algo_config,
        adapter=build_learned_prediction_mpc_adapter(algo_config),
        adapter_name="PredictionMPCPlannerAdapter",
        limitations="diagnostic_learned_short_horizon_prediction_mpc_not_benchmark_evidence",
    )


def _build_sipp_lattice_policy_spec(algo_config: dict[str, Any]) -> AdapterPolicySpec:
    """Build the Slice-2 bounded SIPP state-time search adapter spec.

    The adapter is testing-only/experimental: it produces exploratory
    implementation evidence (#5306 Slice 2), not benchmark superiority evidence.

    Returns:
        AdapterPolicySpec: Adapter construction payload for the map runner.
    """

    raw_opt_in = algo_config.get("allow_testing_algorithms", False)
    opt_in = (
        raw_opt_in
        if isinstance(raw_opt_in, bool)
        else str(raw_opt_in).strip().lower() in {"true", "1", "yes"}
    )
    if not opt_in:
        raise ValueError(
            "sipp_lattice is experimental/testing-only and requires allow_testing_algorithms: true"
        )

    return AdapterPolicySpec(
        algo_key="sipp_lattice",
        algo_config=algo_config,
        adapter=build_sipp_lattice_search_adapter(algo_config),
        adapter_name="SippLatticeSearchPlannerAdapter",
        limitations="experimental_bounded_kinodynamic_sipp_search_testing_only",
    )


_ADAPTER_POLICY_BUILDERS: dict[str, Callable[[dict[str, Any]], AdapterPolicySpec]] = {
    "chance_constrained_mpc": _build_chance_constrained_mpc_policy_spec,
    **dict.fromkeys(LEARNED_PREDICTION_MPC_ALIASES, _build_learned_prediction_mpc_policy_spec),
    "cv_prediction_mpc": _build_prediction_mpc_policy_spec,
    "prediction_aware_mpc": _build_prediction_mpc_policy_spec,
    "prediction_mpc": _build_prediction_mpc_policy_spec,
    "prediction_mpc_cbf": _build_prediction_mpc_policy_spec,
    "dwa": _build_dwa_policy_spec,
    "risk_dwa": _build_risk_dwa_policy_spec,
    "sipp_lattice": _build_sipp_lattice_policy_spec,
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
