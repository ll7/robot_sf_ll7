"""Readiness metadata and profile guards for benchmark algorithm selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from robot_sf.benchmark.algorithm_contract import (
    CONTRACT_RECORDS_BY_NAME,
    MIGRATED_ALGORITHM_RECORDS,
)

AlgorithmTier = Literal["baseline-ready", "experimental", "placeholder"]
BenchmarkProfile = Literal["baseline-safe", "paper-baseline", "experimental"]


@dataclass(frozen=True)
class AlgorithmReadiness:
    """Readiness metadata for one canonical map-benchmark algorithm."""

    canonical_name: str
    tier: AlgorithmTier
    aliases: tuple[str, ...]
    note: str
    requires_explicit_opt_in: bool = False


def _contract_readiness(canonical_name: str) -> AlgorithmReadiness:
    """Build the readiness facade entry from the canonical contract registry.

    The migrated families (issue #7676 first slice) are owned by
    :mod:`robot_sf.benchmark.algorithm_contract`; this adapter keeps the public
    ``AlgorithmReadiness`` surface unchanged.

    Returns:
        AlgorithmReadiness: The registry-derived readiness entry.
    """
    record = CONTRACT_RECORDS_BY_NAME[canonical_name]
    return AlgorithmReadiness(
        canonical_name=record.canonical_name,
        tier=record.tier,
        aliases=record.aliases,
        note=record.note,
        requires_explicit_opt_in=record.requires_explicit_opt_in,
    )


_ALGORITHMS: tuple[AlgorithmReadiness, ...] = (
    AlgorithmReadiness(
        canonical_name="goal",
        tier="baseline-ready",
        aliases=("goal", "simple", "goal_policy", "simple_policy"),
        note="Goal-following heuristic baseline.",
    ),
    AlgorithmReadiness(
        canonical_name="social_force",
        tier="baseline-ready",
        aliases=("social_force", "sf"),
        note="Social-force adapter baseline.",
    ),
    _contract_readiness("orca"),
    _contract_readiness("socnav_orca_nonholonomic"),
    _contract_readiness("socnav_orca_dd"),
    _contract_readiness("socnav_orca_relaxed"),
    _contract_readiness("socnav_hrvo"),
    _contract_readiness("hrvo"),
    AlgorithmReadiness(
        canonical_name="drl_vo",
        tier="experimental",
        aliases=("drl_vo", "drlvo", "drl-vo"),
        note=(
            "DRL-VO hybrid planner (learned policy augmented with velocity obstacle fallback); "
            "prototype stage."
        ),
        requires_explicit_opt_in=True,
    ),
    _contract_readiness("social_navigation_pyenvs_orca"),
    _contract_readiness("social_navigation_pyenvs_socialforce"),
    _contract_readiness("social_navigation_pyenvs_sfm_helbing"),
    _contract_readiness("social_navigation_pyenvs_hsfm_new_guo"),
    AlgorithmReadiness(
        canonical_name="crowdnav_height",
        tier="experimental",
        aliases=("crowdnav_height",),
        note="Upstream CrowdNav_HEIGHT model-only checkpoint wrapper.",
    ),
    AlgorithmReadiness(
        canonical_name="sonic_crowdnav",
        tier="experimental",
        aliases=("sonic_crowdnav", "sonic_gst"),
        note="Upstream SoNIC model-only checkpoint wrapper with fail-fast source asset checks.",
    ),
    _contract_readiness("gensafenav_ours_gst"),
    _contract_readiness("gensafenav_ours_gst_guarded"),
    _contract_readiness("gensafenav_gst_predictor_rand"),
    _contract_readiness("gensafenav_gst_predictor_rand_guarded"),
    AlgorithmReadiness(
        canonical_name="ppo",
        tier="experimental",
        aliases=("ppo",),
        note="Learned PPO baseline (paper profile requires provenance + quality gate).",
    ),
    AlgorithmReadiness(
        canonical_name="sac",
        tier="experimental",
        aliases=("sac",),
        note="Learned SB3 SAC baseline; benchmarkable only after checkpoint-specific quality gate.",
    ),
    AlgorithmReadiness(
        canonical_name="distributional_rl",
        tier="experimental",
        aliases=("distributional_rl", "qr_dqn"),
        note=(
            "Diagnostic QR-DQN-style distributional RL adapter; requires an explicit "
            "smoke checkpoint and is not benchmark or paper evidence."
        ),
    ),
    AlgorithmReadiness(
        canonical_name="brne",
        tier="experimental",
        aliases=("brne",),
        note=(
            "Pinned upstream BRNE corridor-only native diagnostic; requires explicit opt-in and "
            "is not benchmark or paper evidence."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="guarded_ppo",
        tier="experimental",
        aliases=("guarded_ppo",),
        note="PPO baseline with short-horizon safety veto and local fallback.",
    ),
    AlgorithmReadiness(
        canonical_name="socnav_sampling",
        tier="experimental",
        aliases=("socnav_sampling", "sampling"),
        note="SocNav sampling adapter; dependency-sensitive.",
    ),
    AlgorithmReadiness(
        canonical_name="sacadrl",
        tier="experimental",
        aliases=("sacadrl", "sa_cadrl"),
        note="GA3C-CADRL adapter; dependency/model-sensitive.",
    ),
    AlgorithmReadiness(
        canonical_name="sicnav",
        tier="experimental",
        aliases=("sicnav",),
        note="External SICNav MPC wrapper; dependency-sensitive and testing-only.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="dr_mpc",
        tier="experimental",
        aliases=("dr_mpc", "drmpc"),
        note="External DR-MPC wrapper; dependency-sensitive assessment anchor.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="prediction_planner",
        tier="experimental",
        aliases=("prediction_planner",),
        note="RGL-inspired predictive planner; requires trained checkpoint.",
    ),
    AlgorithmReadiness(
        canonical_name="predictive_mppi",
        tier="experimental",
        aliases=("predictive_mppi",),
        note="Learned-prediction sequence optimizer over short action horizons.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="risk_dwa",
        tier="experimental",
        aliases=("risk_dwa",),
        note="Risk-aware dynamic-window planner (non-learning).",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="taga_group_avoidance",
        tier="experimental",
        aliases=("taga_group_avoidance", "group_avoidance"),
        note=(
            "TAGA-like tangent-subgoal wrapper around the goal baseline using declared "
            "social-group o-space metadata; diagnostic group-intrusion surface only."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="risk_surface_dwa",
        tier="experimental",
        aliases=("risk_surface_dwa", "risk_surface_dwa_v0"),
        note=(
            "Deterministic local risk-surface producer wrapped around risk_dwa; "
            "prototype-only and not learned-risk benchmark evidence."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="hybrid_rule_local_planner",
        tier="experimental",
        aliases=(
            "hybrid_rule_local_planner",
            "hybrid_rule_v0_minimal",
            "actuation_aware_hybrid_rule_v0",
        ),
        note=(
            "Deterministic hybrid-rule local planner family; v0 is minimal DWA-style. "
            "Actuation-aware aliases are synthetic diagnostic-only candidates."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="adaptive_proxemic_selector_v0",
        tier="experimental",
        aliases=("adaptive_proxemic_selector_v0",),
        note=(
            "Diagnostic selector over fixed conservative, neutral, and open proxemic "
            "hybrid-rule profile candidates."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="adaptive_proxemic_selector_v1",
        tier="experimental",
        aliases=("adaptive_proxemic_selector_v1",),
        note=(
            "Diagnostic neutral-default selector over fixed proxemic hybrid-rule profiles; "
            "open profile is reserved for sparse low-progress recovery."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="safety_barrier",
        tier="experimental",
        aliases=("safety_barrier",),
        note="Testing-only clean-room static-obstacle safety-barrier planner.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="grid_route",
        tier="experimental",
        aliases=("grid_route",),
        note="Testing-only occupancy-grid route planner for static obstacle slices.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="hybrid_global_rl",
        tier="experimental",
        aliases=(
            "hybrid_global_rl",
            "global_rl_local",
            "route_conditioned_rl",
            "hybrid_route_rl",
        ),
        note=(
            "Diagnostic-only route-conditioned learned local planner; not benchmark evidence "
            "until paired route-scenario comparison is recorded."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="topology_guided_hybrid_rule_v0",
        tier="experimental",
        aliases=("topology_guided_hybrid_rule_v0", "topology_hypothesis_dwa_v0"),
        note=(
            "Diagnostic-only masked-route hypothesis selector feeding the hybrid-rule "
            "local scorer; not benchmark evidence."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="lidar_social_force",
        tier="experimental",
        aliases=("lidar_social_force", "lidar_tracked_social_force"),
        note=(
            "Testing-only LiDAR endpoint-cluster tracked-agent adapter wrapped around "
            "SocialForce; not benchmark evidence without explicit opt-in."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="lidar_grid_route",
        tier="experimental",
        aliases=("lidar_grid_route", "lidar_occupancy_grid_route"),
        note=(
            "Testing-only LiDAR-derived ego occupancy adapter wrapped around grid_route; "
            "not benchmark evidence without explicit opt-in."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="trivial_reference",
        tier="experimental",
        aliases=("trivial_reference", "reference_adapter"),
        note="Diagnostic starter-template adapter for contributor onboarding; not benchmark evidence.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="mppi_social",
        tier="experimental",
        aliases=("mppi_social",),
        note="Sampling-based MPPI/CEM social local planner.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="hybrid_portfolio",
        tier="experimental",
        aliases=("hybrid_portfolio",),
        note="Risk-regime switch between risk_dwa, ORCA, and prediction planner.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="planner_selector_v2_diagnostic",
        tier="experimental",
        aliases=("planner_selector_v2_diagnostic",),
        note=(
            "Diagnostic-only deterministic selector over existing local planner candidates; "
            "not benchmark-strength evidence."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="policy_stack_v1",
        tier="experimental",
        aliases=("policy_stack_v1",),
        note="Minimal non-learning portfolio over goal and risk_dwa proposal sources.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="hybrid_orca_sampler",
        tier="experimental",
        aliases=("hybrid_orca_sampler",),
        note="ORCA primary planner with short-horizon MPPI repair for stalled or unsafe scenes.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="stream_gap",
        tier="experimental",
        aliases=("stream_gap",),
        note="Gap-acceptance local planner for crossing/bottleneck experiments.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="gap_prediction",
        tier="experimental",
        aliases=("gap_prediction",),
        note="Predictive planner with stream-gap veto layer.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="socnav_bench",
        tier="experimental",
        aliases=("socnav_bench",),
        note="SocNav benchmark adapter; dependency-sensitive.",
    ),
    AlgorithmReadiness(
        canonical_name="rvo",
        tier="placeholder",
        aliases=("rvo",),
        note="Placeholder adapter; not benchmark-validated.",
    ),
    AlgorithmReadiness(
        canonical_name="dwa",
        tier="experimental",
        aliases=("dwa",),
        note=(
            "Deterministic in-repository Dynamic Window Approach baseline; implemented but "
            "requires explicit opt-in until benchmark evidence supports promotion."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="teb",
        tier="experimental",
        aliases=("teb",),
        note="Native corridor-commitment planner inspired by TEB-style local optimization.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="nmpc_social",
        tier="experimental",
        aliases=("nmpc_social", "nmpc"),
        note="Native NMPC-style local planner with short-horizon nonlinear optimization.",
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="prediction_mpc",
        tier="experimental",
        aliases=(
            "prediction_mpc",
            "prediction_mpc_cbf",
            "prediction_aware_mpc",
            "cv_prediction_mpc",
        ),
        note=(
            "Prediction-aware MPC local planner with constant-velocity pedestrian futures "
            "as hard time-varying collision constraints."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="learned_prediction_mpc",
        tier="experimental",
        aliases=(
            "learned_prediction_mpc",
            "learned_short_horizon_mpc",
            "model_based_local_planner",
            "learned_prediction_planner",
        ),
        note=(
            "Diagnostic learned short-horizon pedestrian prediction MPC lane; not a full "
            "navigation world model and not benchmark evidence without a trained checkpoint."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="sipp_lattice",
        tier="experimental",
        aliases=("sipp_lattice", "sipp_kinodynamic", "kinodynamic_sipp"),
        note=(
            "Bounded kinodynamic state-time (SIPP-class) local search with time-indexed "
            "pedestrian safe intervals and multi-step commitment (#5306 Slice 2); exploratory "
            "implementation evidence only, not benchmark superiority evidence."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="force_coupled_potential_field",
        tier="experimental",
        aliases=("force_coupled_potential_field",),
        note=(
            "Clean-room force-coupled potential-field local-planner comparator; "
            "implementation-integrity smoke only, not benchmark or reproduction evidence."
        ),
        requires_explicit_opt_in=True,
    ),
    AlgorithmReadiness(
        canonical_name="topology_parallel_nmpc",
        tier="experimental",
        aliases=("topology_parallel_nmpc", "topology_nmpc"),
        note=(
            "Testing-only offline topology-parallel NMPC prototype (#6152 / #5310); evaluates "
            "2-4 deterministic x-y-t maneuver hypotheses with identical NMPC solver, objective, "
            "constraints, tolerances, horizon, and iteration cap. Not benchmark evidence."
        ),
        requires_explicit_opt_in=True,
    ),
)

_ALIAS_INDEX: dict[str, AlgorithmReadiness] = {
    alias: spec for spec in _ALGORITHMS for alias in spec.aliases
}


def get_algorithm_readiness(name: str) -> AlgorithmReadiness | None:
    """Return readiness metadata for an algorithm name or alias."""
    return _ALIAS_INDEX.get(str(name).strip().lower())


def require_algorithm_allowed(
    *,
    algo: str,
    benchmark_profile: BenchmarkProfile,
    ppo_paper_ready: bool,
    allow_testing_algorithms: bool = False,
) -> AlgorithmReadiness | None:
    """Validate algorithm selection against profile gating.

    Returns:
        AlgorithmReadiness | None: Metadata for known algorithms, or ``None``
        when the algorithm is not part of the catalog.

    Raises:
        ValueError: If the algorithm is disallowed by readiness/profile policy.
    """
    spec = get_algorithm_readiness(algo)
    if spec is None:
        return None

    if spec.tier == "placeholder":
        raise ValueError(
            f"Algorithm '{algo}' is marked placeholder and is not allowed for benchmark runs. "
            "Choose a baseline-ready or experimental algorithm.",
        )

    if benchmark_profile == "baseline-safe" and spec.tier != "baseline-ready":
        raise ValueError(
            f"Algorithm '{algo}' is {spec.tier} and blocked by profile 'baseline-safe'. "
            "Use '--benchmark-profile experimental' for exploratory runs.",
        )

    if benchmark_profile == "paper-baseline":
        if spec.canonical_name == "ppo":
            if not ppo_paper_ready:
                raise ValueError(
                    "PPO selected under profile 'paper-baseline' but paper-grade gate failed. "
                    "Provide provenance metadata and quality gate fields in algo config.",
                )
        elif spec.tier != "baseline-ready":
            raise ValueError(
                f"Algorithm '{algo}' is {spec.tier} and blocked by profile 'paper-baseline'.",
            )

    if spec.requires_explicit_opt_in and not allow_testing_algorithms:
        raise ValueError(
            f"Algorithm '{algo}' is marked experimental-testing and blocked by default. "
            "Set 'allow_testing_algorithms: true' in the algo config for exploratory runs.",
        )

    return spec


def paper_baseline_algorithms() -> tuple[str, ...]:
    """Return the canonical publication profile algorithm set.

    The tuple stays literal until every family migrates to the contract
    registry (issue #7676). Registry records flagged ``paper_baseline_eligible``
    must already appear here; the parity gate below fails closed when a newly
    migrated eligible algorithm is missing from the publication profile.
    """
    paper_baselines = ("goal", "social_force", "orca", "ppo")
    registry_eligible = frozenset(
        record.canonical_name
        for record in MIGRATED_ALGORITHM_RECORDS
        if record.paper_baseline_eligible
    )
    missing = sorted(registry_eligible - set(paper_baselines))
    if missing:
        raise ValueError(
            f"Registry marks {missing} as paper-baseline-eligible but they are absent "
            "from paper_baseline_algorithms(); keep membership exact during migration."
        )
    return paper_baselines


__all__ = [
    "AlgorithmReadiness",
    "AlgorithmTier",
    "BenchmarkProfile",
    "get_algorithm_readiness",
    "paper_baseline_algorithms",
    "require_algorithm_allowed",
]
