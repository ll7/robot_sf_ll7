"""Readiness metadata and profile guards for benchmark algorithm selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AlgorithmTier = Literal["baseline-ready", "experimental", "placeholder"]
BenchmarkProfile = Literal["baseline-safe", "paper-baseline", "experimental"]


@dataclass(frozen=True)
class AlgorithmReadiness:
    """Readiness metadata for one canonical map-benchmark algorithm."""

    canonical_name: str
    tier: AlgorithmTier
    aliases: tuple[str, ...]
    note: str


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
    AlgorithmReadiness(
        canonical_name="orca",
        tier="baseline-ready",
        aliases=("orca",),
        note="ORCA baseline (requires rvo2 or explicit fallback policy).",
    ),
    AlgorithmReadiness(
        canonical_name="ppo",
        tier="experimental",
        aliases=("ppo",),
        note="Learned PPO baseline (paper profile requires provenance + quality gate).",
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
        canonical_name="prediction_planner",
        tier="experimental",
        aliases=("prediction_planner", "predictive", "prediction"),
        note="RGL-inspired predictive planner; requires trained checkpoint.",
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
        tier="placeholder",
        aliases=("dwa",),
        note="Placeholder adapter; not benchmark-validated.",
    ),
    AlgorithmReadiness(
        canonical_name="teb",
        tier="placeholder",
        aliases=("teb",),
        note="Placeholder adapter; not benchmark-validated.",
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

    return spec


def paper_baseline_algorithms() -> tuple[str, ...]:
    """Return the canonical publication profile algorithm set."""
    return ("goal", "social_force", "orca", "ppo")


__all__ = [
    "AlgorithmReadiness",
    "AlgorithmTier",
    "BenchmarkProfile",
    "get_algorithm_readiness",
    "paper_baseline_algorithms",
    "require_algorithm_allowed",
]
