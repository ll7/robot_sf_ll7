"""Baseline navigation algorithms for benchmarking."""

from __future__ import annotations

import importlib


def _get_social_force_planner():
    """Lazy import to avoid circular dependencies.

    Returns:
        The SocialForcePlanner class.
    """
    module = importlib.import_module("robot_sf.baselines.social_force")
    return module.SocialForcePlanner


def _get_ppo_planner():
    """Lazy import for PPO baseline adapter.

    Returns:
        The PPOPlanner class.
    """
    module = importlib.import_module("robot_sf.baselines.ppo")
    return module.PPOPlanner


def _get_sac_planner():
    """Lazy import for SAC baseline adapter.

    Returns:
        The SACPlanner class.
    """
    module = importlib.import_module("robot_sf.baselines.sac")
    return module.SACPlanner


def _get_random_planner():
    """Lazy import for Random baseline.

    Returns:
        The RandomPlanner class.
    """
    module = importlib.import_module("robot_sf.baselines.random_policy")
    return module.RandomPlanner


def _get_sicnav_planner():
    """Lazy import for SICNav baseline adapter.

    Returns:
        The SICNavPlanner class.
    """
    module = importlib.import_module("robot_sf.baselines.sicnav")
    return module.SICNavPlanner


def _get_drm_mp_planner():
    """Lazy import for DR-MPC baseline adapter.

    Returns:
        The DRMPCPlanner class.
    """
    module = importlib.import_module("robot_sf.baselines.dr_mpc")
    return module.DRMPCPlanner


def _get_drl_vo_planner():
    """Lazy import for DRL-VO baseline.

    Returns:
        The DrlVoPlanner class.
    """
    module = importlib.import_module("robot_sf.baselines.drl_vo")
    return module.DrlVoPlanner


def _get_brne_planner():
    """Lazy import for BRNE baseline adapter.

    Returns:
        The BRNEPlanner class.
    """
    module = importlib.import_module("robot_sf.baselines.brne")
    return module.BRNEPlanner


# Registry of available baseline algorithms
BASELINES: dict[str, type] = {
    "social_force": _get_social_force_planner,
    "baseline_sf": _get_social_force_planner,  # Backward compatibility
    "ppo": _get_ppo_planner,
    "sac": _get_sac_planner,
    "random": _get_random_planner,
    "sicnav": _get_sicnav_planner,
    "dr_mpc": _get_drm_mp_planner,
    "drl_vo": _get_drl_vo_planner,
    "brne": _get_brne_planner,
}


def get_baseline(name: str) -> type:
    """Get a baseline algorithm class by name.

    Args:
        name: Name of the baseline algorithm

    Returns:
        The baseline class

    Raises:
        KeyError: If the baseline name is not found
    """
    if name not in BASELINES:
        available = list(BASELINES.keys())
        raise KeyError(f"Unknown baseline '{name}'. Available: {available}")

    baseline_factory = BASELINES[name]
    if callable(baseline_factory) and not isinstance(baseline_factory, type):
        # It's a factory function, call it to get the class
        return baseline_factory()
    return baseline_factory


def list_baselines() -> list[str]:
    """List available baseline algorithm names.

    Returns:
        List of baseline algorithm names.
    """
    return list(BASELINES.keys())


# Aliases for the built-in naive goal-seeking policy that the benchmark runner
# constructs directly (see ``robot_sf.benchmark.runner._create_robot_policy``),
# outside the baseline registry above. Kept aligned with the ``goal`` canonical
# planner declared in ``robot_sf.benchmark.algorithm_readiness``.
SIMPLE_POLICY_ALIASES: frozenset[str] = frozenset(
    {"simple_policy", "goal", "simple", "goal_policy"}
)


def is_runnable_algo(name: str) -> bool:
    """Return whether ``robot_sf.benchmark.runner.run_episode`` can execute ``name``.

    The benchmark episode runner resolves a planner either as the built-in
    goal-seeking policy (``SIMPLE_POLICY_ALIASES``) or through the baseline
    registry (``BASELINES``). Planners that live only in other execution
    pipelines (e.g. the map-runner holonomic world-velocity ORCA path backed by
    upstream RVO2) are not reachable here.

    Returns:
        True when the algorithm can be constructed by ``run_episode``.
    """
    return name in SIMPLE_POLICY_ALIASES or name in BASELINES


__all__ = [
    "BASELINES",
    "SIMPLE_POLICY_ALIASES",
    "get_baseline",
    "is_runnable_algo",
    "list_baselines",
]
