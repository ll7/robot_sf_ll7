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


__all__ = ["BASELINES", "get_baseline", "list_baselines"]
