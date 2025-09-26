"""Baseline navigation algorithms for benchmarking."""


def _get_social_force_planner():
    """Lazy import to avoid circular dependencies."""
    from robot_sf.baselines.social_force import SocialForcePlanner

    return SocialForcePlanner


def _get_ppo_planner():
    """Lazy import for PPO baseline adapter."""
    from robot_sf.baselines.ppo import PPOPlanner

    return PPOPlanner


def _get_random_planner():
    """Lazy import for Random baseline."""
    from robot_sf.baselines.random_policy import RandomPlanner

    return RandomPlanner


# Registry of available baseline algorithms
BASELINES: dict[str, type] = {
    "social_force": _get_social_force_planner,
    "baseline_sf": _get_social_force_planner,  # Backward compatibility
    "ppo": _get_ppo_planner,
    "random": _get_random_planner,
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
    """List available baseline algorithm names."""
    return list(BASELINES.keys())


__all__ = ["BASELINES", "get_baseline", "list_baselines"]
