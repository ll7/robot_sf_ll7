"""Baseline algorithms for the Social Navigation Benchmark.

This module provides baseline implementations of navigation algorithms
that can be used for comparison in the benchmark system.
"""

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from robot_sf.baselines.social_force import SocialForcePlanner

def _get_social_force_planner():
    """Lazy import to avoid circular dependencies."""
    from robot_sf.baselines.social_force import SocialForcePlanner
    return SocialForcePlanner

# Registry of available baseline algorithms
BASELINES: Dict[str, Type] = {
    "baseline_sf": _get_social_force_planner,
}


def get_baseline(name: str) -> Type:
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