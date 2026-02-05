"""Compatibility wrapper for the refactored pedestrian environment.

This module preserves the historical import path while re-exporting the
refactored implementation from ``pedestrian_env_refactored``.
"""

from robot_sf.gym_env.pedestrian_env_refactored import RefactoredPedestrianEnv

PedestrianEnv = RefactoredPedestrianEnv

__all__ = ["PedestrianEnv"]
