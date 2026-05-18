"""Regression tests for the consolidated pedestrian environment module."""

from pathlib import Path

from robot_sf.gym_env.pedestrian_env import PedestrianEnv


def test_pedestrian_env_is_defined_in_canonical_module() -> None:
    """PedestrianEnv should no longer be a shim around a refactored module."""
    assert PedestrianEnv.__module__ == "robot_sf.gym_env.pedestrian_env"


def test_refactored_pedestrian_env_file_removed() -> None:
    """The transition-only `_refactored` implementation file should be gone."""
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "robot_sf/gym_env/pedestrian_env_refactored.py").exists()
