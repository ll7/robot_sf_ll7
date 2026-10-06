"""Seed admission around the byte-pinned historical Stage A parameter screen.

Current campaigns use this facade; the original module remains a historical
preregistration input and is not a guarded external execution API.
"""

from __future__ import annotations

from typing import Any

from robot_sf.benchmark.runtime_seed_guard import check_simulation_seed
from robot_sf.research import lane_formation_parameter_screen as _screen
from robot_sf.research.lane_formation_reference import ReferenceProtocol

__all__ = ["run_parameter_screen"]


def run_parameter_screen(
    *,
    protocol: ReferenceProtocol = ReferenceProtocol(),
    seeds: tuple[int, ...] | list[int] = _screen.DEFAULT_REFERENCE_SEEDS,
    n_profiles: int = _screen.DEFAULT_PARAMETER_SCREEN_PROFILES,
    profile_seed: int = 6969,
    sampling_strides: tuple[int, ...] | list[int] = _screen.DEFAULT_SAMPLING_STRIDES,
) -> dict[str, Any]:
    """Admit every seed before profile RNG or the first historical native job.

    Returns:
        The unmodified parameter-screen payload for allowed seeds.
    """
    for seed in seeds:
        check_simulation_seed(seed, boundary="run_parameter_screen")
    check_simulation_seed(profile_seed, boundary="run_parameter_screen.profile_seed")
    return _screen.run_parameter_screen(
        protocol=protocol,
        seeds=seeds,
        n_profiles=n_profiles,
        profile_seed=profile_seed,
        sampling_strides=sampling_strides,
    )
