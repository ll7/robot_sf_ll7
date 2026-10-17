"""Seed admission around the byte-pinned lane-formation reference implementation.

Repository execution uses this facade. Direct external calls to the original
module remain unguarded so that the preregistration's exact source pin is kept.
"""

from __future__ import annotations

from typing import Any

from robot_sf.benchmark.runtime_seed_guard import check_seed_config, check_simulation_seed
from robot_sf.research import lane_formation_reference as _reference
from robot_sf.research.emergent_phenomena import SpeedCalibration, released_default_config
from robot_sf.research.lane_formation_reference import ReferenceProtocol

__all__ = ["run_native_reference", "run_reference_campaign"]


def _check_scene_seeds(sim_config: Any | None, *, boundary: str) -> None:
    config = sim_config if sim_config is not None else released_default_config()
    check_seed_config(config, boundary=boundary)
    check_seed_config(getattr(config, "scene_config", None), boundary=f"{boundary}.scene")


def run_native_reference(
    *,
    protocol: ReferenceProtocol,
    condition: str,
    seed: int,
    calibration: SpeedCalibration,
    sampling_strides: tuple[int, ...] | list[int] = _reference.DEFAULT_SAMPLING_STRIDES,
    sim_config: Any | None = None,
) -> dict[str, Any]:
    """Admit every seed before delegating one native reference run.

    Returns:
        The original implementation's unmodified reference row.
    """
    check_simulation_seed(seed, boundary="run_native_reference")
    _check_scene_seeds(sim_config, boundary="reference")
    return _reference.run_native_reference(
        protocol=protocol,
        condition=condition,
        seed=seed,
        calibration=calibration,
        sampling_strides=sampling_strides,
        sim_config=sim_config,
    )


def run_reference_campaign(
    *,
    protocol: ReferenceProtocol = ReferenceProtocol(),
    seeds: tuple[int, ...] | list[int] = _reference.DEFAULT_REFERENCE_SEEDS,
    conditions: tuple[str, ...] | list[str] = _reference.DEFAULT_REFERENCE_CONDITIONS,
    calibrations: tuple[SpeedCalibration, ...]
    | list[SpeedCalibration] = _reference.DEFAULT_REFERENCE_CALIBRATIONS,
    sampling_strides: tuple[int, ...] | list[int] = _reference.DEFAULT_SAMPLING_STRIDES,
    sim_config: Any | None = None,
) -> dict[str, Any]:
    """Admit the entire campaign before its first reference job.

    Returns:
        The original implementation's unmodified campaign payload.
    """
    for seed in seeds:
        check_simulation_seed(seed, boundary="run_reference_campaign")
    _check_scene_seeds(sim_config, boundary="reference campaign")
    return _reference.run_reference_campaign(
        protocol=protocol,
        seeds=seeds,
        conditions=conditions,
        calibrations=calibrations,
        sampling_strides=sampling_strides,
        sim_config=sim_config,
    )
