"""Export-contract guard for the remaining ``robot_sf.sim`` modules.

Split child of #6477 (repository-wide ``__all__`` sweep), following the pattern
established by #6486 / PR #6762 for the facade and backends package. Guards the
reviewed ``__all__`` surfaces added by issue #6797:

- every declared export resolves to the pre-change object on its pre-change
  import path,
- no declared name is missing or misspelled,
- private, foreign, and stale names never leak into the public surface.
"""

from __future__ import annotations

import importlib

import pytest

SIM_FACADE_ALL = ["SimulatorFactory"]
FAST_PYSF_WRAPPER_ALL = ["FastPysfWrapper"]
PEDESTRIAN_MODEL_VARIANTS_ALL = [
    "HSFM_ALIGNMENT_TORQUE_V1",
    "HSFM_ANISOTROPIC_FOV_V1",
    "HSFM_TOTAL_FORCE_V1",
    "HSFM_TTC_PREDICTIVE_V1",
    "HSFM_ZANLUNGO_COLLISION_PREDICTION_V1",
    "PYSF_POSITION_SLICE",
    "PYSF_VELOCITY_SLICE",
    "SOCIAL_FORCE_DEFAULT",
    "SUPPORTED_PEDESTRIAN_MODELS",
    "anisotropic_fov_total_force",
    "anisotropic_fov_weights",
    "fov_attenuated_total_force",
    "heading_from_total_force",
    "normalize_pedestrian_model",
    "pairwise_fov_attenuated_forces",
    "pairwise_social_force_contributions",
    "pairwise_time_to_collision",
    "step_alignment_torque_heading",
    "step_hsfm_total_force",
    "ttc_predictive_repulsion",
    "wrap_to_pi",
    "zanlungo_collision_prediction_repulsion",
]
PEDESTRIAN_SPEED_TIERS_ALL = [
    "PED_SPEED_TIER_BRISK",
    "PED_SPEED_TIER_HIGH",
    "PED_SPEED_TIER_SLOW",
    "PED_SPEED_TIER_STD",
    "PED_SPEED_TIER_TYPICAL",
    "SUPPORTED_PED_SPEED_TIERS",
    "desired_speed_params_for_tier",
    "normalize_ped_speed_tier",
    "sample_desired_pedestrian_speeds",
]
REGISTRY_ALL = ["get_backend", "list_backends", "register_backend", "select_best_backend"]
SIM_CONFIG_ALL = [
    "AlignmentTorqueConfig",
    "AnisotropicFovConfig",
    "SimulationSettings",
    "TtcPredictiveForceConfig",
    "ZanlungoCollisionPredictionConfig",
]
SIMULATOR_ALL = [
    "PYSF_POSITION_SLICE",
    "PYSF_TAU_INDEX",
    "PYSF_VELOCITY_SLICE",
    "PedSimulator",
    "Simulator",
    "init_ped_simulators",
    "init_simulators",
]

# module_name -> (expected __all__, reviewed list)
REVIEWED_SURFACES: dict[str, list[str]] = {
    "robot_sf.sim.facade": SIM_FACADE_ALL,
    "robot_sf.sim.fast_pysf_wrapper": FAST_PYSF_WRAPPER_ALL,
    "robot_sf.sim.pedestrian_model_variants": PEDESTRIAN_MODEL_VARIANTS_ALL,
    "robot_sf.sim.pedestrian_speed_tiers": PEDESTRIAN_SPEED_TIERS_ALL,
    "robot_sf.sim.registry": REGISTRY_ALL,
    "robot_sf.sim.sim_config": SIM_CONFIG_ALL,
    "robot_sf.sim.simulator": SIMULATOR_ALL,
}


@pytest.mark.parametrize("module_name", sorted(REVIEWED_SURFACES))
def test_module_declares_the_reviewed_export_surface(module_name: str) -> None:
    """Each module exports exactly its reviewed ``__all__`` list."""
    module = importlib.import_module(module_name)
    assert module.__all__ == REVIEWED_SURFACES[module_name]
    assert set(module.__all__) <= set(dir(module))


@pytest.mark.parametrize(
    ("module_name", "name"),
    [(module_name, name) for module_name, names in REVIEWED_SURFACES.items() for name in names],
)
def test_module_exports_resolve_on_pre_change_paths(module_name: str, name: str) -> None:
    """Every declared export resolves to the pre-change module-level binding."""
    module = importlib.import_module(module_name)
    assert getattr(module, name) is module.__dict__[name]


@pytest.mark.parametrize(
    ("module_name", "name"),
    [
        ("robot_sf.sim.facade", "_assert_fast_pysf_initialized"),  # facade guard, not facade.py
        ("robot_sf.sim.fast_pysf_wrapper", "pysf"),  # module-level import, not public
        ("robot_sf.sim.pedestrian_model_variants", "_validate_ttc_inputs"),  # private helper
        ("robot_sf.sim.pedestrian_speed_tiers", "_PED_SPEED_TIER_PARAMS"),  # private map
        ("robot_sf.sim.registry", "_REGISTRY"),  # private registry state
        ("robot_sf.sim.sim_config", "_normalize_ttc_predictive_force_config"),  # private
        ("robot_sf.sim.simulator", "_heading_from_velocity"),  # private helper
    ],
)
def test_module_keeps_private_and_foreign_names_unexported(module_name: str, name: str) -> None:
    """Private, foreign, and stale symbols stay out of every export list."""
    module = importlib.import_module(module_name)
    assert name not in module.__all__
