"""Opt-in simulator truth instrumentation tests for issue #8065.

These tests establish implementation integrity only. They do not estimate goals,
rank inverse-force methods, or provide benchmark or paper-facing evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.ped_npc.adversial_ped_force import AdversarialPedForceConfig
from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig
from robot_sf.ped_npc.residual_adversary import ResidualAdversaryConfig
from robot_sf.prediction.oracle_transition_trace import (
    ForceOperationKind,
    OracleTransitionTraceV1,
)
from robot_sf.sim.pedestrian_model_variants import (
    HSFM_ALIGNMENT_TORQUE_V1,
    HSFM_ANISOTROPIC_FOV_V1,
    HSFM_TOTAL_FORCE_V1,
    HSFM_TTC_PREDICTIVE_V1,
    HSFM_ZANLUNGO_COLLISION_PREDICTION_V1,
)
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import init_simulators
from tests.sim.test_residual_adversary_wiring import _minimal_map


def _build_simulator(
    *,
    oracle_enabled: bool,
    residual_enabled: bool = False,
    pedestrian_model: str | None = None,
    robot_force_enabled: bool = False,
):
    """Build the deterministic one-pedestrian fixture used by the runtime tests."""
    map_def = _minimal_map()
    sim_config = SimulationSettings(
        sim_time_in_secs=4.0,
        time_per_step_in_secs=0.1,
        difficulty=0,
        ped_density_by_difficulty=[0.02, 0.02, 0.02, 0.02],
        population_size=1,
        route_spawn_distribution="spread",
        route_spawn_seed=0,
        oracle_force_trace_enabled=oracle_enabled,
        prf_config=PedRobotForceConfig(is_active=robot_force_enabled),
        apf_config=AdversarialPedForceConfig(is_active=robot_force_enabled),
        residual_adversary=ResidualAdversaryConfig(
            is_active=residual_enabled,
            target_ped_idx=-1,
            max_residual_accel_mps2=1.0,
            max_jerk_mps3=1e9,
        ),
    )
    if pedestrian_model is not None:
        sim_config.pedestrian_model = pedestrian_model
        sim_config.__post_init__()
    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=sim_config,
    )
    return init_simulators(
        config,
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]


def test_oracle_trace_is_opt_in_and_reset_clears_privileged_state() -> None:
    """The trace appears only when enabled and never survives an episode reset."""
    sim_off = _build_simulator(oracle_enabled=False)
    sim_off.step_once([(0.0, 0.0)])
    assert sim_off.last_force_computation is None
    assert sim_off.last_step_diagnostics is None
    assert sim_off.last_oracle_transition_traces is None
    assert sim_off.oracle_force_trace_payload is None

    sim = _build_simulator(oracle_enabled=True)
    sim.step_once([(0.0, 0.0)])

    assert sim.last_force_computation is not None
    assert sim.last_step_diagnostics is not None
    assert sim.last_oracle_transition_traces is not None
    payload = sim.oracle_force_trace_payload
    assert payload is not None
    assert payload["schema_version"] == "oracle_transition_trace.v1"
    assert len(payload["transitions"]) == 1

    trace = OracleTransitionTraceV1.from_dict(payload["transitions"][0])
    assert trace.backend == "pysocialforce"
    assert trace.force_components.component_records
    assert trace.speed_cap.status.value == "not_applied"
    assert trace.force_components.final_pre_cap_force_xy is not None
    np.testing.assert_array_equal(
        sim.last_force_computation.base_total,
        np.asarray(sim.last_ped_forces, dtype=float),
    )

    sim.reset_state()
    assert sim.last_force_computation is None
    assert sim.last_step_diagnostics is None
    assert sim.last_oracle_transition_traces is None
    assert sim.oracle_force_trace_payload is None


def test_oracle_component_records_sum_to_the_force_used_by_integration() -> None:
    """Every low-level component is serialized with stable identity and exact values."""
    sim = _build_simulator(oracle_enabled=True)
    sim.step_once([(0.0, 0.0)])

    result = sim.last_force_computation
    traces = sim.last_oracle_transition_traces
    assert result is not None
    assert traces is not None
    assert len(traces) == 1
    records = traces[0].force_components.component_records
    assert [record.evaluation_order for record in records] == list(range(len(records)))
    assert len({record.component_id for record in records}) == len(records)
    component_sum = np.sum(
        np.asarray([record.force_xy for record in records], dtype=float),
        axis=0,
    )
    np.testing.assert_array_equal(component_sum, result.base_total[0])
    assert traces[0].force_components.registry_total_force_xy == tuple(result.base_total[0])


def test_oracle_residual_stage_records_additive_force_fold() -> None:
    """An enabled residual adversary is represented as a post-registry additive stage."""
    sim = _build_simulator(oracle_enabled=True, residual_enabled=True)
    sim.step_once([(0.0, 0.0)])

    result = sim.last_force_computation
    traces = sim.last_oracle_transition_traces
    assert result is not None
    assert traces is not None
    stage = traces[0].force_components.residual_operation
    assert stage.operation_kind is ForceOperationKind.ADDITIVE
    assert stage.delta_force_xy is not None
    assert stage.result_force_xy is not None
    expected = np.asarray(result.base_total[0]) + np.asarray(stage.delta_force_xy)
    np.testing.assert_allclose(stage.result_force_xy, expected)
    assert traces[0].force_components.final_pre_cap_force_xy == tuple(stage.result_force_xy)


def test_oracle_trace_keeps_robot_force_instances_separate() -> None:
    """Robot-aware force instances retain source identity in the exact roster."""
    sim = _build_simulator(oracle_enabled=True, robot_force_enabled=True)
    sim.step_once([(0.0, 0.0)])

    traces = sim.last_oracle_transition_traces
    assert traces is not None
    records = traces[0].force_components.component_records
    robot_records = [
        record for record in records if record.component_type in {"pedestrian_robot", "adversarial"}
    ]
    assert [(record.component_id, record.source_entity) for record in robot_records] == [
        ("ped_robot:robot_0", "robot:0"),
        ("adversarial:robot_0", "robot:0"),
    ]
    assert all(record.actor_observable is False for record in robot_records)


def test_oracle_instrumentation_preserves_disabled_trajectory_bit_for_bit() -> None:
    """Enabling diagnostics must not alter the default simulation trajectory."""
    sim_off = _build_simulator(oracle_enabled=False)
    sim_on = _build_simulator(oracle_enabled=True)

    for _ in range(5):
        sim_off.step_once([(0.0, 0.0)])
        sim_on.step_once([(0.0, 0.0)])
        np.testing.assert_array_equal(
            sim_off.pysf_state.pysf_states(),
            sim_on.pysf_state.pysf_states(),
        )
        np.testing.assert_array_equal(sim_off.last_ped_forces, sim_on.last_ped_forces)


@pytest.mark.parametrize(
    "pedestrian_model",
    [
        HSFM_TOTAL_FORCE_V1,
        HSFM_TTC_PREDICTIVE_V1,
        HSFM_ZANLUNGO_COLLISION_PREDICTION_V1,
        HSFM_ANISOTROPIC_FOV_V1,
        HSFM_ALIGNMENT_TORQUE_V1,
    ],
)
def test_oracle_trace_records_supported_model_variant_stage(pedestrian_model: str) -> None:
    """Every opt-in HSFM variant exposes a typed final stage without changing integration."""
    sim = _build_simulator(oracle_enabled=True, pedestrian_model=pedestrian_model)
    sim.step_once([(0.0, 0.0)])

    traces = sim.last_oracle_transition_traces
    assert traces is not None
    stage = traces[0].force_components.model_variant_operation
    assert stage.operation_kind.value in {"additive", "transformed"}
    assert stage.result_force_xy is not None
    assert traces[0].force_components.final_pre_cap_force_xy == tuple(stage.result_force_xy)
